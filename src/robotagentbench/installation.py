"""Assemble bundled source and versioned data into a fresh local task workspace."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import resources
from .backends import sources


def _index(manifest, assets):
    name=resources._relative(manifest.get('index','resource-index.json'))
    records = [a['members'][name] for a in manifest['archives'].values()
               if name in a['members']]
    if len(records) != 1:
        raise ValueError('Exactly one pinned resource index is required')
    path = assets / name
    resources._verify_file(path, records[0])
    value = json.loads(path.read_text())
    if (value.get('schema') != 'robotagentbench.asset_index.v1'
            or value.get('version') != manifest['version']
            or not isinstance(value.get('objects'), dict)
            or not isinstance(value.get('locations'), dict)):
        raise ValueError('Code and asset index versions differ')
    for digest, record in value['objects'].items():
        resources._digest(digest)
        if (set(record) != {'bytes', 'sha256', 'path'} or record['sha256'] != digest
                or record['path'] != 'objects/' + digest[:2] + '/' + digest):
            raise ValueError('Invalid indexed content address')
        resources._size(record['bytes'])
    for location, digest in value['locations'].items():
        resources._relative(location)
        if digest not in value['objects']:
            raise ValueError('Resource index references an absent object')
    return value


def _canonical_source(path):
    for prefix in ('api-protocol/resources/', 'api-protocol/all_assets/'):
        if path.startswith(prefix):
            return path[len('api-protocol/'):]
    return path


def _source_identity(path, profile):
    """Restore source-record names below a local root, never at host absolutes."""
    mounts=profile['compatibility_namespaces']
    choices=[row for row in mounts if path==row['source'] or path.startswith(row['source']+'/')]
    if not choices:
        return path
    row=max(choices,key=lambda item:len(item['source']))
    original=row['target'].lstrip('/')+path[len(row['source']):]
    resources._relative(original)
    return original


def configure(manifest, assets, destination, profile):
    """No SDK imports, global mounts, package installs, or physical parameters."""
    source_manifest = sources._manifest()
    if profile not in source_manifest['profiles']:
        raise ValueError('Unknown task backend profile')
    assets, destination = Path(assets).absolute(), Path(destination).absolute()
    if assets.resolve() != assets or destination.parent.resolve() != destination.parent:
        raise ValueError('Canonical installation paths required')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Fresh runtime destination required')
    if shutil.disk_usage(destination.parent).free < 512 * 1024**2:
        raise ValueError('Insufficient runtime reserve')
    index = _index(manifest, assets)
    sources.verify(profile)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=destination.parent))
    source_relative = ('short_tasks/agent_probe_v1/deployment_release/source'
                       if profile == 'room' else 'sources/' + profile)
    source_root = stage / source_relative
    source_root.mkdir(parents=True)
    bindings, file_pins, priorities, checked = {}, {}, {}, set()

    def copy_file(source, relative, digest, priority):
        resources._relative(relative)
        target = stage / relative
        if relative in file_pins:
            if priorities[relative] > priority:
                return
            if priorities[relative] == priority and file_pins[relative] != digest:
                raise ValueError('Different resource versions share a runtime path')
            if file_pins[relative] == digest:
                return
            # Only explicit profile data can replace its older library alias.
            # The source content objects themselves remain immutable.
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target) if priority == 100 else _link_or_copy(source, target)
        file_pins[relative] = digest
        priorities[relative] = priority

    # Code is copied, not linked to the installed package. Original bytes stay
    # intact and generated task files cannot modify the package installation.
    for name, record in source_manifest['profiles'][profile]['files'].items():
        canonical = (_canonical_source(name) if profile == 'room' else
                     _source_identity(name, source_manifest['profiles'][profile]))
        relative = source_relative + '/' + canonical
        copy_file(sources.root(profile) / name, relative, record['sha256'], 100)
    for location, digest in sorted(index['locations'].items()):
        target = None
        priority = 10
        if location.startswith('libraries/'):
            target = location
        prefix = 'profiles/' + profile + '/'
        if location.startswith(prefix):
            part = location[len(prefix):]
            canonical=(_canonical_source(part) if profile=='room'else
                       _source_identity(part, source_manifest['profiles'][profile]))
            target = source_relative + '/' + canonical
            priority = 20
        elif profile == 'room' and location.startswith('libraries/generic_current/'):
            part = _canonical_source(location[len('libraries/generic_current/'):])
            # Also retain the library location for explicit namespace bindings.
            target = source_relative + '/' + part
        elif profile == 'room' and location.startswith('libraries/current_assets/'):
            part = _canonical_source(location[len('libraries/current_assets/'):])
            target = source_relative + '/' + part
            priority = 15
        elif profile == 'room' and location.startswith(
                'libraries/original_absolute/home/lingjun/robot_agent_arena/'):
            part = location[len('libraries/original_absolute/home/lingjun/robot_agent_arena/'):]
            # These two approved scene families are mounted from the local
            # source tree by namespace.command(). Retain their original bytes
            # there as well as in the content-addressed library; otherwise the
            # more-specific mount hides the library's room and sublayer files.
            if part.startswith(('layout_task_activation_v1/',
                                'openend/tasks/all_objects_integration/')):
                target = source_relative + '/' + part
        if target is None:
            continue
        record = index['objects'][digest]
        source = assets / record['path']
        if digest not in checked:
            resources._verify_file(source, record)
            checked.add(digest)
        if location.startswith('libraries/') and target != location:
            copy_file(source, location, digest, 10)
        copy_file(source, target, digest, priority)
        bindings[location] = str(destination / target)
    if profile == 'room':
        # Original shelf helpers derive sibling import locations from __file__.
        # Assemble their exact code/native payloads into the already read-only
        # legacy tree, rather than mounting a new directory over its assets.
        prefix=source_relative+'/vendor/shelf_generation/'
        for path,digest in list(file_pins.items()):
            if path.startswith(prefix):
                alias='libraries/original_absolute/mnt/agent-bench-data/lingjun/ziche/'+path[len(prefix):]
                copy_file(stage/path,alias,digest,100)
        # Preserve the original preparation API's task-local aliases. These
        # point only within this newly created runtime, never outside it.
        for name in ('resources', 'all_assets'):
            (source_root / name).mkdir(exist_ok=True)
            alias = source_root / 'api-protocol' / name
            alias.parent.mkdir(parents=True, exist_ok=True)
            if alias.exists() or alias.is_symlink():
                raise ValueError('Unexpected noncanonical source asset alias')
            alias.symlink_to(Path('..') / name, target_is_directory=True)
    selected_profile=source_manifest['profiles'][profile]
    logical=[dict(target=row['target'],source_relative=_source_identity(row['source'],selected_profile))
             for row in selected_profile['compatibility_namespaces']]
    value = dict(schema='robotagentbench.local_installation.v1', version=manifest['version'],
                 profile=profile, assets=str(assets), runtime=str(destination),
                 source_root=str(destination / source_relative),
                 source_files={path[len(source_relative)+1:]:digest for path,digest in file_pins.items()
                               if path.startswith(source_relative + '/')},
                 library_files={path:digest for path,digest in file_pins.items()
                                if path.startswith('libraries/')},
                 resources=bindings, compatibility_namespaces=logical,
                 container_only_compatibility_paths=True,
                 native_environment='Configure the separate Isaac Sim interpreter and libraries.',
                 parameter_overrides={})
    with (stage / 'runtime.json').open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Runtime destination appeared; partial staging retained')
    os.rename(stage, destination)
    return dict(profile=profile, version=manifest['version'], runtime=str(destination),
                source_root=value['source_root'], source_files=len(value['source_files']),
                resource_bindings=len(bindings), native_started=False)


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)

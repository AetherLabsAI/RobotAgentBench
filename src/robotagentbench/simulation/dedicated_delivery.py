"""CPU-only delivery bindings for the original dedicated task owners.

This produces a namespace PLAN, not a simulator or an admission override. A
supervisor must mount the plan in a fresh container, restore the selected saved
instance there, then seal its generated input directory read-only before native
execution. No original absolute path is created on the host.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from robotagentbench import installation, resources
from .saved_instance_release.restore import select

HERE = Path(__file__).resolve().parent
TEMPLATE_SHA = 'fec6ffbfabab1eeb642311f4096c87bb0086ce950a59ba74a5488fb0cb2ad7f4'
CLOTH_TEMPLATE_SHA = '21097722991dcf616e257da3aaf8fb6aba9815baf70510c1dfd8b313630922b6'
LEGACY_GENERIC = '/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source'
RECOVERY_HOSTS = frozenset(('VM-0-8-ubuntu', 'VM-0-10-ubuntu', 'VM-0-13-ubuntu',
                           'VM-0-11-ubuntu', 'VM-0-9-ubuntu'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate configuration key')
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical(path):
    value = Path(path).absolute()
    if value.resolve(strict=True) != value:
        raise ValueError('Canonical existing input required')
    return value


def template():
    path = HERE / 'dedicated_delivery_template.json'
    if sha(path) != TEMPLATE_SHA:
        raise ValueError('Reviewed delivery role/hash template changed')
    return read(path)


def runtime(path, profile):
    base = canonical(path)
    record = read(base / 'runtime.json')
    source = canonical(record['source_root'])
    if (record.get('schema') != 'robotagentbench.local_installation.v1'
            or record.get('profile') != profile or record.get('runtime') != str(base)
            or not source.is_relative_to(base)):
        raise ValueError('Matching freshly configured Runtime required')
    return base, source, record


def native(value):
    """Verify the caller's separate native install, without importing the SDK."""
    from .dependencies import RuntimeDependencies
    from .environment import NativeEnvironment
    if set(value) != {'dependencies', 'profile'}:
        raise ValueError('Explicit native dependency and environment profile required')
    deps = RuntimeDependencies.from_config(value['dependencies'])
    NativeEnvironment.from_config(deps, value['profile']).verify()
    return deepcopy(value)


def fresh(path):
    out = Path(path).absolute()
    if out.resolve() != out or out.exists() or not out.parent.is_dir():
        raise ValueError('Fresh output beneath an existing canonical directory required')
    out.mkdir()
    return out


def merge_environment(dependencies, value):
    """Rename only dictionary keys; retain exact supplied paths and digests."""
    old = value['dependencies']
    roots = {key: 'native_' + key for key in old['roots']}
    ids = {key: 'native_' + key for key in old['resources']}
    dependencies['roots'].update({roots[k]: v for k, v in old['roots'].items()})
    dependencies['resources'].update({ids[k]: dict(v, root=roots[v['root']])
                                      for k, v in old['resources'].items()})
    profile = deepcopy(value['profile'])
    profile['binary'] = ids[profile['binary']]
    profile['required_resources'] = [ids[k] for k in profile['required_resources']]
    for kind in ('metadata_paths', 'python_paths', 'library_paths', 'plugin_paths',
                 'asset_paths', 'material_paths'):
        profile[kind] = [dict(row, root=roots[row['root']]) for row in profile[kind]]
    return profile


def mountpoint_scaffolds(rows, output):
    """Prepare missing nested bind targets without writing source installations.

    Linux cannot create a child bind mountpoint inside an already read-only
    parent bind. For such parents, make a fresh byte-identical namespace view
    with empty mountpoint placeholders. It remains read-only at execution;
    original Runtime files and their identity/pins are untouched.
    """
    out = canonical(output)
    rows = deepcopy(rows)
    scaffolds = []
    for row in rows:
        source, target = Path(row['source']), Path(row['target'])
        if not row['read_only'] or not source.is_dir():
            continue
        children = [r for r in rows if Path(r['target']) != target
                    and Path(r['target']).is_relative_to(target)]
        missing = [r for r in children if not (source / Path(r['target']).relative_to(target)).exists()]
        # The native lock's exact device/host name is known only to the later
        # supervisor. Reserve a fresh view of its source parent now.
        has_owner_lock = target == Path(LEGACY_GENERIC)
        if not missing and not has_owner_lock:
            continue
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Namespace source view may not contain symbolic links')
        destination = out / 'namespace_scaffolds' / hashlib.sha256(str(target).encode()).hexdigest()
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(source, destination)
        for original in source.rglob('*'):
            if original.is_file() and sha(original) != sha(destination / original.relative_to(source)):
                raise ValueError('Namespace source copy differs')
        for child in children:
            slot = destination / Path(child['target']).relative_to(target)
            if Path(child['source']).is_dir():
                slot.mkdir(parents=True, exist_ok=True)
            else:
                slot.parent.mkdir(parents=True, exist_ok=True)
                if not slot.exists():
                    slot.touch(exist_ok=False)
        row['source'] = str(destination)
        scaffolds.append(dict(source=str(source), view=str(destination), target=str(target)))
    return rows, scaffolds


def prepare_mountpoints(plan_path):
    """Upgrade an unexecuted CPU plan using only its fresh owned directory."""
    path = canonical(plan_path)
    plan = read(path)
    if (plan.get('schema') != 'robotagentbench.dedicated_namespace_plan.v1'
            or plan.get('family') not in ('atomic_coffee', 'failure_recovery')
            or plan.get('execution_ready') is not False or plan.get('namespace_scaffolds')):
        raise ValueError('An unexecuted, unscaffolded dedicated plan is required')
    plan['mounts'], plan['namespace_scaffolds'] = mountpoint_scaffolds(plan['mounts'], path.parent)
    target = path.parent / 'namespace-scaffold-plan.json'
    save(target, plan)
    return target


def external_lock_namespace(rows, lock_root, output):
    """The supervisor owns the host global lease, never a whole RW source bind.

    Original dedicated owners only acquire their narrow local locks. Their
    workspace still expects the configured global-lock directory to exist;
    retain its read-only source view, or an empty read-only placeholder if the
    family does not include that logical directory.
    """
    lock = Path(lock_root)
    rows = [r for r in deepcopy(rows) if not (Path(r['target']) == lock and not r['read_only'])]
    parents = [r for r in rows if lock.is_relative_to(Path(r['target'])) and Path(r['source']).is_dir()]
    if parents:
        parent = max(parents, key=lambda r: len(Path(r['target']).parts))
        if (Path(parent['source']) / lock.relative_to(parent['target'])).is_dir():
            return rows
    directory = canonical(output) / 'external_lock_namespace'
    directory.mkdir()
    rows.append(dict(source=str(directory), target=str(lock), read_only=True))
    return rows


def owner_lock_bindings(plan_path, task, *, hostname, gpu_uuid):
    """Create only the original native owner's single narrow lock-file bind.

    The supervisor supplies its already measured host/device identity. This
    function never queries a GPU, acquires the global lease, or starts a worker.
    A fresh backing is allowed only below this plan's owned output directory.
    It does not replace the external shared GPU lease or make execution ready.
    """
    from .specialized import config
    path = canonical(plan_path)
    plan = read(path)
    if (plan.get('schema') != 'robotagentbench.dedicated_namespace_plan.v1'
            or plan.get('family') not in config.MODULES or plan.get('execution_ready') is not False
            or Path(plan['operator']) != path.parent / 'operator.json'):
        raise ValueError('An unstarted dedicated plan is required')
    operator = config.read_operator(plan['operator'])
    config.selection(operator, plan['family'], task)
    if (hostname not in RECOVERY_HOSTS or not isinstance(gpu_uuid, str)
            or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu_uuid)):
        raise ValueError('Use the real approved host and measured full GPU UUID')
    if plan['family'] == 'atomic_coffee':
        target = Path(LEGACY_GENERIC) / 'short_tasks/agent_probe_v1' / ('gpu-' + gpu_uuid + '.lock')
    elif task in ('failure_recovery_shelf_drink', 'failure_recovery_coffee_bag'):
        target = Path(LEGACY_GENERIC) / 'short_task_robustness/tasks/grasp_failure_recovery' / (
            hostname + '-gpu' + str(operator['gpu']) + '-native.lock')
    else:
        return []  # The selected sofa/cup owners contain no separate flock.
    parents = [r for r in plan.get('namespace_scaffolds', [])
               if target.is_relative_to(Path(r['target']))]
    if parents:
        parent = max(parents, key=lambda r: len(Path(r['target']).parts))
        view = canonical(parent['view'])
        if not view.is_relative_to(path.parent / 'namespace_scaffolds'):
            raise ValueError('Owner lock mountpoint must remain in its fresh namespace view')
        slot = view / target.relative_to(parent['target'])
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.touch(exist_ok=False)
    directory = path.parent / 'owner-locks'
    directory.mkdir(exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError('Owned canonical lock backing required')
    backing = directory / (task + '.' + hostname + '.' + gpu_uuid + '.lock')
    with backing.open('xb'):
        pass
    return [dict(source=str(backing), target=str(target), read_only=False,
                 role='original_native_owner_lock', task=task,
                 external_shared_gpu_lease_still_required=True)]


def specialized(*, assets, manifest, runtimes, bank, output, native_config,
                family, gpu, lock_root, episode=1, native_in_image=False):
    """Bind both source families; emit only the selected family's namespace.

    Bootstrap files hold exact saved launch options and are NEVER runnable.
    The saved-instance owner restores its original attachments next to them,
    replaces all input pins, and applies the normal owner gates afterwards.
    """
    from .specialized import config, camera_release
    if family not in config.MODULES or type(gpu) is not int or gpu < 0:
        raise ValueError('Exact dedicated family and exclusive GPU required')
    approved = template()
    assets = canonical(assets)
    index = installation._index(resources.read_manifest(manifest), assets)
    environments = native(read(native_config))
    lock = canonical(lock_root)
    if not lock.is_dir():
        raise ValueError('Shared existing cooperative GPU lock directory required')
    resolved = {f: runtime(runtimes[f], f) for f in config.MODULES}
    out = fresh(output)
    public = canonical(HERE.parents[1])
    dependencies = dict(schema='robotagentbench.runtime_dependencies.v1',
                        roots={'assets': str(assets)}, resources={}, legacy_sources={})
    by_hash = {}
    mounts = {}
    image_roots = list(map(Path, environments['dependencies']['roots'].values())) if native_in_image else []

    def mount(source, target, *, read_only=True):
        source, target = Path(source), Path(target)
        if any(source == p or source.is_relative_to(p) for p in image_roots):
            if source == target:
                return  # Explicit pinned native image paths, not host bind mounts.
            if not read_only or not source.is_file():
                raise ValueError('Only an exact immutable image file may back a legacy alias')
            digest = sha(source)
            copy = out / 'native_alias_inputs' / digest
            copy.parent.mkdir(exist_ok=True)
            if not copy.exists():
                shutil.copyfile(source, copy)
            if sha(copy) != digest:
                raise ValueError('Pinned image alias copy differs')
            source = copy
        row = dict(source=str(source), target=str(target), read_only=read_only)
        previous = mounts.get(str(target))
        if previous is not None and previous != row:
            raise ValueError('Conflicting namespace target')
        mounts[str(target)] = row

    for f, (base, source, record) in resolved.items():
        dependencies['roots'][f + '_sources'] = str(source)
        mount(base, base)
        for rel, digest in record['source_files'].items():
            path = source / rel
            if path.resolve(strict=True) != path or sha(path) != digest:
                raise ValueError('Configured source member changed')
            by_hash.setdefault(digest, path)
    for digest, record in index['objects'].items():
        by_hash.setdefault(digest, assets / record['path'])
    checked = set()

    def content(digest):
        path = by_hash.get(digest)
        if path is None:
            raise ValueError('Reviewed content is absent from installed assets/sources')
        if digest not in checked:
            if canonical(path) != path or sha(path) != digest:
                raise ValueError('Reviewed content bytes differ')
            checked.add(digest)
        return path

    def resource(key, path, digest, kind):
        root = 'resource_' + key
        dependencies['roots'][root] = str(path.parent)
        dependencies['resources'][key] = dict(root=root, path=path.name,
                                              sha256=digest, kind=kind)

    # Camera upgrades retain the old assembly, not duplicate copies of old
    # executable source. Resolve only the policy's exact before/after roles
    # before populating resources; all other old pins must still be present.
    camera_sources = {}
    for f, record in approved['families'].items():
        bindings = read(content(approved['resources'][record['bindings']]['sha256']))
        for original, change in camera_release.profile(f)['files'].items():
            key = bindings['source_resources'].get(original)
            row = approved['resources'].get(key, {})
            if row.get('kind') != 'source' or row.get('sha256') != change['before']['sha256']:
                raise ValueError('Camera source resource is not the reviewed predecessor')
            path = resolved[f][1] / original.lstrip('/')
            after = change['after']
            if (canonical(path) != path or sha(path) != after['sha256']
                    or path.stat().st_size != after['bytes']):
                raise ValueError('Installed camera source differs from the fixed release')
            value = (path, after['sha256'])
            if key in camera_sources and camera_sources[key] != value:
                raise ValueError('Camera source resource has conflicting family bindings')
            camera_sources[key] = value

    # SDK files are supplied by the verified separate SDK, not downloaded assets.
    sdk_roots = [Path(environments['dependencies']['roots'][r['root']]) / r['path']
                 for r in environments['profile']['metadata_paths']]
    for key, row in approved['resources'].items():
        if key in camera_sources:
            path, digest = camera_sources[key]
            resource(key, path, digest, 'source')
            continue
        digest = row['sha256']
        if 'sdk_relative' in row:
            matches = [root / row['sdk_relative'] for root in sdk_roots
                       if (root / row['sdk_relative']).is_file()
                       and sha(root / row['sdk_relative']) == digest]
            if len(set(matches)) != 1:
                raise ValueError('Original SDK material dependency unavailable')
            by_hash[digest] = matches[0]
        resource(key, content(digest), digest, row['kind'])
    families = {}
    generated = []
    for f, record in approved['families'].items():
        source = resolved[f][1]
        binding_pin = approved['resources'][record['bindings']]['sha256']
        bindings = read(content(binding_pin))
        original_pin = approved['resources'][record['manifest']]['sha256']
        original = content(original_pin)
        derived = out / (f + '-camera-assembly.json')
        with derived.open('xb') as stream:
            stream.write(camera_release.derive_assembly(original, f))
        resource(record['manifest'] + '_upstream', original, original_pin, 'asset')
        resource(record['manifest'], derived, sha(derived), 'asset')
        assembly = read(derived)
        # This path equality is required by the original source verifier.
        for row in assembly['profiles'][f]['files']:
            path = source / row['original'].lstrip('/')
            if canonical(path) != path or sha(path) != row['sha256']:
                raise ValueError('Installed family source identity differs')
            resource(bindings['source_resources'][row['original']], path,
                     row['sha256'], 'source')
        tasks = {}
        for task, row in record['tasks'].items():
            selected = select(bank, task, episode)
            state = selected['state']
            options = state['launch_options']
            if options.get('task_name') != task:
                raise ValueError('Saved launch/task identity differs')
            directory = out / 'generated' / task
            directory.mkdir(parents=True)
            launch = directory / Path(row['launch']).name
            save(launch, options)
            if sha(launch) == state.get('provenance', {}).get('source_launch_sha256'):
                # A bootstrap has no original sidecars. Force the restoration
                # route without changing one saved launch value or source pin.
                with launch.open('a') as stream:
                    stream.write('\n')
            key = 'saved_bootstrap_' + task
            # Logical path resolves only inside the declared fresh container.
            resource(key, Path(row['launch']), sha(launch), 'asset')
            tasks[task] = dict(launch=key, inputs=[key],
                              required_resources=record['required'] + row['extra'],
                              output_parent=row['output_parent'])
            generated.append(dict(task=task, source=str(directory),
                                  target=str(Path(row['launch']).parent), read_only=False))
            # Dependency roots for every task are admitted together, while
            # only the selected family may create restored input sidecars.
            mount(directory, Path(row['launch']).parent, read_only=f != family)
            if f == family:
                destination = out / 'outputs' / task
                destination.mkdir(parents=True)
                mount(destination, row['output_parent'], read_only=False)
        families[f] = dict(manifest=record['manifest'], bindings=record['bindings'],
                           source_root=f + '_sources', tasks=tasks)
        if f == family:
            for row in bindings['namespace_mounts']:
                mount(source / row['source_relative'], row['target'])
            for row in bindings['startup_resources'].values():
                digest = approved['resources'][row['resource']]['sha256']
                mount(content(digest), row['namespace'])
    # Bind existing resource namespaces, but never shadow fresh generated input.
    generated_targets = [Path(r['target']) for r in generated]
    for target, digest in approved['aliases'].items():
        if any(Path(target).is_relative_to(p) for p in generated_targets):
            continue
        mount(content(digest), target)
    environment = merge_environment(dependencies, environments)
    inputs = list(dict.fromkeys([str(public), str(assets), str(out),
        *[str(r[0]) for r in resolved.values()],
        *[str(Path(r['target']) if Path(r['source']).is_dir() else Path(r['target']).parent)
          for r in mounts.values()]]))
    operator = dict(schema=config.SCHEMA, public_root=str(public), dependencies=dependencies,
        environment=environment, workspace=dict(input_roots=inputs, lock_root=str(lock)),
        gpu=gpu, families=families, outer_lifecycle=config.OUTER_CONTRACT)
    save(out / 'operator.json', operator)
    # Both source roots are addressable; only one family's legacy import aliases
    # may be mounted in a process. Runtime/Assets/native inputs remain read-only.
    for path in [public, assets, *map(Path, environments['dependencies']['roots'].values())]:
        mount(path, path)
    mount(out, out, read_only=False)
    mount_rows = external_lock_namespace(list(mounts.values()), lock, out)
    mount_rows, scaffolds = mountpoint_scaffolds(mount_rows, out)
    receipt = dict(schema='robotagentbench.dedicated_namespace_plan.v1',
        family=family, operator=str(out / 'operator.json'), mounts=mount_rows,
        namespace_scaffolds=scaffolds,
        mount_phase='cpu_preparation_only',
        required_native_roots_in_image=list(map(str, image_roots)),
        native_owner_lock_builder='robotagentbench.simulation.dedicated_delivery.owner_lock_bindings',
        external_shared_gpu_lease_still_required=True,
        external_shared_gpu_lease_host_root=str(lock),
        generated_inputs=generated, bootstrap_only=True, execution_ready=False,
        required_next_steps=['mount isolated CPU preparation namespace',
            'saved-instance restore with --owner-config and exact selected task/episode',
            'seal generated task inputs read-only; keep only output and shared lock writable',
            'bind the single original owner lock file using measured host/device identity',
            'unchanged dedicated operator verify and execute'],
        native_started=False, resampled=False, physics_scoring_modified=False)
    save(out / 'namespace-plan.json', receipt)
    return receipt


def cloth(*, assets, manifest, cloth_runtime, geometry_root, output,
          native_config, gpu, lock_root, native_in_image=False):
    """Bind original cloth roles; emit its existing, explicit alias recipe.

    Geometry libraries must be an independent installation with the reviewed
    wheel bytes. Never silently substitute another GEOS/Shapely build.
    """
    from robotagentbench.backends import sources
    from .cloth import config, launcher
    path = HERE / 'dedicated_cloth_template.json'
    if sha(path) != CLOTH_TEMPLATE_SHA:
        raise ValueError('Reviewed cloth roles changed')
    roles = read(path)
    assets = canonical(assets)
    index = installation._index(resources.read_manifest(manifest), assets)
    base, source, record = runtime(cloth_runtime, 'cloth')
    geometry = canonical(geometry_root)
    lock = canonical(lock_root)
    if type(gpu) is not int or gpu < 0 or not lock.is_dir():
        raise ValueError('Explicit device and existing shared lock directory required')
    env = native(read(native_config))
    sources.verify('cloth')
    expected = {p[len('profiles/cloth/'):]: h for p, h in index['locations'].items()
                if p.startswith('profiles/cloth/')}
    expected.update({p: r['sha256'] for p, r in sources._manifest()['profiles']['cloth']['files'].items()})
    if record['source_files'] != expected:
        raise ValueError('Runtime cloth closure differs from installed source/asset release')
    pins = {}
    for relative, digest in expected.items():
        member = source / relative
        if canonical(member) != member or sha(member) != digest:
            raise ValueError('Original cloth dependency changed')
        pins[str(member)] = digest
    for relative, digest in roles['geometry_pins'].items():
        member = geometry / relative
        if canonical(member) != member or sha(member) != digest:
            raise ValueError('Reviewed cloth geometry wheel differs')
        pins[str(member)] = digest
    public = canonical(HERE.parents[1])
    if any((public / 'robotagentbench').rglob('*.pyc')):
        raise ValueError('A clean public source view without bytecode is required')
    for member in (public / 'robotagentbench').rglob('*.py'):
        pins[str(canonical(member))] = sha(member)
    from .dependencies import RuntimeDependencies
    deps = RuntimeDependencies.from_config(env['dependencies'])
    for key, row in deps.resources.items():
        pins[str(deps.file(key))] = row.sha256
    sdk_candidates = [Path(env['dependencies']['roots'][r['root']]) / r['path']
                      for r in env['profile']['metadata_paths']
                      if (Path(env['dependencies']['roots'][r['root']]) / r['path'] / 'isaacsim').is_dir()]
    if len(set(sdk_candidates)) != 1:
        raise ValueError('One explicitly configured original Isaac SDK required')
    out = fresh(output)
    outputs = out / 'outputs'; outputs.mkdir()
    generated = out / 'generated'; generated.mkdir()
    roots = dict(asset_root=str(source / 'cloth_source'), original_root=str(source / 'original'),
        collision_geometry_root=str(source / 'original_geometry'), prepared_root=str(source / 'prepared'),
        owner_root=str(source / 'owner'), geometry_python_root=str(geometry),
        formal_root=str(HERE / 'cloth'), public_root=str(public), sdk_site=str(sdk_candidates[0]))
    templates = {}
    source_manifests = {}
    for task, files in roles['tasks'].items():
        mapped = {key: str(source / value.lstrip('/')) for key, value in files.items()}
        if set(mapped) != config.FILES or any(path not in pins for path in mapped.values()):
            raise ValueError('Every original cloth input role needs its installed release pin')
        from .cloth import source_data
        original_manifest = mapped['source_manifest']
        if original_manifest not in source_manifests:
            derived = source_data.extend_manifest(read(original_manifest), roots['original_root'], pins,
                upstream_sha256=pins[original_manifest])
            destination = generated / ('source_acquisition_' + pins[original_manifest] + '.json')
            save(destination, derived)
            pins[str(destination)] = sha(destination)
            source_manifests[original_manifest] = str(destination)
        mapped['source_manifest'] = source_manifests[original_manifest]
        templates[task] = dict(directories=roots, files=mapped, hashes=pins)
    inputs = [str(base), str(public), str(geometry), str(out),
              *env['dependencies']['roots'].values()]
    operator = dict(schema=launcher.OPERATOR_SCHEMA, public_root=str(public),
        workspace=dict(input_roots=list(dict.fromkeys(inputs)), output_root=str(outputs), lock_root=str(lock)),
        native=env, gpu=gpu, launch_templates=templates,
        outer_lifecycle=dict(validation_watchdog_s=600, exclusive_gpu=True,
            read_only_inputs=True, no_automatic_retry=True, owned_cleanup_required=True))
    save(out / 'operator.json', operator)
    owner = launcher.launch_for(operator, 'collect_tablecloth')['owner']
    image_roots = list(map(Path, env['dependencies']['roots'].values())) if native_in_image else []
    external_native_roots = [] if native_in_image else list(map(Path, env['dependencies']['roots'].values()))
    mounts = [dict(source=str(p), target=str(p), read_only=True)
              for p in dict.fromkeys([base, public, geometry, *external_native_roots])]
    mounts += [dict(source=str(out), target=str(out), read_only=False),
               dict(source=str(lock), target=str(lock), read_only=False)]
    mounts += [dict(r, read_only=True) for r in config.compatibility_recipe(owner)]
    receipt = dict(schema='robotagentbench.dedicated_namespace_plan.v1', family='cloth',
        operator=str(out / 'operator.json'), mounts=mounts,
        mount_phase='cpu_preparation_only',
        required_native_roots_in_image=list(map(str, image_roots)),
        # These eight aliases are the source owner's own fixed recipe, not a
        # containment escape. The supervisor must realize them ONLY in a fresh
        # Runtime copy before sealing it, never in a shared installation.
        source_aliases=config.alias_recipe(owner), execution_ready=False,
        required_next_steps=['realize original aliases only in fresh cloth Runtime',
            'mount isolated CPU preparation namespace',
            'saved-instance restore with --owner-config and exact selected task/episode',
            'seal generated inputs read-only; keep only outputs and shared lock writable',
            'unchanged cloth operator verify and execute'],
        native_started=False, resampled=False, physics_scoring_modified=False)
    save(out / 'namespace-plan.json', receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--atomic-runtime'); p.add_argument('--recovery-runtime')
    p.add_argument('--cloth-runtime'); p.add_argument('--geometry-root')
    p.add_argument('--bank'); p.add_argument('--native-config', required=True)
    p.add_argument('--lock-root', required=True)
    p.add_argument('--family', choices=['atomic_coffee', 'failure_recovery', 'cloth'], required=True)
    p.add_argument('--output', required=True); p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--episode', type=int, default=1)
    p.add_argument('--native-in-image', action='store_true',
                   help='Native config paths are pinned contents of the selected immutable container image')
    a = p.parse_args(argv)
    common = dict(assets=a.assets, manifest=a.manifest, output=a.output,
                  native_config=a.native_config, gpu=a.gpu, lock_root=a.lock_root)
    common['native_in_image'] = a.native_in_image
    if a.family == 'cloth':
        if not a.cloth_runtime or not a.geometry_root:
            raise ValueError('Cloth Runtime and original geometry wheel required')
        result = cloth(**common, cloth_runtime=a.cloth_runtime, geometry_root=a.geometry_root)
    else:
        if not a.atomic_runtime or not a.recovery_runtime or not a.bank:
            raise ValueError('Both specialized Runtime profiles and exact saved bank required')
        result = specialized(**common,
            runtimes=dict(atomic_coffee=a.atomic_runtime, failure_recovery=a.recovery_runtime),
            bank=a.bank, family=a.family, episode=a.episode)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

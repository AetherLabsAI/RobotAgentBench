"""Export a small relocation source overlay without modifying a Runtime."""
import hashlib
import json
import os
from pathlib import Path


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, indent=2)+'\n').encode()


def export_profile(runtime_root, output_root, *, sdk_site, logical_source_root):
    """Export reviewed source patches, original pins and measured SDK data.

    ``sdk_site`` and ``logical_source_root`` are the operator's absolute paths
    *inside* its isolated namespace. No SDK is imported or installed. The
    original Runtime, assets, source pins and records remain read-only.
    """
    from .relocation_baseline import NEW_REVIEW, source_snapshot
    from .relocation_data import original_measurement_bytes, original_shared_preparation_sources
    from .relocation_request import SOURCE_SHA256, _SOURCE
    from .relocation_resources import original_resource_aliases

    runtime = Path(runtime_root).resolve(strict=True)
    output = Path(output_root)
    logical = Path(logical_source_root)
    sdk = Path(sdk_site)
    if (not output.is_absolute() or output.resolve() != output or output.exists()
            or output.is_relative_to(runtime) or not output.parent.is_dir()):
        raise ValueError('A fresh canonical output directory outside the original Runtime is required')
    for value in (logical, sdk):
        if not value.is_absolute() or '..' in value.parts:
            raise ValueError('Use the operator namespace contract for source and SDK paths')
    metadata_raw = (runtime/'runtime.json').read_bytes()
    metadata = json.loads(metadata_raw)
    if (metadata.get('schema') != 'robotagentbench.local_installation.v1'
            or metadata.get('profile') != 'room'):
        raise ValueError('A configured room Runtime is required')
    old_runtime = Path(metadata['runtime'])
    source_relative = Path(metadata['source_root']).relative_to(old_runtime)
    base = runtime/source_relative
    if base.resolve(strict=True) != base or not base.is_dir():
        raise ValueError('Canonical original Runtime source root required')
    old_pins = dict(metadata['source_files'])
    resource_paths = {}
    for relative, recorded in metadata['resources'].items():
        prefix = 'libraries/generic_current/'
        if relative.startswith(prefix):
            resource_paths[relative[len(prefix):]] = runtime/Path(recorded).relative_to(old_runtime)

    snapshot = source_snapshot()
    patches = dict(snapshot)
    patches.update(original_shared_preparation_sources())
    variant = 'short_task_robustness/tasks/stock_wooden_tray_interference'
    canonical = 'short_tasks/tasks/stock_wooden_tray_interference'
    for relative, raw in list(patches.items()):
        if relative.startswith(variant+'/'):
            patches[canonical+relative[len(variant):]] = raw
    original_snapshot = dict(patches)
    measurement = original_measurement_bytes()
    measured = json.loads(measurement)
    helper = Path(__file__).with_name('relocation_sdk.py').read_bytes()
    sdk_input = _json(dict(sdk_site=str(sdk), expected_sdk=measured['sdk_identity'],
                           expected_shape=measured['shape_identity']))
    if _sha(_SOURCE.encode()) != SOURCE_SHA256:
        raise ValueError('Original source-pinned relocation request changed')
    replacements = {
        'generation.py': (
            "    probe=Usd.Stage.CreateInMemory()\n    margin=float(PhysxSchema.PhysxCollisionAPI.Apply(probe.DefinePrim('/Probe')).GetContactOffsetAttr().Get())",
            "    from ._sdk_effective_contact import read_margin\n    margin=read_margin(hand)"),
        'initialization.py': (
            "    probe = Usd.Stage.CreateInMemory()\n    margin = float(PhysxSchema.PhysxCollisionAPI.Apply(probe.DefinePrim('/Probe')).GetContactOffsetAttr().Get())\n    asset = AdaptiveHand()",
            "    asset = AdaptiveHand()\n    from ._sdk_effective_contact import read_margin\n    margin = read_margin(asset)"),
    }
    for home in (variant, canonical):
        patches[home+'/generator_sources.original.json'] = snapshot[variant+'/generator_sources.json']
        patches[home+'/generator_sources.json'] = _json(NEW_REVIEW)
        patches[home+'/_sdk_effective_contact.py'] = helper
        patches[home+'/native_effective_offsets.json'] = measurement
        patches[home+'/sdk_offset_input.json'] = sdk_input
        patches[home+'/request.json'] = _SOURCE.encode()
        for name, (before, after) in replacements.items():
            raw = patches[home+'/'+name].decode()
            if raw.count(before) != 1:
                raise ValueError('Original SDK-read seam differs: '+home+'/'+name)
            patches[home+'/'+name] = raw.replace(before, after).encode()

    aliases = []
    def effective(relative, expected=None):
        if relative in patches:
            return patches[relative]
        path = base/relative
        raw = path.read_bytes() if path.is_file() else None
        if expected is not None and (raw is None or _sha(raw) != expected):
            resource = resource_paths.get(relative)
            if resource is None or not resource.is_file():
                raise ValueError('Install the original shared resource before exporting: '+relative)
            raw = resource.read_bytes()
            if _sha(raw) != expected:
                raise ValueError('Original shared resource differs: '+relative)
            aliases.append(dict(source=str(resource), target=str(logical/relative), read_only=True))
        if raw is None:
            raise ValueError('Original Runtime dependency missing: '+relative)
        return raw

    for home in (variant, canonical):
        relative = home+'/component_sources.json'
        path = base/relative
        if not path.is_file():
            if home == canonical:
                raise ValueError('Original relocation component source manifest required')
            continue
        original = path.read_bytes()
        if relative in old_pins and _sha(original) != old_pins[relative]:
            raise ValueError('Original Runtime source changed: '+relative)
        record = json.loads(original)
        key = next(k for k in ('sources', 'files_sha256', 'source_files') if k in record)
        for name, expected in list(record[key].items()):
            actual = _sha(effective(name, expected))
            if actual != expected and name not in patches:
                raise ValueError('Unreviewed component dependency changed: '+name)
            record[key][name] = actual
        for name in ('_sdk_effective_contact.py', 'native_effective_offsets.json',
                     'sdk_offset_input.json', 'request.json'):
            record[key][home+'/'+name] = _sha(patches[home+'/'+name])
        patches[home+'/component_sources.original.json'] = original
        patches[relative] = _json(record)

    shared = json.loads(snapshot['short_tasks/scene_sources.json'])['sources']
    for relative, expected in shared.items():
        if _sha(effective(relative, expected)) != expected:
            raise ValueError('Original shared source integrity differs: '+relative)
    for relative in patches:
        path = base/relative
        if relative in old_pins and (not path.is_file() or _sha(path.read_bytes()) != old_pins[relative]):
            raise ValueError('Original Runtime source changed: '+relative)
        # The current generator review authorizes its declared preparation
        # sources, not replacement of a different native/scoring backend.
        reviewed = relative in {
            'openend/tasks/scene_requirements/build_shelf.py',
            'openend/tasks/scene_requirements/build_selected_room.py',
            'openend/tasks/scene_requirements/task_choices.py',
            'short_tasks/scene_sources.json',
            variant+'/generator_sources.json', canonical+'/generator_sources.json',
        }
        if relative in original_snapshot and path.is_file() and not reviewed:
            actual = _sha(path.read_bytes())
            if actual not in {_sha(original_snapshot[relative]), _sha(patches[relative])}:
                raise ValueError('Original native/task source version differs: '+relative)
    aliases.extend(original_resource_aliases(runtime))
    unique = {}
    for row in aliases:
        target = row['target']
        if target in unique and unique[target] != row:
            raise ValueError('Conflicting original resource alias: '+target)
        unique[target] = row
    aliases = list(unique.values())

    # A complete canonical source view avoids missing new-file mountpoints
    # below a read-only parent. Reuse unchanged bytes by hardlink, once per
    # source profile (not per episode); never write/chmod a linked file.
    device = output.parent.stat().st_dev
    directories = []
    unchanged = []
    def walk(folder, relative, ancestors):
        real = folder.resolve(strict=True)
        if not real.is_relative_to(runtime) or real in ancestors:
            raise ValueError('Original source link must remain within the configured Runtime')
        if not real.is_dir():
            raise ValueError('Original source directory required')
        directories.append(relative)
        for child in sorted(folder.iterdir()):
            tail = relative/child.name
            actual = child.resolve(strict=True)
            if actual.is_dir():
                walk(child, tail, (*ancestors, real))
            elif actual.is_file():
                if not actual.is_relative_to(runtime) or actual.stat().st_dev != device:
                    raise ValueError('Place the source overlay on the Runtime filesystem')
                unchanged.append((tail.as_posix(), actual))
            else:
                raise ValueError('Regular original source files required')
    walk(runtime, Path(), ())
    internal_resources = {}
    external_aliases = []
    for row in aliases:
        target = Path(row['target'])
        if target.is_relative_to(logical):
            internal_resources[target.relative_to(logical).as_posix()] = Path(row['source'])
        else:
            external_aliases.append(row)
    output.mkdir()
    (output/'OWNER.json').write_bytes(_json(dict(owner='RobotAgentBench-relocation-profile-v1')))
    view = output/source_relative
    for relative in directories:
        (output/relative).mkdir(parents=True, exist_ok=True)
    for relative, original in unchanged:
        if relative in ('runtime.json', 'OWNER.json', 'source_profile.json'):
            continue
        source_tail = (Path(relative).relative_to(source_relative).as_posix()
                       if Path(relative).is_relative_to(source_relative) else None)
        if source_tail not in patches and source_tail not in internal_resources:
            os.link(original, output/relative)
    for relative, original in internal_resources.items():
        target = view/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, target)
    pins = dict(old_pins)
    changes = {}
    for relative, raw in sorted(patches.items()):
        path = view/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        pins[relative] = _sha(raw)
        changes[relative] = dict(previous_sha256=old_pins.get(relative), sha256=pins[relative])
    def rebase(value):
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str) and (value == str(old_runtime) or value.startswith(str(old_runtime)+'/')):
            return str(output)+value[len(str(old_runtime)):]
        return value
    fresh_metadata = rebase(metadata)
    fresh_metadata.update(runtime=str(output), source_root=str(view), source_files=pins)
    # A standard Runtime, not a metadata patch the user must merge by hand.
    (output/'runtime.json').write_bytes(_json(fresh_metadata))
    mounts = [dict(source=str(view), target=str(logical), read_only=True)]
    mounts.extend(original_resource_aliases(output))
    profile = dict(schema='robotagentbench.relocation_source_profile.v1',
        task_name='stock_wooden_tray_interference', runtime_root=str(output),
        source_root=str(logical), source_backing_root=str(view),
        source_pins=pins, readonly_mounts=mounts, sdk_site=str(sdk),
        original_runtime_metadata_sha256=_sha(metadata_raw), changes=changes,
        source_baseline=NEW_REVIEW, measurement_sha256=_sha(measurement),
        request_template_sha256=SOURCE_SHA256, shared_source_gate_preserved=True,
        original_runtime_bytes_modified=False, unchanged_bytes_reused_by_hardlink=True,
        standard_runtime_metadata=True, source_directory_layout_preserved=True,
        scoring_physics_sampler_thresholds_modified=False,
        sdk_imported=False, native_started=False)
    (output/'source_profile.json').write_bytes(_json(profile))
    return profile

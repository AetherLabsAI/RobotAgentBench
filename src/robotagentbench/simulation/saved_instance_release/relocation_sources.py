"""Rebind two frozen shelf-hand source identities to an installed Runtime.

The native reservation validator keys its source hashes by the actual source
path. Saved reservations retain the original source-root names. Only those
names are changed here, after checking the original pins and installed bytes;
all reservation geometry and the original native validator remain unchanged.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

_ORIGINAL_SOURCE = Path('/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source')
_TASK = 'stock_wooden_tray_interference'
_GEOMETRY = (
    ('short_tasks/tasks/stock_wooden_tray_interference/assets/derived/left_hand_side_entry_012.json',
     'd53986cb794bf7ab36980ea6e6ce0627638e45c57bac281b677be4de8fae6f31'),
    ('short_tasks/tasks/stock_wooden_tray_interference/static_short_hand_011/cropped_rig.json',
     'ed9667c4d8a6637f84753b4c7d50a9bf08a26a998e44141bddcde99bf12668e2'),
)


def restore_geometry_sources(manifest, *, source, runtime, metadata):
    """Return a copied manifest plus evidence; never write source or assets."""
    source, runtime = Path(source), Path(runtime)
    if (not source.is_absolute() or not runtime.is_absolute()
            or source.resolve(strict=True) != source or runtime.resolve(strict=True) != runtime
            or not source.is_relative_to(runtime)
            or metadata.get('schema') != 'robotagentbench.local_installation.v1'
            or metadata.get('profile') != 'room'):
        raise ValueError('Canonical installed room Runtime required')
    recorded_root, recorded_source = Path(metadata['runtime']), Path(metadata['source_root'])
    if (not recorded_root.is_absolute() or not recorded_source.is_absolute()
            or '..' in recorded_root.parts or '..' in recorded_source.parts
            or runtime / recorded_source.relative_to(recorded_root) != source):
        raise ValueError('Installed relocation source-root binding differs')
    old, installed, evidence = {}, {}, []
    for relative, expected in _GEOMETRY:
        original = str(_ORIGINAL_SOURCE / relative)
        path = source / relative
        if metadata.get('source_files', {}).get(relative) != expected:
            raise ValueError('Original relocation geometry source pin differs: ' + relative)
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError('Canonical installed relocation geometry file required: ' + relative)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Original relocation geometry bytes differ: ' + relative)
        old[original] = expected
        installed[str(path)] = expected
        evidence.append(dict(original=original, installed=str(path), sha256=expected))
    if len(old) != 2 or len(installed) != 2:
        raise ValueError('Distinct original and installed geometry sources required')

    result = deepcopy(manifest)
    roots = result.get('environment_reservations')
    if not isinstance(roots, list) or len(roots) != 1:
        raise ValueError('Exactly one frozen shelf relocation reservation required')
    root = roots[0]
    rows = [root]
    if not isinstance(root, dict):
        raise ValueError('Frozen shelf relocation reservation required')
    if 'bilateral_routes' in root:
        pool = root['bilateral_routes']
        if (not isinstance(pool, dict) or pool.get('schema') != 'shelf_interference.bilateral_routes.v1'
                or pool.get('coordinate_frame') != 'shelf_local_m'
                or pool.get('selection_at') != 'native_proximity'
                or pool.get('runtime_replanning') is not False):
            raise ValueError('Original frozen bilateral route contract required')
        routes = pool.get('routes')
        if (not isinstance(routes, list) or len(routes) != 2
                or any(not isinstance(row, dict) for row in routes)
                or [row.get('side') for row in routes] != ['left', 'right']):
            raise ValueError('Both original left and right routes required')
        for route in routes:
            child = route.get('reservation')
            if not isinstance(child, dict) or 'bilateral_routes' in child:
                raise ValueError('Nested or missing relocation route is not supported')
            rows.append(child)
    for row in rows:
        if (row.get('schema') != 'arena.human_relocation_reservation.v1'
                or row.get('producer') != {'task_name': _TASK, 'version': 'reserve_first.v1'}
                or row.get('geometry_source_hashes') != old
                or row.get('hand_asset_sha256') != _GEOMETRY[0][1]):
            raise ValueError('Original frozen relocation geometry identities differ')
        row['geometry_source_hashes'] = dict(installed)
    return result, dict(sources=evidence, reservations_rebound=len(rows),
                        source_assets_modified=False, reservation_geometry_modified=False,
                        physics_scoring_modified=False)

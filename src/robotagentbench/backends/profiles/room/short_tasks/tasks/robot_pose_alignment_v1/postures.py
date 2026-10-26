"""Keep each recorded posture intact; fail closed instead of using generic arms.

Existing base stations, object transforms, drivers, clocks and scores are not
owned here. A caller must recheck geometry using the returned actual joints.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE_NAMES = (
    *(f'idx{21+i:02d}_arm_l_joint{i+1}' for i in range(7)),
    *(f'idx{61+i:02d}_arm_r_joint{i+1}' for i in range(7)),
    'idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2',
    'idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1',
)
FAMILIES = {
    'prepare_coffee': (107,), 'bake_bread': (108,),
    'stock_wooden_tray': tuple(range(93,107)), 'hang_cap': (48,),
    'clear_table_trash': (49,), 'fridge_to_heated_box': (116,),
    'clear_tablecloth': (34,), 'prepare_cutlery': (88,),
    'carry_meal_tray': (121,),
}
PERSONAL = {
    'retrieve_from_drawer': {'wallet': (45,), 'glasses_case': (44,)},
    'sofa_to_drawer': {'wallet': (47,), 'notebook_green': (46,)},
    'table_to_storage_basket': {'wallet': (42,), 'glasses_case': (43,)},
    'sofa_to_coffee_table': {'book_red': (118,), 'remote_tv': (119,), 'wallet': (120,),
                             'glasses_case': (0,)},
    'pack_object': {'wallet': (38,), 'power_bank': (37,), 'water_bottle': (39,),
                    'glasses_case': (0,)},
}
PICK_SOURCE_EXTENSIONS = {
    ('wallet','coffee_table.objects'): (11,), ('wallet','small_table'): (12,),
    ('glasses_case','sofa'): (0,), ('glasses_case','coffee_table.objects'): (6,),
    ('glasses_case','small_table'): (7,),
}
UNAVAILABLE = frozenset(('microwave_to_meal_tray','spread_tablecloth',
                         'collect_tablecloth','stock_wooden_tray_interference'))


class MissingSourcePosture(ValueError):
    pass


def read_bank():
    path = HERE/'source_postures.json'
    expected = json.loads((HERE/'source_postures.sha256.json').read_text())['sha256']
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Robot posture bank changed without source review')
    value = json.loads(content)
    if value['active_joint_names'] != list(ACTIVE_NAMES):
        raise ValueError('Dataset active joint order differs')
    return value, expected


def source_indices(task_name, *, asset=None, source=None):
    if task_name in UNAVAILABLE:
        raise MissingSourcePosture('Normal ShortTask is unavailable: '+task_name)
    if task_name in FAMILIES:
        return FAMILIES[task_name]
    if task_name not in PERSONAL or asset not in PERSONAL[task_name]:
        raise MissingSourcePosture('No matching task/asset posture: '+str((task_name,asset)))
    if task_name in ('pack_object','table_to_storage_basket') and source in ('small_table','coffee_table.objects'):
        # Table->basket's own short data takes priority on its original source.
        if task_name == 'table_to_storage_basket' and source == 'coffee_table.objects':
            return PERSONAL[task_name][asset]
        try:
            return PICK_SOURCE_EXTENSIONS[asset,source]
        except KeyError as exc:
            raise MissingSourcePosture('No matching source-pick posture') from exc
    if task_name == 'pack_object' and source != 'sofa':
        raise MissingSourcePosture('Missing or unsupported packing source')
    return PERSONAL[task_name][asset]


def context(plan, manifest=None):
    name = plan['task_name']
    asset = plan.get('asset')
    source = plan.get('source')
    if manifest is not None and name in PERSONAL:
        rows = [r for r in manifest.get('objects',[]) if r.get('task_id') == name]
        if len(rows) != 1:
            raise MissingSourcePosture('Exactly one task-owned personal object required')
        row = rows[0]
        actual_asset = row.get('asset_id',row.get('id'))
        if asset is not None and asset != actual_asset:
            raise MissingSourcePosture('Plan and scene object differ')
        asset, source = actual_asset, row.get('source')
    return {'task_name':name,'asset':asset,'source':source}


def candidates(task_name, *, seed, asset=None, source=None, previous_targets=None):
    """Finite, deterministic order of whole source poses, without angle jitter."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    indices = source_indices(task_name,asset=asset,source=source)
    bank, digest = read_bank()
    rows = [r for r in bank['rows'] if r['task_index'] in indices and r.get('targets_rad_or_m')]
    if task_name == 'carry_meal_tray':
        rows = [r for r in rows if r.get('variant') == 'empty_tray']
    if not rows:
        raise MissingSourcePosture('No authenticated source pose available')
    previous_targets = previous_targets or {}
    def key(row):
        # Sample the task's recorded pose range by seed, rather than snapping
        # every reset to the pose nearest the old generic posture. Keep all
        # joints from one source record; do not invent independent angles.
        return hashlib.sha256(f'{task_name}:{seed}:{row["source_id"]}'.encode()).hexdigest()
    for row in sorted(rows,key=key):
        q = dict(row['targets_rad_or_m'])
        if not set(ACTIVE_NAMES).issubset(q) or not all(math.isfinite(v) for v in q.values()):
            raise ValueError('Incomplete/nonfinite authenticated source posture')
        yield {
            'targets_rad': q,
            'posture_source': row['source_id'],
            'dataset_episode_index': row.get('episode_index'),
            'dataset_task_index': row['task_index'],
            'source_evidence': row['posture_evidence'],
            'source_scope': 'same_task_short_data' if row['task_index'] >= 34 else 'same_object_same_source_atomic_pick_extension',
            'source_bank_sha256': digest,
            'source_provenance_sha256': bank['provenance_sha256'],
            'complete_source_joint_tuple': True,
            'joint_coordinates_independently_sampled': False,
            'joint_coordinates_clipped': False,
            'source_selection': 'seeded_whole_source_pose_within_task_data_range',
            'native_physics_validated': False,
        }


def source_hashes():
    files = [Path(__file__),HERE/'source_postures.json',HERE/'source_postures.sha256.json']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

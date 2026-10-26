"""Two pickup families; task identity is separate from registration readiness."""
import json
from pathlib import Path
from evaluation.tasks.tidy_sofa.specs import TaskSpec

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
BASE = REPO / 'assets/house_scene_l1/demo_sim/scene_room_baseline.usda'
ROBOT = '/World/Robot'
RULE_VERSION = 'arena.personal_item_pick.v1'
APPROVED_TIME_LIMIT_S = 30.0
SMALL_IDS = {'13-1': ('54',), '13-2': ('107',), '13-3': ('111',), '13-5': ('73',)}
BLOCKERS = {
    '13-2': 'Source 5.3/1032 retrieved; initial joint state and current scene compatibility pending audit',
}


def definitions():
    return {t['api_id']: t for t in json.loads((TASK / 'task_definitions.json').read_text())['tasks']}


def task_spec(api_id):
    if api_id not in SMALL_IDS:
        raise ValueError('Unknown personal-item pickup ID: ' + str(api_id))
    data = definitions()[api_id]
    sofa = data['family'] == 'sofa_pick'
    # The baseline has SeatOverlay; worker policy also enables the sofa mesh.
    # The task reader recognizes both exact physical support identities.
    support = '/World/Assets/SeatOverlay' if api_id == '13-1' else data['support_prim']
    furniture = ('/World/Assets/Sofa',) if sofa else (support,)
    return TaskSpec(api_id, data['benchmark_id'], 'pick', data['instruction'].rstrip('。'),
                    data['instruction'], data['target_prim'], SMALL_IDS[api_id],
                    'benchmark_' + data['benchmark_id'].replace('.', '_'),
                    APPROVED_TIME_LIMIT_S, dict(grasp=3, lift=2), 5,
                    support=support, furniture=furniture, hold_s=1.0, lift_m=0.05)


def snapshot_path(api_id):
    return TASK / 'resources' / ('source_initial_' + task_spec(api_id).api_id + '.json')


def require_initial(api_id):
    if api_id in BLOCKERS:
        raise ValueError(BLOCKERS[api_id])
    path = snapshot_path(api_id)
    data = json.loads(path.read_text())
    sp = task_spec(api_id)
    if data['api_id'] != api_id or data['target_path'] != sp.target or data['in_hand'] is not False:
        raise ValueError('Source initial-state identity mismatch')
    return data

"""Existing static room gate, now applied to one fixed recorded task posture."""
import copy
import hashlib
from pathlib import Path

from .geometry import NoFeasibleSourcePosture, checker, update_candidate
from .postures_fixed_v3 import candidates, context, source_hashes


def align_receipt(stage, manifest, plan, receipt, *, seed):
    out = copy.deepcopy(receipt)
    base = [*out['position'][:2], out['yaw_rad']]
    pose = next(candidates(**context(plan, manifest), seed=seed))
    failure = checker(stage, manifest, base)(pose)
    if failure:
        raise NoFeasibleSourcePosture('Fixed task posture rejected; no posture fallback: '
                                     + pose['posture_source'] + ': ' + failure)
    out['previous_posture_validation_scope'] = out.get('method')
    out['targets_rad'] = pose['targets_rad']
    out['robot_posture_alignment'] = {
        **pose, 'rejections': [], 'base_xyyaw_preserved': base,
        'station_ranges_changed': False, 'object_transforms_changed': False,
        'geometry_rechecked_with_source_joints': True,
        'passive_internal_fingers': 'original_room_authored_coordinates_and_existing_coupling',
        'robot_self_collision_validated': False,
    }
    out['posture_source'] = pose['posture_source']
    out['robot_room_geometry_checked'] = True
    out['robot_self_collision_validated'] = False
    out['native_physics_validated'] = False
    update_candidate(out, pose['targets_rad'])
    out['candidate']['sampling'] = pose['source_selection']
    refs = source_hashes()
    for path in (Path(__file__), Path(__file__).with_name('geometry.py')):
        refs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    out.setdefault('reference_hashes', {}).update(refs)
    return out

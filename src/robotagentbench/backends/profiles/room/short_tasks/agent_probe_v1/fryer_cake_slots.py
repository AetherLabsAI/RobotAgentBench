"""Pure initial-slot assignment; no USD writes or running-episode motion.

Input is two freshly sampled source slots in the task workspace frame, BEFORE
the legacy +Y 30 mm second-slot adjustment. Output is final: never apply the old
worker adjustment again. World/layout conversion belongs to the scene owner.
This helper does not make the currently unavailable baking task runnable.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random

POLICY = Path(__file__).with_name('fryer_initial_pose_policy.json')


def finalize_slots(sample, *, seed, target_slot=None):
    if sample.get('stage') != 'sampled_before_legacy_slot_offset':
        raise ValueError('Fresh pre-offset slots required; finalized poses cannot be offset twice')
    policy = json.loads(POLICY.read_text())['cake_randomization']
    if target_slot is None:
        target_slot = random.Random(f'fryer-cake-slots:{seed}').choice(policy['target_slot_choices'])
    if target_slot not in policy['target_slot_choices']:
        raise ValueError('Target slot must be outer or inner')
    center = sample['plate_center_xyz']
    slots = deepcopy(sample['slots'])
    if len(center) != 3 or len(slots) != 2:
        raise ValueError('One plate and exactly two source slots required')
    values = list(center)
    for slot in slots:
        if len(slot['position']) != 3:
            raise ValueError('XYZ position required')
        values.extend(slot['position'])
        values.append(slot['yaw_deg'])
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Finite poses required')
    slots[1]['position'] = [x + dx for x, dx in zip(
        slots[1]['position'], policy['legacy_second_slot_offset_workspace_xyz_m'])]
    radii = [math.hypot(s['position'][0]-center[0], s['position'][1]-center[1]) for s in slots]
    separation = math.dist(slots[0]['position'][:2], slots[1]['position'][:2])
    if max(radii) > policy['maximum_center_radius_on_plate_m']:
        raise ValueError('Final cake center outside source plate radius; resample')
    if separation < policy['minimum_center_separation_m']:
        raise ValueError('Final cake centers too close; resample')
    index = 0 if target_slot == 'outer' else 1
    return dict(stage='final_slots_offset_applied', seed=seed, target_slot=target_slot,
                target=slots[index], distractor=slots[1-index],
                legacy_offset_applied_once=True, center_radii_m=radii,
                center_separation_m=separation, physics_verified=False)

"""Approved coffee reset distance; independent from object randomization.

Training-source audit (51 short.coffee episodes): base-to-machine XY distance
0.7818541--0.8194798 m. User approved the rounded 0.78--0.82 m range.
No arm, head, object, physics, scoring or runtime state changes here.
"""
import math
import random

POLICY = 'prepare_coffee.machine_origin_078_082.v1'
DISTANCE_M = (0.78, 0.82)


def sample(seed, origin, front, *, stand_off_m=None):
    if type(seed) is not int or seed < 0:
        raise ValueError('A non-negative integer seed is required')
    if len(origin) != 2 or len(front) != 2:
        raise ValueError('Expected machine origin XY and horizontal front XY')
    ox, oy = map(float, origin)
    fx, fy = map(float, front)
    norm = math.hypot(fx, fy)
    if not all(math.isfinite(x) for x in (ox, oy, fx, fy)) or norm < 1e-8:
        raise ValueError('Finite machine origin and nonzero front required')
    if stand_off_m is None:
        distance = random.Random(f'{POLICY}:{seed}').uniform(*DISTANCE_M)
    else:
        if isinstance(stand_off_m, bool):
            raise ValueError('Distance must be a number')
        distance = float(stand_off_m)
        if not math.isfinite(distance) or not DISTANCE_M[0] <= distance <= DISTANCE_M[1]:
            raise ValueError('Coffee distance must remain within approved 0.78--0.82 m')
    fx, fy = fx/norm, fy/norm
    return dict(policy=POLICY, seed=seed, distance_range_m=list(DISTANCE_M),
                distance_m=distance, reference='robot_base_origin_to_machine_origin_horizontal',
                machine_origin_xy=[ox, oy], front=[fx, fy],
                initial_base_xyyaw=[ox+fx*distance, oy+fy*distance, math.atan2(-fy,-fx)],
                sampling='seeded_uniform' if stand_off_m is None else 'explicit_in_range',
                resampled=False, joint_targets_changed=False,
                native_collision_acceptance_required=True)

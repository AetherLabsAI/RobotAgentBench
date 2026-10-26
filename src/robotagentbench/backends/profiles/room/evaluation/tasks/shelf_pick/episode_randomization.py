"""Robot initial-pose perturbation for the randomized shelf-pick jobs.

Ported value-for-value from the source generator's policy
`shelf_pick_episode_randomization_v1`
(/mnt/agent-bench-data/lingjun/ziche/work/shelf_training_visible360_20260916:
EPISODE_RANDOMIZATION.json and setup_bulk.py:79-108, read 2026-09-22):

    base XY  ~ U(-0.05, 0.05) m      (both axes, world frame)
    base yaw ~ U(-5, 5) deg          (composed onto the recorded base rotation)
    arm joints ~ U(-4, 4) deg        (every '_arm_' joint of both arms)
    waist idx01/idx02 and head idx11/idx12 stay exactly as recorded

Same distributions, same draw order (arm joints, then base XY, then base yaw) and
the same generator (numpy PCG64 seeded from a sha256 key), so a seed reproduces
one pose. The key differs from the source only in what identifies the episode:
the source keys on slot/attempt, this evaluation keys on the API id and the job
seed, because 202 jobs are identified that way. Recorded in the receipt.

This module only moves the robot. Item layout comes from a held-out accepted
source episode (the user's 2026-09-23 decision); nothing here touches physics
parameters, drives, assets or the trajectory.
"""
import hashlib
import json
import math

import numpy as np

POLICY = 'shelf_pick_episode_randomization_v1'
BASE_XY_UNIFORM_M = 0.05
BASE_YAW_UNIFORM_DEG = 5.0
ARM_JOINT_UNIFORM_DEG = 4.0
FIXED_JOINTS = ('idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2')
ARM_TOKEN = '_arm_'


def generator(api_id, seed):
    """PCG64 seeded exactly like the source policy, keyed by API id and job seed."""
    if not isinstance(api_id, str) or not api_id.strip():
        raise ValueError('Missing API identity for the perturbation key')
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise ValueError('Invalid episode seed')
    key = f'{POLICY}:{api_id}:{seed}:robot-initial'
    digest = hashlib.sha256(key.encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, 'big')), key


def quaternion_yaw(quaternion_wxyz, yaw_deg):
    """Rotate a wxyz quaternion about world +Z, as the source policy does."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    half = math.radians(yaw_deg) / 2
    cw, cz = math.cos(half), math.sin(half)
    # (cw + cz k) * (w + xi + yj + zk)
    result = (cw * w - cz * z, cw * x - cz * y, cw * y + cz * x, cw * z + cz * w)
    norm = math.sqrt(sum(v * v for v in result))
    if not norm > 0:
        raise ValueError('Degenerate base rotation')
    return [v / norm for v in result]


def perturb(snapshot, seed):
    """Return (perturbed snapshot, receipt) for a v2 source-episode snapshot."""
    if snapshot.get('schema') != 'arena.shelf_pick.initial_state.v2':
        raise ValueError('Robot perturbation needs a v2 source-episode snapshot')
    api_id = snapshot['api_id']
    rng, key = generator(api_id, seed)
    joints = dict(snapshot['joint_positions'])
    arm_names = sorted(n for n in joints if ARM_TOKEN in n and n not in FIXED_JOINTS)
    if not arm_names:
        raise ValueError('No arm joints found in the snapshot')
    # Draw order is part of the ported policy: arm joints, base XY, base yaw.
    joint_deltas_deg = rng.uniform(-ARM_JOINT_UNIFORM_DEG, ARM_JOINT_UNIFORM_DEG, len(arm_names))
    base_xy_m = rng.uniform(-BASE_XY_UNIFORM_M, BASE_XY_UNIFORM_M, 2)
    base_yaw_deg = float(rng.uniform(-BASE_YAW_UNIFORM_DEG, BASE_YAW_UNIFORM_DEG))
    for name, delta in zip(arm_names, joint_deltas_deg):
        joints[name] = float(joints[name] + math.radians(float(delta)))
    robot = dict(snapshot['robot'])
    position = [float(v) for v in robot['position_m']]
    position[0] += float(base_xy_m[0])
    position[1] += float(base_xy_m[1])
    robot['position_m'] = position
    robot['quaternion_wxyz'] = quaternion_yaw(robot['quaternion_wxyz'], base_yaw_deg)
    for name in FIXED_JOINTS:
        if name in joints and joints[name] != snapshot['joint_positions'][name]:
            raise ValueError('Fixed joint changed: ' + name)
    receipt = dict(policy=POLICY, seed=seed, seed_key=key,
                   source_policy_file='/mnt/agent-bench-data/lingjun/ziche/work/'
                                      'shelf_training_visible360_20260916/EPISODE_RANDOMIZATION.json',
                   base_xy_delta_m=[float(v) for v in base_xy_m],
                   base_yaw_delta_deg=base_yaw_deg,
                   arm_joint_names=list(arm_names),
                   arm_joint_delta_deg=[float(v) for v in joint_deltas_deg],
                   fixed_joints_unchanged=list(FIXED_JOINTS),
                   layout_source='held-out accepted source episode (user decision 2026-09-23)',
                   appearance_randomization='not enabled (source appearance code is render-only)')
    perturbed = dict(snapshot, robot=robot, joint_positions=joints, episode_randomization=receipt)
    return perturbed, receipt


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    snapshot = json.loads(Path(args.snapshot).read_text())
    perturbed, receipt = perturb(snapshot, args.seed)
    Path(args.output).write_text(json.dumps(perturbed, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

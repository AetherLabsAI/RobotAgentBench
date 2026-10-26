"""Owner-approved continuous initial proposals within measured Short 4 ranges.

This is NOT a physical feasibility check or a shared-preparer integration.
Recorded states are evidence for ranges, not a menu of complete reset states.
"""
import hashlib
import math

import numpy as np

from . import initial_state as source


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def planar_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def yaw_rotate(q, angle):
    x, y, z, w = q
    a, b = math.sin(angle / 2), math.cos(angle / 2)
    return np.array([b*x-a*y, b*y+a*x, b*z+a*w, b*w-a*z])


def local_sample(sample):
    drawer, angle = source.upright_frame(sample['drawer_xyz_xyzw'])
    base, base_q = source.pose(sample['robot_base_xyz_xyzw'])
    target, target_q = source.pose(sample['target_xyz_xyzw'])
    r = planar_rotation(-angle)
    return dict(base_xy=(r @ (base[:2]-drawer[:2])).tolist(),
                base_yaw=wrap(source.yaw(base_q)-angle),
                target_xyz=[*(r @ (target[:2]-drawer[:2])).tolist(), float(target[2]-drawer[2])],
                target_q=yaw_rotate(target_q, -angle).tolist())


def profile(asset):
    if asset not in ('wallet', 'glasses_case'):
        raise ValueError('Drawer source covers only wallet and glasses_case')
    samples = [s for s in source.read_samples() if s['asset'] == asset]
    names = samples[0]['joint_names']
    if any(s['joint_names'] != names for s in samples):
        raise ValueError('Source joint ordering changed')
    local = [local_sample(s) for s in samples]
    # Unwrap about a measured orientation: never sample across the +/-pi seam.
    anchor = source.yaw(local[0]['target_q'])
    target_yaw = [anchor+wrap(source.yaw(v['target_q'])-anchor) for v in local]
    rows = np.array([[*v['base_xy'], v['base_yaw'], *v['target_xyz'][:2], a]
                     for v, a in zip(local, target_yaw)])
    joints = np.array([s['joint_positions_rad_or_m'] for s in samples])
    # Two wallet joints contain <2e-8 rad float32 readback noise, not jitter.
    if asset == 'wallet' and np.any(np.ptp(joints[:, 4:18], axis=0) > np.finfo(np.float32).eps):
        raise ValueError('Wallet source no longer has its audited fixed arm pose')
    radii = np.linalg.norm(rows[:, :2], axis=1)
    return dict(schema='short_tasks.retrieve_from_drawer.range_profile.v1', asset=asset,
                source_count=len(samples), source_episodes=[s['episode_index'] for s in samples],
                source_bank_sha256=hashlib.sha256(source.SOURCE.read_bytes()).hexdigest(),
                frame='upright rigid drawer body; local -X outward, +Y lateral; metres/radians',
                columns=['base_x', 'base_y', 'base_yaw', 'target_x', 'target_y', 'target_yaw'],
                minimum=rows.min(axis=0).tolist(), maximum=rows.max(axis=0).tolist(),
                base_distance_min_max_m=[float(radii.min()), float(radii.max())],
                joint_names=names, joint_minimum=joints.min(axis=0).tolist(),
                joint_maximum=joints.max(axis=0).tolist(),
                samples=samples, local=local, target_yaw_unwrapped=target_yaw)


def sample_candidates(asset, seed, drawer_xyz_xyzw, *, count):
    """Yield bounded continuous proposals; caller must validate actual geometry.

    XY and yaw use independent uniform proposals in the observed intervals;
    the observed radial envelope is additionally enforced. This is a newly
    defined sampler, NOT a claim about the source generator's distribution.
    Target residual orientation and relative support height stay paired with
    the nearest observed yaw. Only glasses-case arm joints are randomized;
    other joints retain that reference's complete measured posture.
    """
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Valid integer seed required')
    if type(count) is not int or count < 1:
        raise ValueError('Positive integer proposal count required')
    destination, angle = source.upright_frame(drawer_xyz_xyzw)
    p = profile(asset)
    digest = hashlib.sha256(f'drawer-ranges-v1:{asset}:{seed}'.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))
    lo, hi = np.array(p['minimum']), np.array(p['maximum'])
    r = planar_rotation(angle)
    produced = 0
    # This is a CPU proposal budget, not a physical threshold or wider range.
    for attempt in range(count * 100):
        values = rng.uniform(lo, hi)
        radius = np.linalg.norm(values[:2])
        if not p['base_distance_min_max_m'][0] <= radius <= p['base_distance_min_max_m'][1]:
            continue
        nearest = int(np.argmin(np.abs(np.array(p['target_yaw_unwrapped'])-values[5])))
        reference = p['samples'][nearest]
        local = p['local'][nearest]
        joints = list(reference['joint_positions_rad_or_m'])
        if asset == 'wallet':
            for i, name in enumerate(p['joint_names']):
                if '_arm_' in name:
                    joints[i] = p['samples'][0]['joint_positions_rad_or_m'][i]
        if asset == 'glasses_case':
            for i, name in enumerate(p['joint_names']):
                if '_arm_' in name:
                    joints[i] = float(rng.uniform(p['joint_minimum'][i], p['joint_maximum'][i]))
        base = destination[:2] + r @ values[:2]
        target = [*(destination[:2] + r @ values[3:5]), destination[2]+local['target_xyz'][2]]
        target_q = yaw_rotate(local['target_q'], values[5]-p['target_yaw_unwrapped'][nearest]+angle)
        produced += 1
        yield dict(schema='short_tasks.retrieve_from_drawer.range_candidate.v1', asset=asset,
                   seed=seed, candidate_index=produced-1, proposal_attempt=attempt,
                   sampling='continuous_measured_ranges_v1',
                   source_bank_sha256=p['source_bank_sha256'],
                   support_posture_reference_episode=reference['episode_index'],
                   initial_base_xyyaw=[float(base[0]), float(base[1]), wrap(values[2]+angle)],
                   source_base_height_m=reference['robot_base_xyz_xyzw'][2],
                   target_xyz_xyzw=list(map(float, [*target, *target_q])),
                   targets_rad=dict(zip(p['joint_names'], joints)),
                   joint_units='rad except idx01_body_joint1 in metres',
                   destination_drawer_xyz_xyzw=list(map(float, drawer_xyz_xyzw)),
                   sampled_local_values=values.tolist(),
                   target_local_height_m=local['target_xyz'][2],
                   target_fit_checked=False, robot_collision_checked=False,
                   native_physics_validated=False, shared_preparer_connected=False,
                   status='proposal_only_not_executable')
        if produced == count:
            return
    raise RuntimeError('Measured radial envelope exhausted proposal budget; no fallback or widening')

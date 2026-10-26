"""Continuous Short 6 initial proposals; hand modes never mixed.

Robot relative pose and cap orientation use measured ranges. The caller can
provide the user-approved middle-two-cushion XY domain; without it this module
retains the historical measured-only sampler for comparison/tests.
Source review and final trajectory qualification are not initial acceptance.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from short_tasks.tasks.retrieve_from_drawer.initial_state import pose, yaw
from short_tasks.tasks.retrieve_from_drawer.range_initial_state import wrap, planar_rotation, yaw_rotate

SOURCE = Path(__file__).with_name('source_initial_samples.json')
APPROVED_CAP_YAW_RAD = (-math.pi / 2, 0.0)
APPROVED_BASE_YAW_HALFSPAN_RAD = math.radians(5)


def sofa_frame(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('Finite sofa world matrix required')
    # This asset is Y-up; its local -Z points into the seat from the front.
    inward = -m[2, :3]
    if np.linalg.norm(inward[:2]) == 0 or abs(inward[2]) > np.linalg.norm(inward)*1e-5:
        raise ValueError('Expected upright canonical sofa geometry')
    return m[3, :3], math.atan2(inward[1], inward[0])


def read_samples():
    data = json.loads(SOURCE.read_text())
    if data['schema'] != 'short_tasks.hang_cap.initial_samples.v1':
        raise ValueError('Wrong source bank')
    rows = data['samples']
    if len(rows) != 50 or len({r['source_id'] for r in rows}) != 50:
        raise ValueError('Expected 50 distinct dataset episodes')
    for r in rows:
        if r['active_hand'] not in ('l', 'r') or r['source_initial_settle_passed'] is not True:
            raise ValueError('Invalid hand/initial evidence')
        names, values = r['joint_names'], r['joint_positions_rad_or_m']
        if len(names) != 34 or len(set(names)) != 34 or len(values) != 34 or not np.isfinite(values).all():
            raise ValueError('Complete finite articulation state required')
        sofa_frame(r['sofa_world_matrix'])
        pose(r['target_xyz_xyzw']); pose(r['robot_base_xyz_xyzw'])
    return rows


def profile(hand):
    if hand not in ('l', 'r'):
        raise ValueError('Hand must be l or r')
    rows = [r for r in read_samples() if r['active_hand'] == hand]
    values, residuals = [], []
    for r in rows:
        origin, heading = sofa_frame(r['sofa_world_matrix'])
        base, bq = pose(r['robot_base_xyz_xyzw']); target, tq = pose(r['target_xyz_xyzw'])
        rotation = planar_rotation(-heading)
        values.append([*(rotation @ (base[:2]-target[:2])), wrap(yaw(bq)-heading),
                       *(rotation @ (target[:2]-origin[:2])), wrap(yaw(tq)-heading)])
        residuals.append(yaw_rotate(tq, -heading))
    a = np.asarray(values)
    a[:,5] = a[0,5] + np.array([wrap(v-a[0,5]) for v in a[:,5]])
    radii = np.linalg.norm(a[:,:2], axis=1)
    joints = np.array([r['joint_positions_rad_or_m'] for r in rows])
    if any(r['joint_names'] != rows[0]['joint_names'] for r in rows):
        raise ValueError('Inconsistent source joint order')
    return dict(hand=hand, source_count=len(rows), minimum=a.min(0), maximum=a.max(0),
                distance_min_max=[radii.min(), radii.max()], values=a,
                joint_minimum=joints.min(0), joint_maximum=joints.max(0),
                rows=rows, residuals=residuals)


def sample_candidates(seed, sofa_world_matrix, *, count, target_local_bounds=None, fixed_station=None,
                      station_offset_bounds=None):
    if type(seed) is not int or not 1 <= seed <= 4294967295 or type(count) is not int or count < 1:
        raise ValueError('Positive seed and proposal count required')
    origin, heading = sofa_frame(sofa_world_matrix)
    if fixed_station is not None:
        station=np.asarray(fixed_station,dtype=float)
        if station.shape!=(3,) or not np.isfinite(station).all() or target_local_bounds is None:
            raise ValueError('Approved domain and finite fixed XY/yaw station required')
    if station_offset_bounds is not None:
        offsets=np.asarray(station_offset_bounds,dtype=float)
        if fixed_station is None or offsets.shape!=(2,2) or not np.isfinite(offsets).all() or np.any(offsets[0]>=offsets[1]):
            raise ValueError('Station center and finite increasing local XY bounds required')
    digest = hashlib.sha256(f'hang-cap-ranges-v1:{seed}'.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))
    modes = {h: profile(h) for h in ('l', 'r')}
    rotation = planar_rotation(heading)
    produced = 0
    for attempt in range(count*100):
        hand = ('l', 'r')[int(rng.integers(2))]
        p = modes[hand]; v = rng.uniform(p['minimum'], p['maximum'])
        if target_local_bounds is not None:
            domain=np.asarray(target_local_bounds,dtype=float)
            if domain.shape!=(2,2) or not np.isfinite(domain).all() or np.any(domain[0]>=domain[1]):
                raise ValueError('Invalid approved target domain')
            v[3:5]=rng.uniform(domain[0],domain[1])
            v[5]=rng.uniform(*APPROVED_CAP_YAW_RAD)
        radius = np.linalg.norm(v[:2])
        if fixed_station is None and not p['distance_min_max'][0] <= radius <= p['distance_min_max'][1]:
            continue
        nearest = int(np.argmin(np.abs(p['values'][:,5]-v[5])))
        reference = p['rows'][nearest]
        joints = dict(zip(reference['joint_names'], reference['joint_positions_rad_or_m']))
        for i,name in enumerate(reference['joint_names']):
            if '_arm_' in name:
                joints[name] = float(rng.uniform(p['joint_minimum'][i], p['joint_maximum'][i]))
        target_xy = origin[:2]+rotation@v[3:5]
        base = target_xy+rotation@v[:2]
        if fixed_station is not None:
            base=station[:2].copy()
            if station_offset_bounds is not None:
                base+=planar_rotation(station[2])@rng.uniform(offsets[0],offsets[1])
            v[:2]=planar_rotation(-heading)@(base-target_xy)
            v[2]=wrap(station[2]-heading+rng.uniform(-APPROVED_BASE_YAW_HALFSPAN_RAD,
                                                    APPROVED_BASE_YAW_HALFSPAN_RAD))
            radius=np.linalg.norm(base-target_xy)
        old_origin,_ = sofa_frame(reference['sofa_world_matrix'])
        z = origin[2]+reference['target_xyz_xyzw'][2]-old_origin[2]
        q = yaw_rotate(p['residuals'][nearest], v[5]-p['values'][nearest,5]+heading)
        yield dict(schema='short_tasks.hang_cap.range_candidate.v1', seed=seed,
            candidate_index=produced, proposal_attempt=attempt, active_hand=hand,
            sampling=('middle_two_cushions_centered_base_xy_yaw_v5' if station_offset_bounds is not None
                      else 'middle_two_cushions_fixed_center_base_yaw5_v4' if fixed_station is not None
                      else 'middle_two_cushions_yaw_minus90_to0_v3' if target_local_bounds is not None
                      else 'continuous_measured_ranges_by_hand_v1'), sampled_local_values=v.tolist(),
            robot_station_xyyaw=None if fixed_station is None else station.tolist(),
            robot_station_offset_bounds_m=None if station_offset_bounds is None else offsets.tolist(),
            robot_yaw_halfspan_deg=None if fixed_station is None else 5.0,
            yaw_sampling_rad=(list(APPROVED_CAP_YAW_RAD) if target_local_bounds is not None
                              else [float(p['minimum'][5]),float(p['maximum'][5])]),
            target_local_bounds_m=None if target_local_bounds is None else domain.tolist(),
            initial_base_xyyaw=[*map(float,base),wrap(v[2]+heading)],
            target_xyz_xyzw=[*map(float,target_xy),float(z),*map(float,q)],
            targets_rad=joints, target_distance_m=float(radius),
            support_posture_reference_episode=reference['episode_index'],
            geometry_reference_source_id=reference['source_id'],
            source_bank_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            native_physics_validated=False, status='proposal_only_not_executable')
        produced += 1
        if produced == count:
            return
    raise RuntimeError('Measured cap envelope exhausted without widening')

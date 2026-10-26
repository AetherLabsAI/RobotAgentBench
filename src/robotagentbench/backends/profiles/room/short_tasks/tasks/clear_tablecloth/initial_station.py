"""User-approved cloth-relative continuous robot initialization; read-only."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source_ready_pose.json'
POLICY = HERE / 'initial_station_policy.json'


def sample_candidates(seed, delta, context, count=100):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    delta = np.asarray(delta, dtype=float)
    if (delta.shape != (4, 4) or not np.isfinite(delta).all()
            or not np.allclose(delta[:3, :3] @ delta[:3, :3].T, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(delta[:3, :3]), 1)
            or not np.allclose(delta[:, 3], [0, 0, 0, 1])
            or not np.allclose(delta[2, :3], [0, 0, 1], atol=1e-7)):
        raise ValueError('Rigid upright table transform required')
    policy = json.loads(POLICY.read_text())
    source = json.loads(SOURCE.read_text())
    table = np.asarray(context['table_bounds'], dtype=float)
    cloth = np.asarray(context['cloth_bounds'], dtype=float)
    if table.shape != (2, 3) or cloth.shape != (2, 3) or not np.isfinite([table, cloth]).all():
        raise ValueError('Finite table and cloth bounds required')
    nominal = np.array([table[0, 0] - policy['nominal_edge_gap_m'],
        cloth[:, 1].mean() - policy['nominal_robot_right_offset_m'], 0.])
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'cloth-station-v1:{seed}'.encode()).digest(), 'big'))
    lo, hi = policy['annulus_m']
    for i in range(count):
        radius = math.sqrt(rng.uniform(lo**2, hi**2))
        theta = rng.uniform(-math.pi, math.pi)
        offset = radius * np.array([math.cos(theta), math.sin(theta)])
        local = nominal.copy()
        local[:2] += offset
        base = np.r_[local, 1.] @ delta
        angle = float(rng.uniform(policy['yaw_min_abs_deg'], policy['yaw_half_width_deg']))
        angle *= -1 if rng.integers(2) else 1
        inward = np.array([1., 0., 0.]) @ delta[:3, :3]
        yaw = math.atan2(inward[1], inward[0]) + math.radians(angle)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        targets = dict(source['nominal_targets'])
        jitter = rng.uniform(-math.radians(policy['arm_half_width_deg']),
                             math.radians(policy['arm_half_width_deg']), len(source['arm_names']))
        for name, value in zip(source['arm_names'], jitter):
            targets[name] += float(value)
        yield dict(candidate_index=i, sampling='approved_cloth_edge_annulus_v1',
            initial_base_xyyaw=[float(base[0]), float(base[1]), yaw],
            canonical_base_xyz=local.tolist(), nominal_base_xyz=nominal.tolist(),
            xy_delta_m=offset.tolist(), annulus_radius_m=radius,
            edge_distance_m=float(table[0, 0]-local[0]),
            robot_right_of_cloth_m=float(cloth[:, 1].mean()-local[1]),
            yaw_deviation_deg=angle, arm_delta_rad=jitter.tolist(), targets_rad=targets,
            joints_sampling='task nominal arms plus independent +/-5 degrees; other source coordinates preserved')


def table_context(stage, manifest):
    from pxr import Gf
    from short_tasks.tasks.clear_table_trash.initial_station import table_context as measured_table
    from openend.tasks.cloth_clear.scene import bounds
    delta, table, baseline = measured_table(stage, manifest)
    rows = [r for r in manifest['objects'] if r['id'] == 'tablecloth_dirty']
    if len(rows) != 1:
        raise ValueError('Exactly one scene-owned cloth required')
    cloth = bounds(stage, rows[0]['path'], Gf.Matrix4d(delta.tolist()))
    if (np.any(cloth[0, :2] < table[0, :2]-1e-5)
            or np.any(cloth[1, :2] > table[1, :2]+1e-5)):
        raise ValueError('Cloth outside actual table')
    return delta, dict(table_bounds=table.tolist(), cloth_bounds=cloth.tolist(),
                      cloth_path=rows[0]['path']), baseline

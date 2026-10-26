"""Data-derived robot station proposals, not an object randomizer or IK proof."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE / 'initial_station_policy.json'
SUPPORTED = ('wallet', 'power_bank', 'water_bottle')


def enabled(manifest):
    rows = [r for r in manifest.get('objects', []) if r.get('task_id') == 'pack_object']
    return len(rows) == 1 and rows[0].get('id') in SUPPORTED and rows[0].get('source') == 'sofa'


def rigid_delta(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(a[:3,:3] @ a[:3,:3].T, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(a[:3,:3]), 1, atol=1e-7)
            or not np.allclose(a[2,:3], [0,0,1], atol=1e-7)
            or abs(a[3,2]) > 1e-7):
        raise ValueError('Sofa transform must be rigid, upright and floor-height preserving')
    return a


def sample_candidates(seed, delta, context, count=100):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    if type(count) is not int or count < 1:
        raise ValueError('Positive candidate count required')
    delta = rigid_delta(delta)
    policy = json.loads(POLICY.read_text())
    asset = context['asset']
    if asset not in SUPPORTED or context.get('source') != 'sofa':
        raise ValueError('No audited data-range robot reset for this source/asset')
    target = np.asarray(context['target_xy'], dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('Actual target XY required')
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(f'pack-sofa-station-v1:{asset}:{seed}'.encode()).digest(), 'big'))
    for i in range(count):
        du = float(rng.uniform(*policy['forward_offset_m']))
        dv = float(rng.uniform(*policy['lateral_offset_m']))
        angle = float(rng.uniform(*policy['yaw_deviation_deg']))
        local = np.array([*policy['canonical_station_xy'], 0., 1.])
        local[:2] += [du, dv]
        base = local @ delta
        inward = np.array([1., 0., 0.]) @ delta[:3,:3]
        yaw = math.atan2(inward[1], inward[0]) + math.radians(angle)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        # This only selects the source gripper-ready variant. It does not bind
        # which hand the policy must use, or award any grasp evidence.
        hand = 'l' if np.dot(target-base[:2], left) >= 0 else 'r'
        profile = policy['profiles'][asset+':'+hand]
        targets = dict(profile['targets_rad'])
        lo, hi = policy['arm_noise_rad'][asset]
        noise = rng.uniform(lo, hi)
        for name, jitter in zip(policy['arm_names'], noise):
            targets[name] += float(jitter)
        front_x = policy['canonical_sofa_center_xy'][0] + policy['canonical_seat_front_u']
        yield dict(candidate_index=i, sampling='continuous_source_sofa_station_v1',
            initial_base_xyyaw=[float(base[0]), float(base[1]), yaw],
            canonical_base_xyz=local[:3].tolist(), forward_delta_m=du,
            lateral_delta_m=dv, yaw_deviation_deg=angle,
            edge_distance_m=float(front_x-local[0]), source_posture_hand=hand,
            targets_rad=targets, arm_delta_rad=noise.tolist(),
            joints_sampling='source nominal posture plus source per-joint uniform bands')


def sofa_context(stage, manifest):
    from pxr import Usd, UsdGeom
    from short_tasks.agent_probe_v1.target_spawn import ROOT
    baseline = ROOT / 'layout_task_activation_v1/calibration_drafts_layout01.json'
    rows = [r for r in json.loads(baseline.read_text())['entries'] if r['id'] == 'sofa']
    if len(rows) != 1:
        raise ValueError('One calibrated sofa reference required')
    ref = rows[0]['target']
    prim = stage.GetPrimAtPath(ref['anchor'])
    if not prim or not prim.IsActive():
        raise ValueError('Actual sofa anchor missing')
    cache = UsdGeom.XformCache()
    reference_scene = Path(manifest['source_layout']).parent / 'ID_01_LIVING_STRAIGHT.usdc'
    original = Usd.Stage.Open(str(reference_scene))
    if not original:
        raise ValueError('Source sofa reference scene missing')
    def meshes(scene):
        transform = UsdGeom.XformCache()
        result = {}
        for child in Usd.PrimRange(scene.GetPrimAtPath(ref['anchor']), Usd.TraverseInstanceProxies()):
            if child.IsA(UsdGeom.Mesh) and child.IsActive():
                points = np.asarray(child.GetAttribute('points').Get(), dtype=float)
                if points.size:
                    matrix = np.asarray(transform.GetLocalToWorldTransform(child))
                    result[str(child.GetPath())] = (np.c_[points, np.ones(len(points))] @ matrix)[:,:3]
        return result
    aa, bb = meshes(original), meshes(stage)
    if not aa or set(aa) != set(bb) or any(aa[k].shape != bb[k].shape for k in aa):
        raise ValueError('Sofa mesh topology differs from source')
    a, b = (np.concatenate([m[k] for k in sorted(m)]) for m in (aa, bb))
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    rotation = u @ vt
    delta = np.eye(4)
    delta[:3,:3] = rotation
    delta[3,:3] = bc-ac@rotation
    delta = rigid_delta(delta)
    if not np.allclose(a@rotation+delta[3,:3], b, atol=1e-5, rtol=0):
        raise ValueError('Sofa vertex transfer is not rigid')
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'], False, True)
    # Root transformation alone is insufficient if someone edited children.
    measured = np.array([a.min(0), a.max(0)])
    expected = np.array([ref['bounds']['min'], ref['bounds']['max']])
    if not np.isfinite(measured).all() or not np.allclose(measured, expected, atol=1e-5, rtol=0):
        raise ValueError('Sofa child geometry differs from the source frame')
    owned = [r for r in manifest['objects'] if r.get('task_id') == 'pack_object']
    if not enabled(manifest):
        raise ValueError('No audited pack_object station for these bindings')
    row = owned[0]
    target = stage.GetPrimAtPath(row['path'])
    if not target or not target.IsActive():
        raise ValueError('Scene-owned pack target missing')
    bounds = bbox.ComputeWorldBound(target).ComputeAlignedRange()
    centre = (np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
    return delta, dict(asset=row['id'], source=row['source'], target_xy=centre[:2].tolist(),
        target_path=row['path'], sofa_path=ref['anchor'],
        reference_files=[str(baseline),str(reference_scene)]), baseline

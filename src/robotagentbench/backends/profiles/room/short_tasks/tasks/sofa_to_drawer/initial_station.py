"""Task-local source-range initialization; never a score or physics policy."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE / 'initial_station_policy.json'
SOURCE = HERE / 'source_initial_samples.json'
TASK_NAME = HERE.name


def policy():
    return json.loads(POLICY.read_text())


def sample_candidates(seed, delta, context, count=100):
    from short_tasks.tasks.sofa_to_coffee_table.initial_station import rigid_delta
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    if type(count) is not int or count < 1:
        raise ValueError('Positive candidate count required')
    delta = np.asarray(rigid_delta(np.asarray(delta).tolist()))
    p = policy()
    if context.get('source') != p['source'] or context.get('asset') not in p['profiles']:
        raise ValueError('Task source and exact asset must match the measured family')
    profile = p['profiles'][context['asset']]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'{TASK_NAME}-initial-v1:{seed}'.encode()).digest(), 'big'))
    for i in range(count):
        local = np.array([rng.uniform(*profile['base_x_m']),
                          rng.uniform(*profile['base_y_m']), 0., 1.])
        calibrated = local.copy()
        calibrated[:3] += np.asarray(context.get('source_frame_offset_xyz', [0.,0.,0.]))
        base = calibrated @ delta
        angle = float(rng.uniform(*profile['yaw_offset_rad']))
        yaw = p['heading_rad'] + angle + math.atan2(delta[0,1], delta[0,0])
        targets = dict(profile['fixed_nonarm_targets'])
        targets.update({name:float(rng.uniform(*limits)) for name,limits in profile['joint_ranges'].items()
                        if '_arm_' in name})
        yield dict(candidate_index=i, sampling='continuous_measured_furniture_frame_v1',
            asset=context['asset'], initial_base_xyyaw=[float(base[0]),float(base[1]),
                math.atan2(math.sin(yaw),math.cos(yaw))],
            canonical_base_xyz=local[:3].tolist(), yaw_deviation_deg=math.degrees(angle),
            targets_rad=targets, native_physics_validated=False)


def enabled(manifest):
    rows = [r for r in manifest['objects'] if r.get('task_id') == TASK_NAME]
    return (len(rows) == 1 and rows[0].get('source') == policy()['source']
            and rows[0].get('asset_id', rows[0].get('id')) in policy()['profiles'])


def support_context(stage, manifest):
    from pxr import Usd, UsdGeom
    from short_tasks.tasks.sofa_to_coffee_table.initial_station import rigid_delta
    if not enabled(manifest):
        raise ValueError('Exactly one supported task-owned source object required')
    row = next(r for r in manifest['objects'] if r.get('task_id') == TASK_NAME)
    if row.get('role') in ('extra','background','infrastructure'):
        raise ValueError('A background object is not a task target')
    anchor = '/World/Assets/Sofa' if policy()['source'] == 'sofa' else '/World/Assets/CoffeeTable'
    baseline = Path(manifest['source_layout']).parent / 'ID_01_LIVING_STRAIGHT.usdc'
    original = Usd.Stage.Open(str(baseline))
    if not original:
        raise ValueError('Canonical furniture reference is missing')
    def meshes(scene):
        root = scene.GetPrimAtPath(anchor)
        if not root or not root.IsActive():
            raise ValueError('Actual source furniture missing')
        cache=UsdGeom.XformCache(); result={}
        for child in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            path=str(child.GetPath())
            if '/black_box_1' in path or not child.IsActive() or not child.IsA(UsdGeom.Mesh):
                continue
            points=np.asarray(child.GetAttribute('points').Get(),dtype=float)
            if points.size:
                result[path]=(np.c_[points,np.ones(len(points))] @
                    np.asarray(cache.GetLocalToWorldTransform(child)))[:,:3]
        return result
    aa,bb=meshes(original),meshes(stage)
    if not aa or set(aa)!=set(bb) or any(aa[k].shape!=bb[k].shape for k in aa):
        raise ValueError('Source furniture topology changed')
    a,b=(np.concatenate([d[k] for k in sorted(d)]) for d in (aa,bb))
    ac,bc=a.mean(0),b.mean(0)
    u,_,vt=np.linalg.svd((a-ac).T @ (b-bc));rotation=u@vt
    delta=np.eye(4);delta[:3,:3]=rotation;delta[3,:3]=bc-ac@rotation
    delta=np.asarray(rigid_delta(delta.tolist()))
    if not np.allclose(a@rotation+delta[3,:3],b,atol=1e-5,rtol=0):
        raise ValueError('Nonrigid furniture transfer is not supported')
    if not stage.GetPrimAtPath(row['path']):
        raise ValueError('Task target is absent')
    offset=np.zeros(3)
    if policy()['source']=='coffee_table.objects':
        measured=json.loads(SOURCE.read_text())['samples'][0]['table_bounds']
        expected=np.asarray(measured)
        if not np.allclose(baseline_size:=a.max(0)[:2]-a.min(0)[:2],expected[1,:2]-expected[0,:2],atol=1e-5):
            raise ValueError('Source table dimensions changed')
        offset[:2]=(a.max(0)[:2]+a.min(0)[:2])/2-expected[:,:2].mean(0)
    return delta, dict(asset=row.get('asset_id',row['id']), source=row['source'],
        source_frame_offset_xyz=offset.tolist(),
        target_path=row['path'], support_path=anchor,
        reference_files=[str(baseline),str(SOURCE),str(POLICY)],
        support_bounds=[a.min(0).tolist(),a.max(0).tolist()]), baseline

"""Read-only source audit: 20 put-task initial scenes and recorded door motion.

No simulation, USD edits, task execution, new success labels or arm-sweep claims.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

HERE = Path(__file__).resolve().parent
CATALOG = Path('/home/lingjun/robot_agent_arena/api-protocol/evaluation/tasks/microwave_put/references/api20_20261002/catalog.json')
LAYOUT = Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
MICRO = '/World/Assets/DiningTableMicrowave'
DOOR = MICRO + '/Asset/door'
TABLE = '/World/Background/Shell/CounterTop_17'
ROBOT = '/World/Robot/base_link'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def corners(lo, hi):
    return np.array([list(v) + [1.] for v in itertools.product(*zip(lo, hi))])

def bounds(points):
    p = np.asarray(points).reshape(-1, 3)
    lo, hi = p.min(0), p.max(0)
    return dict(min=lo.tolist(), max=hi.tolist(), size=(hi-lo).tolist())

def measure(scene):
    stage = Usd.Stage.Open(str(scene))
    assert stage and UsdGeom.GetStageMetersPerUnit(stage) == 1
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    xf = UsdGeom.XformCache()
    def matrix(path):
        p = stage.GetPrimAtPath(path)
        assert p, path
        return np.asarray(xf.GetLocalToWorldTransform(p)).T
    def bb(path):
        p = stage.GetPrimAtPath(path)
        assert p, path
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        assert not b.IsEmpty(), path
        return bounds([b.GetMin(), b.GetMax()])
    micro, table = bb(MICRO), bb(TABLE)
    m, r = matrix(MICRO), matrix(ROBOT)
    is_layout = Path(scene).resolve() == LAYOUT.resolve()
    if not is_layout:
        assert table['min'][0] > r[0, 3], 'This audit assumes the verified west/operator table edge'
    metrics = dict(front_to_operator_edge_m=micro['min'][0]-table['min'][0],
                   root_to_operator_edge_m=m[0,3]-table['min'][0],
                   robot_to_microwave_root_xy_m=None if is_layout else float(np.linalg.norm(m[:2,3]-r[:2,3])),
                   robot_to_operator_edge_m=None if is_layout else table['min'][0]-r[0,3])
    return stage, cache, matrix, dict(metrics=metrics, microwave_bbox=micro,
                                    table_bbox=table, microwave_matrix=m.tolist(), robot_matrix=r.tolist())

def local_visible_bounds(stage, cache, matrix, path):
    inv = np.linalg.inv(matrix(path)); points = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        im = UsdGeom.Imageable(prim)
        if im.ComputeVisibility() == 'invisible' or im.ComputePurpose() not in ('default','render','proxy'):
            continue
        if prim.IsA(UsdGeom.Mesh):
            xyz = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            if xyz.size == 0: continue
            p = np.c_[xyz, np.ones(len(xyz))]
        else:
            r = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if r.IsEmpty(): continue
            p = corners(r.GetMin(), r.GetMax())
        points.append((inv @ matrix(str(prim.GetPath())) @ p.T).T[:,:3])
    return bounds(np.concatenate(points))

def main():
    catalog = json.loads(CATALOG.read_text()); records = []
    for case, info in sorted(catalog['cases'].items()):
        source = Path(info['source']); scene = source/'initial_scene.usda'
        assert sha(scene) == info['source_scene_sha256'], case
        checkpoint = CATALOG.parent/case
        manifest_path = checkpoint/'source_manifest.json'
        assert sha(manifest_path) == info['checkpoint_manifest_sha256'], case
        manifest = json.loads(manifest_path.read_text())
        control = checkpoint/'control_state_60hz.npz'
        expected = next(x['sha256'] for x in manifest['files'] if x['local_name']==control.name)
        assert sha(control) == expected, case
        stage, cache, matrix, result = measure(scene)
        door = local_visible_bounds(stage,cache,matrix,DOOR)
        a = np.load(control,allow_pickle=False)
        paths = list(a['rigid_body_paths']); poses = a['body_pose'][:,paths.index(DOOR),:]
        assert np.isfinite(poses).all() and len(poses)>0
        transforms = np.broadcast_to(np.eye(4),(len(poses),4,4)).copy()
        transforms[:,:3,:3] = Rotation.from_quat(poses[:,3:]).as_matrix()
        transforms[:,:3,3] = poses[:,:3]
        points = np.einsum('nij,kj->nki', np.linalg.inv(matrix(MICRO)) @ transforms,
                           corners(door['min'],door['max']))[:,:,:3]
        result.update(case=case,source=str(source),source_scene_sha256=info['source_scene_sha256'],
                      control_sha256=expected,door_frames=len(poses),door_sweep_in_microwave=bounds(points))
        records.append(result)
        print(case, result['metrics'], flush=True)
    _,_,_,current = measure(LAYOUT)
    union = bounds([v for r in records for v in (r['door_sweep_in_microwave']['min'],r['door_sweep_in_microwave']['max'])])
    m = np.array(current['microwave_matrix'])
    current['put_only_door_sweep_projection'] = bounds((m @ corners(union['min'],union['max']).T).T[:,:3])
    summary = {k:dict(min=min(r['metrics'][k] for r in records),max=max(r['metrics'][k] for r in records),
                      min_case=min(records,key=lambda r:r['metrics'][k])['case'],max_case=max(records,key=lambda r:r['metrics'][k])['case'])
               for k in records[0]['metrics']}
    report = dict(schema='arena.microwave.placement_audit.v1',scope='9.2 put-task source recordings only',
        source_count=len(records),catalog=str(CATALOG),catalog_sha256=sha(CATALOG),summary=summary,
        layout_scene=str(LAYOUT),layout_sha256=sha(LAYOUT),current_layout=current,
        door_sweep_in_microwave=union,door_control_frames=sum(r['door_frames'] for r in records),records=records,
        limitations=['Recorded sources, not twenty successful API replays.',
            'Initial visual aligned bounds, west/operator edge; not collision hull or support validation.',
            'Door bounds enclose visible geometry transformed by recorded 60Hz body poses; not 240Hz substeps or inter-frame certification.',
            'Source door envelope mapped to layout is a geometric preview, not a replay or verified reachability.',
            'No arm/payload swept envelope computed here. Take-task 10.1 not included. No final shared microwave exclusion region or approved distance policy yet.'])
    (HERE/'microwave_placement_audit_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)
    print('CURRENT',json.dumps(current),flush=True)

if __name__=='__main__': main()

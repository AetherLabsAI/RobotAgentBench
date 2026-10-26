"""Read-only sampled workspace audit. No USD edits, simulation or task execution.

Outputs are confined to this directory. Local AABBs enclose source geometry;
hand spheres enclose all finger configurations using joint-chain lengths.
Neither is a continuous-time collision certificate.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
REPO = Path('/home/lingjun/robot_agent_arena/api-protocol')
EPISODES = REPO / 'evaluation/tasks/air_fryer/resources/source_episodes'
ROOT = '/World/Robot'
STATION = '/World/Task66_67'
FRYER = STATION + '/AirFryerPlacement/AirFryer'
TRAY = FRYER + '/Centered/Tray'
ROLES = {'tray': TRAY, 'target': STATION + '/Cakes/PlateCakeA',
         'other': STATION + '/Cakes/PlatePlainCakeA'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(flat):
    return np.asarray(flat, float).reshape(4, 4).T


def corners(lo, hi):
    return np.array([list(x) + [1.] for x in itertools.product(*zip(lo, hi))])


def joint_frame(p, i):
    q = p.GetAttribute('physics:localRot' + str(i)).Get()
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
    m[:3, 3] = p.GetAttribute('physics:localPos' + str(i)).Get()
    return m


def box(points):
    p = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])[..., :3]
    return {'min': p.min(axis=0).tolist(), 'max': p.max(axis=0).tolist(),
            'size': np.ptp(p, axis=0).tolist()}


def union(boxes):
    return box([p for b in boxes for p in (b['min'], b['max'])])


def audit(path):
    manifest = json.loads(path.read_text())
    producer = manifest['producer']
    trace_path = Path(producer['trace'])
    assert sha(trace_path) == producer['trace_sha256'], trace_path
    assert sha(producer['scene']) == producer['scene_sha256'], producer['scene']
    rows = json.loads(trace_path.read_text())
    names = json.loads((path.parent / 'producer_targets.json').read_text())['joints']
    qs = np.asarray([r['measured_joint_positions'] for r in rows], float)
    assert qs.shape == (len(rows), len(names)) and np.isfinite(qs).all()
    assert len(rows) == producer['trace_rows']
    times = np.array([r['t'] for r in rows])
    assert np.allclose(np.diff(times), .05, atol=1e-6)
    stage = Usd.Stage.Open(producer['scene'])
    assert stage and UsdGeom.GetStageMetersPerUnit(stage) == 1
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], useExtentsHint=False)

    def world(p):
        return np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T

    def local_corners(p):
        r = bbox.ComputeUntransformedBound(stage.GetPrimAtPath(p)).ComputeAlignedRange()
        if r.IsEmpty():
            return np.array([[0., 0., 0., 1.]])
        return corners(r.GetMin(), r.GetMax())

    anchor = matrix(manifest['world']['fryer'])
    inv_anchor = np.linalg.inv(anchor)
    parents = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint, UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint)):
            continue
        a, b = [p.GetRelationship('physics:body' + str(i)).GetTargets() for i in (0, 1)]
        if len(a) != 1 or len(b) != 1:
            continue
        a, b = str(a[0]), str(b[0])
        assert b not in parents
        f0, f1 = joint_frame(p, 0), joint_frame(p, 1)
        moving = not p.IsA(UsdPhysics.FixedJoint)
        axis = 'XYZ'.index(p.GetAttribute('physics:axis').Get()) if moving else None
        angular = p.IsA(UsdPhysics.RevoluteJoint)
        relative = np.linalg.inv(world(a) @ f0) @ world(b) @ f1
        q0 = (Rotation.from_matrix(relative[:3, :3]).as_rotvec()[axis] if angular else relative[axis, 3]) if moving else 0.
        parents[b] = dict(parent=a, name=p.GetName(), f0=f0, f1inv=np.linalg.inv(f1),
                          axis=axis, angular=angular, q0=q0)
    frames = {ROOT + '/base_link': np.broadcast_to(world(ROOT + '/base_link'), (len(rows), 4, 4))}

    def fk(body):
        if body in frames:
            return frames[body]
        c = parents[body]
        t = fk(c['parent']) @ c['f0']
        motion = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
        if c['axis'] is not None:
            # Do not substitute targets or authored angles for unmeasured arm joints.
            assert c['name'] in names, c['name']
            values = qs[:, names.index(c['name'])]
            if c['angular']:
                axis = np.eye(3)[c['axis']]
                motion[:, :3, :3] = Rotation.from_rotvec(values[:, None] * axis).as_matrix()
            else:
                motion[:, c['axis'], 3] = values
        frames[body] = t @ motion @ c['f1inv']
        return frames[body]

    arm_boxes, hand_boxes, residuals = {}, {}, {}
    for side in ('l', 'r'):
        bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI) and (p.GetName().startswith('arm_' + side + '_') or p.GetName() == 'gripper_' + side + '_base_link')]
        arm_boxes[side] = union([box(np.einsum('nij,kj->nki', inv_anchor @ fk(b), local_corners(b))) for b in bodies])
        # Conservative all-configuration finger enclosure, not guessed finger angles.
        base = ROOT + '/gripper_' + side + '_base_link'
        radius = 0.
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
            body = str(p.GetPath())
            if not p.HasAPI(UsdPhysics.RigidBodyAPI) or not p.GetName().startswith('gripper_' + side + '_'):
                continue
            length = float(np.linalg.norm(local_corners(body)[:, :3], axis=1).max())
            cursor = body
            while cursor != base:
                c = parents[cursor]
                assert c['axis'] is None or c['angular']
                length += np.linalg.norm(c['f0'][:3, 3]) + np.linalg.norm(c['f1inv'][:3, 3])
                cursor = c['parent']
            radius = max(radius, length)
        centers = (inv_anchor @ fk(base))[:, :3, 3]
        hand_boxes[side] = dict(radius_m=radius, bounds=box(np.concatenate([centers - radius, centers + radius])))
        hand = ROOT + '/gripper_' + side + '_center_link'
        predicted = fk(hand)[:, :3, 3]
        if all('tcp_world' in r and side in r['tcp_world'] for r in rows):
            measured = np.asarray([r['tcp_world'][side] for r in rows])
            err = np.linalg.norm(predicted - measured, axis=1)
            residuals[side] = dict(max_m=float(err.max()), median_m=float(np.median(err)))

    objects = {}
    objects['fryer_closed'] = box(local_corners(FRYER))
    plate = STATION + '/DisplayPlatePlacement/Plate'
    objects['plate'] = box((inv_anchor @ matrix(manifest['world']['plate']) @ local_corners(plate).T).T)
    for role, prim in ROLES.items():
        poses = [r['poses'][prim] for r in rows]
        q = np.asarray([p['quaternion'] for p in poses])
        transforms = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
        transforms[:, :3, :3] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        transforms[:, :3, 3] = [p['position'] for p in poses]
        objects[role] = box(np.einsum('nij,kj->nki', inv_anchor @ transforms, local_corners(prim)))
    return dict(key=manifest['key'], label=manifest['label'], frames=len(rows),
                manifest_sha256=sha(path), scene_sha256=producer['scene_sha256'], trace=str(trace_path),
                trace_sha256=producer['trace_sha256'], source_status=manifest['producer_status_label'],
                object_bounds=objects, object_union=union(list(objects.values())), arm_geometry_bounds=arm_boxes,
                hand_conservative_bounds=hand_boxes, tcp_fk_residuals=residuals,
                combined_union=union([*objects.values(), *arm_boxes.values(), *[h['bounds'] for h in hand_boxes.values()]]))


def main():
    destination = HERE / 'task2_workspace_audit_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    results = []
    for p in sorted(EPISODES.glob('*/*/episode.json')):
        results.append(audit(p))
        print(json.dumps({'done': results[-1]['key'], 'label': results[-1]['label'],
                          'frames': results[-1]['frames'], 'residuals': results[-1]['tcp_fk_residuals']}), flush=True)
    report = dict(schema='arena.task2.sampled_workspace.v1', episodes=len(results),
                  frames=sum(r['frames'] for r in results), sample_rate_hz=20,
                  coordinate_frame='Each episode fryer root; +X drawer outward, +Y lateral, +Z up. Not robot-centered.',
                  per_task={k: dict(count=sum(r['key']==k for r in results),
                                   object_bounds=union([r['object_union'] for r in results if r['key']==k]),
                                   combined_bounds=union([r['combined_union'] for r in results if r['key']==k]))
                            for k in sorted({r['key'] for r in results})},
                  object_bounds=union([r['object_union'] for r in results]),
                  combined_bounds=union([r['combined_union'] for r in results]), records=results,
                  limitations=['20 selected source episodes, not the entire task2 dataset or a continuous full workflow.',
                               '20Hz measured joints and object poses, not all 240Hz physics substeps; inter-frame sweeps not certified.',
                               'Source geometry AABB enclosures, not exact convex collision hulls/contact offsets.',
                               'Unrecorded finger articulation is enclosed by conservative all-configuration joint-chain spheres.',
                               'No environment/self collision, target-layout support, navigation or physical validation.',
                               'No extra safety margin has been applied. Source task parameters/assets were not changed.',
                               'Physical controller/base constraints may differ on new layout; FK residuals must be reviewed.'])
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','limitations']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

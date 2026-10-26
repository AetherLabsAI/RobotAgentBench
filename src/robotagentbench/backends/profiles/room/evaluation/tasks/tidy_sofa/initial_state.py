"""Task-owned fixed reset layer for the tidy_sofa family. Activation and transforms only.

Joint coordinates are returned separately for the public RobotInitialization hook. Historical
initializers and their physics setters are never imported. The drawer slide is NOT authored
here: it lives in the public scene layer (assets/.../livingroom_drawer_slide.usda) composed by
BASE; this module only verifies that the composed drawer matches the source frame-0 pose.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from .specs import spec as task_spec, ROBOT, DRAWER, DRAWER_JOINT, DRAWER_INITIAL_LIP_M

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
BASELINE = REPO/'assets/house_scene_l1/demo_sim/scene_room_baseline.usda'
SLIDE_LAYER = REPO/'assets/house_scene_l1/demo_sim/livingroom_drawer_slide.usda'
TASK_LAYERS = (SLIDE_LAYER,)
BASE = REPO/'assets/house_scene_l1/demo_sim/scene_room_livingroom_cleanup.usda'
SCHEMA = 'arena.livingroom_cleanup.initial_state.v1'
RESET_OP = 'xformOp:transform:livingroom_reset'
CLAMP_LIMIT_RAD = 1e-3
PUBLIC_RESET_TOLERANCE_RAD = math.radians(.001)
DRAWER_POSE_TOLERANCE_M = 0.003


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_path(sp):
    return TASK/'resources'/f'source_initial_{sp.api_id}.json'


def load_snapshot(sp):
    data = json.loads(snapshot_path(sp).read_text())
    if data['schema'] != SCHEMA or data['api_id'] != sp.api_id or data['target_path'] != sp.target:
        raise ValueError('Wrong initial-state snapshot for ' + sp.api_id)
    poses = [data['target'], data['robot'], *data['scenery'].values()]
    if sp.kind == 'drawer':
        poses.append(data['drawer_initial'])
    for pose in poses:
        p, q = pose['position_m'], pose['quaternion_wxyz']
        if len(p) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in p+q) or abs(sum(x*x for x in q)-1) > 1e-5:
            raise ValueError('Invalid source pose')
    if set(data['scenery']) != set(sp.scenery):
        raise ValueError('Snapshot scenery differs from the API binding')
    return data


def compose_task_layers(environment, output):
    """Task scene = TASK_LAYERS over a prepared runs/ environment (tools/build_l1_scene.py only accepts BASELINE)."""
    from pxr import Usd
    environment, output = Path(environment).resolve(strict=True), Path(output).resolve()
    if output.exists() or not output.is_relative_to(REPO/'runs'):
        raise ValueError('Use a new runs/... scene for the composed task layers')
    source = Usd.Stage.Open(str(environment))
    stage = Usd.Stage.CreateNew(str(output))
    for key in ('upAxis', 'metersPerUnit', 'timeCodesPerSecond', 'framesPerSecond'):
        if source.HasAuthoredMetadata(key): stage.SetMetadata(key, source.GetMetadata(key))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output.parent) for p in (*TASK_LAYERS, environment)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetRootLayer().Save()
    return dict(scene=str(output), environment=str(environment), task_layers={str(p): digest(p) for p in TASK_LAYERS})


def audit_layer(layer, allowed_paths):
    """Reject any unexpected spec, relationship, metadata, or physics opinion in the reset layer."""
    from pxr import Sdf
    violations = []
    def inspect(path):
        s = layer.GetObjectAtPath(path)
        if isinstance(s, Sdf.PrimSpec):
            for key in s.ListInfoKeys():
                if key not in (('subLayers', 'subLayerOffsets', 'upAxis', 'metersPerUnit', 'defaultPrim', 'timeCodesPerSecond', 'framesPerSecond') if path == Sdf.Path.absoluteRootPath else ('specifier', 'active')):
                    violations.append(f'{path}: metadata {key}')
            if s.HasInfo('active') and str(path) not in allowed_paths:
                violations.append(f'{path}: activation outside reset instances')
        elif isinstance(s, Sdf.AttributeSpec):
            joint_frame = str(path.GetPrimPath()) == DRAWER_JOINT and s.name in ('physics:localPos0', 'physics:localRot0')
            if not joint_frame and (str(path.GetPrimPath()) not in allowed_paths or s.name not in (RESET_OP, 'xformOpOrder')):
                violations.append(f'{path}: forbidden attribute')
        elif isinstance(s, Sdf.RelationshipSpec):
            violations.append(f'{path}: forbidden relationship')
    layer.Traverse('/', inspect)
    if violations:
        raise ValueError('State-layer audit failed: ' + '; '.join(violations))


def clamp_recorded_joints(stage, joint_positions):
    from pxr import UsdPhysics
    clamped, result = {}, dict(joint_positions)
    for name, value in joint_positions.items():
        prim = stage.GetPrimAtPath(ROBOT+'/joints/'+name)
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint): continue
        lo, hi = (math.radians(prim.GetAttribute('physics:'+s+'Limit').Get()) for s in ('lower', 'upper'))
        target = min(max(value, lo), hi)
        if abs(target-value) > PUBLIC_RESET_TOLERANCE_RAD:
            if abs(target-value) > CLAMP_LIMIT_RAD:
                raise ValueError(f'Recorded joint {name} is {abs(target-value):.4f} rad outside its native limit')
            result[name] = target; clamped[name] = dict(recorded=value, limit=target, delta_rad=target-value)
    return result, clamped


def layout_delta(reference_stage, stage, anchor):
    """Rigid transform taking canonical world poses to the (possibly randomized) layout, from one furniture anchor."""
    from pxr import Gf, UsdGeom
    ref = UsdGeom.Xformable(reference_stage.GetPrimAtPath(anchor)).ComputeLocalToWorldTransform(0)
    cur = UsdGeom.Xformable(stage.GetPrimAtPath(anchor)).ComputeLocalToWorldTransform(0)
    if (Gf.Transform(ref).GetScale()-Gf.Transform(cur).GetScale()).GetLength() > 1e-6:
        raise ValueError('Anchor scale changed; state adapter requires a rigid layout transform: ' + anchor)
    return ref.RemoveScaleShear().GetInverse()*cur.RemoveScaleShear()


def prepare(output, *, sp, base=BASE):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    out = Path(output).resolve()
    if not out.is_relative_to(TASK/'artifacts') or out.exists():
        raise ValueError('Use a new directory under tidy_sofa/artifacts/')
    source = Path(base).resolve(strict=True)
    source_hash = digest(source)
    data = load_snapshot(sp)
    out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [os.path.relpath(source, out)]
    baseline = Usd.Stage.Open(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    for key in ('timeCodesPerSecond', 'framesPerSecond'):
        if baseline.HasAuthoredMetadata(key): stage.SetMetadata(key, baseline.GetMetadata(key))
    def require(path):
        prim = stage.GetPrimAtPath(path)
        if not prim: raise ValueError('Missing existing instance ' + path)
        return prim
    anchor = sp.support if sp.kind == 'pick' else sp.destination
    if not require(anchor).IsActive(): raise ValueError('Support/destination inactive in baseline: ' + anchor)
    reference = Usd.Stage.Open(str(BASE))
    frame_delta = layout_delta(reference, stage, anchor)
    paths = [sp.target, ROBOT, *sp.scenery]
    for path in paths: require(path).SetActive(True)
    def physics_facts():
        return {str(p.GetPath()): dict(schemas=list(p.GetAppliedSchemas()),
            attributes={a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:', 'physx'))},
            relationships={r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships() if r.GetName().startswith(('physics:', 'physx', 'material:binding'))})
            for path in paths for p in Usd.PrimRange(require(path))}
    before = physics_facts()
    hashes = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath) != out/'initial.usda'}
    def author_world(path, position, quaternion):
        prim = require(path)
        old = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        scale = Gf.Transform(old).GetScale()
        q = Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])).GetNormalized()
        world = Gf.Matrix4d(1).SetRotate(q); world.SetTranslateOnly(Gf.Vec3d(*position))
        world = world*frame_delta
        # Keep the instance's authored scale (Remote/BaseballCap/cushion carry cm→m scales); the
        # source rigid-body pose is the unscaled body frame, so re-apply the scale first.
        scaled = Gf.Matrix4d(1).SetScale(scale)*world
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='livingroom_reset').Set(scaled*parent.GetInverse())
        return [float(v) for v in scale]
    scales = {}
    scales[sp.target] = author_world(sp.target, data['target']['position_m'], data['target']['quaternion_wxyz'])
    scales[ROBOT] = author_world(ROBOT, data['robot']['position_m'], data['robot']['quaternion_wxyz'])
    for path, pose in data['scenery'].items():
        scales[path] = author_world(path, pose['position_m'], pose['quaternion_wxyz'])
    for path in (sp.target, *sp.scenery):
        body = UsdPhysics.RigidBodyAPI(require(path))
        if not body or body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
            raise ValueError('Task objects must already be dynamic rigid bodies: ' + path)
    drawer_check = None
    if sp.kind == 'drawer':
        joint = stage.GetPrimAtPath(DRAWER_JOINT)
        if not joint or not joint.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Drawer slide layer is not composed into ' + str(source))
        # The public slide layer anchors the joint frame (body0 = world) at the canonical layout; move that
        # frame rigidly with the table so the drawer stays in its compartment in randomized layouts.
        # Joint limits, friction, drive and materials are untouched (state only).
        pos0 = joint.GetAttribute('physics:localPos0').Get(); rot0 = joint.GetAttribute('physics:localRot0').Get()
        new_pos0 = frame_delta.Transform(Gf.Vec3d(*pos0))
        delta_rot = frame_delta.RemoveScaleShear().ExtractRotationQuat()
        new_rot0 = delta_rot * Gf.Quatd(rot0.GetReal(), Gf.Vec3d(*rot0.GetImaginary()))
        joint.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*[float(v) for v in new_pos0]))
        joint.GetAttribute('physics:localRot0').Set(Gf.Quatf(float(new_rot0.GetReal()), *[float(v) for v in new_rot0.GetImaginary()]))
        pos = UsdGeom.Xformable(require(DRAWER)).ComputeLocalToWorldTransform(0).ExtractTranslation()
        expected = frame_delta.Transform(Gf.Vec3d(*data['drawer_initial']['position_m']))
        error = float((pos-expected).GetLength())
        if error > DRAWER_POSE_TOLERANCE_M:
            raise ValueError(f'Composed drawer pose differs from the source frame 0 by {error:.4f} m')
        drawer_check = dict(composed_world_m=[float(v) for v in pos], source_world_m=[float(v) for v in expected], error_m=error,
                            initial_lip_m=DRAWER_INITIAL_LIP_M, joint=DRAWER_JOINT, joint_frame_world=dict(localPos0=[float(v) for v in new_pos0], localRot0_wxyz=[float(new_rot0.GetReal()), *[float(v) for v in new_rot0.GetImaginary()]]))
    errors = [str(e) for p in stage.Traverse() for e in p.GetPrimIndex().localErrors]
    if errors: raise ValueError('Composed scene has unresolved references: ' + str(errors[:5]))
    audit_layer(layer, set(paths))
    if before != physics_facts(): raise ValueError('Physics changed while authoring state')
    if any(digest(p) != h for p, h in hashes.items()) or digest(source) != source_hash:
        raise ValueError('Source files changed during preparation')
    layer.Save()
    joints, clamped = clamp_recorded_joints(stage, data['joint_positions'])
    record = dict(schema='arena.livingroom_cleanup.initial_plan.v1', api_id=sp.api_id, kind=sp.kind, scene=str(out/'initial.usda'),
                  source_scene=str(source), source_sha256=source_hash, dependency_sha256=hashes,
                  snapshot_sha256=digest(snapshot_path(sp)), target=sp.target, scenery=list(sp.scenery),
                  support=sp.support, destination=sp.destination, in_hand=sp.in_hand,
                  robot_initial_positions=joints, recorded_joints_clamped_to_native_limits=clamped,
                  instance_scales=scales, drawer_check=drawer_check,
                  source_to_layout_matrix=[list(row) for row in frame_delta],
                  physics_unchanged=True, authored_state_only=True, state_paths=sorted(paths), physical_acceptance=False)
    (out/'initial-state.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--base', default=str(BASE)); p.add_argument('--api', required=True)
    a = p.parse_args()
    result = prepare(a.output, sp=task_spec(a.api), base=a.base)
    print(json.dumps({k: result[k] for k in ('api_id', 'scene', 'physics_unchanged', 'physical_acceptance', 'drawer_check')}))


if __name__ == '__main__':
    main()

"""Task-local pre-physics reset: poses only, native imported assets retained."""
import hashlib
import json
import os
from pathlib import Path
from .initial_state import TASK_DIR, TARGET, ROBOT, FRIDGE
from .randomization import build_plan, fixed_plan
from arena_api.food_box_assets import ASSETS as SHARED_ASSETS, verify_files

EXTRAS = {'salad': '/World/Assets/SaladBoxR', 'food': '/World/Assets/BinFoodR'}
ASSETS = {kind: path.name for kind, path in SHARED_ASSETS.items()}
PARTS = {'shelf4': ('col_shelf4_floor', 'col_shelf4_lip', 'col_shelf4_panel'),
         'bin34': ('col_bin34_floor', 'col_shelf3_lip', 'col_bin34_panel')}


def validate_flags(flags):
    if any(v for k,v in flags.items() if k != 'layout'):
        raise ValueError('Refrigerator take only supports task-local layout randomization')


def physical_snapshot(stage):
    result = {}
    for p in stage.Traverse():
        for a in p.GetAttributes():
            if a.GetName().startswith(('physics:', 'physx', 'drive:')):
                result[str(a.GetPath())] = str(a.Get())
        for rel in p.GetRelationships():
            if 'physics' in rel.GetName() or rel.GetName().startswith('physx'):
                result[str(rel.GetPath())] = str(rel.GetTargets())
    return result


def build(scene, output, seed, *, fixed=False):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    verify_files()
    output = Path(output).resolve()
    if not output.is_relative_to(TASK_DIR.resolve()) or output == TASK_DIR.resolve():
        raise ValueError('Task-local output directory required')
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'random_scene.usda'
    if destination.exists():
        raise FileExistsError(destination)
    plan = fixed_plan() if fixed else build_plan(seed)
    source = Usd.Stage.Open(str(Path(scene).resolve(strict=True)))
    before = physical_snapshot(source)
    layer = Sdf.Layer.CreateAnonymous()
    layer.subLayerPaths = [str(Path(scene).resolve())]
    stage = Usd.Stage.Open(layer)
    for key in ('upAxis', 'metersPerUnit', 'framesPerSecond', 'timeCodesPerSecond'):
        stage.SetMetadata(key, source.GetMetadata(key))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    cache = UsdGeom.XformCache()
    door = stage.GetPrimAtPath(FRIDGE+'/right_door')
    door_tf = cache.GetLocalToWorldTransform(door)
    if not door:
        raise ValueError('Missing closed right door')

    def place(path, position, orientation):
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        scale = [op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale]
        xf.ClearXformOpOrder()
        t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, 'randomReset')
        q = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, 'randomReset')
        t.Set(Gf.Vec3d(*position)); q.Set(orientation.GetNormalized())
        xf.SetXformOpOrder([t,q,*scale], resetXformStack=True)

    dependencies = {}
    for kind, relative in ASSETS.items():
        path = EXTRAS[kind]
        if stage.GetPrimAtPath(path):
            raise ValueError('Background box path already exists: '+path)
        asset = SHARED_ASSETS[kind].resolve(strict=True)
        dependencies[relative] = hashlib.sha256(asset.read_bytes()).hexdigest()
        root = stage.DefinePrim(path, 'Xform')
        body = stage.DefinePrim(path+'/Asset', 'Xform')
        body.GetReferences().AddReference(str(asset))
        # USD references drop relationships pointing outside the referenced
        # root. Rebind the exact source's external material target in this
        # episode layer; never choose a replacement or change material values.
        original_asset = Usd.Stage.Open(str(asset))
        original_root = original_asset.GetDefaultPrim()
        for original in Usd.PrimRange(original_root):
            targets = original.GetRelationship('material:binding:physics').GetTargets()
            if targets:
                for target in targets:
                    if not stage.GetPrimAtPath(target):
                        raise ValueError('Missing original external physics material: '+str(target))
                relative_prim = original.GetPath().MakeRelativePath(original_root.GetPath())
                destination_prim = stage.GetPrimAtPath(body.GetPath().AppendPath(relative_prim))
                destination_prim.CreateRelationship('material:binding:physics').SetTargets(targets)
        # Shared assets bake the exact original fridge dimensions. No second scale.
        if (not body.HasAPI(UsdPhysics.RigidBodyAPI)
                or UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()):
            raise ValueError('Imported box must already be a dynamic rigid body')
        for p in Usd.PrimRange(root):
            for target in p.GetRelationship('material:binding:physics').GetTargets():
                if not stage.GetPrimAtPath(target):
                    raise ValueError('Unresolved native physics material: '+str(target))
    # Preserve all source asset physics and joint reset/drive settings exactly.
    added_physics = physical_snapshot(stage)
    for kind, box in plan['boxes'].items():
        path = TARGET if kind == 'beef' else EXTRAS[kind]
        q = box['door_local_quaternion_wxyz']
        quat = door_tf.ExtractRotationQuat()*Gf.Quatd(q[0], Gf.Vec3d(*q[1:]))
        p = door_tf.Transform(Gf.Vec3d(*box['door_local_position_m']))
        place(path, p, quat)
        box['root'] = path
        box['body'] = path if kind=='beef' else path+'/Asset'
        box['world_position_m'] = list(p)
    z = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT)).ExtractTranslation()[2]
    x,y,yaw = plan['robot']['world_xy_yaw_m_deg']
    place(ROBOT, [x,y,z], Gf.Rotation(Gf.Vec3d(0,0,1), yaw).GetQuat())
    if physical_snapshot(stage) != added_physics:
        raise ValueError('Pose randomization changed physical parameters')
    for key,value in before.items():
        if added_physics.get(key) != value:
            raise ValueError('Existing scene physics changed: '+key)
    # Anchor the layer before relativizing, so USD never resolves relative
    # references against an anonymous layer/current working directory.
    layer.Export(str(destination))
    exported = Usd.Stage.Open(str(destination))
    for kind, relative in ASSETS.items():
        refs = exported.GetPrimAtPath(EXTRAS[kind]+'/Asset').GetReferences()
        refs.ClearReferences()
        refs.AddReference(os.path.relpath(SHARED_ASSETS[kind], output))
    exported.GetRootLayer().subLayerPaths = [os.path.relpath(scene, output)]
    exported.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination))
    if physical_snapshot(reopened) != added_physics:
        raise ValueError('Reopened reset changed physics/dependencies')
    plan['scene_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    plan['asset_sha256'] = dependencies
    plan['physics_parameters_changed'] = False
    plan['native_asset_configuration'] = 'fridge_food_boxes_v1: original fridge dimensions baked into geometry; unit instances; native physics and existing GripMat retained'
    with (output/'randomization_plan.json').open('x') as f:
        json.dump(plan,f,indent=2,allow_nan=False)
    return destination, plan


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from .api_runtime import PREPARED
    result,plan = build(PREPARED/'scene.usda',args.output,args.seed)
    print(json.dumps({'scene': str(result), 'boxes': {k:v['slot'] for k,v in plan['boxes'].items()}}))

"""Fixed tabletop resets: existing assets, state only, no physics overrides."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROBOT = '/World/Robot'
SUPPORT = '/World/Assets/CoffeeTable'
RESET_OP = 'xformOp:transform:tabletopInitial'
SPECS = {
    'wallet': ('7-2', '7.13', '/World/Assets/Wallet'),
    'glasses_case': ('7-1', '7.15', '/World/Assets/GlassesCase'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_facts(stage, paths):
    from pxr import Usd
    return {str(p.GetPath()): {
        'schemas': list(p.GetAppliedSchemas()),
        'attributes': {a.GetName(): str(a.Get()) for a in p.GetAttributes()
                       if a.GetName().startswith(('physics:', 'physx'))},
        'relationships': {r.GetName(): [str(v) for v in r.GetTargets()]
                          for r in p.GetRelationships()
                          if r.GetName().startswith(('physics:', 'physx', 'material:binding'))},
    } for path in paths for p in Usd.PrimRange(stage.GetPrimAtPath(path))}


def audit_layer(layer, allowed):
    from pxr import Sdf
    errors = []
    def check(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec):
            if str(path.GetPrimPath()) not in allowed or spec.name not in (RESET_OP, 'xformOpOrder'):
                errors.append(str(path))
        elif isinstance(spec, Sdf.RelationshipSpec):
            errors.append(str(path))
        elif isinstance(spec, Sdf.PrimSpec):
            for key in spec.ListInfoKeys():
                permitted = ('specifier', 'active') if path != Sdf.Path.absoluteRootPath else (
                    'subLayers', 'subLayerOffsets', 'upAxis', 'metersPerUnit', 'defaultPrim')
                if key not in permitted or (key == 'active' and str(path) not in allowed):
                    errors.append(str(path)+':'+key)
    layer.Traverse('/', check)
    if errors:
        raise ValueError('Unexpected reset opinions: '+repr(errors))


def robot_plan(stage, data):
    from pxr import UsdPhysics
    from arena_api.robot_initialization import validate
    from arena_api.task_runtime import RobotInitialization
    names = data['robot_dof_names'].astype(str).tolist()
    q = data['robot_joint_position_rad_or_m'][0]
    positions = dict(zip(names, map(float, q)))
    command = {}
    arm_names = [f'idx{start+i}_arm_{hand}_joint{i+1}'
                 for hand, start in [('l', 21), ('r', 61)] for i in range(7)]
    channels = [(arm_names, 'command_arm_rad'),
                (['idx02_body_joint2', 'idx01_body_joint1'], 'command_waist_rad_m'),
                (['idx11_head_joint1', 'idx12_head_joint2'], 'command_head_rad'),
                (['idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1'], 'command_grip_rad')]
    for keys, channel in channels:
        values = data[channel][0]
        if len(keys) != len(values):
            raise ValueError('Source command channel shape mismatch')
        command.update(zip(keys, map(float, values)))
    limits, driven, tolerances, hold = {}, [], {}, {}
    for name, value in positions.items():
        p = stage.GetPrimAtPath(ROBOT+'/joints/'+name)
        if not p or not (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            raise ValueError('Source joint missing: '+name)
        angular = p.IsA(UsdPhysics.RevoluteJoint)
        axis = 'angular' if angular else 'linear'
        bounds = [p.GetAttribute('physics:'+x+'Limit').Get() for x in ('lower', 'upper')]
        limits[name] = [math.radians(v) for v in bounds] if angular else bounds
        tolerances[name] = math.radians(.001) if angular else .001  # Existing public audit precision.
        if p.HasAPI(UsdPhysics.DriveAPI, axis):
            driven.append(name)
            hold[name] = command.get(name, value)
    if not set(command) <= set(driven):
        raise ValueError('Source commands include undriven joints')
    plan = RobotInitialization(positions, hold)
    validate(plan, names, limits, driven, tolerances)
    return dict(positions=positions, hold_targets=hold)


def prepare(output, kind):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from arena_api import l1_scene
    from .inspect_initial import source_pose
    if kind not in SPECS:
        raise ValueError('Only wallet and glasses_case are in the current scope')
    out = Path(output).resolve()
    if not out.is_relative_to(TASK/'artifacts') or out == TASK/'artifacts' or out.exists():
        raise ValueError('Use a fresh directory inside tabletop_pick/artifacts')
    api_id, source_id, target = SPECS[kind]
    core = TASK/'references/l1_20260916'/source_id/'core'
    recorded_audit = json.loads((TASK/'source_audit.json').read_text())
    hashes = {row['path']: row['sha256'] for row in recorded_audit['files']}
    for name in ('full_state.npz', 'initial_scene.usda', 'action.npz'):
        p = core/name
        if digest(p) != hashes[str(p.relative_to(TASK))]:
            raise ValueError('Source snapshot changed: '+name)
    source_layer = Sdf.Layer.FindOrOpen(str(core/'initial_scene.usda'))
    out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [os.path.relpath(p, out) for p in (*l1_scene.LAYERS, l1_scene.BASE)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    l1_scene.verify(stage)
    source_table = source_pose(source_layer, SUPPORT)
    table = UsdGeom.Xformable(stage.GetPrimAtPath(SUPPORT)).ComputeLocalToWorldTransform(0)
    if not Gf.IsClose(Gf.Transform(table).GetScale(), Gf.Transform(source_table).GetScale(), 1e-6):
        raise ValueError('Support scale differs; do not silently rescale the task')
    delta = source_table.RemoveScaleShear().GetInverse()*table.RemoveScaleShear()
    prim = stage.GetPrimAtPath(target)
    if not prim:
        raise ValueError('Missing existing shared target')
    prim.SetActive(True)
    body = UsdPhysics.RigidBodyAPI(prim)
    if not body or not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
        raise ValueError('Shared target must already be dynamic')
    before = physics_facts(stage, (ROBOT, target, SUPPORT))
    dep_hashes = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers()
                  if l.realPath and Path(l.realPath) != out/'initial.usda'}
    authored = {}
    for path in (target, ROBOT):
        p = stage.GetPrimAtPath(path)
        current = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
        scale = Gf.Transform(current).GetScale()
        world = Gf.Matrix4d(1).SetScale(scale)*source_pose(source_layer, path).RemoveScaleShear()*delta
        parent = UsdGeom.Xformable(p.GetParent()).ComputeLocalToWorldTransform(0)
        xf = UsdGeom.Xformable(p)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='tabletopInitial').Set(world*parent.GetInverse())
        authored[path] = dict(world_matrix=[list(row) for row in world], preserved_scale=list(scale))
    with np.load(core/'full_state.npz', allow_pickle=False) as data:
        plan = robot_plan(stage, data)
    colliders = [str(p.GetPath()) for p in Usd.PrimRange(prim)
                 if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    if not colliders:
        raise ValueError('Target has no active collisions')
    errors = [str(e) for p in stage.Traverse() for e in p.GetPrimIndex().localErrors]
    if errors:
        raise ValueError('USD composition errors: '+repr(errors[:5]))
    audit_layer(layer, {target, ROBOT})
    if before != physics_facts(stage, (ROBOT, target, SUPPORT)):
        raise ValueError('Physics opinions changed during reset')
    if any(digest(p) != h for p, h in dep_hashes.items()):
        raise ValueError('Dependency changed during construction')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    layer.Save()
    report = dict(api_id=api_id, kind=kind, target=target, support=SUPPORT,
                  source_id=source_id, source_core=str(core.relative_to(TASK)),
                  poses=authored, robot_initialization=plan, active_colliders=colliders,
                  target_mass_kg=UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
                  target_bbox_m=[list(bbox.GetMin()), list(bbox.GetMax())],
                  relative_frame='source CoffeeTable rigid frame -> current CoffeeTable rigid frame',
                  physics_unchanged=True, external_sources_unchanged=True,
                  source_dependencies_sha256=dep_hashes,
                  physics_validated=False, evaluation_ready=False, scoring_status='not_defined',
                  scene_sha256=digest(out/'initial.usda'))
    (out/'initial.json').write_text(json.dumps(report, indent=2)+'\n')
    return out/'initial.usda'


class InitialStateHooks:
    def __init__(self, output, kind, *, seed=None):
        if kind not in SPECS:
            raise ValueError('Unsupported task kind')
        self.output, self.kind, self.plan = Path(output), kind, None
        if seed is not None:
            from .randomization import sample
            sample(seed)
        self.seed, self.scene, self.randomized = seed, None, False

    def initialize(self, source):
        from arena_api import l1_scene
        if Path(source).resolve() != l1_scene.BASE.resolve() or self.plan is not None:
            raise ValueError('Fixed shared baseline and one initialization required')
        scene = prepare(self.output/'initial', self.kind)
        self.plan = json.loads((scene.parent/'initial.json').read_text())
        self.scene = scene
        return scene

    def randomize(self, scene):
        if self.scene is None or Path(scene).resolve() != self.scene.resolve():
            raise ValueError('Use this episode initialized scene')
        if self.randomized:
            raise ValueError('Randomization can only be applied once')
        if self.seed is None:
            return scene
        from .randomization import prepare as prepare_random
        result = prepare_random(scene, self.output/'randomized', seed=self.seed)
        self.plan = json.loads((result.parent/'initial.json').read_text())
        self.randomized = True
        return result

    def robot_initialization(self):
        from arena_api.task_runtime import RobotInitialization
        if self.plan is None:
            raise RuntimeError('Initialize first')
        return RobotInitialization(**self.plan['robot_initialization'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=tuple(SPECS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.kind))

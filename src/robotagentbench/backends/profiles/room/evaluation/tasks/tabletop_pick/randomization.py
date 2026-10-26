"""Approved robot perturbations and robot-biased tabletop target sampling."""
import argparse
import copy
import json
import math
import os
from pathlib import Path

from .initial_state import TASK, ROBOT, SUPPORT, SPECS, audit_layer, digest, physics_facts
from .placement import tabletop_rectangle, visible_bounds, sample_placement, approved_points

SCHEMA = 'arena.tabletop_pick.initial_randomization.v4'
ARM_NAMES = tuple(f'idx{start+i}_arm_{hand}_joint{i+1}'
                  for hand, start in (('l', 21), ('r', 61)) for i in range(7))


def sample(seed):
    import numpy as np
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    # Base draw and arm seed match L1 _family4_initial_state.sample.
    # Arm ranges are preserved; no claim of old producer trajectory equivalence.
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.05, 0.05, 2).round(4).tolist()
    yaw = round(float(rng.uniform(-5., 5.)), 3)
    arm_seed = seed * 7 + 1
    arm = np.random.default_rng(arm_seed).uniform(-math.radians(4), math.radians(4), 14)
    return dict(seed=seed, base_dxy_m=xy, base_dyaw_deg=yaw,
                arm_noise_seed=arm_seed, arm_delta_rad=dict(zip(ARM_NAMES, arm.tolist())))


def perturbed_plan(stage, initial, draw):
    from pxr import UsdPhysics
    from arena_api.robot_initialization import validate
    from arena_api.task_runtime import RobotInitialization
    plan = copy.deepcopy(initial)
    for name, delta in draw['arm_delta_rad'].items():
        for field in ('positions', 'hold_targets'):
            plan[field][name] += delta
    limits, tolerances, driven = {}, {}, []
    for name in set(plan['positions']) | set(plan['hold_targets']):
        prim = stage.GetPrimAtPath(ROBOT+'/joints/'+name)
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        if not angular and not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Unsupported reset joint: '+name)
        values = [prim.GetAttribute('physics:'+side+'Limit').Get() for side in ('lower', 'upper')]
        limits[name] = [math.radians(v) for v in values] if angular else values
        tolerances[name] = math.radians(.001) if angular else .001
        if prim.HasAPI(UsdPhysics.DriveAPI, 'angular' if angular else 'linear'):
            driven.append(name)
    # Same public validation, no clipping/resampling or changes to native limits.
    validate(RobotInitialization(**plan), list(limits), limits, driven, tolerances)
    return plan


def matrix(stage, path):
    from pxr import UsdGeom
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)


def prepare(source, output, *, seed):
    from pxr import Gf, Usd, UsdGeom
    draw = sample(seed)
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    root = (TASK/'artifacts').resolve()
    if not source.is_relative_to(root):
        raise ValueError('Source must be a task-local fixed initial scene')
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError('Use a fresh output directory inside task artifacts')
    report = json.loads((source.parent/'initial.json').read_text())
    if report.get('randomization'):
        raise ValueError('Do not accumulate perturbations; use a fixed initial scene')
    if report.get('kind') not in SPECS or digest(source) != report['scene_sha256']:
        raise ValueError('Invalid or changed initial scene')
    nominal = Usd.Stage.Open(str(source))
    target = SPECS[report['kind']][2]
    if report['target'] != target:
        raise ValueError('Initial target mismatch')
    before = physics_facts(nominal, (ROBOT, target, SUPPORT))
    old = matrix(nominal, ROBOT)
    support_world = matrix(nominal, SUPPORT)
    target_world = matrix(nominal, target)
    dependencies = {layer.realPath: digest(layer.realPath)
                    for layer in nominal.GetUsedLayers() if layer.realPath}
    plan = perturbed_plan(nominal, report['robot_initialization'], draw)
    # World-XY offsets and yaw about the base's own origin, not about the room.
    origin = old.ExtractTranslation()
    shift = Gf.Vec3d(*draw['base_dxy_m'], 0.)
    rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), draw['base_dyaw_deg']))
    world = old * Gf.Matrix4d(1).SetTranslate(-origin) * rotation
    world *= Gf.Matrix4d(1).SetTranslate(origin+shift)
    surface = tabletop_rectangle(nominal, SUPPORT+'/Mesh')
    points, scan_sha = approved_points(report['kind'], surface)
    placement = sample_placement(seed, surface,
                                 visible_bounds(nominal, target),
                                 list(world.ExtractTranslation())[:2], candidates=points)
    placement['approved_region_source_report_sha256'] = scan_sha
    placement['physics_validation_status'] = 'not_run_user_accepted_continuous_region'
    target_world *= Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*placement['translation_m']))
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema=SCHEMA, sample=draw, status='preparing',
                  frame='current shared scene world XY; yaw about robot base origin',
                  reference='user-reviewed fixed initial scene (not the old unperturbed producer pose)',
                  bounds=dict(base_xy_each_m=.05, base_yaw_deg=5., arm_each_deg=4.),
                  target_randomized=True, object_placement=placement, whole_house_randomized=False,
                  trajectory_regenerated=False, physics_validated=False)
    def save():
        (output/'randomization.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        path = output/'randomized.usda'
        stage = Usd.Stage.CreateNew(str(path))
        layer = stage.GetRootLayer()
        layer.subLayerPaths = [os.path.relpath(source, output)]
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        for prim_path, new_world in ((ROBOT, world), (target, target_world)):
            prim = stage.GetPrimAtPath(prim_path)
            parent = matrix(stage, str(prim.GetParent().GetPath()))
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='tabletopInitial').Set(new_world*parent.GetInverse())
        audit_layer(layer, {ROBOT, target})
        if not Gf.IsClose(matrix(stage, ROBOT), world, 1e-9):
            raise ValueError('Base transform mismatch')
        if not Gf.IsClose(matrix(stage, SUPPORT), support_world, 1e-9):
            raise ValueError('Table moved')
        if not Gf.IsClose(matrix(stage, target), target_world, 1e-9):
            raise ValueError('Target transform mismatch')
        import numpy as np
        bounds = visible_bounds(stage, target)
        if not np.allclose(bounds, placement['after_bbox_m'], rtol=0, atol=1e-9):
            raise ValueError('Target geometry placement mismatch')
        if physics_facts(stage, (ROBOT, target, SUPPORT)) != before:
            raise ValueError('Physics parameters changed')
        if any(digest(p) != h for p, h in dependencies.items()):
            raise ValueError('Source dependency changed')
        layer.Save()
        report['robot_initialization'] = plan
        report['poses'][ROBOT]['world_matrix'] = [list(row) for row in world]
        report['poses'][target]['world_matrix'] = [list(row) for row in target_world]
        report['target_bbox_m'] = bounds.tolist()
        report['target_bbox_method'] = 'exact visible mesh world vertices'
        report['randomization'] = record.copy()
        report['randomization']['status'] = 'prepared_physics_pending'
        report['scene_sha256'] = digest(path)
        report['source_dependencies_sha256'] = dependencies
        report['physics_validated'] = False
        report['evaluation_ready'] = False
        (output/'initial.json').write_text(json.dumps(report, indent=2)+'\n')
        record.update(status='prepared_physics_pending', source_sha256=digest(source),
                      scene_sha256=digest(path), physics_unchanged=True,
                      support_unchanged=True, target_rotation_scale_unchanged=True,
                      source_dependencies_unchanged=True)
        save()
        return path
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output, seed=args.seed))

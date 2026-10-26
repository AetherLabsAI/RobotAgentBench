"""Reset-only local jitter, shared by all 17 coffee-table placement variants.

Reuse dining_place's approved L1 sampler, not its room/layout initializer.
Only the robot and its held group move; joints, assets and destinations do not.
"""
import json
import os
from pathlib import Path

from .source_initial import TASK
from .initial_state import audit_layer, digest, physics_facts
from .bindings import binding_for

SCHEMA = 'arena.coffee_table_place.local_randomization.v1'


def replay_contract(record):
    """Comparable reset identity without this episode's disposable output path."""
    import copy
    if record is None:
        return None
    return copy.deepcopy({k: v for k, v in record.items()
                          if k not in ('scene_task_relative', 'scene_sha256')})


def sample_offsets(seed):
    # Same seed means the same offset across variants and dining_place.
    from evaluation.tasks.dining_place.randomization import sample_offsets as dining_offsets
    return dining_offsets(seed)


def group_delta(robot_world, offsets):
    from pxr import Gf
    center = robot_world.ExtractTranslation()
    before = Gf.Matrix4d(1).SetTranslate(-center)
    rotate = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), offsets['yaw_deg']))
    after = Gf.Matrix4d(1).SetTranslate(center + Gf.Vec3d(offsets['dx_m'], offsets['dy_m'], 0))
    return before * rotate * after


def verify_record(record):
    """Fail closed if this episode's authored state was changed after preparation."""
    scene = (TASK / record['scene_task_relative']).resolve(strict=True)
    if (record['schema'] != SCHEMA or not scene.is_relative_to((TASK/'artifacts').resolve())
            or digest(scene) != record['scene_sha256']):
        raise ValueError('Local randomized scene changed or escaped task artifacts')


def prepare(task_id, source, output, *, seed):
    from pxr import Gf, Sdf, Usd, UsdGeom
    source = Path(source).resolve(strict=True)
    out = Path(output).resolve()
    if not source.is_relative_to((TASK/'resources').resolve()):
        raise ValueError('Local random requires an immutable packaged reset')
    if out.exists() or not out.is_relative_to((TASK/'artifacts').resolve()) or out == TASK/'artifacts':
        raise ValueError('Use a new directory inside coffee_table_place/artifacts')
    offsets = sample_offsets(seed)
    binding = binding_for(task_id)
    stage = Usd.Stage.Open(str(source))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected a metres/Z-up reset')
    roots = [binding.robot, binding.held_body] + ([binding.payload] if binding.payload else [])
    matrix = lambda path: UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    original = {p: matrix(p) for p in roots}
    destination = matrix(binding.destination)
    before = physics_facts(stage, roots + [binding.destination])
    dependencies = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers() if l.realPath}
    delta = group_delta(original[binding.robot], offsets)
    layer = stage.GetSessionLayer()
    stage.SetEditTarget(layer)
    for path in roots:
        prim = stage.GetPrimAtPath(path)
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
        xf = UsdGeom.Xformable(prim)
        # Preserve the complete authored affine matrix. Decomposing measured
        # reset matrices into rotation/scale can perturb their calibration.
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='coffee_place_initial').Set(original[path] * delta * parent.GetInverse())
    audit_layer(layer, set(roots))
    if physics_facts(stage, roots + [binding.destination]) != before:
        raise ValueError('Local jitter changed physics/material configuration')
    if not Gf.IsClose(matrix(binding.destination), destination, 1e-12):
        raise ValueError('Local jitter moved the destination')
    for path in roots:
        if not Gf.IsClose(matrix(path), original[path] * delta, 1e-9):
            raise ValueError('Local jitter changed group relation or scale: ' + path)
    if any(digest(p) != h for p, h in dependencies.items()):
        raise ValueError('Source layer changed during preparation')
    out.mkdir(parents=True)
    scene = out/'initial.usda'
    exported = Sdf.Layer.CreateAnonymous('local_random.usda')
    exported.TransferContent(layer)
    exported.subLayerPaths = [os.path.relpath(source, out)]
    exported.defaultPrim = 'World'
    exported.pseudoRoot.SetInfo('metersPerUnit', 1.)
    exported.pseudoRoot.SetInfo('upAxis', 'Z')
    audit_layer(exported, set(roots))
    exported.Export(str(scene))
    reopened = Usd.Stage.Open(str(scene))
    if not reopened or physics_facts(reopened, roots + [binding.destination]) != before:
        raise ValueError('Reopened randomized state changed physics')
    for path in roots + [binding.destination]:
        actual = UsdGeom.Xformable(reopened.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
        if not Gf.IsClose(actual, matrix(path), 1e-9):
            raise ValueError('Pose changed when reopening randomized state')
    record = dict(schema=SCHEMA, api_id=task_id, seed=seed, offsets=offsets,
        sampler='evaluation.tasks.dining_place.randomization.sample_offsets',
        bounds=dict(xy_each_m=.020, yaw_deg=8., z_m=0.),
        translation_frame='world_xy', yaw_pivot='robot_base_origin',
        transformed_roots=roots, destination=binding.destination,
        group_delta_world=[list(row) for row in delta],
        poses_before={p: [list(row) for row in m] for p, m in original.items()},
        poses_after={p: [list(row) for row in matrix(p)] for p in roots},
        source_task_relative=str(source.relative_to(TASK)), source_sha256=digest(source),
        scene_task_relative=str(scene.relative_to(TASK)), scene_sha256=digest(scene),
        joint_reset_and_targets_unchanged=True, destination_unchanged=True,
        physics_parameters_unchanged=True, whole_house_randomized=False,
        live_initial_acceptance_required=True, physics_validated=False)
    (out/'randomization.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    verify_record(record)
    return scene, record

"""Prepare local task 10.2 reuse inputs against the current dining geometry.

CPU only. Writes evidence/plans into a fresh output directory; never edits USD,
starts Isaac, connects to L2, registers an evaluator or labels a physics pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from evaluation.tasks.dining_place import initial_state as task
from evaluation.tasks.dining_place.source_reference import (
    load_reference, initial_reference, command_reference,
)


def inspect_destination(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-9:
        raise ValueError('Expected current metre-scale Z-up scene')
    p = stage.GetPrimAtPath(task.DESTINATION)
    if (not p or not p.IsA(UsdGeom.Cube) or not p.HasAPI(UsdPhysics.CollisionAPI)
            or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()):
        raise ValueError('Expected actual colliding dining tabletop, not CoffeeTable or a proxy')
    xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    normal = xf.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
    if (normal - Gf.Vec3d(0, 0, 1)).GetLength() > 1e-6:
        raise ValueError('Tilted tabletop requires a different calibration')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeWorldBound(p).ComputeAlignedRange()
    low, high = list(bounds.GetMin()), list(bounds.GetMax())
    if any(not math.isfinite(v) for v in low + high) or any(a >= b for a, b in zip(low, high)):
        raise ValueError('Invalid tabletop extent')
    # The lift offset is only exact when the native prismatic axis is world Z.
    joint = stage.GetPrimAtPath(task.ROBOT + '/joints/idx01_body_joint1')
    if not joint or not joint.IsA(UsdPhysics.PrismaticJoint) or joint.GetAttribute('physics:axis').Get() != 'Z':
        raise ValueError('Expected native vertical lift joint')
    bodies = joint.GetRelationship('physics:body0').GetTargets()
    if len(bodies) != 1:
        raise ValueError('Unknown lift parent')
    parent = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = Gf.Rotation(Gf.Quatd(joint.GetAttribute('physics:localRot0').Get()))
    axis = parent.TransformDir(rotation.TransformDir(Gf.Vec3d(0, 0, 1))).GetNormalized()
    if (axis - Gf.Vec3d(0, 0, 1)).GetLength() > 1e-6:
        raise ValueError('Lift is not world vertical; no automatic height compensation')
    limits = [float(joint.GetAttribute('physics:' + side + 'Limit').Get()) for side in ('lower', 'upper')]
    return dict(prim=task.DESTINATION, world_bounds_m=[low, high], support_top_m=high[2],
                lift_limits_m=limits, lift_world_axis=list(axis))


def prepare(base, output):
    from pxr import Usd, UsdPhysics
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest, motion = load_reference()
    stage = Usd.Stage.Open(str(base))
    if not stage:
        raise ValueError('Cannot load base scene')
    layers = {str(Path(layer.realPath)): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in stage.GetUsedLayers() if layer.realPath}
    destination = inspect_destination(stage)
    source = initial_reference(motion)
    initial = task.height_candidate(source, destination['support_top_m'], destination['lift_limits_m'])
    commands = command_reference(motion)
    # Verify all reused commands against CURRENT native joint limits. Reject,
    # never clip the recorded arms to create a plausible-looking candidate.
    for name in commands['targets'][0]:
        joint = stage.GetPrimAtPath(task.ROBOT + '/joints/' + name)
        if not joint or not joint.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError('Missing current revolute joint ' + name)
        lo, hi = [math.radians(joint.GetAttribute('physics:' + side + 'Limit').Get())
                  for side in ('lower', 'upper')]
        values = [row[name] for row in commands['targets']]
        values.append(source['arm_targets_rad'].get(name, source['gripper_command_reference_rad'].get(name)))
        if min(values) < lo or max(values) > hi:
            raise ValueError('Source command outside current limits: ' + name)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in layers.items())
    if not unchanged:
        raise ValueError('Scene source changed during read-only preparation')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='arena.dining_place.reference_preparation.v1', task_id='10.2',
        status='source_reused_calibration_pending', physics_validated=False,
        api_registered=False, policy_executed=False, base_scene=str(base),
        base_scene_sha256=layers[str(base)], source_layers_unchanged=True,
        source_manifest=manifest, destination=destination,
        source_initial=source, current_initial_candidate=initial,
        control_steps=commands['control_steps'], duration_s=commands['duration_s'],
        source_layer_sha256=layers)
    for name, value in [('reference_preparation.json', report),
                        ('initial_candidate.json', initial), ('commands_reference.json', commands)]:
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.base, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'physics_validated', 'api_registered',
        'destination', 'control_steps', 'duration_s')}, indent=2))


if __name__ == '__main__':
    main()

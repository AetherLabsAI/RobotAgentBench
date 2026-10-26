"""Read-only beef reset planning against the composed shared scene.

Produces the existing RobotInitialization contract, not an accepted reset.
Does not author USD, tune physics, replay actions, or register a task.
"""
import math

import numpy as np

from arena_api import robot_initialization, robot_profiles
from arena_api.task_runtime import RobotInitialization
from .recorded_checkpoint import rigid_matrix

MICROWAVE_ROOT = "/World/Assets/DiningTableMicrowave"


def robot_reset_plan(checkpoint, stage):
    """Keep measured native34 q; send targets only to existing active drives.

    The fourteen passive coordinates remain in positions. Their recorded
    native_target fields are not actuator commands. Existing Mimic coupling
    is validated, never replaced or supplemented with a drive.
    """
    from pxr import UsdPhysics

    expected = set(robot_profiles.NATIVE_JOINTS)
    if (checkpoint.variant not in ('beef_box', 'leftover_box')
            or checkpoint.source_manifest.get('payload_variant') != checkpoint.variant):
        raise ValueError("beef/leftover checkpoint identity mismatch")
    for values in (checkpoint.positions, checkpoint.velocities, checkpoint.hold_targets):
        if set(values) != expected:
            raise ValueError("Expected complete native34 checkpoint")
    limits, units, tolerances, driven = {}, {}, {}, []
    for name in robot_profiles.NATIVE_JOINTS:
        prim = stage.GetPrimAtPath("/World/Robot/joints/" + name)
        if not prim or not prim.IsActive():
            raise ValueError("Missing active robot joint: " + name)
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        if not angular and not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError("Unsupported reset joint: " + name)
        units[name] = "rad" if angular else "m"
        bounds = [prim.GetAttribute("physics:" + side + "Limit").Get()
                  for side in ("lower", "upper")]
        if any(v is None or not math.isfinite(v) for v in bounds):
            raise ValueError("Missing finite joint limits: " + name)
        limits[name] = tuple(math.radians(v) for v in bounds) if angular else tuple(bounds)
        # Match the existing worker's native reset precision, not a new gate.
        tolerances[name] = math.radians(.001) if angular else .001
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular" if angular else "linear"):
            driven.append(name)
    plan = RobotInitialization(dict(checkpoint.positions),
                               {n: checkpoint.hold_targets[n] for n in driven})
    robot_initialization.validate(plan, expected, limits, driven, tolerances)
    adapter = robot_profiles.NativeAdapter(
        robot_profiles.DEFAULT, robot_profiles.NATIVE_JOINTS, units, driven,
        plan.hold_targets, limits,
        mimic_followers=robot_profiles.inspect_mimic_followers(stage))
    commands = {n: checkpoint.hold_targets[n] for n in adapter.profile.joints}
    expanded = adapter.expand(commands)
    if expanded != plan.hold_targets:
        raise ValueError("Recorded targets disagree with the existing gripper coupling")
    return plan


def relative_placement_plan(checkpoint, stage):
    """Use the appliance's rigid frame; preserve its separate cavity scaling.

    This scene's body transform carries width-only geometry scale. Do not feed
    that scale into the recorded rigid-body reference frame, or multiply the
    hand/box offsets by 1.25. Unknown body rotation/translation/shear fails.
    Nothing is moved here, including the plate stand and room objects.
    """
    from pxr import UsdGeom

    if (checkpoint.variant not in ('beef_box', 'leftover_box')
            or checkpoint.source_manifest.get('payload_variant') != checkpoint.variant):
        raise ValueError("beef/leftover checkpoint identity mismatch")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Expected a metre-authored scene")
    root = stage.GetPrimAtPath(MICROWAVE_ROOT)
    body = stage.GetPrimAtPath(MICROWAVE_ROOT + "/Asset/body")
    if not root or not body or not root.IsActive() or not body.IsActive():
        raise ValueError("Missing active microwave root/body")
    cache = UsdGeom.XformCache()
    world = rigid_matrix(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    relative = np.linalg.inv(world) @ np.asarray(cache.GetLocalToWorldTransform(body)).T
    diagonal = np.diag(relative)
    if (not np.isfinite(relative).all() or np.any(diagonal[:3] <= 0)
            or not np.allclose(relative, np.diag(diagonal), rtol=0, atol=1e-10)
            or not np.isclose(diagonal[3], 1, rtol=0, atol=1e-10)):
        raise ValueError("Microwave body has an unknown rigid offset or shear")
    return checkpoint.placement_plan(world), diagonal[:3].tolist()


def inspect_composed_scene(checkpoint, stage):
    """Report fixed-property differences, not a complete live-worker audit."""
    from pxr import UsdPhysics
    from arena_api import policy_execution

    plan = robot_reset_plan(checkpoint, stage)
    placements, shape_scale = relative_placement_plan(checkpoint, stage)
    differences, checked = [], 0
    for name, attrs in checkpoint.metadata["parameters"].items():
        prim = stage.GetPrimAtPath("/World/Robot/joints/" + name)
        for attr, expected in attrs.items():
            if attr.endswith((":targetPosition", ":targetVelocity")):
                continue
            value = prim.GetAttribute(attr).Get() if prim else None
            try:
                equal = float(value) == float(expected)
            except (TypeError, ValueError):
                equal = str(value) == str(expected)
            checked += 1
            if not equal:
                differences.append(dict(joint=name, attribute=attr,
                                        source=expected, current=str(value)))
    cameras = []
    for path in checkpoint.metadata["camera_collision_revision"]["paths"]:
        prim = stage.GetPrimAtPath(path)
        enabled = bool(prim and prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
                       and prim.GetAttribute("physics:collisionEnabled").Get() is not False)
        cameras.append(dict(path=path, collision_enabled=enabled))
    protocol = policy_execution.spec()
    return dict(
        scope="shared_L1_layers_plus_approved_plate_not_live_worker",
        robot_initialization=dict(positions=plan.positions, hold_targets=plan.hold_targets),
        source_passive_targets_not_sent={n: v for n, v in checkpoint.hold_targets.items()
                                        if n not in plan.hold_targets},
        native_positions=len(plan.positions), existing_drive_targets=len(plan.hold_targets),
        robot_profile=robot_profiles.DEFAULT, reset_coordinates_within_current_limits=True,
        placements_column_vector={k: v.tolist() for k, v in placements.items()},
        microwave_geometry_scale_preserved=shape_scale,
        fixed_joint_properties_checked=checked, fixed_joint_differences=differences,
        source_wrist_colliders_in_composed_scene=cameras,
        clock=dict(source_physics_hz=checkpoint.metadata["physics_hz"],
                   source_control_hz=checkpoint.metadata["command_hz"],
                   current_physics_hz=1 / protocol["physics_dt_s"],
                   current_control_hz=1 / protocol["control_dt_s"]),
        source_max_abs_joint_velocity=max(abs(v) for v in checkpoint.velocities.values()),
        source_body_velocities=checkpoint.body_velocities,
        reset_velocity_policy="existing RobotInitialization zeros restored joint velocities",
        exact_dynamic_checkpoint_supported=False, scene_authored=False, physics_run=False,
        physical_initial_state_accepted=False, api_registered=False)

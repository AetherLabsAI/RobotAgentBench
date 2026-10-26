"""Shared Isaac execution defaults; task duration is independent of substep count."""
import math

PROFILE = 'arena.physics.l1_240hz_32_4.v1'
PHYSICS_DT = 1 / 240
POSITION_ITERATIONS = 32
VELOCITY_ITERATIONS = 4


def steps_for_seconds(seconds):
    count = round(seconds / PHYSICS_DT)
    if seconds < 0 or not math.isclose(count * PHYSICS_DT, seconds, abs_tol=1e-10):
        raise ValueError('Duration must be an integer number of physics steps')
    return count


def apply_robot(stage):
    from pxr import PhysxSchema
    from arena_api import robot_drive_defaults
    root = stage.GetPrimAtPath('/World/Robot')
    if not root:
        raise RuntimeError('Missing robot for shared solver defaults')
    api = PhysxSchema.PhysxArticulationAPI.Apply(root)
    api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)
    robot_drive_defaults.apply(stage)


def verify_runtime(world, robot, *, expected_dt=PHYSICS_DT):
    receipt = dict(profile=PROFILE, physics_dt_s=float(world.get_physics_dt()),
        solver_position_iterations=int(robot.get_solver_position_iteration_count()),
        solver_velocity_iterations=int(robot.get_solver_velocity_iteration_count()))
    if expected_dt != PHYSICS_DT:
        receipt['profile'] = 'arena.physics.variable_rate_32_4.v1'
    if (not math.isclose(receipt['physics_dt_s'], expected_dt, abs_tol=1e-10)
        or receipt['solver_position_iterations'] != POSITION_ITERATIONS
        or receipt['solver_velocity_iterations'] != VELOCITY_ITERATIONS):
        raise RuntimeError('Runtime physics defaults mismatch: ' + repr(receipt))
    return receipt

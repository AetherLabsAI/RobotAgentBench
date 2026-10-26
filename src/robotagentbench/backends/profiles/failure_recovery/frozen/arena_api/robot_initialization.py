"""One-time, trusted reset adapter; no task paths, target geometry or scoring."""
import math
from .task_runtime import RobotInitialization


def validate(plan, names, limits, driven, tolerances=None):
    if not isinstance(plan, RobotInitialization):
        raise ValueError('Expected trusted RobotInitialization')
    if not plan.positions or not plan.hold_targets:
        raise ValueError('Empty robot reset')
    for field, values in (('positions', plan.positions), ('hold_targets', plan.hold_targets)):
        for name, value in values.items():
            if name not in names or name not in limits:
                raise ValueError('Unknown reset joint: '+name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Nonfinite reset coordinate')
            lo,hi=limits[name]
            tolerance=(tolerances or {}).get(name,1e-6)
            if not lo-tolerance <= value <= hi+tolerance:
                raise ValueError('Reset outside native limits: '+name)
            if field=='hold_targets' and name not in driven:
                raise ValueError('Reset cannot introduce a new drive: '+name)
    return plan


def apply(robot, stage, plan):
    from pxr import UsdPhysics
    import numpy as np
    limits={};driven=[];tolerances={}
    if not isinstance(plan,RobotInitialization):raise ValueError('Expected trusted RobotInitialization')
    for name in set(plan.positions)|set(plan.hold_targets):
        if name not in robot.dof_names:raise ValueError('Unknown reset joint: '+name)
        p=stage.GetPrimAtPath(str(robot.prim_path)+'/joints/'+name)
        angular=p.IsA(UsdPhysics.RevoluteJoint)
        if not angular and not p.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Unsupported robot initialization joint')
        bounds=[p.GetAttribute('physics:'+s+'Limit').Get() for s in ('lower','upper')]
        if any(x is None or not math.isfinite(x) for x in bounds):
            raise ValueError('Missing native reset limits')
        limits[name]=[math.radians(x) for x in bounds] if angular else bounds
        # Match the existing worker's +/-0.001 native-unit initial-state audit.
        # Recorded passive coordinates can differ by a few microradians; do
        # not clip them or silently change the accepted contact checkpoint.
        tolerances[name]=math.radians(.001) if angular else .001
        if p.HasAPI(UsdPhysics.DriveAPI,'angular' if angular else 'linear'):driven.append(name)
    validate(plan,robot.dof_names,limits,driven,tolerances)
    indices=np.asarray([robot.dof_names.index(n) for n in plan.positions])
    robot.set_joint_positions(np.asarray(list(plan.positions.values())),joint_indices=indices)
    robot.set_joint_velocities(np.zeros(len(indices)),joint_indices=indices)
    return dict(robot_joint_state_writes=1,object_pose_writes=0,
                positions=dict(plan.positions),hold_targets=dict(plan.hold_targets),
                phase='reset_before_settle_and_evaluation')

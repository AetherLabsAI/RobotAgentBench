"""Read-only native admission checks, same checks as the source worker."""
import math


def stage_issues(stage):
    from pxr import UsdPhysics
    from arena_api.isaac_backend import contact_offset_issue
    issues = []
    for prim in stage.Traverse():
        issue = contact_offset_issue(prim.GetAttribute('physxCollision:contactOffset').Get(),
                                     prim.GetAttribute('physxCollision:restOffset').Get())
        if issue:
            issues.append(dict(issue, prim=str(prim.GetPath())))
        if prim.GetAttribute('physics:kinematicEnabled').Get() is True:
            parent = prim
            while parent and not parent.IsPseudoRoot():
                if parent.HasAPI(UsdPhysics.ArticulationRootAPI):
                    issues.append(dict(kind='kinematic_body_under_articulation', prim=str(prim.GetPath())))
                    break
                parent = parent.GetParent()
    return issues


def joint_issues(stage, robot):
    from pxr import UsdPhysics
    issues = []
    for name, value in zip(robot.dof_names, robot.get_joint_positions()):
        prim = stage.GetPrimAtPath('/World/Robot/joints/'+name)
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        if not revolute and not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Unmapped original joint: '+name)
        low = prim.GetAttribute('physics:lowerLimit').Get()
        high = prim.GetAttribute('physics:upperLimit').Get()
        native = math.degrees(value) if revolute else float(value)
        if not math.isfinite(native):
            raise ValueError('Nonfinite original joint: '+name)
        if low is not None and high is not None and not low-1e-3 <= native <= high+1e-3:
            issues.append(dict(kind='initial_joint_outside_limit', joint=name, initial_value=native,
                               limits=[low,high], unit='deg' if revolute else 'm'))
    return issues

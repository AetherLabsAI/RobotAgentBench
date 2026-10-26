"""Only arm-link physical pairs may be filtered; never mutate shared assets."""
import itertools
import re

ARM = re.compile(r'^/World/Robot/arm_([lr])_link([1-6])$')

def eligible_pairs(body_paths, scope):
    if scope not in ('same_and_cross_arm', 'cross_arm_only'):
        raise ValueError('Explicit arm collision scope required')
    arms=sorted({str(p) for p in body_paths if ARM.fullmatch(str(p))})
    return [(a,b) for a,b in itertools.combinations(arms,2)
            if scope=='same_and_cross_arm' or ARM.fullmatch(a)[1]!=ARM.fullmatch(b)[1]]

def apply_session_filters(stage, scope):
    from pxr import UsdPhysics
    if stage.GetEditTarget().GetLayer()!=stage.GetSessionLayer():
        raise ValueError('Must author only in the independent session layer')
    bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    arms=[p for p in bodies if ARM.fullmatch(p)]
    expected={f'/World/Robot/arm_{side}_link{i}' for side in ('l','r') for i in range(1,7)}
    if set(arms)!=expected:raise ValueError('Expected the verified six arm-link bodies per side; gripper bases are excluded')
    added=[]
    for a,b in eligible_pairs(arms,scope):
        prim=stage.GetPrimAtPath(a)
        existing=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
        reverse=UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(b)).GetFilteredPairsRel()
        if b in [str(x) for x in existing.GetTargets()] or a in [str(x) for x in reverse.GetTargets()]:continue
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(b)
        added.append([a,b])
    return dict(scope=scope,added_pairs=added,global_self_collision_setting_unchanged=True,
                gripper_torso_head_environment_filters_unchanged=True)

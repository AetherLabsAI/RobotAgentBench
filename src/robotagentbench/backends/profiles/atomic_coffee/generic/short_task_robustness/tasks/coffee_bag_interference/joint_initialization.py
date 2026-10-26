"""Task-local start/end planning inside the unchanged training domain.

The producer's bag proposal anchors the endpoint. Sample one continuous
direction in the away-facing 180-degree semicircle, then reverse-plan a start.
Route rejection is logged; no fallback to fixed axes or hidden resampling.
"""
import math
import random
from .training_domain import contains_origin

SEMICIRCLE_OFFSET_DEG = (-90.,90.)


def finite_point(value):
    if len(value)!=3 or not all(math.isfinite(v) for v in value):
        raise ValueError('Finite world-space 3D position required')
    return [float(v) for v in value]


def direction_check(start,end,gripper,*,reference_kind='measured_triggering_jaw'):
    """Straight XY movement cannot initially approach the triggering jaw.

    Nonnegative derivative at the start makes squared jaw distance monotonic
    on the full straight segment. The jaw reference is frozen at trigger;
    this does not assume the robot stops or promise dynamic future clearance.
    """
    if reference_kind not in ('measured_triggering_jaw','original_robot_base'):
        raise ValueError('Explicit direction reference identity required')
    start,end,gripper=map(finite_point,(start,end,gripper))
    delta=[end[i]-start[i] for i in range(2)]
    away=[start[i]-gripper[i] for i in range(2)]
    distance2=sum(v*v for v in delta)
    projection=sum(a*b for a,b in zip(delta,away))
    defined=distance2>0. and sum(v*v for v in away)>0.
    angle=math.degrees(math.atan2(away[0]*delta[1]-away[1]*delta[0],projection)) if defined else None
    return dict(allowed=defined and projection>=0.,direction_defined=defined,
                angle_offset_deg=angle,semicircle_offset_deg=list(SEMICIRCLE_OFFSET_DEG),
                initial_distance_derivative_half_m2=projection,
                horizontal_displacement_m=math.sqrt(distance2),
                start_origin_m=start,end_origin_m=end,reference_position_m=gripper,
                reference_kind=reference_kind,
                reference='supplied XY reference held fixed for straight-path directional admission')


def pair_candidates(seed,endpoint,training,robot):
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer seed required')
    endpoint=finite_point(endpoint);position=finite_point(robot['root_xyz'])
    if not contains_origin(endpoint,training):raise ValueError('Producer endpoint outside training domain')
    if not math.isfinite(robot['yaw']):raise ValueError('Original robot heading required')
    away=[endpoint[i]-position[i] for i in range(2)];norm=math.hypot(*away)
    if norm==0:raise ValueError('Robot and target XY cannot coincide')
    lower,upper=training['origin_bounds_world_xy_m'];rng=random.Random(seed)
    offset=rng.uniform(*SEMICIRCLE_OFFSET_DEG)
    heading=math.atan2(away[1],away[0])+math.radians(offset)
    axis=[math.cos(heading),math.sin(heading)]
    # Start=end-distance*axis, entirely inside the unchanged rectangle. The
    # dot-product cap keeps the angle relative to the resulting START within
    # the same semicircle, not just the preselected endpoint's radial vector.
    limits=[(endpoint[i]-lower[i])/v if v>0 else (upper[i]-endpoint[i])/-v
            for i,v in enumerate(axis) if v!=0.]
    free=min(*limits,sum(v*a for v,a in zip(axis,away)))
    fraction=rng.random()
    metadata=dict(direction='continuous_away_semicircle',sampled_angle_offset_deg=offset,
                  direction_world_xy_unit=axis,semicircle_offset_deg=list(SEMICIRCLE_OFFSET_DEG),
                  angle_sampling='uniform [-90,90] relative to endpoint/robot radial direction')
    if free<=0. or fraction==0.:
        return [dict(metadata,admitted_domain=False,reason='no_nonapproaching_start_interval')]
    distance=free*fraction;start=[endpoint[i]-distance*axis[i] for i in range(2)]+[endpoint[2]]
    check=direction_check(start,endpoint,position,reference_kind='original_robot_base')
    if check['horizontal_displacement_m']==0.:
        return [dict(metadata,admitted_domain=False,reason='no_representable_nonzero_displacement')]
    if not contains_origin(start,training) or not check['allowed']:
        raise ValueError('Invalid paired proposal; never clip or silently resample')
    return [dict(metadata,admitted_domain=True,start_origin_m=start,
        destination_origin_m=endpoint.copy(),distance_m=distance,available_distance_m=free,
        reference_robot=robot,initial_direction_check=check,
        sampling='one continuous semicircle angle and one feasible distance; original producer endpoint')]


def author_pair(stage,initial_plan):
    """Before physics, move bag AND all its beans to the admitted start."""
    from pxr import Gf,UsdGeom
    from evaluation.tasks.coffee_machine import randomization
    from evaluation.tasks.coffee_machine.specs import BAG,BEANS
    from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
    from .training_domain import domain
    from .preview_geometry import plan
    training=domain(initial_plan)
    endpoint=list(randomization.origin_of(stage,BAG))
    candidates=pair_candidates(initial_plan['seed'],endpoint,training,initial_plan['robot_base'])
    paths=[BAG,*[str(p.GetPath()) for p in stage.GetPrimAtPath(BEANS).GetChildren()]]
    if len(paths)!=101:raise ValueError('Paired initialization needs the bag and all 100 beans')
    cache=UsdGeom.XformCache()
    baseline={path:cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) for path in paths}
    asset=AdaptiveHand();rejected=[]
    def restore():
        for path,matrix in baseline.items():
            parent=UsdGeom.XformCache().GetParentToWorldTransform(stage.GetPrimAtPath(path))
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(matrix*parent.GetInverse())
    for pair in candidates:
        if not pair['admitted_domain']:
            rejected.append(pair);continue
        restore()
        delta=Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*[a-b for a,b in zip(pair['start_origin_m'],endpoint)]))
        randomization.move(stage,paths,delta)
        try:reservation=plan(stage,asset,initial_plan['seed'],initial_plan,pair=pair)
        except ValueError as error:
            rejected.append(dict(pair,reason=str(error)));continue
        import numpy as np
        current_cache=UsdGeom.XformCache()
        errors=[float(np.max(np.abs(np.asarray(current_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                                 -np.asarray(matrix*delta)))) for path,matrix in baseline.items()]
        if max(errors)>1e-9:raise ValueError('Bag/beans did not retain the same rigid translation')
        return dict(pair=pair,reservation=reservation,original_producer_endpoint_m=endpoint,
                    rejected_directions=rejected,candidates=candidates,beans_moved_with_bag=100,
                    bag_and_beans_translation_m=[a-b for a,b in zip(pair['start_origin_m'],endpoint)],
                    translation_matrix_max_error=max(errors),translated_body_count=len(paths),
                    domain=training,physical_parameters_modified=False)
    restore()
    raise ValueError('No admitted paired start/end: '+str(rejected))

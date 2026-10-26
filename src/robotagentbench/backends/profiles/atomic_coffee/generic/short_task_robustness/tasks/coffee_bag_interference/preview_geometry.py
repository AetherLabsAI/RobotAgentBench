"""Read-only countertop reservation for this new human actor.

The endpoint is sampled inside the original training domain, with no 10 cm
minimum displacement. Fixture/robot AABBs are
conservative admission, never a substitute for native collision evidence.
"""
import numpy as np
from .motion import RelocationMotion
from short_task_robustness.tasks.stock_wooden_tray_interference.retreat_geometry import check_transport


def collider_boxes(stage, roots, exclude_body_roots=()):
    from pxr import Usd, UsdGeom, UsdPhysics
    cache=UsdGeom.BBoxCache(0,['default','render','proxy']);result=[]
    for root in roots:
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            path=str(p.GetPath())
            if any(path==b or path.startswith(b+'/') for b in exclude_body_roots):continue
            if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            b=cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not b.IsEmpty():result.append([list(b.GetMin()),list(b.GetMax())])
    return result


def robot_visual_boxes(stage,excluded_grippers,native_body_poses):
    """Keep true arm/torso visual envelopes when gripper collisions are off.

    This never creates missing collisions. Existing source robot policies may
    leave only gripper shapes enabled; treating an empty collider list as no
    robot would admit visibly bad routes. Use current native link transforms.
    """
    from itertools import product
    from pxr import Usd,UsdGeom
    cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(0,['default','render','proxy']);result=[]
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
        path=str(p.GetPath())
        if any(path==b or path.startswith(b+'/') for b in excluded_grippers):continue
        if not p.IsA(UsdGeom.Boundable) or UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
        b=bbox.ComputeWorldBound(p).ComputeAlignedRange()
        if b.IsEmpty():continue
        owner=p
        while owner and str(owner.GetPath()) not in native_body_poses:owner=owner.GetParent()
        if not owner:raise ValueError('Native robot body transform missing for '+path)
        corners=np.array(list(product(*zip(b.GetMin(),b.GetMax()))))
        usd_inverse=np.asarray(cache.GetLocalToWorldTransform(owner).GetInverse())
        native=np.asarray(native_body_poses[str(owner.GetPath())])
        local=corners@usd_inverse[:3,:3]+usd_inverse[3,:3]
        current=local@native[:3,:3]+native[3,:3]
        result.append([current.min(0).tolist(),current.max(0).tolist()])
    if not result:raise ValueError('Actual robot visual geometry missing')
    return result


def rear_side_normal(narrow_axis, center, robot_position):
    """Select the narrow side facing away from the original robot station."""
    normal=np.array(narrow_axis,dtype=float).copy();normal[2]=0.
    away=np.asarray(center,dtype=float)-np.asarray(robot_position,dtype=float);away[2]=0.
    if not np.isfinite(normal).all() or not np.isfinite(away).all() or np.linalg.norm(normal)==0:
        raise ValueError('Measured horizontal bag axis and original robot position required')
    normal/=np.linalg.norm(normal)
    # The palm's fitted position lies on its normal side; entry extends farther
    # outward on that same side, not through the bag from the opposite side.
    if np.dot(normal,away)<0:normal=-normal
    if abs(np.dot(normal,away))<1e-12:
        raise ValueError('Rear narrow side is ambiguous relative to the robot')
    return normal


def rear_entry(asset,orientation,grip,outward,fixtures,target_box):
    """Use the midpoint of actual free rear space, not a long fixed detour.

    Ray/AABB admission uses the open anatomical groups and unchanged fixtures.
    This selects the new actor's route, never changes collision parameters.
    """
    points=asset.sample(0.)[0]@orientation.T+grip
    shapes=np.array([[points[g].min(0),points[g].max(0)]
                     for g in asset.collision_groups().values()])
    free=2*max(np.asarray(target_box)[1,:2]-np.asarray(target_box)[0,:2])
    for shape in shapes:
        for obstacle in np.asarray(fixtures):
            enter,leave=-np.inf,np.inf
            for axis in range(3):
                if abs(outward[axis])<1e-12:
                    if shape[1,axis]<=obstacle[0,axis] or obstacle[1,axis]<=shape[0,axis]:
                        leave=-np.inf;break
                else:
                    a=(obstacle[0,axis]-shape[1,axis])/outward[axis]
                    b=(obstacle[1,axis]-shape[0,axis])/outward[axis]
                    enter=max(enter,min(a,b));leave=min(leave,max(a,b))
            if enter<leave and leave>0:free=min(free,max(0.,enter))
    if free<=0:raise ValueError('No fixture-clear rear entry from fitted palm')
    return grip+.5*free*outward,dict(
        method='midpoint_of_fixture_clear_rear_ray',available_rear_distance_m=float(free),
        selected_entry_distance_m=float(.5*free))


def plan(stage, asset, seed, initial_plan,*,pair=None):
    from pxr import UsdGeom
    from evaluation.tasks.coffee_machine.specs import BAG, COUNTER, MACHINE, CUP
    from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders
    cache=UsdGeom.XformCache();tf=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BAG)))
    local=np.concatenate([v for v,s in body_colliders(stage,BAG).values() if v is not None])
    vertices=local@tf[:3,:3]+tf[3,:3];box=np.array([vertices.min(0),vertices.max(0)]);center=box.mean(0)
    from .training_domain import domain,contains_origin
    from .relocation_profile import sample_destination
    training=domain(initial_plan)
    start_origin=tf[3,:3]
    if not contains_origin(start_origin,training):raise ValueError('Initial bag origin is outside its original data domain')
    counter=collider_boxes(stage,[COUNTER])[0];fixtures=collider_boxes(stage,[MACHINE,CUP])
    if not (np.all(box[0,:2]>=np.asarray(counter)[0,:2]) and np.all(box[1,:2]<=np.asarray(counter)[1,:2])):
        raise ValueError('Paired start footprint outside original counter')
    # The narrow bag axis is the source's local X, not world X.
    robot_position=initial_plan['robot_base']['root_xyz']
    normal=rear_side_normal(tf[0,:3],center,robot_position)
    vertical=np.array([0.,0.,1.]);finger=np.cross(normal,vertical)
    # User chose lateral approach/closure, not an upright top-down palm.
    orientation=np.column_stack((vertical,finger,normal))
    from .side_grip import fit_side_anchor
    section=(vertices-center)@orientation
    anchor,grip_fit=fit_side_anchor(asset,section[:,1:])
    grip=center-orientation@anchor
    # Fit may put the palm on the other normal side: choose the actual wrist,
    # not merely the orientation sign, relative to the robot.
    if np.dot((grip-center)[:2],(center-np.asarray(robot_position))[:2])<=0:
        normal=-normal;finger=np.cross(normal,vertical)
        orientation=np.column_stack((vertical,finger,normal))
        section=(vertices-center)@orientation
        anchor,grip_fit=fit_side_anchor(asset,section[:,1:])
        grip=center-orientation@anchor
    if np.dot((grip-center)[:2],(center-np.asarray(robot_position))[:2])<=0:
        raise ValueError('No rear-side fitted palm position')
    outward=grip-center;outward[2]=0;outward/=np.linalg.norm(outward)
    # User chose to appear directly at the rear-side open grip and close,
    # rather than spending time entering from farther behind the bag.
    entry=grip.copy()
    raised=grip.copy()
    pair=pair or initial_plan.get('relocation_pair')
    if pair is not None:
        if not np.allclose(start_origin,pair['start_origin_m'],rtol=0,atol=1e-9):
            raise ValueError('Scene start differs from frozen joint initialization')
        from .joint_initialization import direction_check
        if not direction_check(start_origin,pair['destination_origin_m'],robot_position,
                               reference_kind='original_robot_base')['allowed']:
            raise ValueError('Initial paired route approaches robot reference')
        proposal=dict(pair,seed=seed,height_preserved=True,minimum_distance_m=None)
    else:
        proposal=sample_destination(seed,start_origin,training)  # historical source diagnostics only
    distance=proposal['distance_m']
    rejected=[];groups=list(asset.collision_groups().values())
    for endpoint in [proposal['destination_origin_m']]:
        delta=np.asarray(endpoint)-start_origin;final=box+delta
        if not contains_origin(start_origin+delta,training):
            rejected.append(dict(endpoint=endpoint,reason='outside_original_bag_data_domain'));continue
        if not (np.all(final[0,:2]>=np.asarray(counter)[0,:2]) and np.all(final[1,:2]<=np.asarray(counter)[1,:2])):
            rejected.append(dict(endpoint=endpoint,reason='outside_counter'));continue
        destination=grip+delta
        exit_pose,exit_admission=rear_entry(asset,orientation,destination,outward,fixtures,final)
        motion=RelocationMotion(entry,grip,raised,raised+delta,destination,exit_pose)
        # Preview starts without robot teleportation. Robot geometry is checked
        # separately after restoring the source preparation joints.
        allowed,reason=check_transport(motion,asset,orientation,[np.array([[-1e6]*3,[-1e6+1]*3])],fixtures,box)
        if not allowed:rejected.append(dict(endpoint=endpoint,reason=reason));continue
        # Source bag starts on its support, so its zero-lift carry boundary
        # intentionally touches the counter. Do not exempt the human: check
        # its anatomical hulls against that slab throughout every phase.
        hand_clear=True
        slab=np.asarray(counter)
        for t in np.linspace(0,motion.duration_s,61):
            sample=motion.sample(float(t));pts=asset.sample(sample.closure)[0]@orientation.T+sample.wrist_position
            shapes=np.array([[pts[g].min(0),pts[g].max(0)] for g in groups])
            if np.any(np.all(shapes[:,0]<slab[1],-1)&np.all(slab[0]<shapes[:,1],-1)):
                hand_clear=False;break
        if not hand_clear:rejected.append(dict(endpoint=endpoint,reason='hand_crosses_counter'));continue
        p,_=asset.sample(0.)
        clear=True
        for fraction in np.linspace(0,1,49):
            pts=p@orientation.T+destination+fraction*(exit_pose-destination)
            shapes=np.array([[pts[g].min(0),pts[g].max(0)] for g in groups])
            obs=np.asarray(fixtures)
            if np.any(np.all(shapes[:,None,0]<obs[None,:,1],-1)&np.all(obs[None,:,0]<shapes[:,None,1],-1)):
                clear=False;break
        if not clear:rejected.append(dict(endpoint=endpoint,reason='retreat_fixture'));continue
        return dict(seed=seed,distance_m=distance,delta_m=delta.tolist(),target_bounds=box.tolist(),
            destination_bounds=final.tolist(),orientation=orientation.tolist(),waypoints=[list(p) for p in motion.points],
            fixture_boxes=fixtures,counter_bounds=counter,hand_counter_checked=True,
            original_support_contact_expected=True,grip_anchor_local=anchor.tolist(),grip_fit=grip_fit,rejected=rejected,
            training_domain=training,source_bag_origin=start_origin.tolist(),planned_bag_origin=(start_origin+delta).tolist(),
            endpoint_sampling=proposal,
            motion_duration_s=motion.duration_s,grasp_side='rear relative to original robot station',
            grasp_side_reference_robot_position=list(robot_position),
            constant_height_horizontal_translation=True,
            appearance='absent until trigger; appears open at fitted rear-side wrist',
            rear_exit_admission=exit_admission,
            joint_pair=pair,
            method='frozen_countertop_footprint_and_full_hand_route',native_validation_required=True)
    raise ValueError('No admitted countertop route: '+str(rejected))

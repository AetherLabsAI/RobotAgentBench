"""Initialization-reserved coffee routes, selected once from fresh native facts.

No robot commands, target pose writes, physics stepping, timer or scoring edits.
The original single-route actor remains unchanged for old/combined initials.
"""
from copy import deepcopy
import math

SCHEMA = 'coffee_bag_interference.reserved_routes.v1'


def endpoint_options(start, training, original_endpoint):
    """Original endpoint plus interior points toward domain corners/edge midpoints.

    These are not new initial-object draws and do not extend the training range.
    The candidate count follows the rectangle's geometry, not a tuned distance.
    """
    lower, upper = training['origin_bounds_world_xy_m']
    if any(not math.isfinite(v) for v in [*start,*lower,*upper,*original_endpoint]):
        raise ValueError('Finite training rectangle and target positions required')
    if any(lower[i] >= upper[i] or not lower[i] <= start[i] <= upper[i] for i in range(2)):
        raise ValueError('Original bag start must lie in its unchanged training rectangle')
    points = [list(original_endpoint)]
    for x in (lower[0],(lower[0]+upper[0])/2,upper[0]):
        for y in (lower[1],(lower[1]+upper[1])/2,upper[1]):
            # Use each ray's midpoint inside the domain, not an exact boundary
            # landing whose tiny native release drift could exit the range.
            points.append([(x+start[0])/2,(y+start[1])/2,start[2]])
    unique=[]
    for p in points:
        if not (all(lower[i] <= p[i] <= upper[i] for i in range(2)) and p[2]==start[2]):
            raise ValueError('Reserved endpoint outside original domain or height')
        if p != list(start) and p not in unique:unique.append(p)
    return unique


def _hit(shapes, obstacles):
    import numpy as np
    shapes=np.asarray(shapes,dtype=float);obstacles=np.asarray(obstacles,dtype=float)
    if not len(obstacles):return False
    return bool(np.any(np.all(shapes[:,None,0]<obstacles[None,:,1],-1)&
                       np.all(obstacles[None,:,0]<shapes[:,None,1],-1)))


def _motion_samples(asset, orientation, motion, target_box):
    """Same 49 transport samples as the existing shared checker, cached once."""
    import numpy as np
    groups=list(asset.collision_groups().values());hand=[];carried=[]
    for t in np.linspace(0,2.4,49):
        s=motion.sample(float(t))
        pts=asset.sample(s.closure)[0]@orientation.T+np.asarray(s.wrist_position)
        hand.extend([[pts[g].min(0).tolist(),pts[g].max(0).tolist()] for g in groups])
        if s.phase in ('carry','release'):
            carried.append((np.asarray(target_box)+np.asarray(s.wrist_position)-motion.points[1]).tolist())
    return dict(hand=hand,carried=carried,sample_count=49)


def reserve_routes(stage, initial, template):
    """Admit the complete route BEFORE physics; never search new endpoints live."""
    import numpy as np
    from .motion import RelocationMotion
    from .preview_geometry import rear_entry
    from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
    asset=AdaptiveHand();groups=list(asset.collision_groups().values())
    orientation=np.asarray(template['orientation']);box=np.asarray(template['target_bounds'])
    start=template['source_bag_origin'];grip=np.asarray(template['waypoints'][1])
    outward=grip-box.mean(0);outward[2]=0;outward/=np.linalg.norm(outward)
    fixtures=template['fixture_boxes'];counter=np.asarray(template['counter_bounds'])
    routes=[];rejected=[]
    for endpoint in endpoint_options(start,template['training_domain'],template['planned_bag_origin']):
        delta=np.asarray(endpoint)-start;final=box+delta
        if not (np.all(final[0,:2]>=counter[0,:2]) and np.all(final[1,:2]<=counter[1,:2])):
            rejected.append(dict(endpoint=endpoint,reason='target_footprint_outside_counter'));continue
        destination=grip+delta
        try:
            exit_pose,exit_admission=rear_entry(asset,orientation,destination,outward,fixtures,final)
            motion=RelocationMotion(grip,grip,grip,destination,destination,exit_pose)
            samples=_motion_samples(asset,orientation,motion,box)
            if _hit(samples['hand'],fixtures) or _hit(samples['carried'],fixtures):
                raise ValueError('hand_or_carried_bag_crosses_fixture')
            # Preserve the original 61-point counter and 49-point retreat checks.
            for t in np.linspace(0,motion.duration_s,61):
                sample=motion.sample(float(t))
                pts=asset.sample(sample.closure)[0]@orientation.T+sample.wrist_position
                shapes=[[pts[g].min(0),pts[g].max(0)] for g in groups]
                if _hit(shapes,[counter]):raise ValueError('hand_crosses_counter')
            pts=asset.sample(0.)[0]@orientation.T
            for fraction in np.linspace(0,1,49):
                current=pts+destination+fraction*(exit_pose-destination)
                if _hit([[current[g].min(0),current[g].max(0)] for g in groups],fixtures):
                    raise ValueError('retreat_crosses_fixture')
        except ValueError as error:
            rejected.append(dict(endpoint=endpoint,reason=str(error)));continue
        reservation=deepcopy(template)
        reservation.update(planned_bag_origin=list(endpoint),delta_m=delta.tolist(),
            distance_m=float(np.linalg.norm(delta)),destination_bounds=final.tolist(),
            waypoints=[list(p) for p in motion.points],rear_exit_admission=exit_admission,
            endpoint_sampling=dict(method='initialization_reserved_rectangle_design_points',
                                   destination_origin_m=list(endpoint)),
            joint_pair=None,selection_requires_fresh_jaws=True)
        routes.append(dict(id='route_%02d'%len(routes),reservation=reservation,samples=samples))
    if len(routes)<2:
        raise ValueError('Multiple fixture-clear routes required: '+str(rejected))
    return dict(schema=SCHEMA,source_bag_origin=list(start),training_domain=deepcopy(template['training_domain']),
        routes=routes,rejected=rejected,endpoints_frozen_at_initialization=True,
        robot_initialization_changed=False,live_endpoint_generation=False,
        selection='largest nearest-jaw endpoint clearance gain among currently clear reserved routes')


def _segment_minimum_distance2(start, end, point):
    d=[end[i]-start[i] for i in range(2)]
    length=sum(v*v for v in d)
    if length==0:raise ValueError('No zero-displacement route')
    fraction=max(0.,min(1.,sum((point[i]-start[i])*d[i] for i in range(2))/length))
    return sum((start[i]+fraction*d[i]-point[i])**2 for i in range(2))


def rank_routes(pool, facts):
    """Primary jaw must be in the away half-plane; neither jaw may be crossed."""
    from .joint_initialization import direction_check
    from .trigger_rules import distance_to_box
    start=facts['target_origin_world_m'];jaws=facts['gripper_centers_world_m']
    if set(jaws)!= {'l','r'}:raise ValueError('Both fresh native jaws required')
    if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in [start,*jaws.values()]):
        raise ValueError('Finite current target/jaw positions required')
    local=facts['gripper_centers_target_local_m']
    primary=min(local,key=lambda h:distance_to_box(local[h],facts['target_bounds_local_m']))
    initial=min(sum((start[i]-p[i])**2 for i in range(2)) for p in jaws.values())
    choices=[]
    for route in pool['routes']:
        end=route['reservation']['planned_bag_origin']
        direction=direction_check(start,end,jaws[primary])
        if not direction['allowed']:continue
        # Exact XY point/segment minimum, not just endpoint distance.
        path_min=min(_segment_minimum_distance2(start,end,p) for p in jaws.values())
        if path_min<initial:continue
        final=min(sum((end[i]-p[i])**2 for i in range(2)) for p in jaws.values())
        if final<=initial:continue
        choices.append(dict(route=route,direction=dict(direction,measured_hand=primary),
            clearance_gain_squared_m2=final-initial,minimum_path_distance_squared_m2=path_min))
    return sorted(choices,key=lambda x:(-x['clearance_gain_squared_m2'],x['route']['id']))


class ReservedRouteTrigger:
    """Evaluate only pre-reserved routes on proximity, then lock one forever."""
    def __init__(self,target_path,pool):
        from .trigger_rules import ProximityTrigger
        if pool.get('schema')!=SCHEMA or len(pool.get('routes',[]))<2:
            raise ValueError('Fresh initialization reservation pool required')
        self.pool=deepcopy(pool);self.trigger=ProximityTrigger(target_path)
        self.armed=False;self.selected=None;self.locked_direction=None;self.selection_evidence=None

    def arm(self):
        if self.armed:raise RuntimeError('Arm route selector once')
        self.armed=True

    def observe(self,facts,*,route_clear):
        from .trigger_rules import TRIGGER_DISTANCE_M,distance_to_box
        if not self.armed:raise RuntimeError('Native initial acceptance must arm routes')
        if not callable(route_clear):raise ValueError('Measured full-route checker required')
        distances={h:distance_to_box(p,facts['target_bounds_local_m'])
                   for h,p in facts['gripper_centers_target_local_m'].items()}
        eligible=(not self.trigger.fired and not self.trigger.robot_touched and
            not facts['robot_contacting_target'] and not facts['robot_holding_target'] and
            facts['contacts_complete'] and facts['target_supported'] and min(distances.values())<=TRIGGER_DISTANCE_M)
        choice=None;checked=[]
        if eligible:
            for candidate in rank_routes(self.pool,facts):
                clear,reason=route_clear(candidate['route'])
                if type(clear) is not bool:raise ValueError('Explicit current route admission required')
                checked.append(dict(route_id=candidate['route']['id'],clear=clear,reason=reason))
                if clear:choice=candidate;break
        event=self.trigger.observe(facts,route_clear=choice is not None)
        if event['start']:
            self.selected=deepcopy(choice['route']);self.locked_direction=deepcopy(choice['direction'])
            self.selection_evidence=dict(route_id=self.selected['id'],direction_check=self.locked_direction,
                jaw_positions_world_m=deepcopy(facts['gripper_centers_world_m']),
                selected_endpoint_m=self.selected['reservation']['planned_bag_origin'],
                clearance_gain_squared_m2=choice['clearance_gain_squared_m2'],checked_routes=checked,
                simulation_time_s=facts['simulation_time_s'],endpoint_was_pre_reserved=True)
            event.update(self.selection_evidence)
        elif eligible:event.update(reason='no_currently_clear_away_reserved_route',checked_routes=checked)
        return event


class LiveRouteGeometry:
    """Original collision/visual envelopes, transformed by current native poses."""
    def __init__(self,stage,view,excluded_grippers):
        from pxr import Usd,UsdGeom,UsdPhysics
        from itertools import product
        from evaluation.tasks.coffee_machine.specs import MACHINE,CUP
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        self.stage,self.excluded=stage,excluded_grippers
        self.fixtures=[];cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(0,['default','render','proxy'])
        for root in (MACHINE,CUP):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:continue
                bounds=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                if bounds.IsEmpty():continue
                owner=prim
                while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
                corners=list(product(*zip(bounds.GetMin(),bounds.GetMax())))
                if not owner:raise ValueError('Fixture collision has no measured rigid owner')
                import numpy as np
                inverse=np.asarray(cache.GetLocalToWorldTransform(owner).GetInverse())
                local=np.asarray(corners)@inverse[:3,:3]+inverse[3,:3]
                self.fixtures.append((_BoundBodies(view,[str(owner.GetPath())]),local))

    def snapshot(self,robot_native_poses):
        import numpy as np
        from evaluation.tasks.shelf_pick.geometry import rotation
        from .preview_geometry import robot_visual_boxes
        boxes=robot_visual_boxes(self.stage,self.excluded,robot_native_poses)
        for view,local in self.fixtures:
            p,q=view.get_world_pose()
            corners=rotation(q).apply(local)+np.asarray(p)
            boxes.append([corners.min(0).tolist(),corners.max(0).tolist()])
        return boxes

    @staticmethod
    def check(route,obstacles):
        samples=route['samples']
        clear=not (_hit(samples['hand'],obstacles) or _hit(samples['carried'],obstacles))
        return clear,dict(method='same_49_transport_samples_vs_current_native_obstacles',
            sample_count=samples['sample_count'],native_validation_required=True)

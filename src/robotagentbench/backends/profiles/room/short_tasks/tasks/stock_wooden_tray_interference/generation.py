"""Reserve-first shelf policy. Only the independent interference task opts in."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np

from .initialization import plan_route, reviewed_orientation, background_candidate_allowed
from .route_contract import require_rear_entry, robot_obstacle_shape
from .placement_geometry import body_bounds, stage_bounds

TASK='stock_wooden_tray_interference'
POLICY={'task_name':TASK,'version':'reserve_first.v1'}


def validate_policy(policy, requests):
    if {k:v for k,v in policy.items() if k not in ('robot_route_reference','relocation_distance')} != POLICY:
        raise ValueError('Unknown task-specific shelf generation policy')
    from .distance_bands import checked_rule
    checked_rule(policy.get('relocation_distance'))
    if 'robot_route_reference' in policy:
        path=Path(policy['robot_route_reference']).resolve()
        if not path.is_relative_to(Path(__file__).resolve().parent) or not path.is_file():
            raise ValueError('Task-owned route reference required')
    from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
    if (len(requests)!=1 or requests[0].get('type')!='snack_drink_to_tray'
            or requests[0].get('product') not in PRODUCTS
            or PRODUCTS[requests[0]['product']].tier!='tier3'):
        raise ValueError('Interference generation requires exactly one declared drink')


def validate_output(source, root):
    allowed=(Path(root)/'short_tasks/tasks'/TASK/'generated_rooms').resolve()
    source=Path(source).resolve()
    if source.parent.parent!=allowed:
        raise ValueError('Interference output must be a fresh task-owned generated-room folder')


def policy_for_plan(plan):
    from .distance_bands import checked_rule
    if plan.get('task_name') != TASK:
        raise ValueError('Distance selection belongs only to the interference task')
    return dict(POLICY, relocation_distance=checked_rule(plan.get('relocation_distance')))


def reserve_then_pack(*, rng, catalog, supports, fixtures, counts, pools,
                      pose_item, separated, planned, sample_scene, target_product,
                      reserve_candidate, target_allowed=None):
    """Use existing target proposals and fixed-target/background callback hooks.

    No background is placed until reserve_candidate has admitted the route
    and frozen both endpoint grasp trapezoids.
    Retain the existing inventory/counts/heading distributions/spacing/retries.
    An exhausted packing run fails, never reduces background counts.
    """
    occurrences=[sid for sid,names in planned.items() for name in names if name==target_product]
    if occurrences!=['tier3']:
        raise ValueError('Target must occur exactly once in the planned inventory')
    accepted={};rejected={}
    def admit(item, _placed):
        if target_allowed is not None and not target_allowed(item):
            rejected['outside_source_range']=rejected.get('outside_source_range',0)+1
            return False
        try:
            reservation=reserve_candidate(item)
            from .endpoint_lanes import validate_endpoint_lanes
            validate_endpoint_lanes(reservation)
        except ValueError as exc:
            rejected['last_reason']=str(exc)
            rejected['count']=rejected.get('count',0)+1
            if rejected['count']==1:
                import sys
                print('First reservation rejection: '+str(exc),file=sys.stderr,flush=True)
            return False
        accepted['reservation']=reservation
        return True
    try:
        target_rows,target_metrics=sample_scene(rng,catalog,supports,fixtures,
            {'tier3':1},{'tier3':[target_product]},pose_item,separated,
            planned_inventory={'tier3':[target_product]},candidate_allowed=admit)
    except RuntimeError as exc:
        raise RuntimeError('Reserve-first target admission exhausted: '+str(rejected)) from exc
    if len(target_rows)!=1 or 'reservation' not in accepted:
        raise ValueError('Target proposal did not produce an admitted reservation')
    reservation=accepted['reservation']
    def allowed(item,_placed):
        return background_candidate_allowed(reservation,body_bounds(item[1]))
    placed,metrics=sample_scene(rng,catalog,supports,fixtures,counts,pools,pose_item,separated,
        planned_inventory=planned,fixed_target=target_rows[0],candidate_allowed=allowed)
    if len(placed)!=sum(counts.values()):
        raise ValueError('Reserved packing changed the inventory count')
    for item in placed[1:]:
        if not allowed(item,[]):raise ValueError('Background crossed the frozen reservation')
    metrics.update(interference_policy='reserve_first.v1',target_sampling=target_metrics,
                   reserve_before_background=True, endpoint_grasp_trapezoids=2)
    if 'bilateral_routes' in reservation:
        metrics.update(endpoint_grasp_trapezoids=3,reserved_routes=2,
                       target_lateral_sampling='source_bounded_triangular_center_preference')
    return placed,metrics,reservation


def sample_reserved(policy,stage,requests,seed,catalog,supports,fixtures,counts,pools,
                    pose_item,separated,planned,sample_scene):
    validate_policy(policy,requests)
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    from repair_room_placement_v2 import cs,shape_mesh
    from .adaptive_hand import AdaptiveHand
    shelf='/World/RoomEpisode/Shelf'
    frame=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(shelf))
    if not np.allclose(np.asarray(frame),np.asarray(frame.RemoveScaleShear()),atol=1e-7,rtol=0):
        raise ValueError('Non-rigid shelf frame')
    inverse=frame.GetInverse();cache=UsdGeom.XformCache()
    robot=stage.GetPrimAtPath('/World/Robot')
    if not robot:raise ValueError('Robot geometry missing')
    robot_bounds=[]
    for p in Usd.PrimRange(robot):
        if not robot_obstacle_shape(p):continue
        robot_bounds.append(stage_bounds(p,cache,inverse))
    # Use separate physical shelf pieces, never a solid box around the hollow
    # shelf's entire visual mesh. Native/full-room admission is still required.
    shelf_fixtures=[b for b in fixtures if b.name.startswith(shelf+'/')
        and stage.GetPrimAtPath(b.name).HasAPI(UsdPhysics.CollisionAPI)
        and stage.GetPrimAtPath(b.name).GetAttribute('physics:collisionEnabled').Get() is not False]
    board=[b for b in shelf_fixtures if b.name==supports['tier3']['prim_path']]
    if len(board)!=1:raise ValueError('Original shelf support collider missing')
    from .shelf_geometry import collider_component_boxes
    fixture_boxes=[box for b in shelf_fixtures for box in collider_component_boxes(
        stage.GetPrimAtPath(b.name),inverse,np.array([b.lo,b.hi]))]
    board_box=np.array([board[0].lo,board[0].hi])
    hand=AdaptiveHand()
    # Cache read-only deformations across candidate checks. No native state.
    class CachedHand:
        def __init__(self):self.cache={}
        def sample(self,c):
            if c not in self.cache:self.cache[c]=hand.sample(c)
            return self.cache[c]
        def collision_groups(self):return hand.collision_groups()
        def palm_center(self,r):return hand.palm_center(r)
    asset=CachedHand()
    probe=Usd.Stage.CreateInMemory()
    margin=float(PhysxSchema.PhysxCollisionAPI.Apply(probe.DefinePrim('/Probe')).GetContactOffsetAttr().Get())
    orientation=reviewed_orientation(asset,margin)
    from .source_range import contract, contains, destination_allowed
    from short_tasks.tasks.stock_wooden_tray.initial_station import shelf_frame
    source_frame=shelf_frame(stage,{'objects':[{'id':'snack_shelf','path':shelf}]})
    range_rule=contract(requests[0]['product'],source_frame['local_bounds'])
    route_reference=None
    if 'robot_route_reference' in policy:
        import json
        route_reference=json.loads(Path(policy['robot_route_reference']).read_text())
        if (route_reference.get('schema')!='recorded_gripper_route_reference.v1'
                or route_reference['seed']!=seed
                or not np.allclose(route_reference['shelf_frame'],np.asarray(frame),atol=1e-7,rtol=0)):
            raise ValueError('Recorded route reference belongs to another room')
    def in_range(item):
        row=item[0]
        from shapely.geometry import Polygon
        return (contains(range_rule,row['xyz_m'][:2],row['yaw_deg'])
                and Polygon(supports['tier3']['polygon_xy_m']).covers(item[2]))
    def reserve(item):
        row,bodies,_=item;box=body_bounds(bodies)
        ceilings=[b[0,2] for b in fixture_boxes if b[0,2]>box[1,2]]
        if not ceilings:raise ValueError('Upper shelf missing')
        def route(side):
            center=box.mean(0)
            return plan_route(asset=asset,target_path=row['prim_path']+'/Asset',
            shelf_level_path=supports['tier3']['prim_path'],seed=seed,shelf_bounds=board_box,
            target_bounds=box,stock_bounds=[],robot_bounds=robot_bounds,
            fixture_bounds=fixture_boxes,base_orientation=orientation,ceiling_z=float(min(ceilings)),
            destination_allowed=lambda xy: destination_allowed(range_rule,row['xyz_m'][:2],box,xy,row['yaw_deg']),
            rear_short=True,
            distance_rule=policy.get('relocation_distance'),
            approach_points=None,
            destination_filter=lambda xy: side*(xy[0]-center[0])>0
                and np.isclose(xy[1],center[1],atol=1e-9,rtol=0))
        alternatives=[route(-1),route(1)]
        metadata=dict(producer=POLICY,task_id=requests[0]['id'],product=requests[0]['product'],
            target_wrapper=row['prim_path'],shelf_path=shelf,
            shelf_local_to_world_row_matrix=np.asarray(frame).tolist(),
            target_initial_pose=deepcopy(row),source_product_range=deepcopy(range_rule),
            hand_asset_sha256=hashlib.sha256(hand.path.read_bytes()).hexdigest())
        metadata['geometry_source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(hand.path),Path(__file__).parent/'static_short_hand_011/cropped_rig.json')}
        for reservation in alternatives:
            reservation.update(deepcopy(metadata))
        from .bilateral_routes import combine
        return combine(*alternatives)
    # Center preference changes only this variant's target proposal, before any
    # background packing. Original range/heading/depth and counts remain intact.
    center_rng=np.random.default_rng(seed)
    def centered_pose(name,xy,yaw,sid,role):
        original=pose_item(name,xy,yaw,sid,role)
        if name!=requests[0]['product']:
            return original
        from .bilateral_routes import centered_x
        proposed=list(xy)
        proposed[0]=centered_x(center_rng,original_xy=xy,
            target_bounds=body_bounds(original[1]),shelf_bounds=board_box,source_rule=range_rule)
        return pose_item(name,proposed,yaw,sid,role)
    return reserve_then_pack(rng=np.random.default_rng(seed),catalog=catalog,supports=supports,
        fixtures=fixtures,counts=counts,pools=pools,pose_item=centered_pose,separated=separated,
        planned=planned,sample_scene=sample_scene,target_product=requests[0]['product'],
        reserve_candidate=reserve,target_allowed=in_range)


def validate_frozen(reservation, *, target_path, shelf_level_path, target_box, shelf_box,
                    stock_boxes, robot_boxes, fixture_boxes, frame, asset, source_shelf_bounds=None):
    """Recheck generated geometry without selecting another destination."""
    if 'bilateral_routes' in reservation:
        from .bilateral_routes import reservations
        for route in reservations(reservation):
            validate_frozen(route,target_path=target_path,shelf_level_path=shelf_level_path,
                target_box=target_box,shelf_box=shelf_box,stock_boxes=stock_boxes,
                robot_boxes=robot_boxes,fixture_boxes=fixture_boxes,frame=frame,
                asset=asset,source_shelf_bounds=source_shelf_bounds)
        return deepcopy(reservation)
    from .reservation import slot_is_clear, freeze_reservation
    from .motion import RelocationMotion
    require_rear_entry(reservation)
    if reservation.get('producer')!=POLICY:raise ValueError('Unrecognized generated reservation')
    if reservation.get('coordinate_frame')!='shelf_local_m':raise ValueError('Shelf-local reservation required')
    if 'source_product_range' in reservation:
        from .source_range import validate_record
        if source_shelf_bounds is None:raise ValueError('Measured source range shelf frame required')
        validate_record(reservation,source_shelf_bounds)
    if reservation['target_path']!=target_path or reservation['shelf_level_path']!=shelf_level_path:
        raise ValueError('Generated reservation identity changed')
    if not np.allclose(reservation['initial_target_bounds'],target_box,atol=1e-6,rtol=0):
        raise ValueError('Generated target geometry changed: planned='+str(reservation['initial_target_bounds'])+
                         '; composed='+str(np.asarray(target_box).tolist()))
    if not np.allclose(reservation['shelf_local_to_world_row_matrix'],frame,atol=1e-7,rtol=0):
        raise ValueError('Generated shelf frame changed')
    expected=freeze_reservation(target_path=target_path,shelf_level_path=shelf_level_path,
        seed=reservation['seed'],shelf_bounds=shelf_box,target_bounds=target_box,
        destination_xy=reservation['destination_xy'],obstacle_bounds=stock_boxes,
        waypoints=reservation['waypoints'],route_evidence=reservation['route_evidence'],
        distance_rule=reservation.get('relocation_distance'))
    if not np.allclose(expected['reserved_bounds'],reservation['reserved_bounds'],atol=1e-6,rtol=0):
        raise ValueError('Generated destination geometry changed')
    from .endpoint_lanes import validate_endpoint_lanes
    validate_endpoint_lanes(reservation)
    if hasattr(asset,'path') and reservation.get('hand_asset_sha256')!=hashlib.sha256(Path(asset.path).read_bytes()).hexdigest():
        raise ValueError('Generated human geometry changed')
    if hasattr(asset,'path'):
        for p in (Path(asset.path),Path(__file__).parent/'static_short_hand_011/cropped_rig.json'):
            if reservation.get('geometry_source_hashes',{}).get(str(p))!=hashlib.sha256(p.read_bytes()).hexdigest():
                raise ValueError('Generated human source changed: '+str(p))
    slot=np.asarray(reservation['reserved_bounds'])
    if np.any(slot[0,:2]<shelf_box[0,:2]-1e-9) or np.any(slot[1,:2]>shelf_box[1,:2]+1e-9):
        raise ValueError('Reserved destination outside shelf')
    if not slot_is_clear(reservation,stock_boxes) or any(not background_candidate_allowed(reservation,b) for b in stock_boxes):
        raise ValueError('Generated reservation occupied')
    orientation=np.asarray(reservation['hand_orientation'])
    if orientation.shape!=(3,3) or not np.allclose(orientation@orientation.T,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(orientation),1):
        raise ValueError('Invalid hand rotation')
    motion=RelocationMotion(*reservation['waypoints'])
    delta=slot.mean(0)-np.asarray(target_box).mean(0)
    if not np.allclose(np.asarray(motion.points[4])-motion.points[1],delta,atol=1e-9,rtol=0):
        raise ValueError('Route no longer ends in reserved destination')
    groups=list(asset.collision_groups().values());obstacles=np.asarray([*robot_boxes,*fixture_boxes,*stock_boxes])
    if not robot_boxes:raise ValueError('Robot geometry missing')
    for t in np.linspace(0,3,61):
        state=motion.sample(float(t));points=asset.sample(state.closure)[0]@orientation.T+state.wrist_position
        boxes=np.array([[points[g].min(0),points[g].max(0)] for g in groups])
        if state.phase in ('carry','release'):
            carried=np.asarray(target_box)+np.asarray(state.wrist_position)-motion.points[1]
            boxes=np.concatenate((boxes,carried[None]))
        if np.any(np.all(boxes[:,None,0]<obstacles[None,:,1],axis=-1)&np.all(obstacles[None,:,0]<boxes[:,None,1],axis=-1)):
            raise ValueError('Generated human route blocked at '+str(t))
    return deepcopy(reservation)

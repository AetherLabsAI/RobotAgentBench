"""CPU geometry admission for this task, before native physics attachment.

Reuses the reviewed hand and existing side-entry/carry/retreat algorithms.
No object is moved to make a blocked room pass. All calculations are in the
rigid shelf frame; authored robot geometry is NOT a settled-robot certificate.
"""
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np

from .motion import RelocationMotion, seeded_destinations
from .reservation import freeze_reservation, with_reservation
from .retreat_geometry import choose_side_entry, check_transport, choose_retreat


def background_candidate_allowed(reservation, candidate_bounds):
    """Task-local callback for a future generator hook; shelf-local metres.

    Preserve both endpoint grasp trapezoids, and the sampled hand /
    carried-object corridor, including boxes connecting adjacent samples.
    Never modifies a candidate or the reservation, and never awards points.
    """
    if 'bilateral_routes' in reservation:
        from .bilateral_routes import reservations
        return all(background_candidate_allowed(r, candidate_bounds)
                   for r in reservations(reservation))
    from .reservation import _bounds, _overlap, SCHEMA
    if reservation.get('schema') != SCHEMA or reservation.get('coordinate_frame') != 'shelf_local_m':
        raise ValueError('An explicitly shelf-local reservation is required')
    candidate = _bounds(candidate_bounds)
    from .endpoint_lanes import clear_of_endpoint_lanes
    if not clear_of_endpoint_lanes(reservation, candidate):
        return False
    protected = [reservation['reserved_bounds'], reservation['initial_target_bounds']]
    samples = reservation.get('corridor_samples')
    if not samples: raise ValueError('Hand corridor missing from reservation')
    previous = []
    for sample in samples:
        shapes = sample['bounds']
        protected.extend(shapes)
        for a, b in zip(previous, shapes):
            protected.append([np.minimum(a[0],b[0]),np.maximum(a[1],b[1])])
        previous = shapes
    return not any(_overlap(candidate, _bounds(box)) for box in protected)


def reviewed_orientation(asset, contact_margin):
    """Same anatomical front-plane admission used by the native diagnostic."""
    points, bones = asset.sample(0.)
    groups = asset.collision_groups()
    palm = points[groups['hand.R']]
    arm = points[groups['forearm.R'] + groups.get('upper_arm.R', [])]
    elbow = bones['forearm.R'][:3, 3]
    angle = math.atan2(elbow[2], -elbow[1])
    def clear(a):
        return ((arm[:, 1]*math.sin(a)+arm[:, 2]*math.cos(a)).max()+contact_margin
                <= (palm[:, 1]*math.sin(a)+palm[:, 2]*math.cos(a)).max())
    if not clear(angle):
        lo, hi = angle, math.pi/2
        if not clear(hi):
            raise ValueError('No front-plane-clear anatomical arm pose')
        for _ in range(32):
            mid = (lo+hi)/2
            if clear(mid): hi = mid
            else: lo = mid
        angle = hi
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[0,-1,0],[0,0,1],[-1,0,0]]) @ np.array([[1,0,0],[0,c,-s],[0,s,c]])


def plan_route(*, asset, target_path, shelf_level_path, seed, shelf_bounds,
               target_bounds, stock_bounds, robot_bounds, fixture_bounds,
               base_orientation, ceiling_z, destination_allowed=None,
               rear_short=True, approach_points=None, distance_rule=None,
               destination_filter=None):
    """Choose once at initialization. Reject blocked stock; never delete it."""
    if not rear_short:
        raise ValueError('Rear entry is required for this interference task')
    shelf = np.asarray(shelf_bounds, dtype=float)
    box = np.asarray(target_bounds, dtype=float)
    for b in [shelf, box, *stock_bounds, *robot_bounds, *fixture_bounds]:
        b = np.asarray(b)
        if b.shape != (2,3) or not np.isfinite(b).all() or np.any(b[1] < b[0]):
            raise ValueError('Finite ordered shelf-frame geometry required')
    if not robot_bounds or not math.isfinite(ceiling_z):
        raise ValueError('Robot geometry and upper shelf required')
    if np.any(box[0,:2] < shelf[0,:2]) or np.any(box[1,:2] > shelf[1,:2]):
        raise ValueError('Initial target footprint extends outside its shelf')
    center = box.mean(0)
    from .endpoint_lanes import build_endpoint_lanes, current_rule
    lane_rule = current_rule()
    # Only obstacles at the target's height block the destination footprint.
    same_level = [np.asarray(b)[:,:2] for b in [*stock_bounds, *fixture_bounds]
                  if b[0][2] < box[1,2] and b[1][2] > box[0,2]]
    destinations = seeded_destinations(shelf[:,:2], box[:,:2], same_level, seed)
    from .distance_bands import checked_rule, allows
    distance_rule = checked_rule(distance_rule)
    destinations = [xy for xy in destinations if allows(distance_rule, center[:2], xy)]
    if not destinations:
        raise ValueError('No same-level destination in selected relocation distance band: '+str(distance_rule))
    if destination_allowed is not None:
        destinations = [xy for xy in destinations if destination_allowed(xy)]
        if not destinations:
            raise ValueError('No separate destination within source product range')
    if destination_filter is not None:
        destinations = [xy for xy in destinations if destination_filter(xy)]
        if not destinations:
            raise ValueError('No same-depth destination on the required side')
    if approach_points is not None:
        from .recorded_clearance import away_destinations
        destinations=away_destinations(box,destinations,approach_points)
        if not destinations:
            raise ValueError('No destination away from the recorded approaching gripper')
    obstacles = [np.asarray(b) for b in [*stock_bounds, *fixture_bounds]]
    radius = float(max(box[1,:2]-box[0,:2])/2)
    orientation, grip, entry, entry_info = choose_side_entry(
        asset, base_orientation, asset.palm_center(radius), box, obstacles,rear_short=rear_short,
        approach_points=approach_points)
    hand_top = max(float((asset.sample(c)[0]@orientation.T+grip)[:,2].max()) for c in (0.,1.))
    free = ceiling_z-max(hand_top, box[1,2])
    if free <= 0:
        raise ValueError('No clearance below the upper shelf')
    height = float(box[1,2]-box[0,2])
    lift = min(height, free/2)  # existing geometric ceiling-limited lift
    raised = grip + [0,0,lift]
    open_points, _ = asset.sample(0.)
    rejected = []
    for destination in destinations:
        delta = np.r_[np.asarray(destination)-center[:2], 0.]
        final_box = box+delta
        try:
            endpoint_lanes = build_endpoint_lanes(box, final_box, lane_rule)
        except ValueError as exc:
            rejected.append(dict(destination=destination, reason=str(exc)))
            continue
        dest_grip = grip+delta
        motion = RelocationMotion(entry, grip, raised, raised+delta, dest_grip, dest_grip)
        valid, transport = check_transport(motion, asset, orientation, robot_bounds, obstacles, box)
        if not valid:
            rejected.append(dict(destination=destination, reason=transport))
            if transport['phase'] == 'approach': break
            continue
        try:
            exit_pose, retreat = choose_retreat(dest_grip, open_points@orientation.T,
                robot_bounds, obstacles, 2*height, list(asset.collision_groups().values()), final_box,rear_short=rear_short)
        except ValueError as exc:
            rejected.append(dict(destination=destination, reason=str(exc)))
            continue
        motion = RelocationMotion(entry, grip, raised, raised+delta, dest_grip, exit_pose)
        # Publish actual sampled hand/body envelopes, not just wrist points.
        # These are conservative background exclusion volumes, not colliders.
        corridor = []
        groups = list(asset.collision_groups().values())
        for t in np.linspace(0, 3, 61):
            state = motion.sample(float(t))
            points = asset.sample(state.closure)[0]@orientation.T + state.wrist_position
            shapes = [[points[g].min(0).tolist(), points[g].max(0).tolist()] for g in groups]
            if state.phase in ('carry', 'release'):
                shapes.append((box + np.asarray(state.wrist_position)-grip).tolist())
            corridor.append(dict(time_s=float(t), phase=state.phase, bounds=shapes))
        reservation = freeze_reservation(target_path=target_path, shelf_level_path=shelf_level_path,
            seed=seed, shelf_bounds=shelf, target_bounds=box, destination_xy=destination,
            obstacle_bounds=stock_bounds, waypoints=motion.points,
            distance_rule=distance_rule,
            route_evidence=dict(transport_clear=True, retreat_clear=True, entry=entry_info,
                transport=transport, retreat=retreat))
        reservation.update(coordinate_frame='shelf_local_m', hand_orientation=orientation.tolist(),
            corridor_samples=corridor, endpoint_grasp_lanes=endpoint_lanes,
            candidate_count=len(destinations), rejected=rejected,
            robot_pose_evidence='authored_geometry_only',
            native_ready_pose_recheck_required=True, native_collision_recheck_required=True)
        if any(not background_candidate_allowed(reservation,b) for b in stock_bounds):
            rejected.append(dict(destination=destination,reason='background intersects protected corridor'))
            continue
        return reservation
    raise ValueError('No clear initialization route; do not start human: '+str(rejected))


def prepare_reservation(plan, stage, manifest):
    """Read live USD geometry; no stale manifest AABBs or object pose writes."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
    from .adaptive_hand import AdaptiveHand
    existing = manifest.get('environment_reservations', [])
    from .distance_bands import checked_rule, require_manifest_band
    distance_rule = checked_rule(plan.get('relocation_distance'))
    if existing:
        require_manifest_band(plan, manifest)
        from .generation import POLICY
        if len(existing) != 1 or existing[0].get('producer') != POLICY:
            raise ValueError('Existing reservation must not be silently replanned')
        declaration=plan['runtime_components'][0]
        if (existing[0].get('task_id')!=declaration['instance_id']
                or existing[0].get('product')!=declaration['parameters']['product']
                or existing[0].get('seed')!=manifest['seed']):
            raise ValueError('Generated reservation belongs to a different task or seed')
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Expected the existing metre/Z-up room')
    def one(values, role):
        if len(values) != 1: raise ValueError('Exactly one '+role+' required')
        return values[0]
    shelf_path = one([o['path'] for o in manifest['objects'] if o['id']=='snack_shelf'], 'shelf')
    target_wrapper = manifest['interference_background_exclusion']['target_path']
    target = one([p for p in Usd.PrimRange(stage.GetPrimAtPath(target_wrapper))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)], 'target rigid body')
    if target.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Target must be a dynamic rigid body')
    shelf_prim = stage.GetPrimAtPath(shelf_path)
    if not shelf_prim: raise ValueError('Shelf absent from scene')
    cache = UsdGeom.XformCache()
    frame = cache.GetLocalToWorldTransform(shelf_prim)
    rigid = frame.RemoveScaleShear()
    if not np.allclose(np.asarray(frame), np.asarray(rigid), atol=1e-7, rtol=0):
        raise ValueError('Scaled shelf must be resolved by the scene owner')
    if not np.allclose(rigid.TransformDir(Gf.Vec3d(0,0,1)), [0,0,1], atol=1e-7, rtol=0):
        raise ValueError('Tilted shelf unsupported')
    inverse = rigid.GetInverse()
    geometry_cache = {}
    from .placement_geometry import stage_bounds
    shape_cache = {}
    def bounds(prim):
        path = str(prim.GetPath())
        if path in geometry_cache: return geometry_cache[path]
        result = stage_bounds(prim,cache,inverse,shape_cache=shape_cache)
        geometry_cache[path] = result
        return result
    colliders = [p for p in Usd.PrimRange(shelf_prim) if p.HasAPI(UsdPhysics.CollisionAPI)
                 and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    board = one([p for p in colliders if p.GetName()=='Tier3Overlay'], 'drink shelf board')
    shelf_box, target_box = bounds(board), bounds(target)
    from .shelf_geometry import collider_component_boxes
    fixtures = [box for p in colliders for box in collider_component_boxes(p,inverse,bounds(p))]
    ceilings = [b[0,2] for b in fixtures if b[0,2] > target_box[1,2]]
    if not ceilings: raise ValueError('Upper shelf geometry missing')
    robot = stage.GetPrimAtPath('/World/Robot')
    if not robot: raise ValueError('Robot geometry missing')
    from .route_contract import robot_obstacle_shape
    robot_boxes = [bounds(p) for p in Usd.PrimRange(robot) if robot_obstacle_shape(p)]
    stock_boxes = [bounds(stage.GetPrimAtPath(o['path'])) for o in manifest['objects']
                   if o['path'] != target_wrapper and
                   (o.get('source')=='snack_shelf' or o.get('instance_id','').startswith('shelf/'))
                   and o['path'] != shelf_path]
    # Read the existing human collision schema fallback on an anonymous probe;
    # never apply it to or change offsets on any original scene asset.
    probe = Usd.Stage.CreateInMemory()
    margin = float(PhysxSchema.PhysxCollisionAPI.Apply(probe.DefinePrim('/Probe')).GetContactOffsetAttr().Get())
    asset = AdaptiveHand()
    if existing:
        from copy import deepcopy
        from .generation import validate_frozen
        source_shelf_bounds=None
        if 'source_product_range' in existing[0]:
            from .source_range import validate_record
            from short_tasks.tasks.stock_wooden_tray.initial_station import shelf_frame
            source_shelf_bounds=shelf_frame(stage,manifest)['local_bounds']
            tf=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target_wrapper))*inverse)
            validate_record(existing[0],source_shelf_bounds,tf[3,:2],math.degrees(math.atan2(tf[0,1],tf[0,0])))
        checked = validate_frozen(existing[0], target_path=str(target.GetPath()),
            shelf_level_path=str(board.GetPath()), target_box=target_box, shelf_box=shelf_box,
            stock_boxes=stock_boxes, robot_boxes=robot_boxes, fixture_boxes=fixtures,
            frame=np.asarray(rigid), asset=asset,source_shelf_bounds=source_shelf_bounds)
        result = deepcopy(manifest)
        result['environment_reservations'] = [checked]
        return result
    reservation = plan_route(asset=asset, target_path=str(target.GetPath()),
        shelf_level_path=str(board.GetPath()), seed=manifest['seed'], shelf_bounds=shelf_box,
        target_bounds=target_box, stock_bounds=stock_boxes, robot_bounds=robot_boxes,
        fixture_bounds=fixtures, base_orientation=reviewed_orientation(asset, margin),
        distance_rule=distance_rule,
        ceiling_z=float(min(ceilings)))
    reservation.update(shelf_path=shelf_path, shelf_local_to_world_row_matrix=np.asarray(rigid).tolist(),
        target_wrapper=target_wrapper, hand_asset_sha256=hashlib.sha256(Path(asset.path).read_bytes()).hexdigest())
    reservation['geometry_source_hashes'] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(asset.path), Path(__file__).parent/'static_short_hand_011/cropped_rig.json')}
    return with_reservation(manifest, reservation)

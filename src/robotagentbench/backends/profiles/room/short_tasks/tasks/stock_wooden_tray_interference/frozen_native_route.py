"""Consume an initialization reservation unchanged; no runtime destination search."""
import numpy as np
from .motion import RelocationMotion
from .route_contract import require_rear_entry, robot_obstacle_shape


def world_motion(reservation):
    require_rear_entry(reservation)
    if reservation.get('coordinate_frame') != 'shelf_local_m':
        raise ValueError('Shelf-local initialization reservation required')
    frame = np.asarray(reservation['shelf_local_to_world_row_matrix'], dtype=float)
    if frame.shape != (4, 4) or not np.isfinite(frame).all():
        raise ValueError('Invalid shelf transform')
    rot = frame[:3, :3]
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rot), 1):
        raise ValueError('Rigid shelf transform required')
    points = np.asarray(reservation['waypoints']) @ rot + frame[3, :3]
    orientation = rot.T @ np.asarray(reservation['hand_orientation'])
    destination = np.asarray(reservation['reserved_bounds']).mean(0) @ rot + frame[3, :3]
    return RelocationMotion(*points), orientation, destination


def route_hits(boxes, obstacles, *, carried_index=None, support_indices=(), lift=None, support_verified=False):
    hits=np.all(boxes[:,None,0]<obstacles[None,:,1],axis=-1)&np.all(obstacles[None,:,0]<boxes[:,None,1],axis=-1)
    # A natively supported body starts with solver contact overlap. Lifting
    # or returning to that same height does not newly cross its support.
    # Never exempt the hand, another fixture, or a downward displacement.
    if support_verified and carried_index is not None and lift is not None and lift>=0:
        for index in support_indices: hits[carried_index,index]=False
    return np.argwhere(hits)


def check_live_route(stage, manifest, reservation, asset, *, source_support_verified=False,
                     robot_link_poses=None):
    """Recheck all six frozen segments against current native-readback geometry.

    Build-time identity/shape checks run separately before physics. Here bodies
    may have settled: measure their real vertices instead of treating a tiny
    settling displacement as a changed asset, or moving them back to the plan.
    """
    from itertools import product
    from pxr import Usd, UsdGeom, UsdPhysics
    from .shelf_geometry import collider_component_boxes
    require_rear_entry(reservation)
    frame = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(reservation['shelf_path']))
    if not np.allclose(np.asarray(frame), reservation['shelf_local_to_world_row_matrix'], atol=1e-7, rtol=0):
        raise ValueError('Shelf frame changed after native attachment')
    inverse = frame.GetInverse()
    cache = UsdGeom.XformCache()
    from .placement_geometry import stage_bounds
    shape_cache = {}
    def bounds(prim):
        return stage_bounds(prim,cache,inverse,robot_link_poses=robot_link_poses,shape_cache=shape_cache)
    robot = [bounds(p) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
             if robot_obstacle_shape(p)]
    stock = [bounds(stage.GetPrimAtPath(o['path'])) for o in manifest['objects']
             if o['path'] not in (reservation['target_wrapper'], reservation['shelf_path'])
             and (o.get('source') == 'snack_shelf' or o.get('instance_id','').startswith('shelf/'))]
    fixture_rows = [(str(p.GetPath()),box) for p in Usd.PrimRange(stage.GetPrimAtPath(reservation['shelf_path']))
                if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
                for box in collider_component_boxes(p, inverse, bounds(p))]
    fixtures=[box for _,box in fixture_rows]
    support_indices=[len(robot)+len(stock)+i for i,(path,_) in enumerate(fixture_rows)
                     if path==reservation['shelf_level_path']]
    if not source_support_verified or not support_indices:
        raise ValueError('Verified native contact with the exact original board required')
    if not robot: raise ValueError('Missing native robot geometry')
    target = bounds(stage.GetPrimAtPath(reservation['target_path']))
    motion = RelocationMotion(*reservation['waypoints'])
    orient = np.asarray(reservation['hand_orientation'])
    obstacles = np.asarray([*robot,*stock,*fixtures])
    groups = list(asset.collision_groups().values())
    for t in np.linspace(0,3,61):
        sample = motion.sample(float(t))
        points = asset.sample(sample.closure)[0] @ orient.T + sample.wrist_position
        boxes = np.array([[points[g].min(0),points[g].max(0)] for g in groups])
        if sample.phase in ('carry','release'):
            carried = target + np.asarray(sample.wrist_position)-motion.points[1]
            boxes = np.concatenate((boxes,carried[None]))
        hits=route_hits(boxes,obstacles,
                        carried_index=len(groups) if len(boxes)>len(groups) else None,
                        support_indices=support_indices,lift=sample.wrist_position[2]-motion.points[1][2],
                        support_verified=source_support_verified)
        if len(hits):
            bi,oi=map(int,hits[0])
            role='robot' if oi<len(robot) else ('stock' if oi<len(robot)+len(stock) else 'fixture')
            depth=np.minimum(boxes[bi,1],obstacles[oi,1])-np.maximum(boxes[bi,0],obstacles[oi,0])
            raise ValueError('Frozen route blocked by native geometry at '+str(t)+' s; no replanning; '+
                             str(dict(body='carried_target' if bi==len(groups) else list(asset.collision_groups())[bi],
                                      obstacle_role=role,obstacle_index=oi,overlap_m=depth.tolist(),
                                      body_bounds=boxes[bi].tolist(),obstacle_bounds=obstacles[oi].tolist())))
    return dict(samples=61,robot_bounds=len(robot),stock_bounds=len(stock),fixture_bounds=len(fixtures),
                runtime_replanning=False,settled_target_bounds=target.tolist(),passed=True,
                robot_pose_source='native_tensor' if robot_link_poses is not None else 'usd_readback')

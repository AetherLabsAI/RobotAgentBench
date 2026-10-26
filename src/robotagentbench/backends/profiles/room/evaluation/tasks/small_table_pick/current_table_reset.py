"""New wallet reset on current geometry, not reconstruction of the old replay.

Only root poses are authored. The existing small-table robot template supplies
the starting posture; no source wallet position or historical actions are used.
"""
from .definitions import TASK, SUPPORT, ROBOT


def polygon_area(polygon):
    return abs(sum(a[0]*b[1]-a[1]*b[0]
                   for a, b in zip(polygon, polygon[1:]+polygon[:1]))) / 2


def frontal_station(polygon, old_xy):
    """Face the real tabletop long edge; keep its existing normal stand-off.

    Center the base along the tabletop width, not along the root's rotated
    asset axes. This is a fixed task initial pose, not a randomization range.
    """
    import math
    if len(polygon) < 3 or not all(math.isfinite(x) for p in polygon for x in p):
        raise ValueError('Invalid support polygon')
    a, b = max(zip(polygon, polygon[1:]+polygon[:1]),
               key=lambda ab: math.dist(*ab))
    length = math.dist(a, b)
    if length == 0 or not all(math.isfinite(x) for x in old_xy):
        raise ValueError('Invalid station geometry')
    u = [(b[k]-a[k])/length for k in range(2)]
    n = [-u[1], u[0]]
    dot = lambda p, v: sum(x*y for x,y in zip(p,v))
    center = [sum(p[k] for p in polygon)/len(polygon) for k in range(2)]
    if dot([old_xy[k]-center[k] for k in range(2)], n) < 0:
        n = [-x for x in n]
    front = max(dot(p, n) for p in polygon)
    old_normal = dot(old_xy, n)
    if old_normal <= front:
        raise ValueError('Reference station is not outside the front edge')
    mid_u = (min(dot(p, u) for p in polygon)+max(dot(p, u) for p in polygon))/2
    xy = [mid_u*u[k]+old_normal*n[k] for k in range(2)]
    return dict(xy_m=xy, yaw_deg=math.degrees(math.atan2(-n[1], -n[0])),
                front_normal_xy=n, long_edge_axis_xy=u,
                base_to_front_edge_m=old_normal-front,
                reference_xy_m=list(old_xy))


def clipped_area(polygon, lo, hi):
    """Area of a top face inside the wallet's conservative rectangular footprint."""
    points = polygon
    for axis, boundary, sign in ((0,lo[0],1),(0,hi[0],-1),(1,lo[1],1),(1,hi[1],-1)):
        result = []
        for a, b in zip(points, points[1:]+points[:1]):
            da, db = sign*(a[axis]-boundary), sign*(b[axis]-boundary)
            if da >= 0:
                result.append(a)
            if (da >= 0) != (db >= 0):
                t = da/(da-db)
                result.append([a[k]+t*(b[k]-a[k]) for k in range(2)])
        points = result
    return polygon_area(points)


def near_halfway(polygon, half_size, robot_xy):
    """Halfway from surface center to footprint-safe boundary toward robot.

    This defines a single initial placement, not a randomization range. The
    complete axis-aligned target footprint must fit strictly inside the top.
    """
    import math
    center = [sum(v[k] for v in polygon)/len(polygon) for k in range(2)]
    direction = [robot_xy[k]-center[k] for k in range(2)]
    if not all(math.isfinite(v) for v in (*center, *half_size, *robot_xy)):
        raise ValueError('Nonfinite placement geometry')
    if any(v <= 0 for v in half_size) or math.hypot(*direction) == 0:
        raise ValueError('Invalid footprint or robot coincides with surface center')
    limit = math.inf
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        ex, ey = b[0]-a[0], b[1]-a[1]
        slack = ex*(center[1]-a[1])-ey*(center[0]-a[0])
        slack -= abs(ey)*half_size[0]+abs(ex)*half_size[1]
        if slack <= 0:
            raise ValueError('Target cannot fit at surface center')
        rate = ex*direction[1]-ey*direction[0]
        if rate < 0:
            limit = min(limit, slack / -rate)
    if not math.isfinite(limit):
        raise ValueError('Invalid convex support polygon')
    return [center[k]+0.5*limit*direction[k] for k in range(2)]


def flat_surface(stage):
    """Measured planar top faces, not the authored extent or an assumed box."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from evaluation.tasks.tabletop_pick.placement import mesh_points, region_hull
    meshes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(SUPPORT)):
        if (prim.IsA(UsdGeom.Mesh)
                and UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible'
                and UsdGeom.Imageable(prim).ComputePurpose() in ('default', 'render')):
            meshes.append((prim, mesh_points(stage, prim)))
    if not meshes:
        raise ValueError('No visible small-table geometry')
    z = max(float(pts[:, 2].max()) for _, pts in meshes)
    faces, polygons, area = [], [], 0.
    for prim, points in meshes:
        mesh = UsdGeom.Mesh(prim)
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        offset = 0
        for i, count in enumerate(mesh.GetFaceVertexCountsAttr().Get()):
            vertices = points[list(indices[offset:offset+count])]
            offset += count
            if i in holes or count < 3 or not np.all(np.abs(vertices[:, 2]-z) < 1e-6):
                continue
            xy = vertices[:, :2].tolist()
            area += polygon_area(xy)
            faces.extend(xy)
            polygons.append(xy)
    if not faces:
        raise ValueError('No planar top faces')
    hull = region_hull(faces).tolist()
    return dict(z_m=z, polygon_xy_m=hull, area_m2=area, top_faces_xy_m=polygons)


def wallet_poses(stage, target_path):
    import numpy as np
    import json
    from pxr import Gf, UsdGeom, UsdPhysics, Usd
    from evaluation.tasks.tabletop_pick.initial_state import robot_plan
    from evaluation.tasks.tabletop_pick.placement import visible_bounds, mesh_points
    from .fixed_station import robot_world, definition as station_definition
    core = TASK / 'references' / 'wallet'
    station = json.loads((TASK / 'references' / 'sidetable_station.json').read_text())
    table = UsdGeom.Xformable(stage.GetPrimAtPath(SUPPORT)).ComputeLocalToWorldTransform(0)
    current_robot = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)).ComputeLocalToWorldTransform(0)
    robot = robot_world(table, current_robot)
    with np.load(core / 'full_state.npz', allow_pickle=False) as data:
        if str(data['task_id'].item()) != '60':
            raise ValueError('Wrong robot posture template')
        if not np.array_equal(data['robot_initial_joint_position_rad_or_m'],
                              data['robot_joint_position_rad_or_m'][0]):
            raise ValueError('Source first sample does not represent its explicit robot initial state')
        plan = robot_plan(stage, data)
    with np.load(core / 'pick_contact_trace.npz', allow_pickle=False) as contacts:
        if np.any(contacts['pad_contacts'][0]):
            raise ValueError('Preparation posture must not be an already-grasped source state')
    target = stage.GetPrimAtPath(target_path)
    bounds = visible_bounds(stage, target_path)
    collision_points = []
    for prim in Usd.PrimRange(target):
        if (prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim.GetAttribute('physics:collisionEnabled').Get() is not False):
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError('Non-mesh target collision requires explicit geometry inspection')
            collision_points.extend(mesh_points(stage, prim))
    if not collision_points:
        raise ValueError('Missing wallet collider points')
    collision_points = np.asarray(collision_points)
    # Fit both visible geometry and source collider geometry (not cooked hulls).
    lo = np.minimum(bounds[0], collision_points.min(axis=0))
    hi = np.maximum(bounds[1], collision_points.max(axis=0))
    surface = flat_surface(stage)
    # Keep the previously reviewed wallet placement; only the robot changes.
    anchor = Gf.Matrix4d(station['anchor_table_world_matrix'])
    delta = anchor.GetInverse() * table.RemoveScaleShear()
    xy = station['wallet_reference_center_xy_m']
    center = list(delta.Transform(Gf.Vec3d(xy[0], xy[1], surface['z_m'])))[:2]
    half = (hi[:2]-lo[:2])/2
    footprint_lo, footprint_hi = np.asarray(center)-half, np.asarray(center)+half
    covered = sum(clipped_area(face, footprint_lo, footprint_hi)
                  for face in surface.pop('top_faces_xy_m'))
    required = float(np.prod(hi[:2]-lo[:2]))
    if not np.isclose(covered, required, rtol=1e-5, atol=1e-8):
        raise ValueError('Wallet footprint is not fully covered by actual flat top faces')
    surface['footprint_area_m2'] = required
    surface['covered_footprint_area_m2'] = covered
    shift = np.r_[np.asarray(center)-(lo[:2]+hi[:2])/2, surface['z_m']-lo[2]]
    current = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(0)
    wallet = current * Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*shift))
    report = dict(mode='current_table_geometry_not_old_wallet_replay', surface=surface,
                  target_center_xy_m=center, translation_m=shift.tolist(),
                  visible_bounds_m=(bounds+shift).tolist(),
                  collider_source_bounds_m=(np.array([collision_points.min(0), collision_points.max(0)])+shift).tolist(),
                  placement='preserved_wallet_current_table_v1_relative_to_table',
                  extra_spawn_clearance_m=0., robot_template='references/wallet:explicit_robot_initial_state',
                  robot_station=station_definition()['id'],
                  robot_station_geometry=station_definition(),
                  uses_old_wallet_pose=False, uses_old_wallet_actions=False,
                  cooked_collision_and_reachability_validated=False)
    return {ROBOT: robot, target_path: wallet}, plan, report

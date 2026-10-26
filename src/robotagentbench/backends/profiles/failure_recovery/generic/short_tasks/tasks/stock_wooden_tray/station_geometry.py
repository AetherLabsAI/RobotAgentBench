"""Read-only FK clearance for source station proposals, with no scene edits.

Uses the existing drawer FK and the same per-collider floor/triangle checks as
the accepted trash initializer. It does not change joint limits or clearances.
"""
import numpy as np
from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners, fk

ROBOT = '/World/Robot'


def geometry(stage, manifest):
    from pxr import Usd, UsdGeom, UsdPhysics
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    floors = floor_paths(stage)
    polygons = manifest.get('overview', {}).get('floors', [])
    if not floors or not polygons:
        raise ValueError('Measured floor geometry required')
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'], False, True)
    robot_parts, obstacles = [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if (not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.GetAttribute('physics:collisionEnabled').Get() is False):
            continue
        path = str(prim.GetPath())
        extent = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        box = np.array([extent.GetMin(), extent.GetMax()])
        if not np.isfinite(box).all() or np.any(box[0] > box[1]):
            raise ValueError('Invalid collider bounds: ' + path)
        if path.startswith(ROBOT + '/'):
            body = prim
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.IsPseudoRoot():
                raise ValueError('Collider without robot body: ' + path)
            # Match the existing fridge station checker: start with body-local
            # bounds, then apply FK once. Inverse-transforming a world AABB
            # inflates a rotated link and can falsely reject a valid station.
            relative = bbox.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            local_box = np.array([relative.GetMin(), relative.GetMax()])
            if not np.isfinite(local_box).all() or np.any(local_box[0] > local_box[1]):
                raise ValueError('Invalid body-local collider bounds: ' + path)
            local = np.c_[corners(local_box), np.ones(8)]
            robot_parts.append((path, str(body.GetPath()), local))
            continue
        if any(path == floor or path.startswith(floor + '/') for floor in floors):
            continue
        triangles = None
        if prim.IsA(UsdGeom.Mesh):
            vertices = prim.GetAttribute('points').Get()
            indices = list(prim.GetAttribute('faceVertexIndices').Get() or [])
            faces, offset = [], 0
            for count in prim.GetAttribute('faceVertexCounts').Get() or []:
                face = indices[offset:offset + count]
                offset += count
                faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
            if vertices is not None and len(vertices) and faces:
                vertices = (np.c_[np.array(vertices), np.ones(len(vertices))]
                            @ np.array(cache.GetLocalToWorldTransform(prim)))[:, :3]
                triangles = vertices[np.array(faces)]
        obstacles.append((path, box, triangles))
    if not robot_parts:
        raise ValueError('No robot collider geometry')
    return robot_parts, obstacles, polygons


def rejection(stage, scene_geometry, targets, position_xy, yaw_rad):
    from scipy.spatial.transform import Rotation
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    robot_parts, obstacles, polygons = scene_geometry
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('z', yaw_rad).as_matrix()
    base[:3, 3] = [*position_xy, 0.]
    links = fk(stage, targets, base)  # Missing joints/limits are errors, not retries.
    for path, body, local in robot_parts:
        vertices = (local @ links[body].T)[:, :3]
        box = np.array([vertices.min(0), vertices.max(0)])
        perimeter = [(a, b) for a in np.linspace(box[0, 0], box[1, 0], 11)
                     for b in (box[0, 1], box[1, 1])]
        perimeter += [(a, b) for b in np.linspace(box[0, 1], box[1, 1], 11)
                      for a in (box[0, 0], box[1, 0])]
        if not all(any(inside(v, polygon) for polygon in polygons) for v in perimeter):
            return 'robot_outside_floor:' + path
        hit = next((p for p, b, triangles in obstacles if intersects(box, b)
                    and (triangles is None or triangles_hit_box(triangles, box))), None)
        if hit:
            return 'robot_room_overlap:' + path + ':' + hit
    return None

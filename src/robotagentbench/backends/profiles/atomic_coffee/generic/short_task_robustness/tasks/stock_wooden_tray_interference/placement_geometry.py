"""Read-only geometry shared by shelf packing, preparation and native admission.

Primitive tessellation and enabled analytic-collider envelopes match the
existing shelf sampler's shape_mesh/_proxies rules. Transform actual vertices,
never the corners of an already axis-aligned primitive extent.
"""
import numpy as np


def point_bounds(points, row_transform):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite shape vertices required')
    transform = np.asarray(row_transform, dtype=float)
    moved = points @ transform[:3, :3] + transform[3, :3]
    return np.array([moved.min(0), moved.max(0)])


def body_bounds(bodies):
    boxes = [point_bounds(b.v, np.asarray(b.transform).T) for b in bodies]
    if not boxes:
        raise ValueError('Missing sampled object geometry')
    return np.array([np.min([b[0] for b in boxes], axis=0),
                     np.max([b[1] for b in boxes], axis=0)])


def enabled_collider(prim):
    from pxr import UsdPhysics
    return (prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetAttribute('physics:collisionEnabled').Get() is not False)


def included_shape(prim):
    from pxr import UsdGeom
    if not prim.IsA(UsdGeom.Boundable):
        return False
    if enabled_collider(prim):
        return True
    image = UsdGeom.Imageable(prim)
    path = str(prim.GetPath()).lower()
    return (image.ComputeVisibility() != 'invisible'
            and image.ComputePurpose() not in ('proxy', 'guide')
            and '/collisions/' not in path
            and 'collision' not in prim.GetName().lower()
            and 'proxy' not in prim.GetName().lower())


def shape_points(prim):
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Mesh):
        return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    import trimesh
    if prim.IsA(UsdGeom.Cube):
        mesh = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
    elif prim.IsA(UsdGeom.Sphere):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
    elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
        cylinder = prim.IsA(UsdGeom.Cylinder)
        g = UsdGeom.Cylinder(prim) if cylinder else UsdGeom.Capsule(prim)
        args = dict(radius=float(g.GetRadiusAttr().Get()), height=float(g.GetHeightAttr().Get()))
        mesh = (trimesh.creation.cylinder(**args, sections=48) if cylinder
                else trimesh.creation.capsule(**args, count=[16, 32]))
        axis = str(g.GetAxisAttr().Get())
        if axis == 'X':
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        if axis == 'Y':
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    else:
        raise ValueError('Unsupported shape geometry: ' + str(prim.GetPath()))
    if enabled_collider(prim) and not prim.IsA(UsdGeom.Cube):
        # Reuse the sampler's circumscribed envelope, not an inscribed mesh.
        mesh = mesh.convex_hull
        normals = mesh.face_normals
        distances = np.einsum('ij,ij->i', normals, mesh.triangles[:, 0])
        if prim.IsA(UsdGeom.Sphere):
            support = np.full(len(normals), float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        else:
            axis_index = 'XYZ'.index(str(g.GetAxisAttr().Get()))
            axial = np.abs(normals[:, axis_index])
            radial = np.sqrt(np.maximum(0., 1. - axial ** 2))
            support = args['radius'] * (radial if cylinder else np.ones(len(normals))) + args['height'] / 2 * axial
        mesh.vertices *= max(1., float(np.max(support / distances)))
    return np.asarray(mesh.vertices, dtype=float)


def stage_bounds(prim, cache, inverse_frame, *, robot_link_poses=None, shape_cache=None):
    from pxr import Usd, UsdPhysics
    boxes = []
    shape_cache = {} if shape_cache is None else shape_cache
    for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not included_shape(shape):
            continue
        path = str(shape.GetPath())
        if path not in shape_cache:
            shape_cache[path] = shape_points(shape)
        if robot_link_poses is not None and path.startswith('/World/Robot/'):
            body = shape
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or str(body.GetPath()) not in robot_link_poses:
                raise ValueError('Robot shape lacks a current native body pose: ' + path)
            local = np.asarray(cache.GetLocalToWorldTransform(shape) * cache.GetLocalToWorldTransform(body).GetInverse())
            transform = local @ np.asarray(robot_link_poses[str(body.GetPath())]).T @ np.asarray(inverse_frame)
        else:
            transform = np.asarray(cache.GetLocalToWorldTransform(shape) * inverse_frame)
        boxes.append(point_bounds(shape_cache[path], transform))
    if not boxes:
        raise ValueError('Missing geometry: ' + str(prim.GetPath()))
    return np.array([np.min([b[0] for b in boxes], axis=0),
                     np.max([b[1] for b in boxes], axis=0)])

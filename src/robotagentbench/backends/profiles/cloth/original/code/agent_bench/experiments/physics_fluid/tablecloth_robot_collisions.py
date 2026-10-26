"""Export non-hand USD contacts as native primitives with convex mesh fallback.

Curved CAD primitives use 64 azimuth and 32 polar segments. Their inscribed
convex hulls approximate the authored surface; transforms include USD scale.
The export contains geometry, not task semantics or a second source registry.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def convex_contact_mesh(points):
    """Return a compact, closed, outward-facing hull of the CAD vertices.

    Newton's add_shape_convex_hull labels a supplied mesh; it does not rebuild
    its triangles. Soft contacts still query those triangles for inside/outside,
    so passing a CAD triangle soup can produce reversed contact normals.
    """
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    tri = points[faces]
    inward = np.einsum('ij,ij->i', np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                        hull.equations[:, :3]) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    used, indices = np.unique(faces, return_inverse=True)
    return points[used].astype(np.float32), indices.reshape(-1, 3).astype(np.int32)


def native_primitive(kind, spec, local):
    """Preserve CAD dimensions, axis and scale when a native shape is exact."""
    scale = np.linalg.norm(local[:3, :3], axis=0)
    rotation = local[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        return None
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1  # All supported primitives are reflection symmetric.
    result = dict(position=local[:3, 3].tolist())
    if kind == 'Cube':
        result.update(type='box', half_extents=(spec['half'] * scale).tolist())
    elif kind == 'Cylinder':
        axis = spec['axis']
        radial = [i for i in range(3) if i != axis]
        if not np.isclose(scale[radial[0]], scale[radial[1]], atol=1e-8):
            return None
        # Newton's cylinder axis is local Z.
        align = [Rotation.from_euler('y', np.pi / 2).as_matrix(),
                 Rotation.from_euler('x', -np.pi / 2).as_matrix(), np.eye(3)][axis]
        rotation = rotation @ align
        result.update(type='cylinder', radius=float(spec['radius'] * scale[radial[0]]),
                      half_height=float(spec['half'] * scale[axis]))
    elif kind == 'Sphere' and np.allclose(scale, scale[0], atol=1e-8):
        result.update(type='sphere', radius=float(spec['radius'] * scale[0]))
    else:
        return None
    result['quaternion_xyzw'] = Rotation.from_matrix(rotation).as_quat().tolist()
    return result


def export_nonhand_collisions(stage, link_names):
    from pxr import Usd, UsdGeom, UsdPhysics

    shapes = []
    angles = np.arange(64) * (2 * np.pi / 64)
    for link in stage.GetDefaultPrim().GetChildren():
        name = link.GetName()
        if name not in link_names or name.startswith('gripper_'):
            continue
        inverse = np.array(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).GetInverse()).T
        for prim in Usd.PrimRange(link):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            kind = prim.GetTypeName()
            spec = {}
            if kind == 'Cube':
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                spec = dict(half=half)
                points = np.array([[x, y, z] for x in (-half, half)
                                   for y in (-half, half) for z in (-half, half)])
            elif kind == 'Cylinder':
                cylinder = UsdGeom.Cylinder(prim)
                radius = float(cylinder.GetRadiusAttr().Get())
                half = float(cylinder.GetHeightAttr().Get()) / 2
                axis = 'XYZ'.index(cylinder.GetAxisAttr().Get())
                spec = dict(radius=radius, half=half, axis=axis)
                transverse = [i for i in range(3) if i != axis]
                points = np.zeros((128, 3))
                points[:, axis] = np.repeat([-half, half], 64)
                points[:, transverse[0]] = np.tile(radius * np.cos(angles), 2)
                points[:, transverse[1]] = np.tile(radius * np.sin(angles), 2)
            elif kind == 'Sphere':
                radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                spec = dict(radius=radius)
                points = [[0., 0., -radius], [0., 0., radius]]
                for polar in np.arange(1, 32) * np.pi / 32:
                    points.extend(np.column_stack((radius * np.sin(polar) * np.cos(angles),
                                                   radius * np.sin(polar) * np.sin(angles),
                                                   np.full(64, radius * np.cos(polar)))))
                points = np.asarray(points)
            elif kind == 'Mesh':
                points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            else:
                raise ValueError(f'Unsupported robot collider: {kind} {prim.GetPath()}')
            world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())).T
            local = inverse @ world
            primitive = native_primitive(kind, spec, local)
            points = points @ local[:3, :3].T + local[:3, 3]
            hull = ConvexHull(points)
            triangles = hull.simplices.copy()
            p = points[triangles]
            inward = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                                hull.equations[:, :3]) < 0
            triangles[inward] = triangles[inward][:, [0, 2, 1]]
            shapes.append(dict(link=name, prim_path=str(prim.GetPath()), kind=kind,
                               primitive=primitive,
                               approximation='native_USD_primitive' if primitive else 'convex_hull_of_USD_primitive_or_mesh',
                               vertices=points.tolist(), triangles=triangles.tolist()))
    if not shapes:
        raise ValueError('No non-hand robot collision geometry exported')
    return shapes


def add_robot_collision(builder, body, shape, cfg):
    """Use analytic particle SDFs for CAD primitives, including deep interiors."""
    import newton
    import warp as wp

    primitive = shape.get('primitive')
    if primitive:
        pose = wp.transform(primitive['position'], primitive['quaternion_xyzw'])
        if primitive['type'] == 'box':
            x, y, z = primitive['half_extents']
            return builder.add_shape_box(body=body, xform=pose, hx=x, hy=y, hz=z, cfg=cfg)
        if primitive['type'] == 'cylinder':
            return builder.add_shape_cylinder(body=body, xform=pose, radius=primitive['radius'],
                                              half_height=primitive['half_height'], cfg=cfg)
        if primitive['type'] == 'sphere':
            return builder.add_shape_sphere(body=body, xform=pose, radius=primitive['radius'], cfg=cfg)
        raise ValueError(f"Unsupported native primitive: {primitive['type']}")
    mesh = newton.Mesh(np.asarray(shape['vertices'], dtype=np.float32),
                       np.asarray(shape['triangles'], dtype=np.int32).reshape(-1))
    return builder.add_shape_convex_hull(body=body, mesh=mesh, cfg=cfg)

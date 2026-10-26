"""Geometry-preserving preparation of static cloth contact meshes before reset.

Integrates the delivered support60_probe's transform-bake repair. Run only after
the episode layout is resolved and before physics starts. Never save this
prepared stage back to House or move its parents afterwards.
"""
from __future__ import annotations

import numpy as np


def prepare_static_contact_mesh(stage, path='/World/Assets/CoffeeTable/Mesh'):
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError('Missing static contact mesh: ' + path)
    if any(p != prim and UsdGeom.Xformable(p) for p in Usd.PrimRange(prim)):
        raise ValueError('Cannot bake a contact mesh with descendants')
    current = prim
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Cannot bake a mesh in a rigid body')
        for op in UsdGeom.Xformable(current).GetOrderedXformOps():
            if op.GetAttr().GetNumTimeSamples():
                raise ValueError('Cannot bake animated contact geometry')
        current = current.GetParent()
    mesh = UsdGeom.Mesh(prim)
    if any(a.GetNumTimeSamples() for a in (mesh.GetPointsAttr(), mesh.GetNormalsAttr(),
                                          mesh.GetFaceVertexIndicesAttr(), mesh.GetFaceVertexCountsAttr())):
        raise ValueError('Cannot bake time-varying mesh geometry')
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid contact mesh vertices')
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError('Contact transform must be finite and orientation preserving')
    world = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
    baked = world.astype(np.float32)
    error = float(np.abs(baked.astype(np.float64) - world).max())
    if error >= 3e-7:
        raise ValueError('Contact bake exceeds existing world-vertex tolerance')
    normals = mesh.GetNormalsAttr().Get()
    transformed_normals = None
    if normals is not None and len(normals):
        transformed_normals = np.asarray(normals) @ np.linalg.inv(matrix[:3, :3]).T
        lengths = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
        if not np.isfinite(lengths).all() or np.any(lengths == 0):
            raise ValueError('Invalid contact mesh normals')
        transformed_normals = (transformed_normals / lengths).astype(np.float32)
    # Same triangles at the same world positions; no convexification or support.
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(baked))
    mesh.CreateExtentAttr().Set(UsdGeom.Mesh.ComputeExtent(Vt.Vec3fArray.FromNumpy(baked)))
    if transformed_normals is not None:
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(transformed_normals))
    return dict(path=path, method='geometry_preserving_world_transform_bake',
                vertices=len(points), source_polygons=len(mesh.GetFaceVertexCountsAttr().Get()),
                world_vertex_error_max_m=error, source_transform=matrix.tolist(),
                source_triangles_preserved=True, collision_approximation_unchanged=True,
                added_support_bodies=0, applied_before_reset=True,
                layout_must_be_resolved_before_preparation=True)

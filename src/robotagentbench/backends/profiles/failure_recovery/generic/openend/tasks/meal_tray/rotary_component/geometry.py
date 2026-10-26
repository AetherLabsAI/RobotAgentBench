"""Source load_clearance.collect_body_points, scoped to the approved mechanism.

Uses visible OR collision geometry, including instance proxies, as the original
rotary native probe did. A finger without a collider is not a missing body.
"""
import itertools
import numpy as np


def collect_body_points(stage,body_path):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache();body=stage.GetPrimAtPath(body_path);assert body
    matrix=cache.GetLocalToWorldTransform(body)
    rigid=Gf.Transform(matrix)
    origin=np.array(rigid.GetTranslation())
    rotation=np.array(Gf.Matrix3d(rigid.GetRotation())).T
    rows=[];paths=[]
    for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
        visible=UsdGeom.Imageable(prim).ComputeVisibility()!='invisible' if prim.IsA(UsdGeom.Imageable) else False
        collision=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        if not (visible or collision):continue
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        elif prim.IsA(UsdGeom.Boundable):
            extent=UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim),0)
            if extent is None or len(extent)!=2:continue
            points=np.array(list(itertools.product(*zip(extent[0],extent[1]))))
        else:continue
        if not len(points):continue
        transform=np.array(cache.GetLocalToWorldTransform(prim))
        world=(np.column_stack([points,np.ones(len(points))])@transform)[:,:3]
        rows.append((world-origin)@rotation);paths.append(str(prim.GetPath()))
    assert rows,body_path
    points=np.concatenate(rows)
    try:
        from scipy.spatial import ConvexHull
        points=points[ConvexHull(points).vertices]
    except ImportError:pass
    return points,paths

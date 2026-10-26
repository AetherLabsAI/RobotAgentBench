"""Read-only bounds of disconnected static triangle-mesh components.

An authored shelf mesh may contain all four separated boards. Its root AABB
is NOT a solid collider. Only split native triangle meshes (approximation
none); convex collision approximations retain their full conservative box.
"""
import numpy as np


def triangle_component_bounds(points, faces):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points=np.asarray(points,dtype=float);faces=np.asarray(faces,dtype=int)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Invalid mesh points')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces):raise ValueError('Triangle mesh required')
    if faces.min()<0 or faces.max()>=len(points):raise ValueError('Mesh indices out of bounds')
    # Exact welding affects topology of the read-only query only. No rounding,
    # simplification, collider editing, or object transforms are authored.
    used=np.unique(faces);vertices,inverse=np.unique(points[used],axis=0,return_inverse=True)
    mapping=np.full(len(points),-1,dtype=int);mapping[used]=inverse
    triangles=mapping[faces]
    a=triangles.reshape(-1);b=triangles[:,[1,2,0]].reshape(-1)
    graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(vertices),len(vertices))).tocsr()
    count,labels=connected_components(graph,directed=False)
    lo=np.full((count,3),np.inf);hi=np.full((count,3),-np.inf)
    np.minimum.at(lo,labels,vertices);np.maximum.at(hi,labels,vertices)
    return np.stack((lo,hi),axis=1)


def collider_component_boxes(prim, inverse_frame, fallback_bounds):
    from pxr import UsdGeom
    if not prim.IsA(UsdGeom.Mesh):return [np.asarray(fallback_bounds)]
    approx=prim.GetAttribute('physics:approximation').Get()
    if approx not in (None,'none'):return [np.asarray(fallback_bounds)]
    mesh=UsdGeom.Mesh(prim);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not len(counts) or not np.all(counts==3):return [np.asarray(fallback_bounds)]
    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    tf=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)*inverse_frame)
    points=points@tf[:3,:3]+tf[3,:3]
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    return list(triangle_component_bounds(points,faces))

"""Planar occupancy and height maps of a room USD (CPU USD, numpy).

All visible render geometry except the robot is sampled. A cell is blocked for the base when any
geometry lies in the height band swept by the robot body. The same samples give a top-down height
map that is used as a schematic map in the web viewer.
"""
import numpy as np
from pxr import Usd, UsdGeom

ROBOT = '/World/Robot'


def _matrix(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float).T


def _triangles(counts, indices):
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if len(counts) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if (counts == 3).all():
        return indices.reshape(-1, 3)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for n in np.unique(counts):
        if n < 3:
            continue
        s = starts[counts == n]
        for k in range(1, n - 1):
            tris.append(np.stack([indices[s], indices[s + k], indices[s + k + 1]], axis=1))
    return np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)


def _sample(points, tris, step):
    """Points covering every triangle with spacing <= step (vertices, plus interior samples for large ones)."""
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
    out = [points[np.unique(tris)]]
    large = edge > step
    if large.any():
        a, b, c, edge = a[large], b[large], c[large], edge[large]
        divisions = np.minimum(np.ceil(edge / step).astype(int), 2000)
        for n in np.unique(divisions):
            sel = divisions == n
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
            keep = i + j <= n
            u, v = i[keep] / n, j[keep] / n
            w = 1 - u - v
            pts = (a[sel][:, None, :] * w[None, :, None] + b[sel][:, None, :] * u[None, :, None]
                   + c[sel][:, None, :] * v[None, :, None])
            out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def sample_scene(stage, step=0.02, z_max=2.2, skip_prefixes=(ROBOT,)):
    """World-space surface samples (N, 3) of all visible geometry below z_max."""
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(0, ['default', 'render'])
    chunks = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        path = str(prim.GetPath())
        if any(path.startswith(p) for p in skip_prefixes) or not prim.IsA(UsdGeom.Gprim):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if image.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts:
                continue
            world = _matrix(cache.GetLocalToWorldTransform(prim))
            points = np.asarray(pts, dtype=float) @ world[:3, :3].T + world[:3, 3]
            tris = _triangles(counts, indices)
            tris = tris[(tris < len(points)).all(axis=1)]
            if len(tris) == 0:
                continue
            z = points[tris][:, :, 2]
            tris = tris[z.min(axis=1) < z_max]
            if len(tris):
                chunks.append(_sample(points, tris, step))
        else:  # analytic shapes: sample the faces of the world-aligned bound
            box = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
            if lo[2] >= z_max or not np.isfinite(lo).all() or not np.isfinite(hi).all():
                continue
            n = np.maximum(np.ceil((hi - lo) / step).astype(int), 1)
            n = np.minimum(n, 600)
            gx, gy = np.linspace(lo[0], hi[0], n[0] + 1), np.linspace(lo[1], hi[1], n[1] + 1)
            x, y = np.meshgrid(gx, gy)
            for z in (lo[2], hi[2]):
                chunks.append(np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1))
    points = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    return points[points[:, 2] < z_max]


def build_maps(points, resolution=0.04, body_band=(0.04, 1.30), margin=0.3):
    """Rasterize samples into a base-blocking mask and a top-down height map (row 0 = y_min)."""
    lo = points[:, :2].min(axis=0) - margin
    hi = points[:, :2].max(axis=0) + margin
    shape = np.ceil((hi - lo) / resolution).astype(int)[::-1]  # (rows=y, cols=x)
    col = np.clip(((points[:, 0] - lo[0]) / resolution).astype(int), 0, shape[1] - 1)
    row = np.clip(((points[:, 1] - lo[1]) / resolution).astype(int), 0, shape[0] - 1)
    flat = row * shape[1] + col
    height = np.full(shape[0] * shape[1], -1.0, dtype=np.float32)
    np.maximum.at(height, flat, points[:, 2].astype(np.float32))
    in_band = (points[:, 2] > body_band[0]) & (points[:, 2] < body_band[1])
    blocked = np.zeros(shape[0] * shape[1], dtype=bool)
    blocked[flat[in_band]] = True
    floor = np.zeros(shape[0] * shape[1], dtype=bool)
    floor[flat[points[:, 2] <= body_band[0]]] = True
    return dict(origin=lo.tolist(), resolution=resolution, shape=[int(shape[0]), int(shape[1])],
                blocked=blocked.reshape(shape), height=height.reshape(shape), floor=floor.reshape(shape))

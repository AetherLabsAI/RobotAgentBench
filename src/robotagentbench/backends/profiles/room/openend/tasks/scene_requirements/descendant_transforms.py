"""Check composed descendants; never infer furniture placement from its root alone."""
import numpy as np
from pxr import Usd, UsdGeom, Gf


def geometry(stage, path, exclude=()):
    root = stage.GetPrimAtPath(path)
    if not root:
        raise ValueError('Missing furniture: ' + path)
    cache = UsdGeom.XformCache()
    result = {}
    for prim in Usd.PrimRange(root):
        relative = str(prim.GetPath())[len(path):]
        if any(s in relative for s in exclude) or not prim.IsA(UsdGeom.Mesh):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == 'invisible' or image.ComputePurpose() in ('proxy', 'guide'):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if points.size:
            result[relative] = (np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
    return result


def rigid_delta(source, target, path, exclude=()):
    aa, bb = geometry(source, path, exclude), geometry(target, path, exclude)
    if not aa or aa.keys() != bb.keys() or any(aa[k].shape != bb[k].shape for k in aa):
        raise ValueError('Furniture geometry differs from approved sampler: ' + path)
    a, b = [np.concatenate([values[k] for k in sorted(values)]) for values in (aa, bb)]
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    rot = u @ vt
    translation = bc-ac @ rot
    error = float(np.linalg.norm(a @ rot+translation-b, axis=1).max())
    if np.linalg.det(rot) < 0 or not np.allclose(rot[2], [0,0,1], atol=1e-7) or error > 1e-5:
        raise ValueError('Furniture transform must remain rigid and upright: ' + path)
    matrix = np.eye(4); matrix[:3,:3] = rot; matrix[3,:3] = translation
    return Gf.Matrix4d(*matrix.reshape(-1).tolist()), dict(path=path, meshes=len(aa), points=len(a), max_residual_m=error, matrix=matrix.tolist(), method='all_visible_mesh_vertices_composed_descendant_world_transforms')


def audit(stage):
    cache = UsdGeom.XformCache()
    authored = []; checked = 0
    for prim in stage.Traverse():
        xf = UsdGeom.Xformable(prim)
        if not xf:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite descendant transform: ' + str(prim.GetPath()))
        checked += 1
        ops = xf.GetOrderedXformOps()
        if ops or xf.GetResetXformStack():
            authored.append(dict(path=str(prim.GetPath()), ops=[str(o.GetOpName()) for o in ops], reset_stack=xf.GetResetXformStack()))
    return dict(checked_xformable_nodes=checked, authored_nodes=authored, finite=True, includes_descendants=True,
                limitations='Transform audit and rigid relocation evidence only; not a collision or physical settling certificate')

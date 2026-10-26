"""User-approved, temporary source cloth geometry control. Not a new default.

Runs before physics initialization. Copies geometry only, relative to the same
CoffeeTable frame as the original pose transfer. No material or body edits.
"""
import hashlib
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent
SOURCE = '/World/Assets/Task69Items/DirtyTablecloth'
DEST = '/World/Assets/TableclothRuleState/Surface'
TABLE = '/World/Assets/CoffeeTable'


def triangles(counts, indices):
    result, offset = [], 0
    for count in counts:
        if count < 3:
            raise ValueError('Degenerate source face')
        face = indices[offset:offset + count]
        result.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        offset += count
    if offset != len(indices):
        raise ValueError('Face topology mismatch')
    return result


def apply(stage):
    from pxr import Gf, Usd, UsdGeom
    if not stage.GetEditTarget().GetLayer().anonymous:
        raise RuntimeError('Only the anonymous test layer may be changed')
    root = stage.GetPrimAtPath('/World/Assets/TableclothRuleState')
    if not root or root.HasAPI('OmniPhysicsDeformableBodyAPI'):
        raise RuntimeError('Requires the existing static test cloth, before physics')
    path = TASK/'source_16_seed1000000/initial_scene.usdc'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected = json.loads((TASK/'artifacts/initial_v1/initial.json').read_text())['source_sha256']['initial_scene.usdc']
    if digest != expected:
        raise ValueError('Source snapshot changed')
    source = Usd.Stage.Open(str(path))
    src = UsdGeom.Mesh(source.GetPrimAtPath(SOURCE))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(DEST))
    if not src or not mesh:
        raise ValueError('Missing source/destination cloth')
    cache = UsdGeom.XformCache()
    old_table = cache.GetLocalToWorldTransform(source.GetPrimAtPath(TABLE))
    new_table = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(TABLE))
    delta = old_table.RemoveScaleShear().GetInverse() * new_table.RemoveScaleShear()
    source_world = cache.GetLocalToWorldTransform(src.GetPrim())
    dest_world = cache.GetLocalToWorldTransform(mesh.GetPrim())
    transform = source_world * delta * dest_world.GetInverse()
    points = [Gf.Vec3f(transform.Transform(Gf.Vec3d(p))) for p in src.GetPointsAttr().Get()]
    counts, indices = list(src.GetFaceVertexCountsAttr().Get()), list(src.GetFaceVertexIndicesAttr().Get())
    if counts != [4] or indices != [0, 1, 2, 3]:
        raise ValueError('Unexpected pinned source topology')
    tris = triangles(counts, indices)
    before = dict(vertices=len(mesh.GetPointsAttr().Get()), faces=len(mesh.GetFaceVertexCountsAttr().Get()))
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetOrientationAttr().Set(src.GetOrientationAttr().Get())
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    # Keep dormant rest/bind-pose data consistent; this probe never activates it.
    for name, value in {
        'omniphysics:restShapePoints': points,
        'omniphysics:restTriVtxIndices': [Gf.Vec3i(*t) for t in tris],
        'deformablePose:default:omniphysics:points': points,
    }.items():
        attr = mesh.GetPrim().GetAttribute(name)
        if attr:
            attr.Set(value)
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    if uv:
        uv.Set([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    actual = [dest_world.Transform(Gf.Vec3d(p)) for p in mesh.GetPointsAttr().Get()]
    expected_points = [(source_world * delta).Transform(Gf.Vec3d(p)) for p in src.GetPointsAttr().Get()]
    error = max((a-b).GetLength() for a, b in zip(actual, expected_points))
    if error > 1e-7:
        raise RuntimeError('Source cloth geometry readback mismatch')
    return dict(scope=DEST, source_sha256=digest, before=before,
                vertices=4, faces=1, diagnostic_triangles=2,
                world_points_m=[list(p) for p in actual], max_vertex_error_m=error,
                material_changed=False, robot_changed=False, shared_asset_saved=False,
                dynamic_transition_attempted=False)

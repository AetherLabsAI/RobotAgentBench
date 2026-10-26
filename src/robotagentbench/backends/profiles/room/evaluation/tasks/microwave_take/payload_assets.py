"""Read-only checks of the user-confirmed native MicrowaveFoodbox asset.

No collision generation, mass edits, scaling edits or fallbacks to beef assets.
Resolve the pinned native source from the composed prim's actual layer stack.
"""
import hashlib
from pathlib import Path

from . import initial_state as m


def verify_leftover_asset(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(m.LEFTOVER_PAYLOAD)
    geo = stage.GetPrimAtPath(m.LEFTOVER_PAYLOAD + '/geo')
    if not root or not root.IsActive() or not geo:
        raise ValueError('Missing active native MicrowaveFoodbox')
    paths = {Path(s.layer.realPath) for prim in (root, geo) for s in prim.GetPrimStack()
             if s.layer.realPath}
    native = next((p for p in paths if p.name == 'foodbox.usda'), None)
    physics = next((p for p in paths if p.name == 'foodbox_task108_physics_v2.usda'), None)
    if native is None or physics is None:
        raise ValueError('Foodbox is not the native L1 asset')
    files = dict(payload_native_mesh=native, payload_native_physics=physics,
                 payload_native_texture=native.parent / 'textures/box.jpg')
    for role, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != m.LEFTOVER_ASSET_SHA256[role]:
            raise ValueError('Native foodbox source changed: ' + role)
    expected = Usd.Stage.Open(str(physics))
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    scale = Gf.Transform(cache.GetLocalToWorldTransform(root)).GetScale()
    if not Gf.IsClose(scale, Gf.Vec3d(0.94), 1e-6):
        raise ValueError('Native foodbox scene scale changed')
    for suffix in ('', '/geo', '/col_box'):
        actual = stage.GetPrimAtPath(m.LEFTOVER_PAYLOAD + suffix)
        source = expected.GetPrimAtPath('/Foodbox' + suffix)
        if not actual or actual.GetTypeName() != source.GetTypeName():
            raise ValueError('Native foodbox prim changed: ' + suffix)
        for attr in source.GetAttributes():
            name = attr.GetName()
            # Dynamic state is measured separately; movement is not an asset edit.
            if name in ('physics:velocity', 'physics:angularVelocity'):
                continue
            if name.startswith(('physics:', 'physxCollision:', 'physxConvexDecomposition:')) or name in (
                    'points', 'faceVertexCounts', 'faceVertexIndices', 'normals', 'primvars:st'):
                if actual.GetAttribute(name).Get() != attr.Get():
                    raise ValueError('Composed native foodbox attribute changed: ' + name)
        if suffix:
            tf = cache.GetLocalToWorldTransform(actual) * inverse
            original = cache.GetLocalToWorldTransform(source)
            if not Gf.IsClose(tf, original, 1e-8):
                raise ValueError('Native foodbox local geometry moved')
    enabled = [str(p.GetPath()) for p in Usd.PrimRange(root)
               if p.HasAPI(UsdPhysics.CollisionAPI)
               and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    if enabled != [m.LEFTOVER_PAYLOAD + '/geo']:
        raise ValueError('Unexpected native foodbox collision geometry')
    return dict(m.LEFTOVER_ASSET_SHA256)

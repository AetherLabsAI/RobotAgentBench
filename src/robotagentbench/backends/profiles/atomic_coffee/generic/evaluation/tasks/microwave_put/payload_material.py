"""Restore only the approved missing payload-to-shared-material relationship.

No material values, geometry, mass, drive, contact offsets or other instances
are authored. Shared GripMat parameters remain the public policy's job.
"""
from arena_api.finger_pad_material import SOURCE_MATERIAL

PAYLOAD = "/World/Assets/MicrowaveBeefbox"
POLICY = "microwave_put.existing_gripmat_binding.v1"


def configuration(stage, *, require_bound=True, payload_variant="beef_box"):
    from pxr import Usd, UsdPhysics, UsdShade

    from .payload_binding import payload_for
    payload = payload_for(payload_variant)
    root = stage.GetPrimAtPath(payload)
    material = stage.GetPrimAtPath(SOURCE_MATERIAL)
    if not root or not root.IsActive() or not root.IsLoaded() or root.IsInstanceProxy():
        raise ValueError("Missing editable active selected payload")
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Existing shared physical material is missing")
    bindings = {}
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if prim.IsInstanceProxy() or not prim.IsLoaded():
            raise ValueError("Cannot validate payload collider")
        bound, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        path = str(bound.GetPath()) if bound else None
        if path not in (None, SOURCE_MATERIAL):
            raise ValueError("Unexpected payload physics material: " + str(prim.GetPath()))
        if require_bound and path != SOURCE_MATERIAL:
            raise ValueError("Unbound payload collider: " + str(prim.GetPath()))
        bindings[str(prim.GetPath())] = dict(material=path,
            relationship=str(relation.GetPath()) if relation else None)
    if not bindings:
        raise ValueError("No active payload colliders")
    return dict(policy_id=POLICY, payload=payload, material=SOURCE_MATERIAL,
        active_colliders=len(bindings),
        bound_colliders=sum(v["material"] == SOURCE_MATERIAL for v in bindings.values()),
        collider_bindings=bindings,
        material_parameters={a.GetName(): str(a.Get()) for a in material.GetAttributes()
            if a.GetName().startswith(("physics:", "physxMaterial:"))})


def bind_existing(stage):
    """Fill a missing binding, fail on a conflicting material, and remain idempotent."""
    from pxr import UsdShade

    before = configuration(stage, require_bound=False)
    if before["bound_colliders"] == before["active_colliders"]:
        return dict(configuration(stage), binding_added=False, material_values_changed=False)
    layer = stage.GetEditTarget().GetLayer()
    snapshot = layer.ExportToString()
    try:
        root = stage.GetPrimAtPath(PAYLOAD)
        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            UsdShade.Material(stage.GetPrimAtPath(SOURCE_MATERIAL)),
            UsdShade.Tokens.weakerThanDescendants, "physics")
        after = configuration(stage)
        if before["material_parameters"] != after["material_parameters"]:
            raise RuntimeError("Binding changed shared material values")
        if set(before["collider_bindings"]) != set(after["collider_bindings"]):
            raise RuntimeError("Binding changed collider set")
    except Exception:
        layer.ImportFromString(snapshot)
        raise
    return dict(after, binding_added=True, material_values_changed=False)

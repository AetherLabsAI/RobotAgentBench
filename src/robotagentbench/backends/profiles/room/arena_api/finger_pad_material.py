"""One shared GripMat and pad policy, independent of task, hand, model and seed.

Apply the user-approved 2.2/2.0/0/0 to the shared scene material as well as
all four pads before physics reset. Preserve other materials and source files.
Accept the known old shared values only as a load-time migration; reject
unknown inputs and task-time overrides.
"""
import math

from .finger_pad_contact import PAIRS

POLICY = "house.shared_grip_material.v3_rigid_2p2_2p0"
SOURCE_MATERIAL = "/World/PhysicsMaterials/GripMat"
MATERIAL = "/World/PhysicsMaterials/ArenaFingerPadMat"
LEGACY = "/World/Task106GripperPadPhysics"
LEGACY_SOURCE_PARAMETERS = {
    "physics:staticFriction": 1.2,
    "physics:dynamicFriction": 1.0,
    "physics:restitution": 0.0,
    "physxMaterial:compliantContactStiffness": 8000.0,
    "physxMaterial:compliantContactDamping": 300.0,
    "physxMaterial:frictionCombineMode": "max",
}
PARAMETERS = {
    **LEGACY_SOURCE_PARAMETERS,
    "physics:staticFriction": 2.2,
    "physics:dynamicFriction": 2.0,
    "physxMaterial:compliantContactStiffness": 0.0,
    "physxMaterial:compliantContactDamping": 0.0,
}
SOURCE_PARAMETERS = dict(PARAMETERS)
SHARED_OVERRIDES = (
    "physics:staticFriction", "physics:dynamicFriction",
    "physxMaterial:compliantContactStiffness", "physxMaterial:compliantContactDamping",
)


def _material_configuration(stage, path, parameters):
    from pxr import UsdPhysics
    material = stage.GetPrimAtPath(path)
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Physics material is missing: " + path)
    schemas = material.GetMetadata("apiSchemas")
    if not schemas or "PhysxMaterialAPI" not in schemas.GetAppliedItems():
        raise ValueError("Physics material must retain its PhysX schema: " + path)
    for name, expected in parameters.items():
        value = material.GetAttribute(name).Get()
        valid = (value == expected if isinstance(expected, str) else
                 isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-7))
        if not valid:
            raise ValueError("Physics material differs from its fixed policy: " + path + "." + name)
    return dict(material_attributes={
        a.GetName(): str(a.Get()) for a in material.GetAttributes()
        if a.GetName().startswith(("physics:", "physxMaterial:"))},
        material_schemas=str(material.GetMetadata("apiSchemas")))


def _pad_configuration(stage):
    from pxr import UsdPhysics, UsdShade
    pads = {}
    for pair in PAIRS:
        for path in pair:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.IsActive() or not prim.IsLoaded() or prim.IsInstanceProxy()
                    or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True):
                raise ValueError("Expected active finger-pad collider: " + path)
            bound, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if not bound or not relation:
                raise ValueError("Finger pad has no resolved physics material: " + path)
            pads[path] = dict(material=str(bound.GetPath()), binding=str(relation.GetPath()),
                              strength=str(relation.GetMetadata("bindMaterialAs")))
    return pads


def _source_configuration_before_load(stage):
    """Only the reviewed old baseline or the current shared policy may load."""
    try:
        return _material_configuration(stage, SOURCE_MATERIAL, SOURCE_PARAMETERS)
    except ValueError:
        return _material_configuration(stage, SOURCE_MATERIAL, LEGACY_SOURCE_PARAMETERS)


def configuration(stage):
    return dict(pads=_pad_configuration(stage),
                **_material_configuration(stage, MATERIAL, PARAMETERS),
                source_material=_material_configuration(stage, SOURCE_MATERIAL, SOURCE_PARAMETERS))


def apply(stage):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError("Global finger-pad material requires meter-scale stage")
    reference = _source_configuration_before_load(stage)
    before = _pad_configuration(stage)
    exists = bool(stage.GetPrimAtPath(MATERIAL))
    if exists:
        _material_configuration(stage, MATERIAL, PARAMETERS)
    allowed_legacy = {right: right.split("/collisions/")[0] + "/collisions.material:binding:physics"
                      for _, right in PAIRS}
    changes = []
    for path, row in before.items():
        if row["material"] == MATERIAL:
            continue
        if row["material"] != SOURCE_MATERIAL and not (
                row["material"] == LEGACY and row["binding"] == allowed_legacy.get(path)):
            raise ValueError("Unreviewed finger-pad material override: " + path)
        changes.append(dict(path=path, binding=path + ".material:binding:physics",
                            previous_binding=row["binding"], before=row["material"], after=MATERIAL))
    # Update GripMat itself (including its non-pad consumers), but do not
    # retarget non-pad bindings or unrelated materials. Roll back everything
    # if a stronger ancestor defeats the four explicit pad bindings.
    session = stage.GetSessionLayer()
    session_before = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            source = stage.GetPrimAtPath(SOURCE_MATERIAL)
            for name in SHARED_OVERRIDES:
                source.GetAttribute(name).Set(SOURCE_PARAMETERS[name])
            material = UsdShade.Material.Define(stage, MATERIAL)
            if not exists:
                prim = material.GetPrim()
                UsdPhysics.MaterialAPI.Apply(prim)
                prim.AddAppliedSchema("PhysxMaterialAPI")
                for attr in stage.GetPrimAtPath(SOURCE_MATERIAL).GetAttributes():
                    if attr.GetName().startswith(("physics:", "physxMaterial:")) and attr.Get() is not None:
                        prim.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=False).Set(
                            PARAMETERS.get(attr.GetName(), attr.Get()))
            for row in changes:
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(row["path"])).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants, "physics")
        after = configuration(stage)
        if any(row["material"] != MATERIAL for row in after["pads"].values()):
            raise RuntimeError("Shared/pad material policy did not resolve all four pad bindings")
    except Exception:
        session.ImportFromString(session_before)
        raise
    return dict(policy_id=POLICY, changes=changes, configuration=after,
                task_overrides_allowed=False, material_parameters_changed=True,
                source_material_parameters_changed=after["source_material"] != reference,
                source_material_before=reference, shared_material_path=SOURCE_MATERIAL,
                shared_material_scope="all_colliders_resolving_to_GripMat",
                dedicated_material_created=not exists,
                geometry_mass_drives_offsets_changed=False, source_assets_modified=False)


def verify(stage, expected):
    """Reject runtime mutation; never silently retune or repair during a task."""
    try:
        current = configuration(stage)
    except ValueError as exc:
        raise RuntimeError("Global finger-pad material changed during execution") from exc
    if current != expected or any(row["material"] != MATERIAL for row in current["pads"].values()):
        raise RuntimeError("Task-time finger-pad material override is forbidden")

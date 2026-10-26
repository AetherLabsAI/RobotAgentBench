"""Explicit collision-offset A/B for a newly owned diagnostic cloth.

This authors USD before PhysX attachment, never moves nodes or changes the
source candidate. Native support gaps must still be measured after attachment.
"""
import hashlib
import math

from evaluation.tasks.tabletop_pick.soft_cloth import native as source_native


PRESETS = ("source", "thin_contact_v1")
MESH_PATH = source_native.ROOT + "/Surface"
CONTACT_OFFSET_M = .002
REST_OFFSET_M = .0005
FIELDS = ("contactOffset", "restOffset")


def _contact_state(prim):
    values = {}
    for name in FIELDS:
        attr = prim.GetAttribute("physxCollision:" + name)
        value = attr.Get() if attr else None
        automatic = isinstance(value, (int, float)) and not math.isfinite(value)
        values[name] = dict(attribute_available=bool(attr),
            authored=bool(attr and attr.HasAuthoredValueOpinion()),
            value_m=None if automatic else value,
            automatic=automatic,
            schema_fallback="-inf" if automatic and value < 0 else "+inf" if automatic else None)
    return dict(physx_collision_api_applied=prim.HasAPI("PhysxCollisionAPI"), values=values)


def _protected_digest(stage):
    """Compare every composed USD opinion except the two permitted offsets/API.

    The temporary in-memory copy is never attached, saved or returned. Both
    snapshots normalize only this exact mesh's collision API and offset attrs.
    Thus material, rest state, poses, unrelated geometry and physics stay bound
    to the same digest even when the new API reveals extra schema fallbacks.
    """
    from pxr import Sdf, Usd
    copy = Usd.Stage.Open(stage.Flatten())
    mesh = copy.GetPrimAtPath(MESH_PATH)
    for name in FIELDS:
        mesh.RemoveProperty("physxCollision:" + name)
    schemas = [name for name in mesh.GetAppliedSchemas() if name != "PhysxCollisionAPI"]
    mesh.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
    return hashlib.sha256(copy.GetRootLayer().ExportToString().encode()).hexdigest()


def apply(stage, manifest, *, preset="source", physics_attached=False):
    """Author an explicit diagnostic preset and return JSON-safe USD evidence."""
    if preset not in PRESETS:
        raise ValueError("Unknown diagnostic contact preset")
    if physics_attached is not False:
        raise ValueError("Contact selection must precede physics attachment")
    if not isinstance(manifest, dict):
        raise ValueError("An explicit scene-owner manifest is required")
    from pxr import PhysxSchema, UsdGeom, UsdPhysics
    cloth = manifest.get("gpu_cloth", {})
    if not isinstance(cloth, dict):
        raise ValueError("An explicit scene-owned cloth descriptor is required")
    layer = stage.GetEditTarget().GetLayer()
    if (manifest.get("layout_scope") != "isolated_component_preview"
            or manifest.get("initialization_owner") != "gpu_cloth_scene_owner.v1"
            or cloth.get("body_root") != source_native.ROOT
            or cloth.get("mesh") != MESH_PATH
            or not layer.anonymous
            or not layer.GetPrimAtPath(source_native.ROOT)
            or not layer.GetPrimAtPath(MESH_PATH)):
        raise ValueError("Contact A/B requires newly authored isolated cloth in its anonymous owner layer")
    if (UsdGeom.GetStageUpAxis(stage) != "Z"
            or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.)):
        raise ValueError("Contact offsets require metre-scale Z-up coordinates")
    body, mesh = stage.GetPrimAtPath(source_native.ROOT), stage.GetPrimAtPath(MESH_PATH)
    if (not body.HasAPI("OmniPhysicsDeformableBodyAPI")
            or body.GetAttribute("omniphysics:kinematicEnabled").Get() is not False
            or not mesh.HasAPI("OmniPhysicsSurfaceDeformableSimAPI")
            or not mesh.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError("Expected the existing dynamic native cloth and collision mesh")
    material = stage.GetPrimAtPath(source_native.ROOT + "/PhysicsMaterial")
    thickness = material.GetAttribute("omniphysics:surfaceThickness").Get() if material else None
    if thickness is None or not math.isclose(thickness, .001, rel_tol=0, abs_tol=1e-9):
        raise ValueError("This diagnostic preserves the existing 1 mm source thickness")
    before, protected_before = _contact_state(mesh), _protected_digest(stage)
    snapshot = layer.ExportToString()
    try:
        if preset == "thin_contact_v1":
            api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
            if (not api or not api.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
                    or not api.CreateRestOffsetAttr().Set(REST_OFFSET_M)):
                raise RuntimeError("Could not author diagnostic cloth contact offsets")
        after, protected_after = _contact_state(mesh), _protected_digest(stage)
        if protected_before != protected_after:
            raise RuntimeError("Contact selection changed unrelated geometry, rest state, material or physics")
        if preset == "source" and layer.ExportToString() != snapshot:
            raise RuntimeError("Source contact preset must not author any USD opinion")
    except Exception:
        layer.ImportFromString(snapshot)
        raise
    changed = [name for name in FIELDS if before["values"][name] != after["values"][name]]
    return dict(schema="openend.short_task.gpu_cloth_contact_ab.v1",
        preset=preset, diagnostics_only=True, contact_calibrated=False,
        benchmark_contact_validated=False, mesh_path=MESH_PATH,
        source_values=before, authored_values=after, changed_fields=changed,
        allowed_authored_paths=[MESH_PATH] if preset == "thin_contact_v1" else [],
        allowed_authored_properties=["apiSchemas", "physxCollision:contactOffset", "physxCollision:restOffset"]
            if preset == "thin_contact_v1" else [],
        only_new_cloth_subtree=True, protected_usd_sha256=protected_after,
        geometry_rest_state_material_mass_unchanged=True,
        source_thickness_m=float(thickness), source_config_changed=False,
        physics_attached=False, runtime_contact_writes=0, native_offsets_verified=False,
        note="Explicit USD offset selection only; actual native support gap and contact behavior remain unvalidated.")

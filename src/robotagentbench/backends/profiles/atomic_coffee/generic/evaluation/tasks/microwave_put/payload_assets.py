"""Use approved shared payload revisions; never author task-local scale.

This is a read-only asset contract for the future reset/worker adapter, not a
physically accepted initial state or a registered API. USD imports are lazy.
"""
from dataclasses import dataclass
from pathlib import Path

from .variants import variant_for


@dataclass(frozen=True)
class PayloadAsset:
    revision: str
    path: Path
    prim_path: str
    size_m: tuple


def resolve_payload_asset(variant):
    selected = variant_for(variant)
    if selected.name == "leftover_box":
        from arena_api import food_box_assets
        data = food_box_assets.verify_files()['variants'][selected.name]
        return PayloadAsset(food_box_assets.REVISION,
                            food_box_assets.DIRECTORY/data['asset'],
                            data['root'], tuple(data['size_m']))
    from arena_api import beef_asset
    beef_asset.verify_files()
    return PayloadAsset(beef_asset.REVISION, beef_asset.ASSET, "/Beefbox", beef_asset.SIZE_M)


def validate_payload_geometry(stage, prim_path, variant):
    """Check a composed instance, including ancestors and per-mesh overrides.

    Numerical tolerances below only accommodate USD float representation;
    these are not contact, success, or reset acceptance thresholds.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    spec = resolve_payload_asset(variant)
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Payload scene must use metres")
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsActive():
        raise ValueError("Missing active payload instance: " + prim_path)
    if variant == "leftover_box":
        from arena_api import food_box_assets
        data = food_box_assets.verify_instance(stage, prim_path, variant)
        enabled = [p for p in Usd.PrimRange(root)
                   if p.HasAPI(UsdPhysics.CollisionAPI)
                   and p.GetAttribute('physics:collisionEnabled').Get() is not False]
        return {
            'revision': spec.revision, 'asset': str(spec.path), 'instance': prim_path,
            'size_m': list(spec.size_m), 'instance_scale': 1,
            'active_colliders': len(enabled), 'geometry_verified': True,
            'mass_kg_observed_not_overridden': root.GetAttribute('physics:mass').Get(),
            'shared_mass_verified': True, 'native_physics_verified': True,
            'source_sha256': data['source_sha256'],
            'physical_initial_state_accepted': False,
        }
    from arena_api import beef_asset
    mass = root.GetAttribute("physics:mass").Get()
    if mass is None or not np.isclose(mass, beef_asset.MASS_KG, rtol=0, atol=1e-7):
        raise ValueError("Payload must use the user-approved shared mass of 0.20 kg")
    cache = UsdGeom.XformCache()
    matrix = np.asarray(cache.GetLocalToWorldTransform(root))
    basis = matrix[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(basis), 1, rtol=0, atol=1e-6)):
        raise ValueError("Shared beef instance and ancestors must have scale=1, no shear/reflection")
    canonical = Usd.Stage.Open(str(spec.path))
    canonical_root = canonical.GetDefaultPrim()
    canonical_cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    canonical_inverse = canonical_cache.GetLocalToWorldTransform(canonical_root).GetInverse()
    expected_meshes = {str(p.GetPath().MakeRelativePath(canonical_root.GetPath())): p
                       for p in Usd.PrimRange(canonical_root) if p.IsA(UsdGeom.Mesh)}
    actual_meshes = {str(p.GetPath().MakeRelativePath(root.GetPath())): p
                     for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)}
    if set(actual_meshes) != set(expected_meshes):
        raise ValueError("Shared beef visual/contact mesh set differs")

    def transformed_points(prim, transforms, inv):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        tf = np.asarray(transforms.GetLocalToWorldTransform(prim) * inv)
        return points @ tf[:3, :3] + tf[3, :3]

    for relative, expected in expected_meshes.items():
        actual = actual_meshes[relative]
        points = transformed_points(actual, cache, inverse)
        target = transformed_points(expected, canonical_cache, canonical_inverse)
        if points.shape != target.shape or not np.allclose(points, target, rtol=0, atol=1e-7):
            raise ValueError("Payload geometry differs from shared revision: " + relative)
        for name in ("faceVertexCounts", "faceVertexIndices"):
            if actual.GetAttribute(name).Get() != expected.GetAttribute(name).Get():
                raise ValueError("Payload topology differs: " + relative)
    enabled = {str(p.GetPath().MakeRelativePath(root.GetPath())) for p in Usd.PrimRange(root)
               if p.HasAPI(UsdPhysics.CollisionAPI)
               and p.GetAttribute("physics:collisionEnabled").Get() is not False}
    expected_enabled = {str(p.GetPath().MakeRelativePath(canonical_root.GetPath()))
                        for p in Usd.PrimRange(canonical_root)
                        if p.HasAPI(UsdPhysics.CollisionAPI)
                        and p.GetAttribute("physics:collisionEnabled").Get() is not False}
    if enabled != expected_enabled:
        raise ValueError("Payload active collider set differs; legacy/double collision not accepted")
    return {
        "revision": spec.revision, "asset": str(spec.path), "instance": prim_path,
        "size_m": list(spec.size_m), "instance_scale": 1,
        "visual_mesh_count": len(expected_meshes) - len(expected_enabled),
        "active_colliders": len(enabled), "geometry_verified": True,
        "mass_kg_observed_not_overridden": mass,
        "mass_revision": beef_asset.MASS_REVISION, "shared_mass_verified": True,
        "physical_initial_state_accepted": False,
    }

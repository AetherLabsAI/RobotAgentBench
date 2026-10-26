"""Session-only, visible-mesh static obstacles for the standalone House scene.

Apply after variant/layout selection and BEFORE physics reset. No rigid bodies,
proxy hulls, geometry edits, material/drive changes, or scoring rules are added.
"""
from collections import Counter
from arena_randomization.shelf_supports import BOARD, BOARD_SUPPORT_PATHS

POLICY = "house.visible_static_obstacles.v2"
SOFA = "/World/Assets/Sofa"
SHELF = "/World/Assets/ShelvingUnit"
PLANTS = "/World/Assets/LivingRoomPlantStands"
LAMPS = "/World/Assets/LivingRoomLampReplacements"
ROOTS = (SOFA, SHELF, PLANTS, LAMPS)
FURNITURE = (
    SOFA + "/Obj/Meshes/FAB_converted_model/root/GLTF_SceneRootNode/__seat_sofa_0/Object_4/Object_0",
    SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/shelf_boards_0/Object_4/Object_0",
    SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/horizontal_brackets_1/Object_6/Object_1",
    SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/vertical_beams_2/Object_8/Object_2",
    SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/vertical_beams_2/Object_9/Object_3",
    SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/plastic_feet_3/Object_11/Object_4",
)


def category(path):
    if path in BOARD_SUPPORT_PATHS:
        return "shelf"
    for root, name in ((SOFA, "sofa"), (SHELF, "shelf"), (LAMPS, "lamp")):
        if path == root or path.startswith(root + "/"):
            return name
    if path.startswith(PLANTS + "/"):
        return "plant_stand" if "/Stand/" in path else "plant"
    return None


def rigid_ancestor(prim):
    from pxr import UsdPhysics
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def inventory(stage, validate_geometry=False):
    """Preflight the complete scope before authoring any changes."""
    from pxr import Usd, UsdGeom, UsdPhysics
    rows, furniture = [], set()
    aligned = any(stage.GetPrimAtPath(p) for p in (BOARD_SUPPORT_PATHS[0], BOARD_SUPPORT_PATHS[3]))
    if aligned:
        for path in BOARD_SUPPORT_PATHS:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.IsActive() or not prim.IsA(UsdGeom.Cube)
                    or not prim.HasAPI(UsdPhysics.CollisionAPI)):
                raise ValueError("Incomplete L1 shelf supports: " + path)
    for root_path in ROOTS + (BOARD_SUPPORT_PATHS[1:3] if aligned else ()):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsActive() or not root.IsLoaded():
            raise ValueError("Expected loaded House obstacle root: " + root_path)
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            path = str(prim.GetPath())
            if prim.IsInstanceProxy():
                raise ValueError("Instance obstacle needs explicit asset preparation: " + path)
            if prim.GetTypeName() not in ("Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"):
                raise ValueError("Unsupported visible collision geometry: " + path)
            if rigid_ancestor(prim):
                raise ValueError("Static obstacle unexpectedly belongs to a rigid body: " + path)
            support = aligned and path in BOARD_SUPPORT_PATHS
            if root_path in (SOFA, SHELF) and not support:
                furniture.add(path)
                if (path not in FURNITURE or not prim.HasAPI(UsdPhysics.CollisionAPI)
                        or prim.GetAttribute("physics:approximation").Get() != "none"):
                    raise ValueError("Unreviewed furniture collision topology: " + path)
            # This is a coplanar texture decal above the solid tabletop, not
            # a separate physical sheet. Never create a duplicate contact plane.
            decal = root_path == PLANTS and path.endswith("/Stand/TopTexture")
            visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
            if validate_geometry and prim.IsA(UsdGeom.Mesh) and not decal:
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
                if (not points or not counts or not indices or sum(counts) != len(indices)
                        or min(counts) < 3 or min(indices) < 0 or max(indices) >= len(points)):
                    raise ValueError("Invalid obstacle mesh topology: " + path)
            rows.append(dict(path=path, category=category(path), type=prim.GetTypeName(),
                             enabled=support or (visible and not decal and not (aligned and path == BOARD)),
                             excluded_reason="l1_board_slabs_replace_mesh" if aligned and path == BOARD
                             else None if support else "texture_decal" if decal else "hidden" if not visible else None))
    if furniture != set(FURNITURE):
        raise ValueError("House sofa/shelf structure is incomplete")
    return rows


def configuration(stage):
    """Cheap runtime guard: visibility, variant topology, static transforms and physics."""
    from pxr import Usd, UsdGeom, UsdPhysics
    rows = inventory(stage)
    cache = UsdGeom.XformCache()
    for row in rows:
        prim = stage.GetPrimAtPath(row["path"])
        row.update(collision_api=prim.HasAPI(UsdPhysics.CollisionAPI),
                   collision_enabled=prim.GetAttribute("physics:collisionEnabled").Get(),
                   approximation=prim.GetAttribute("physics:approximation").Get(),
                   world_transform=str(cache.GetLocalToWorldTransform(prim)))
    variants = {}
    for root in ROOTS:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for name in prim.GetVariantSets().GetNames():
                variants[str(prim.GetPath()) + ":" + name] = prim.GetVariantSet(name).GetVariantSelection()
    return dict(colliders=rows, variants=variants)


def apply(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-9:
        raise ValueError("House static obstacles require meter-scale stage")
    rows = inventory(stage, validate_geometry=True)
    changes = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for row in rows:
            prim = stage.GetPrimAtPath(row["path"])
            had_api = prim.HasAPI(UsdPhysics.CollisionAPI)
            old = prim.GetAttribute("physics:collisionEnabled").Get() if had_api else None
            # A decal without collision stays completely untouched.
            if row["excluded_reason"] == "texture_decal" and not had_api:
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(row["enabled"])
            if row["type"] == "Mesh":
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            if not had_api or old != row["enabled"]:
                changes.append(dict(path=row["path"], before=old, after=row["enabled"]))
    result = configuration(stage)
    return dict(policy_id=POLICY, mode="static_visible_geometry", configuration=result,
                changes=changes, enabled_counts=dict(Counter(r["category"] for r in rows if r["enabled"])),
                source_assets_modified=False, geometry_changed=False, dynamic_bodies_added=False,
                legacy_support_overlays_changed=False, scoring_implemented=False)


def verify(stage, expected):
    if configuration(stage) != expected:
        raise RuntimeError("Obstacle variants/visibility/physics changed after initialization; reset and reapply")

"""Measure visible shelf faces, collision supports and realized placements.

The Git-owned alignment layer moves individual boards and their brackets, never
scales the rack or changes archived trajectories. Validation reads face vertices
rather than stale mesh extents (the asset retains unused fifth-tier vertices).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .l1_shelf_geometry import SHELF_BOARD_TOP_Z_M

SHELF = "/World/Assets/ShelvingUnit"
BOARD = SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/shelf_boards_0/Object_4/Object_0"
BRACKETS = SHELF + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/horizontal_brackets_1/Object_6/Object_1"
BOARD_SUPPORT_PATHS = (
    SHELF + "/BoardSupports/Bottom", "/World/Assets/Tier2Overlay",
    "/World/Assets/Tier3Overlay", SHELF + "/BoardSupports/Top",
)
SURFACES = {
    "shelf_snack": (1, "/World/Assets/Tier2Overlay", "snack_surface_z_m"),
    "shelf_beverage": (2, "/World/Assets/Tier3Overlay", "beverage_surface_z_m"),
    "shelf_top": (3, BOARD_SUPPORT_PATHS[3], "top_surface_z_m"),
}


def capture_source_support(stage, target_path):
    """Measure the actual shelf collider; the rack's AABB is not a support."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    parent = str(target_path).rsplit("/", 1)[0]
    path = {
        "/World/Assets/ShelfItems": "/World/Assets/Tier3Overlay",
        "/World/Assets/Tier2Items": "/World/Assets/Tier2Overlay",
    }.get(parent)
    if path is None:
        return {}
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive():
        raise ValueError("source support is missing: " + path)
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"],
        useExtentsHint=False, ignoreVisibility=True,
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError("source support bound is empty: " + path)
    count = sum(bool(p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
                for p in Usd.PrimRange(prim))
    return {
        "source_support_body_path": np.asarray(path),
        "source_support_aabb_min_xyz_m": np.asarray(bounds.GetMin(), dtype=float),
        "source_support_aabb_max_xyz_m": np.asarray(bounds.GetMax(), dtype=float),
        "source_support_collision_path_count": np.asarray(count, dtype=np.int64),
    }


def _mesh_world_points(stage, path):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"shelf mesh missing: {path}")
    mesh = UsdGeom.Mesh(prim)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
    return mesh, matrix, points


def measure_board_faces(stage):
    import numpy as np
    from pxr import UsdGeom
    mesh, _, points = _mesh_world_points(stage, BOARD)
    if not mesh.GetPrim().IsActive() or UsdGeom.Imageable(mesh).ComputeVisibility() == "invisible":
        raise ValueError("shelf boards are not visible and active")
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = {}
    offset = 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        polygon = points[indices[offset:offset + count]]
        offset += count
        if count < 3 or np.ptp(polygon[:, 2]) > 1e-5:
            continue
        area = sum(np.linalg.norm(np.cross(polygon[i] - polygon[0], polygon[i+1] - polygon[0])) / 2
                   for i in range(1, count-1))
        z = round(float(polygon[:, 2].mean()), 6)
        row = faces.setdefault(z, {"area": 0., "points": []})
        row["area"] += float(area)
        row["points"].extend(polygon.tolist())
    broad = sorted(z for z, row in faces.items() if row["area"] > .2)
    groups = []
    for z in broad:
        if groups and z - groups[-1][-1] < .02:
            groups[-1].append(z)
        else:
            groups.append([z])
    if len(groups) != 4:
        raise ValueError(f"expected four visible board levels, observed {groups}")
    result = []
    for group in groups:
        top = max(group)
        p = np.array(faces[top]["points"])
        result.append({"z_m": top, "area_m2": faces[top]["area"],
                       "xy_min_m": p[:, :2].min(axis=0).tolist(),
                       "xy_max_m": p[:, :2].max(axis=0).tolist()})
    return result


def validate_shelf_supports(stage, *, policy=None, placements=(), tolerance_m=.001):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    boards = measure_board_faces(stage)
    errors = []
    mesh_collision = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(BOARD))
    if mesh_collision and mesh_collision.GetCollisionEnabledAttr().Get():
        errors.append("duplicate shelf board mesh collision is enabled")
    for i, expected in enumerate(SHELF_BOARD_TOP_Z_M):
        if abs(boards[i]["z_m"] - expected) > tolerance_m:
            errors.append(f"visible board {i+1}: {boards[i]['z_m']} != {expected}")
    supports = {}
    for surface, (index, path, key) in SURFACES.items():
        board = boards[index]
        row = {"visible_top_z_m": board["z_m"]}
        if path:
            prim = stage.GetPrimAtPath(path)
            collision = UsdPhysics.CollisionAPI(prim) if prim else None
            if not prim or not prim.IsA(UsdGeom.Cube) or not prim.IsActive() or not collision or not collision.GetCollisionEnabledAttr().Get():
                errors.append(f"enabled shelf collision support missing: {path}")
            else:
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                corners = np.array([xf.Transform(Gf.Vec3d(x*size/2, y*size/2, z*size/2))
                                    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                row["collision_top_z_m"] = float(corners[:, 2].max())
                if abs(row["collision_top_z_m"] - board["z_m"]) > tolerance_m:
                    errors.append(f"{surface}: collision top does not match visible board")
                if (np.any(corners[:, :2].min(axis=0) < np.array(board['xy_min_m']) - .005)
                        or np.any(corners[:, :2].max(axis=0) > np.array(board['xy_max_m']) + .005)):
                    errors.append(f"{surface}: collider extends beyond visible board")
                if (np.any(corners[:, :2].min(axis=0) > np.array(board['xy_min_m']) + .001)
                        or np.any(corners[:, :2].max(axis=0) < np.array(board['xy_max_m']) - .001)):
                    errors.append(f"{surface}: collider does not cover the full visible board")
        if policy is not None:
            z = float(policy["shelf"][key])
            row["configured_z_m"] = z
            if abs(z - board["z_m"]) > tolerance_m:
                errors.append(f"{surface}: configured height {z} differs from visible board {board['z_m']}")
        supports[surface] = row
    # Test actual world-space bounds after _set_pose, including parent offsets.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], False, True)
    checked = 0
    for placement in placements:
        surface = placement.get("surface")
        if surface not in SURFACES:
            continue
        prim = stage.GetPrimAtPath(placement["prim_path"])
        if not prim or not prim.IsActive():
            errors.append(f"shelf placement missing/inactive: {placement['prim_path']}")
            continue
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
        if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            errors.append(f"shelf placement has invalid bounds: {prim.GetPath()}")
            continue
        board = boards[SURFACES[surface][0]]
        gap = .003 if placement.get("dynamic") else .001
        if abs(float(lo[2]) - board["z_m"] - gap) > tolerance_m:
            errors.append(f"{prim.GetPath()}: bottom gap {lo[2] - board['z_m']:.6f} != {gap}")
        if (np.any(lo[:2] < np.array(board['xy_min_m']) - .005)
                or np.any(hi[:2] > np.array(board['xy_max_m']) + .005)):
            errors.append(f"{prim.GetPath()}: footprint extends beyond its visible board")
        checked += 1
    if errors:
        raise ValueError("shelf support inconsistency: " + "; ".join(errors))
    return {"schema": "real2sim.shelf_support_geometry.v1", "passed": True,
            "method": "visible_face_vertices_and_collision_corners", "boards": boards,
            "supports": supports, "placements_checked": checked,
            "physics_replay_performed": False}


def build_collision_layer(stage, destination):
    """Author one full-size slab per board, without coplanar mesh duplicates.

    This layer replaces the baseline's old Tier2/3 cube definitions. Bottom/top
    supports follow the rack as children; existing overlay identities remain.
    """
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    faces = measure_board_faces(stage)
    mesh, _, points = _mesh_world_points(stage, BOARD)
    points = points[sorted(set(mesh.GetFaceVertexIndicesAttr().Get()))]
    labels = np.argmin(np.abs(points[:, 2, None]
                             - np.array([b["z_m"] for b in faces])[None]), axis=1)
    layer = Sdf.Layer.CreateAnonymous("shelf_support_collisions.usda")
    output = Usd.Stage.Open(stage.GetRootLayer(), layer)
    output.SetEditTarget(layer)
    UsdPhysics.CollisionAPI(output.GetPrimAtPath(BOARD)).CreateCollisionEnabledAttr(False)
    UsdGeom.Scope.Define(output, SHELF + "/BoardSupports")
    for index, path in enumerate(BOARD_SUPPORT_PATHS):
        vertices = points[labels == index]
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        cube = UsdGeom.Cube.Define(output, path)
        cube.CreateSizeAttr(1.)
        cube.CreateVisibilityAttr("invisible")
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        if index in (1, 2):
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lower + upper) / 2)))
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float, upper - lower)))
        else:
            world = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, upper - lower)))
            world.SetTranslateOnly(Gf.Vec3d(*map(float, (lower + upper) / 2)))
            parent = UsdGeom.Xformable(cube.GetPrim().GetParent()).ComputeLocalToWorldTransform(0)
            xf.AddTransformOp().Set(world * parent.GetInverse())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        spec = layer.GetPrimAtPath(path)
        spec.specifier, spec.typeName = Sdf.SpecifierDef, "Cube"
    layer.comment = "Measured full-board slabs; visible mesh collision disabled to avoid duplicate contacts."
    layer.Export(str(destination))


def build_alignment_layer(stage, destination):
    """Author only points/extents in a source sublayer; preserve rack transforms."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    tops = [b["z_m"] for b in measure_board_faces(stage)]
    target = list(SHELF_BOARD_TOP_Z_M)
    layer = Sdf.Layer.CreateAnonymous("shelf_support_alignment.usda")
    output = Usd.Stage.Open(layer)
    for path in (BOARD, BRACKETS):
        mesh, matrix, points = _mesh_world_points(stage, path)
        used = set(mesh.GetFaceVertexIndicesAttr().Get())
        inverse = matrix.GetInverse()
        authored = list(mesh.GetPointsAttr().Get())
        for i in used:
            tier = min(range(4), key=lambda j: abs(points[i, 2] - tops[j]))
            delta = target[tier] - tops[tier]
            shifted = points[i].copy()
            shifted[2] += delta
            authored[i] = Gf.Vec3f(inverse.Transform(Gf.Vec3d(*map(float, shifted))))
        override = UsdGeom.Mesh(output.OverridePrim(path))
        override.CreatePointsAttr(Vt.Vec3fArray(authored))
        a = np.array(authored)
        override.CreateExtentAttr(Vt.Vec3fArray([
            Gf.Vec3f(*map(float, a.min(axis=0))), Gf.Vec3f(*map(float, a.max(axis=0)))]))
    layer.comment = "User-approved per-tier board/bracket translation. No rack scaling or trajectory edits."
    layer.Export(str(destination))


def main():
    from pxr import Usd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--build-alignment", type=Path)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.scene))
    if args.build_alignment:
        build_alignment_layer(stage, args.build_alignment)
    else:
        policy = json.loads(args.policy.read_text()) if args.policy else None
        print(json.dumps(validate_shelf_supports(stage, policy=policy), indent=2))


if __name__ == "__main__":
    main()

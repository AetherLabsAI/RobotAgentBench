#!/usr/bin/env python3
"""Render-only USD overlay and appearance placement for 50 House topologies.

The topology catalog is still a candidate catalog.  This module deliberately
authors only disposable working scenes: it replaces the visible shell, moves
available interaction-module roots, materializes catalog-bound assets and
supports, creates clearly labelled proxies only for genuinely undeclared
modules, and relocates render-only decor.  It does not promote a candidate or
claim live navigation, collision, or manipulation success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

try:
    from . import house_layout, house_topology
except ImportError:  # Direct execution with production/ on sys.path.
    import house_layout
    import house_topology


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCHEMA = "real2sim.house_topology_visual_plan.v1"
OVERLAY_SCHEMA = "real2sim.house_topology_visual_overlay.v1"
ROOT_PATH = "/World/HouseTopologyCandidate"
WALL_HEIGHT_M = 3.0
WALL_THICKNESS_M = 0.12
FLOOR_THICKNESS_M = 0.10
MIN_ROOM_LIGHT_INTENSITY = 2000.0
AMBIENT_LIGHT_INTENSITY = 360.0

# A topology module must have an explicit asset or generated support.  Reusing
# the canonical monolithic kitchen as a small counter made its 2.28 x 4.62 m
# composed footprint cross the generated room boundary, so there are no
# geometry fallbacks here.
VISUAL_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {}

HIDDEN_CANONICAL_PRIMS = (
    "/World/Assets/Door",
    "/World/Assets/Fireplace",
    "/World/Assets/HouseEntryDoor",
    "/World/Assets/LivingRoomRug",
    "/World/Assets/LivingRoomWallpaper",
    "/World/Assets/PantrySlidingDoor",
    "/World/Assets/WestWindowTrim",
    "/World/Assets/EastWindowTrim",
)

DECOR_SLOT_ROOTS = {
    "TvSouthWestWindowLeft": (
        "/World/Assets/LivingRoomPlantStands/TvSouthWestWindowLeft",
        "/World/Assets/LivingRoomLampReplacements/TvSouthWestWindowLeft",
    ),
    "TvNorth": (
        "/World/Assets/LivingRoomPlantStands/TvNorth",
        "/World/Assets/LivingRoomLampReplacements/TvNorth",
    ),
    "WestWindowRight": (
        "/World/Assets/LivingRoomPlantStands/WestWindowRight",
    ),
    "EastWindowLeft": (
        "/World/Assets/LivingRoomPlantStands/EastWindowLeft",
    ),
    "EastWindowRight": (
        "/World/Assets/LivingRoomPlantStands/EastWindowRight",
    ),
}


def stage_declared_assets(
    catalog: Mapping[str, Any], render_demo: Path | str,
) -> Path:
    """Stage catalog-bound USD bundles under the renderer-mounted scene root."""

    render_root = Path(render_demo).resolve()
    root = Path(tempfile.mkdtemp(
        prefix="house_topology_assets_", dir=render_root))
    root.chmod(0o755)
    source_directories: set[tuple[Path, Path]] = set()
    for module in catalog["interaction_modules"].values():
        spec = module.get("asset_materialization")
        if spec is None:
            continue
        relative = Path(str(spec["reference_path"]))
        source = (REPO / relative).resolve()
        try:
            source.relative_to(REPO)
        except ValueError as exc:
            raise ValueError(
                f"topology asset escapes repository root: {relative}"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        bundle_relative = Path(str(spec.get(
            "stage_bundle_root", relative.parent)))
        bundle_source = (REPO / bundle_relative).resolve()
        try:
            source.relative_to(bundle_source)
            bundle_source.relative_to(REPO)
        except ValueError as exc:
            raise ValueError(
                f"topology asset bundle does not contain asset: {bundle_relative}"
            ) from exc
        if not bundle_source.is_dir():
            raise FileNotFoundError(bundle_source)
        required_sizes = spec.get("required_file_sizes_bytes", {})
        for required_relative in map(Path, spec.get("required_files", ())):
            required_source = (bundle_source / required_relative).resolve()
            try:
                required_source.relative_to(bundle_source)
            except ValueError as exc:
                raise ValueError(
                    "topology asset dependency escapes its bundle: "
                    f"{required_relative}"
                ) from exc
            if not required_source.is_file():
                raise FileNotFoundError(required_source)
            expected_size = int(required_sizes[str(required_relative)])
            actual_size = required_source.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    "topology asset dependency size mismatch: "
                    f"{required_source} is {actual_size}, expected {expected_size}"
                )
        source_directories.add((bundle_source, bundle_relative))
    for source, relative_parent in sorted(
        source_directories, key=lambda row: str(row[1])
    ):
        destination = root / relative_parent
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    return root


def candidate_for_id(
    topology_id: str,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = dict(catalog or house_topology.load_catalog())
    matches = [
        dict(row) for row in effective["candidates"]
        if row["id"] == str(topology_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown House topology {topology_id!r}")
    return matches[0], effective


def candidate_for_ordinal(
    ordinal: int,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("topology ordinal must be an integer")
    if not 1 <= ordinal <= house_topology.CANDIDATE_COUNT:
        raise ValueError("topology ordinal must be in 1..50")
    effective = dict(catalog or house_topology.load_catalog())
    return dict(effective["candidates"][ordinal - 1]), effective


def load_overlay_catalog(path: Path | str) -> dict[str, Any]:
    """Load a bounded delivery catalog without treating it as task identity."""

    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = catalog.get("candidates")
    modules = catalog.get("interaction_modules")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("topology overlay catalog must contain candidates")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("topology overlay catalog must contain interaction_modules")
    ids = [row.get("id") for row in candidates]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("topology overlay candidate IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("topology overlay candidate IDs must be unique")
    return catalog


def _topology_frame_aliases(
    modules: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    aliases = {
        str(module["canonical_frame"]): str(module_id)
        for module_id, module in modules.items()
    }
    aliases.update({
        "shelf_beverage": "snack_shelf_cluster",
        "shelf_snack": "snack_shelf_cluster",
        "shelf_top": "snack_shelf_cluster",
        "kitchen": "kitchen_counter_cluster",
        "world": None,
    })
    return aliases


def topology_layout(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    *,
    available_prim_paths: set[str] | None = None,
) -> house_layout.HouseLayout:
    """Convert a full-House candidate into the existing SE(2) contract."""

    transforms: dict[str, house_layout.SE2] = {}
    prim_roots: dict[str, tuple[str, ...]] = {}
    for module_id, module in modules.items():
        pivot = tuple(map(float, module["canonical_pivot_xy_m"]))
        pose = candidate["module_poses"][module_id]
        target = tuple(map(float, pose["center_xy_m"]))
        transforms[module_id] = house_layout.SE2(
            pivot_xy_m=pivot,
            translate_xy_m=(target[0] - pivot[0], target[1] - pivot[1]),
            yaw_deg=float(pose["yaw_deg"]),
        )
        requested = tuple(map(str, module["source_prim_roots"]))
        fallback = tuple(VISUAL_FALLBACK_ROOTS.get(module_id, ()))
        roots = requested + fallback
        if available_prim_paths is not None:
            roots = tuple(path for path in roots if path in available_prim_paths)
        prim_roots[module_id] = roots
    return house_layout.HouseLayout(
        ordinal=int(candidate["ordinal"]),
        layout_id=str(candidate["id"]),
        status=house_topology.STATUS,
        transforms=transforms,
        frame_aliases=_topology_frame_aliases(modules),
        cluster_prim_roots=prim_roots,
    )


def _rects(candidate: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    rows = [tuple(map(float, space["rect_xywh_m"])) for space in candidate["spaces"]]
    for corridor in candidate["corridors"]:
        rows.extend(tuple(map(float, rect)) for rect in corridor["rects_xywh_m"])
    return rows


def _inside(point: tuple[float, float], rect: Sequence[float]) -> bool:
    x, y, width, height = map(float, rect)
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _union_grid(candidate: Mapping[str, Any]) -> tuple[
    list[float], list[float], set[tuple[int, int]]
]:
    rects = _rects(candidate)
    xs = sorted({value for x, _y, width, _height in rects for value in (x, x + width)})
    ys = sorted({value for _x, y, _width, height in rects for value in (y, y + height)})
    occupied = set()
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            center = ((xs[xi] + xs[xi + 1]) / 2.0, (ys[yi] + ys[yi + 1]) / 2.0)
            if any(_inside(center, rect) for rect in rects):
                occupied.add((xi, yi))
    return xs, ys, occupied


def floor_slabs(candidate: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    """Return a compact non-overlapping rectangle union for floor/ceiling."""

    xs, ys, occupied = _union_grid(candidate)
    bands: list[tuple[float, float, float, float]] = []
    for yi in range(len(ys) - 1):
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for xi in range(len(xs) - 1):
            filled = (xi, yi) in occupied
            if filled and start is None:
                start = xi
            if start is not None and (not filled or xi == len(xs) - 2):
                end = xi + 1 if filled else xi
                runs.append((start, end))
                start = None
        for left, right in runs:
            bands.append((
                xs[left], ys[yi],
                xs[right] - xs[left], ys[yi + 1] - ys[yi],
            ))
    # Semantic zones and corridor rectangles introduce many internal grid
    # lines into one continuous open-plan floor.  Merge vertically adjacent
    # bands with the same horizontal extent so a tote or robot does not rest
    # across several coplanar collision bodies.  This preserves the exact
    # union boundary and area; only the collider partition changes.
    slabs: list[tuple[float, float, float, float]] = []
    for x, y, width, height in bands:
        for index, (old_x, old_y, old_width, old_height) in enumerate(slabs):
            if (
                math.isclose(x, old_x, abs_tol=1.0e-8)
                and math.isclose(width, old_width, abs_tol=1.0e-8)
                and math.isclose(y, old_y + old_height, abs_tol=1.0e-8)
            ):
                slabs[index] = (old_x, old_y, old_width, old_height + height)
                break
        else:
            slabs.append((x, y, width, height))
    return sorted(slabs, key=lambda row: (row[1], row[0], row[2], row[3]))


def exterior_wall_segments(
    candidate: Mapping[str, Any],
) -> list[tuple[str, float, float, float]]:
    """Return merged axis-aligned exterior boundary segments.

    Rows are ``(axis, fixed_coordinate, start, end)`` where axis ``x`` means a
    horizontal segment and axis ``y`` means a vertical segment.
    """

    xs, ys, occupied = _union_grid(candidate)
    raw: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for xi, yi in occupied:
        for neighbor, key, interval in (
            ((xi, yi - 1), ("x", ys[yi]), (xs[xi], xs[xi + 1])),
            ((xi, yi + 1), ("x", ys[yi + 1]), (xs[xi], xs[xi + 1])),
            ((xi - 1, yi), ("y", xs[xi]), (ys[yi], ys[yi + 1])),
            ((xi + 1, yi), ("y", xs[xi + 1]), (ys[yi], ys[yi + 1])),
        ):
            if neighbor not in occupied:
                raw.setdefault(key, []).append(interval)
    result: list[tuple[str, float, float, float]] = []
    for (axis, fixed), intervals in sorted(raw.items()):
        ordered = sorted(intervals)
        start, end = ordered[0]
        for next_start, next_end in ordered[1:]:
            if math.isclose(next_start, end, abs_tol=1.0e-8):
                end = next_end
            else:
                result.append((axis, fixed, start, end))
                start, end = next_start, next_end
        result.append((axis, fixed, start, end))
    return result


def _rotated_half_extents(size: Sequence[float], yaw_deg: float) -> tuple[float, float]:
    width, depth = map(float, size)
    angle = math.radians(float(yaw_deg))
    return (
        (abs(math.cos(angle)) * width + abs(math.sin(angle)) * depth) / 2.0,
        (abs(math.sin(angle)) * width + abs(math.cos(angle)) * depth) / 2.0,
    )


def _decor_positions(
    candidate: Mapping[str, Any], modules: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float | list[float]]], float]:
    living = next(space for space in candidate["spaces"] if space["kind"] == "living")
    x, y, width, height = map(float, living["rect_xywh_m"])
    inset = 0.55
    pool = [
        (x + inset, y + inset), (x + width - inset, y + inset),
        (x + inset, y + height - inset), (x + width - inset, y + height - inset),
        (x + width * 0.25, y + inset), (x + width * 0.75, y + inset),
        (x + width * 0.25, y + height - inset),
        (x + width * 0.75, y + height - inset),
        (x + inset, y + height * 0.5), (x + width - inset, y + height * 0.5),
    ]
    obstacles = []
    for module_id, pose in candidate["module_poses"].items():
        # Logical spaces overlap in open-plan layouts; every module is physical.
        module = modules[module_id]
        bounds = module.get("footprint_bounds_xy_m")
        if bounds is None:
            fixture_width, fixture_depth = map(float, module["fixture_size_xy_m"])
            bounds = (-fixture_width / 2, -fixture_depth / 2,
                      fixture_width / 2, fixture_depth / 2)
        min_x, min_y, max_x, max_y = map(float, bounds)
        angle = math.radians(float(pose["yaw_deg"]))
        cosine, sine = math.cos(angle), math.sin(angle)
        center_x, center_y = map(float, pose["center_xy_m"])
        corners = [
            (center_x + cosine * px - sine * py,
             center_y + sine * px + cosine * py)
            for px in (min_x, max_x) for py in (min_y, max_y)
        ]
        world_min_x, world_max_x = min(p[0] for p in corners), max(p[0] for p in corners)
        world_min_y, world_max_y = min(p[1] for p in corners), max(p[1] for p in corners)
        obstacles.append(((world_min_x + world_max_x) / 2,
                          (world_min_y + world_max_y) / 2,
                          (world_max_x - world_min_x) / 2,
                          (world_max_y - world_min_y) / 2))

    def clearance(point: tuple[float, float]) -> float:
        values = []
        for ox, oy, half_x, half_y in obstacles:
            dx = max(abs(point[0] - ox) - half_x, 0.0)
            dy = max(abs(point[1] - oy) - half_y, 0.0)
            values.append(math.hypot(dx, dy))
        return min(values, default=99.0)

    selected: list[tuple[float, float]] = []
    for point in sorted(pool, key=lambda item: (-clearance(item), item)):
        if clearance(point) < 0.48:
            continue
        if any(math.dist(point, other) < 0.9 for other in selected):
            continue
        selected.append(point)
        if len(selected) == len(DECOR_SLOT_ROOTS):
            break
    if len(selected) != len(DECOR_SLOT_ROOTS):
        # Keep valid original slots, then fill gaps deterministically. The module
        # clearance, decor spacing, inset, yaw, theme and seeds remain unchanged.
        step = 0.2
        fallback = [
            (round(x + inset + ix * step, 8), round(y + inset + iy * step, 8))
            for ix in range(max(0, math.floor((width - 2 * inset) / step) + 1))
            for iy in range(max(0, math.floor((height - 2 * inset) / step) + 1))
        ]
        for point in sorted(fallback, key=lambda item: (-clearance(item), item)):
            if clearance(point) < 0.48:
                continue
            if any(math.dist(point, other) < 0.9 for other in selected):
                continue
            selected.append(point)
            if len(selected) == len(DECOR_SLOT_ROOTS):
                break
    if len(selected) != len(DECOR_SLOT_ROOTS):
        raise ValueError(f"{candidate['id']} has no collision-safe living-room decor slots")
    slots = {}
    for name, point in zip(DECOR_SLOT_ROOTS, selected, strict=True):
        nearest = min(
            (
                (point[1] - y, 180.0),
                (y + height - point[1], 0.0),
                (point[0] - x, 90.0),
                (x + width - point[0], 270.0),
            ),
            key=lambda item: item[0],
        )
        slots[name] = {"xy_m": [round(point[0], 5), round(point[1], 5)], "yaw_deg": nearest[1]}
    return slots, min(clearance(point) for point in selected)


def build_visual_plan(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    appearance_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if appearance_plan.get("schema") != "real2sim.livingroom_appearance_plan.v3":
        raise ValueError("topology visual plan requires whole-House appearance v3")
    slots, clearance = _decor_positions(candidate, modules)
    scheme_lights = list(appearance_plan["lights"])
    intensity_scale = sum(float(row["intensity_scale"]) for row in scheme_lights) / max(len(scheme_lights), 1)
    temperatures = [float(row["color_temperature_k"]) for row in scheme_lights]
    room_lights = []
    for space in candidate["spaces"]:
        x, y, width, height = map(float, space["rect_xywh_m"])
        room_lights.append({
            "space_id": str(space["id"]),
            "kind": str(space["kind"]),
            "xyz_m": [x + width / 2.0, y + height / 2.0, WALL_HEIGHT_M - 0.12],
            "size_xy_m": [max(1.0, min(width * 0.42, 3.0)), max(1.0, min(height * 0.42, 3.0))],
            # The generated shells are much larger than the canonical living
            # room.  Keep evening palettes warm, but do not let a task camera
            # fall below the production luminance gate merely because its
            # selected module moved away from the old fixed fixtures.
            "intensity": round(
                max(MIN_ROOM_LIGHT_INTENSITY, 2250.0 * intensity_scale), 4),
            "color_temperature_k": round(sum(temperatures) / max(len(temperatures), 1), 1),
        })
    return {
        "schema": SCHEMA,
        "status": house_topology.STATUS,
        "topology_id": str(candidate["id"]),
        "topology_ordinal": int(candidate["ordinal"]),
        "appearance_style": str(appearance_plan["style"]),
        "lighting_scheme": str(appearance_plan["lighting_scheme"]),
        "decor_slots": slots,
        "minimum_decor_to_module_clearance_m": round(clearance, 5),
        "room_lights": room_lights,
        "invariants": {
            "accepted_action_state_untouched": True,
            "render_only_decor": True,
            "physics_acceptance_claimed": False,
        },
    }


def _bind(stage: Any, prim: Any, material_path: str) -> None:
    from pxr import UsdShade

    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim and material_prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim))


def _cube(
    stage: Any, path: str, *, center: Sequence[float], size: Sequence[float],
    material: str, collision: bool,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xformable.AddScaleOp().Set(Gf.Vec3d(*map(float, size)))
    _bind(stage, cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube.GetPrim()


FLOOR_UV_SCHEMA = "real2sim.world_aligned_wood_floor_uv.v1"
FLOOR_TEXTURE_TILE_M = 2.4


def _floor_slab(
    stage: Any, path: str, *, x: float, y: float, width: float, height: float,
    material: str,
) -> Any:
    """Author one collision slab with world-aligned face-varying wood UVs."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    x0, y0 = float(x), float(y)
    x1, y1 = x0 + float(width), y0 + float(height)
    z0, z1 = -FLOOR_THICKNESS_M, 0.0
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr([
        Gf.Vec3f(x0, y0, z0), Gf.Vec3f(x1, y0, z0),
        Gf.Vec3f(x1, y1, z0), Gf.Vec3f(x0, y1, z0),
        Gf.Vec3f(x0, y0, z1), Gf.Vec3f(x1, y0, z1),
        Gf.Vec3f(x1, y1, z1), Gf.Vec3f(x0, y1, z1),
    ])
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr([
        0, 3, 2, 1,  # bottom
        4, 5, 6, 7,  # textured top
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ])
    mesh.CreateExtentAttr([
        Gf.Vec3f(x0, y0, z0), Gf.Vec3f(x1, y1, z1),
    ])
    u0, v0 = x0 / FLOOR_TEXTURE_TILE_M, y0 / FLOOR_TEXTURE_TILE_M
    u1, v1 = x1 / FLOOR_TEXTURE_TILE_M, y1 / FLOOR_TEXTURE_TILE_M
    top_uv = [
        Gf.Vec2f(u0, v0), Gf.Vec2f(u1, v0),
        Gf.Vec2f(u1, v1), Gf.Vec2f(u0, v1),
    ]
    side_uv = [
        Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0),
    ]
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying,
    )
    st.Set([*top_uv, *top_uv, *side_uv, *side_uv, *side_uv, *side_uv])
    prim = mesh.GetPrim()
    prim.CreateAttribute(
        "real2sim:floorUvSchema", Sdf.ValueTypeNames.String,
    ).Set(FLOOR_UV_SCHEMA)
    prim.CreateAttribute(
        "real2sim:floorTextureTileM", Sdf.ValueTypeNames.Double,
    ).Set(FLOOR_TEXTURE_TILE_M)
    _bind(stage, prim, material)
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    return prim


def _catalog_asset_path(relative: str, asset_root: Path | str) -> Path:
    root = Path(asset_root).resolve()
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"topology module asset escapes its asset root: {relative}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"catalog-bound topology module asset is missing: {path}"
        )
    return path


def _materialize_declared_module_supports(
    stage: Any,
    modules: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Author exact-size counter supports in each module's canonical frame."""

    from pxr import Gf, Sdf, UsdGeom

    rows: list[dict[str, Any]] = []
    for module_id, module in sorted(modules.items()):
        spec = module.get("support_materialization")
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"{module_id} support_materialization must be an object")
        root_path = str(spec["root_prim_path"])
        support_path = str(spec["support_prim_path"])
        if root_path not in tuple(map(str, module.get("source_prim_roots", ()))):
            raise ValueError(
                f"{module_id} generated support root is not a module source root"
            )
        if not support_path.startswith(root_path.rstrip("/") + "/"):
            raise ValueError(f"{module_id} support must be below its root")
        if stage.GetPrimAtPath(root_path) or stage.GetPrimAtPath(support_path):
            raise RuntimeError(f"{module_id} generated support already exists")
        size = tuple(map(float, spec["size_xyz_m"]))
        if len(size) != 3 or any(value <= 0.0 for value in size):
            raise ValueError(f"{module_id} support size must be three positive values")
        pivot_x, pivot_y = map(float, module["canonical_pivot_xy_m"])
        root = UsdGeom.Xform.Define(stage, root_path)
        UsdGeom.Xformable(root).AddTranslateOp(opSuffix="canonicalPivot").Set(
            Gf.Vec3d(pivot_x, pivot_y, 0.0)
        )
        root.GetPrim().CreateAttribute(
            "real2sim:moduleId", Sdf.ValueTypeNames.String
        ).Set(str(module_id))
        support = _cube(
            stage,
            support_path,
            center=(0.0, 0.0, size[2] / 2.0),
            size=size,
            material="/World/Background/Shell/WorktopMat",
            collision=True,
        )
        support.CreateAttribute(
            "real2sim:generatedModuleSupport", Sdf.ValueTypeNames.Bool
        ).Set(True)
        support.CreateAttribute(
            "real2sim:moduleId", Sdf.ValueTypeNames.String
        ).Set(str(module_id))
        rows.append({
            "module_id": str(module_id),
            "root_prim_path": root_path,
            "support_prim_path": support_path,
            "size_xyz_m": list(size),
            "canonical_pivot_xy_m": [pivot_x, pivot_y],
            "dimensions_source": "catalog_exact_generated_geometry",
        })
    return rows


def _materialize_declared_module_assets(
    stage: Any,
    modules: Mapping[str, Mapping[str, Any]],
    *,
    scene_path: Path,
    asset_root: Path | str,
) -> list[dict[str, Any]]:
    """Create catalog-bound asset instances before the SE(2) overlay.

    A declared asset is a hard dependency: if it cannot be resolved, authoring
    fails instead of silently replacing it with a proxy.  This distinction is
    important for the selected Chefman air-fryer asset, whose USD exists in the
    current repository asset root even though it is not part of the baseline
    stage until a task/session layer composes it.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    rows: list[dict[str, Any]] = []
    for module_id, module in sorted(modules.items()):
        spec = module.get("asset_materialization")
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"{module_id} asset_materialization must be an object")
        relative_asset = str(spec["reference_path"])
        asset = _catalog_asset_path(relative_asset, asset_root)
        asset_identifier = os.path.relpath(asset, scene_path.parent)
        asset_stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadNone)
        expected_default = str(spec["reference_default_prim"])
        if (
            asset_stage is None
            or not asset_stage.GetDefaultPrim()
            or str(asset_stage.GetDefaultPrim().GetPath()) != expected_default
        ):
            raise ValueError(
                f"{module_id} asset default prim is not {expected_default}: {asset}"
            )

        source_roots = tuple(map(str, module.get("source_prim_roots", ())))
        module_root_path = str(spec["module_root_path"])
        if module_root_path not in source_roots:
            raise ValueError(
                f"{module_id} materialized root is not a module source root"
            )
        placement_path = str(spec["placement_prim_path"])
        instance_path = str(spec["instance_prim_path"])
        support_path = str(spec["support_prim_path"])
        root_prefix = module_root_path.rstrip("/") + "/"
        if not all(
            path.startswith(root_prefix)
            for path in (placement_path, instance_path, support_path)
        ):
            raise ValueError(
                f"{module_id} materialized prims must be children of {module_root_path}"
            )
        if (
            stage.GetPrimAtPath(placement_path)
            or stage.GetPrimAtPath(instance_path)
            or stage.GetPrimAtPath(support_path)
        ):
            raise RuntimeError(
                f"{module_id} materialized prim already exists in the input scene"
            )

        pivot_x, pivot_y = map(float, module["canonical_pivot_xy_m"])
        root = UsdGeom.Xform.Define(stage, module_root_path)
        root_xform = UsdGeom.Xformable(root)
        root_xform.AddTranslateOp(opSuffix="canonicalPivot").Set(
            Gf.Vec3d(pivot_x, pivot_y, 0.0)
        )
        root.GetPrim().CreateAttribute(
            "real2sim:moduleId", Sdf.ValueTypeNames.String
        ).Set(str(module_id))
        root.GetPrim().CreateAttribute(
            "real2sim:materializedAsset", Sdf.ValueTypeNames.Bool
        ).Set(True)

        offset = tuple(map(float, spec["asset_local_translate_m"]))
        if len(offset) != 3:
            raise ValueError(f"{module_id} asset_local_translate_m must have 3 values")
        placement = UsdGeom.Xform.Define(stage, placement_path)
        placement_xform = UsdGeom.Xformable(placement)
        placement_xform.AddTranslateOp(opSuffix="supportPlacement").Set(
            Gf.Vec3d(*offset)
        )
        placement_xform.AddRotateZOp(opSuffix="interactionFaceAlignment").Set(
            float(spec["asset_local_yaw_deg"])
        )
        scale = tuple(map(float, spec.get("asset_local_scale_xyz", (1.0, 1.0, 1.0))))
        if len(scale) != 3 or any(value <= 0.0 for value in scale):
            raise ValueError(f"{module_id} asset_local_scale_xyz must be positive")
        placement_xform.AddScaleOp(opSuffix="assetScale").Set(
            Gf.Vec3f(*scale)
        )

        instance = UsdGeom.Xform.Define(stage, instance_path)
        if not instance.GetPrim().GetReferences().AddReference(asset_identifier):
            raise RuntimeError(f"could not reference {asset} at {instance_path}")
        bbox_center = tuple(map(float, spec["asset_bbox_center_xy_m"]))
        if len(bbox_center) != 2:
            raise ValueError(f"{module_id} asset_bbox_center_xy_m must have 2 values")
        UsdGeom.Xformable(instance).AddTranslateOp(opSuffix="bboxCentering").Set(
            Gf.Vec3d(-bbox_center[0], -bbox_center[1], 0.0)
        )
        instance.GetPrim().CreateAttribute(
            "real2sim:selectedAsset", Sdf.ValueTypeNames.String
        ).Set(str(spec["selected_asset"]))
        instance.GetPrim().CreateAttribute(
            "real2sim:assetReference", Sdf.ValueTypeNames.Asset
        ).Set(Sdf.AssetPath(asset_identifier))
        disabled_rigid_bodies = []
        disabled_collisions = []
        disabled_joints = []
        for descendant in Usd.PrimRange(instance.GetPrim()):
            if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(descendant).CreateRigidBodyEnabledAttr(
                    False
                )
                disabled_rigid_bodies.append(str(descendant.GetPath()))
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(
                    False
                )
                disabled_collisions.append(str(descendant.GetPath()))
            joint = UsdPhysics.Joint(descendant)
            if joint:
                joint.CreateJointEnabledAttr(False)
                disabled_joints.append(str(descendant.GetPath()))
        instance.GetPrim().CreateAttribute(
            "real2sim:renderMaterializationPhysicsDisabled",
            Sdf.ValueTypeNames.Bool,
        ).Set(True)

        support_size = tuple(map(float, spec["support_size_xyz_m"]))
        if len(support_size) != 3 or any(value <= 0 for value in support_size):
            raise ValueError(f"{module_id} support_size_xyz_m must be 3 positive values")
        support = _cube(
            stage,
            support_path,
            center=(0.0, 0.0, support_size[2] / 2.0),
            size=support_size,
            material="/World/Background/Shell/WorktopMat",
            collision=True,
        )
        support.CreateAttribute(
            "real2sim:generatedModuleSupport", Sdf.ValueTypeNames.Bool
        ).Set(True)
        support.CreateAttribute(
            "real2sim:moduleId", Sdf.ValueTypeNames.String
        ).Set(str(module_id))
        rows.append({
            "module_id": str(module_id),
            "module_root": module_root_path,
            "placement_prim_path": placement_path,
            "instance_prim_path": instance_path,
            "support_prim_path": support_path,
            "canonical_asset_reference": str(
                (REPO / relative_asset).resolve()
            ),
            "usd_asset_reference": asset_identifier,
            "asset_default_prim": expected_default,
            "selected_asset": str(spec["selected_asset"]),
            "asset_local_yaw_deg": float(spec["asset_local_yaw_deg"]),
            "asset_local_scale_xyz": list(scale),
            "selection_evidence": str(spec["selection_evidence"]),
            "task_variant_policy": str(spec["task_variant_policy"]),
            "render_materialization_physics": str(
                spec["render_materialization_physics"]
            ),
            "disabled_rigid_body_paths": disabled_rigid_bodies,
            "disabled_collision_paths": disabled_collisions,
            "disabled_joint_paths": disabled_joints,
        })
    return rows


def _audit_composed_module_footprints(
    stage: Any,
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """Fail closed when composed furniture geometry crosses its room walls."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    spaces = {
        str(row["id"]): tuple(map(float, row["rect_xywh_m"]))
        for row in candidate["spaces"]
    }
    rows = []
    violations = []
    for module_id, module in sorted(modules.items()):
        pose = candidate["module_poses"][module_id]
        room = spaces[str(pose["space_id"])]
        observed_ranges = []
        measured_roots = []
        for path in map(str, module["source_prim_roots"]):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = tuple(map(float, aligned.GetMin()))
            maximum = tuple(map(float, aligned.GetMax()))
            if (
                any(abs(value) >= 1.0e20 for value in minimum + maximum)
                or any(minimum[index] > maximum[index] for index in range(3))
            ):
                continue
            observed_ranges.append((minimum, maximum))
            measured_roots.append(path)
        if not observed_ranges:
            rows.append({
                "module_id": module_id,
                "measured": False,
                "reason": "no_active_composed_geometry",
            })
            continue
        observed = (
            min(row[0][0] for row in observed_ranges),
            min(row[0][1] for row in observed_ranges),
            max(row[1][0] for row in observed_ranges),
            max(row[1][1] for row in observed_ranges),
        )
        expected_rect = house_topology._fixture_rect(module, pose)
        expected = (
            expected_rect[0], expected_rect[1],
            expected_rect[0] + expected_rect[2],
            expected_rect[1] + expected_rect[3],
        )
        footprint_tolerance = 2.5e-3
        within_declared = bool(
            observed[0] >= expected[0] - footprint_tolerance
            and observed[1] >= expected[1] - footprint_tolerance
            and observed[2] <= expected[2] + footprint_tolerance
            and observed[3] <= expected[3] + footprint_tolerance
        )
        room_min_x = room[0] + house_topology.ROOM_INTERIOR_INSET_M
        room_min_y = room[1] + house_topology.ROOM_INTERIOR_INSET_M
        room_max_x = room[0] + room[2] - house_topology.ROOM_INTERIOR_INSET_M
        room_max_y = room[1] + room[3] - house_topology.ROOM_INTERIOR_INSET_M
        within_room = bool(
            observed[0] >= room_min_x - footprint_tolerance
            and observed[1] >= room_min_y - footprint_tolerance
            and observed[2] <= room_max_x + footprint_tolerance
            and observed[3] <= room_max_y + footprint_tolerance
        )
        row = {
            "module_id": module_id,
            "measured": True,
            "measured_prim_roots": measured_roots,
            "observed_bbox_xyxy_m": [round(value, 6) for value in observed],
            "declared_bbox_xyxy_m": [round(value, 6) for value in expected],
            "within_declared_exact_footprint": within_declared,
            "within_room_interior": within_room,
        }
        rows.append(row)
        if not within_declared or not within_room:
            violations.append(row)
    report = {
        "schema": "real2sim.house_topology_composed_footprint_audit.v1",
        "passed": not violations,
        "wall_thickness_m": house_topology.WALL_THICKNESS_M,
        "furniture_to_wall_gap_m": house_topology.FURNITURE_TO_WALL_GAP_M,
        "measured_module_count": sum(bool(row["measured"]) for row in rows),
        "violation_count": len(violations),
        "rows": rows,
    }
    if violations and raise_on_violation:
        failed = ", ".join(row["module_id"] for row in violations)
        raise RuntimeError(f"composed furniture footprint audit failed: {failed}")
    return report


def apply_topology_overlay(
    scene: Path | str,
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    *,
    task_frame: str | None = None,
    robot_path: str | None = None,
    extra_prim_frames: Mapping[str, str] | None = None,
    asset_root: Path | str = REPO,
    physics_overlay: bool = False,
) -> dict[str, Any]:
    """Author one candidate shell and available module transforms."""

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    scene_path = Path(scene).resolve()
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open topology scene {scene_path}")
    if stage.GetPrimAtPath(ROOT_PATH):
        raise RuntimeError(f"scene already contains {ROOT_PATH}")
    materialized_supports = _materialize_declared_module_supports(
        stage, modules)
    materialized_assets = _materialize_declared_module_assets(
        stage,
        modules,
        scene_path=scene_path,
        asset_root=asset_root,
    )
    if (materialized_supports or materialized_assets) and not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save materialized topology assets {scene_path}")
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen topology assets in {scene_path}")
    all_paths = {str(prim.GetPath()) for prim in stage.TraverseAll()}
    layout = topology_layout(candidate, modules, available_prim_paths=all_paths)
    missing_modules = {
        module_id: [
            path for path in module["source_prim_roots"] if path not in all_paths
        ]
        for module_id, module in modules.items()
        if not any(path in all_paths for path in (
            *module["source_prim_roots"], *VISUAL_FALLBACK_ROOTS.get(module_id, ()),
        ))
    }
    moved = house_layout.apply_scene_overlay(
        scene_path,
        layout,
        frame=task_frame or "world",
        robot_path=robot_path,
        extra_prim_frames=extra_prim_frames,
    )
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen topology scene {scene_path}")
    footprint_audit = _audit_composed_module_footprints(
        stage, candidate, modules, raise_on_violation=False)
    if not footprint_audit["passed"]:
        violations = [
            row for row in footprint_audit["rows"]
            if row.get("measured")
            and (
                row.get("within_declared_exact_footprint") is not True
                or row.get("within_room_interior") is not True
            )
        ]
        raise RuntimeError(
            "composed furniture footprint audit failed: "
            + json.dumps(violations, ensure_ascii=False, sort_keys=True)
        )
    root = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    root.CreateAttribute("real2sim:topologyId", Sdf.ValueTypeNames.String).Set(str(candidate["id"]))
    root.CreateAttribute("real2sim:candidateStatus", Sdf.ValueTypeNames.String).Set(house_topology.STATUS)
    hidden = []
    disabled_canonical_collisions: list[str] = []

    def hide_imageables(prim: Any) -> None:
        for descendant in Usd.PrimRange(prim):
            if not descendant.IsA(UsdGeom.Imageable):
                continue
            UsdGeom.Imageable(descendant).CreateVisibilityAttr().Set("invisible")
            hidden.append(str(descendant.GetPath()))

    def disable_collisions(prim: Any) -> None:
        if not physics_overlay:
            return
        for descendant in Usd.PrimRange(prim):
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(
                False)
            disabled_canonical_collisions.append(
                str(descendant.GetPath()))

    shell = stage.GetPrimAtPath("/World/Background/Shell")
    movable_shell = {
        path for paths in layout.cluster_prim_roots.values() for path in paths
        if path.startswith("/World/Background/Shell/")
    }
    if shell:
        for child in shell.GetChildren():
            path = str(child.GetPath())
            if child.GetTypeName() in {"Cube", "Mesh", "Xform"} and path not in movable_shell:
                UsdGeom.Imageable(child).CreateVisibilityAttr().Set("invisible")
                hidden.append(path)
                disable_collisions(child)
    for path in HIDDEN_CANONICAL_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if prim:
            hide_imageables(prim)
            disable_collisions(prim)
    old_lights = stage.GetPrimAtPath("/World/Background/Lights")
    if old_lights:
        hide_imageables(old_lights)

    slabs = floor_slabs(candidate)
    for index, (x, y, width, height) in enumerate(slabs):
        _floor_slab(
            stage, f"{ROOT_PATH}/Shell/Floors/Floor_{index:03d}",
            x=x, y=y, width=width, height=height,
            material="/World/Background/Shell/FloorMat",
        )
        _cube(
            stage, f"{ROOT_PATH}/Shell/Ceilings/Ceiling_{index:03d}",
            center=(x + width / 2.0, y + height / 2.0, WALL_HEIGHT_M + WALL_THICKNESS_M / 2.0),
            size=(width, height, WALL_THICKNESS_M),
            material="/World/Background/Shell/CeilMat", collision=True,
        )
    walls = exterior_wall_segments(candidate)
    for index, (axis, fixed, start, end) in enumerate(walls):
        if axis == "x":
            center = ((start + end) / 2.0, fixed, WALL_HEIGHT_M / 2.0)
            size = (end - start, WALL_THICKNESS_M, WALL_HEIGHT_M)
        else:
            center = (fixed, (start + end) / 2.0, WALL_HEIGHT_M / 2.0)
            size = (WALL_THICKNESS_M, end - start, WALL_HEIGHT_M)
        _cube(
            stage, f"{ROOT_PATH}/Shell/Walls/Wall_{index:03d}",
            center=center, size=size,
            material="/World/Background/Shell/WallMat", collision=True,
        )

    proxy_rows = []
    for module_id in sorted(missing_modules):
        module = modules[module_id]
        pose = candidate["module_poses"][module_id]
        width, depth = map(float, module["fixture_size_xy_m"])
        proxy = _cube(
            stage, f"{ROOT_PATH}/RenderOnlyModuleProxies/{module_id}",
            center=(*map(float, pose["center_xy_m"]), 0.45),
            size=(width, depth, 0.9),
            material="/World/Background/Shell/WorktopMat", collision=False,
        )
        proxy.CreateAttribute("real2sim:renderOnlyProxy", Sdf.ValueTypeNames.Bool).Set(True)
        proxy.CreateAttribute("real2sim:moduleId", Sdf.ValueTypeNames.String).Set(module_id)
        proxy_rows.append({"module_id": module_id, "prim_path": str(proxy.GetPath())})
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save topology scene {scene_path}")
    reopened = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadNone)
    if reopened is None or not reopened.GetPrimAtPath(ROOT_PATH):
        raise RuntimeError("topology scene failed author/save/reopen")
    return {
        "schema": OVERLAY_SCHEMA,
        "ok": True,
        "scene": str(scene_path),
        "topology_id": str(candidate["id"]),
        "topology_ordinal": int(candidate["ordinal"]),
        "candidate_status": house_topology.STATUS,
        "floor_slab_count": len(slabs),
        "generated_floor_uv_schema": FLOOR_UV_SCHEMA,
        "generated_floor_texture_tile_m": FLOOR_TEXTURE_TILE_M,
        "exterior_wall_count": len(walls),
        "module_scene_overlay": moved,
        "composed_module_footprint_audit": footprint_audit,
        "materialized_module_supports": materialized_supports,
        "materialized_module_assets": materialized_assets,
        "render_only_module_proxies": proxy_rows,
        "hidden_canonical_prims": sorted(set(hidden)),
        "physics_overlay_mode": bool(physics_overlay),
        "disabled_canonical_collision_paths": sorted(set(
            disabled_canonical_collisions)),
        "usd_author_save_reopen": "passed",
        "live_navigation_validated": False,
        "live_collision_validated": False,
        "physics_replay_validated": False,
    }


def _visible_world_bbox_xy(cache: Any, prim: Any) -> tuple[float, float, float, float] | None:
    """Return one active, visible prim's finite world-space XY bounds."""

    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = tuple(map(float, aligned.GetMin()))
    maximum = tuple(map(float, aligned.GetMax()))
    if (
        any(abs(value) >= 1.0e20 for value in minimum + maximum)
        or any(minimum[index] > maximum[index] for index in range(3))
    ):
        return None
    return minimum[0], minimum[1], maximum[0], maximum[1]


def _bbox_center_xy(bounds: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bounds[0]) + float(bounds[2])) / 2.0,
        (float(bounds[1]) + float(bounds[3])) / 2.0,
    )


def _fit_rotated_bbox_center_in_rect(
    bounds: Sequence[float], target: Sequence[float], yaw_deg: float,
    rect_xywh_m: Sequence[float], *, inset_m: float = 0.02,
) -> tuple[float, float]:
    """Move a render-only object's center just far enough inside one room.

    The artwork variants have different depths and widths.  Centering every
    variant directly on a wall-mounted TV can therefore leave the thicker
    variants partly outside a generated floor slab.  Compute the conservative
    rotated AABB about its current visible-geometry center, then clamp only the
    requested center; the asset geometry and selected appearance stay intact.
    """

    center_x, center_y = _bbox_center_xy(bounds)
    radians = math.radians(float(yaw_deg))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    offsets = []
    for x in (float(bounds[0]), float(bounds[2])):
        for y in (float(bounds[1]), float(bounds[3])):
            dx = x - center_x
            dy = y - center_y
            offsets.append((cosine * dx - sine * dy, sine * dx + cosine * dy))
    minimum_dx = min(point[0] for point in offsets)
    maximum_dx = max(point[0] for point in offsets)
    minimum_dy = min(point[1] for point in offsets)
    maximum_dy = max(point[1] for point in offsets)
    room_x, room_y, room_width, room_height = map(float, rect_xywh_m)
    minimum_x = room_x + inset_m - minimum_dx
    maximum_x = room_x + room_width - inset_m - maximum_dx
    minimum_y = room_y + inset_m - minimum_dy
    maximum_y = room_y + room_height - inset_m - maximum_dy
    if minimum_x > maximum_x or minimum_y > maximum_y:
        raise RuntimeError("decor geometry is larger than its generated room")
    return (
        min(max(float(target[0]), minimum_x), maximum_x),
        min(max(float(target[1]), minimum_y), maximum_y),
    )


def _living_rect_for_point(
    candidate: Mapping[str, Any], point: Sequence[float],
) -> Sequence[float]:
    matches = []
    for space in candidate["spaces"]:
        if space["kind"] != "living":
            continue
        x, y, width, height = map(float, space["rect_xywh_m"])
        if x <= float(point[0]) <= x + width and y <= float(point[1]) <= y + height:
            matches.append(space["rect_xywh_m"])
    if not matches:
        raise RuntimeError("TV artwork target is outside every generated living room")
    return min(matches, key=lambda row: float(row[2]) * float(row[3]))


def apply_visual_plan(
    scene: Path | str,
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Relocate decor and author topology room lights after appearance v3."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    if plan.get("schema") != SCHEMA or plan.get("topology_id") != candidate["id"]:
        raise ValueError("topology visual plan identity mismatch")
    scene_path = Path(scene).resolve()
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetPrimAtPath(ROOT_PATH):
        raise RuntimeError("topology overlay must be authored before appearance")

    decor_transforms: dict[str, house_layout.SE2] = {}
    decor_roots: dict[str, tuple[str, ...]] = {}
    decor_targets: dict[str, tuple[float, float]] = {}
    decor_target_kinds: dict[str, str] = {}
    source_bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=False,
    )
    for slot, placement in plan["decor_slots"].items():
        roots = tuple(path for path in DECOR_SLOT_ROOTS[slot] if stage.GetPrimAtPath(path))
        target = tuple(map(float, placement["xy_m"]))
        for root_index, path in enumerate(roots):
            prim = stage.GetPrimAtPath(path)
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            root_source = world.ExtractTranslation()
            source_bounds = _visible_world_bbox_xy(source_bbox_cache, prim)
            if source_bounds is None:
                source = (float(root_source[0]), float(root_source[1]))
                target_kind = "root_origin_no_visible_geometry"
            else:
                source = _bbox_center_xy(source_bounds)
                target_kind = "visible_geometry_bbox_center"
            transform_id = f"{slot}_{root_index:02d}"
            decor_transforms[transform_id] = house_layout.SE2(
                source,
                (target[0] - source[0], target[1] - source[1]),
                float(placement["yaw_deg"]),
            )
            decor_roots[transform_id] = (path,)
            decor_targets[transform_id] = target
            decor_target_kinds[transform_id] = target_kind
    artwork = "/World/Assets/FireplaceArtwork"
    if stage.GetPrimAtPath(artwork):
        tv = candidate["module_poses"]["tv_cluster"]
        prim = stage.GetPrimAtPath(artwork)
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        root_source = world.ExtractTranslation()
        source_bounds = _visible_world_bbox_xy(source_bbox_cache, prim)
        if source_bounds is None:
            source = (float(root_source[0]), float(root_source[1]))
            target_kind = "root_origin_no_visible_geometry"
        else:
            source = _bbox_center_xy(source_bounds)
            target_kind = "visible_geometry_bbox_center"
        target = tuple(map(float, tv["center_xy_m"]))
        if source_bounds is not None:
            target = _fit_rotated_bbox_center_in_rect(
                source_bounds,
                target,
                float(tv["yaw_deg"]),
                _living_rect_for_point(candidate, target),
            )
        decor_transforms["artwork"] = house_layout.SE2(
            source,
            (target[0] - source[0], target[1] - source[1]),
            float(tv["yaw_deg"]),
        )
        decor_roots["artwork"] = (artwork,)
        decor_targets["artwork"] = target
        decor_target_kinds["artwork"] = target_kind
    decor_layout = house_layout.HouseLayout(
        ordinal=int(candidate["ordinal"]),
        layout_id=f"{candidate['id']}_decor",
        status=house_topology.STATUS,
        transforms=decor_transforms,
        frame_aliases={"world": None},
        cluster_prim_roots=decor_roots,
    )
    if decor_roots:
        house_layout.apply_scene_overlay(
            scene_path, decor_layout, frame="world", robot_path=None)
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    readback_bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=False,
    )
    decor_readback = []
    footprint_rows = []
    footprint_violations = []
    slabs = floor_slabs(candidate)

    def inside_floor_union(point: tuple[float, float]) -> bool:
        tolerance = 1.0e-5
        return any(
            slab_x - tolerance <= point[0] <= slab_x + slab_width + tolerance
            and slab_y - tolerance <= point[1] <= slab_y + slab_height + tolerance
            for slab_x, slab_y, slab_width, slab_height in slabs
        )

    for transform_id, roots in decor_roots.items():
        prim = stage.GetPrimAtPath(roots[0])
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        observed_bounds = _visible_world_bbox_xy(readback_bbox_cache, prim)
        target_kind = decor_target_kinds[transform_id]
        if target_kind == "visible_geometry_bbox_center":
            if observed_bounds is None:
                raise RuntimeError(
                    f"decor root {roots[0]} lost its visible geometry after placement"
                )
            observed = _bbox_center_xy(observed_bounds)
        else:
            root_observed = world.ExtractTranslation()
            observed = (float(root_observed[0]), float(root_observed[1]))
        target = decor_targets[transform_id]
        error = math.dist(observed, target)
        if error > 1.0e-5:
            raise RuntimeError(
                f"decor geometry {roots[0]} missed its topology slot by {error} m")
        decor_readback.append({
            "prim_path": roots[0],
            "target_kind": target_kind,
            "target_xy_m": list(target),
            "observed_xy_m": list(observed),
            "error_m": error,
        })
        if observed_bounds is None:
            footprint_rows.append({
                "prim_path": roots[0],
                "measured": False,
                "reason": "no_visible_geometry",
            })
            continue
        check_points = [
            (observed_bounds[0], observed_bounds[1]),
            (observed_bounds[0], observed_bounds[3]),
            (observed_bounds[2], observed_bounds[1]),
            (observed_bounds[2], observed_bounds[3]),
            _bbox_center_xy(observed_bounds),
        ]
        within_floor = all(inside_floor_union(point) for point in check_points)
        row = {
            "prim_path": roots[0],
            "measured": True,
            "world_bbox_xyxy_m": list(observed_bounds),
            "within_floor_union": within_floor,
        }
        footprint_rows.append(row)
        if not within_floor:
            footprint_violations.append(row)
    if footprint_violations:
        paths = ", ".join(row["prim_path"] for row in footprint_violations)
        raise RuntimeError(f"decor geometry crosses the generated floor: {paths}")
    lights = []
    for index, row in enumerate(plan["room_lights"]):
        light = UsdLux.RectLight.Define(
            stage, f"{ROOT_PATH}/Lights/Room_{index:02d}_{row['kind']}")
        light.CreateIntensityAttr(float(row["intensity"]))
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(row["color_temperature_k"]))
        light.CreateWidthAttr(float(row["size_xy_m"][0]))
        light.CreateHeightAttr(float(row["size_xy_m"][1]))
        xform = UsdGeom.Xformable(light)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, row["xyz_m"])))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        light.GetPrim().CreateAttribute(
            "real2sim:topologySpaceId", Sdf.ValueTypeNames.String
        ).Set(str(row["space_id"]))
        lights.append(str(light.GetPath()))
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT_PATH}/Lights/Ambient")
    dome.CreateIntensityAttr(AMBIENT_LIGHT_INTENSITY)
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(float(plan["room_lights"][0]["color_temperature_k"]))
    lights.append(str(dome.GetPath()))
    try:
        from .visual_scene_geometry import visible_bounds
    except ImportError:
        from visual_scene_geometry import visible_bounds
    from physical_distractors import ensure_scene_distractor_collisions
    decor_collisions = ensure_scene_distractor_collisions(stage,
        [p for group in decor_roots.values() for p in group
         if visible_bounds(stage, p) is not None])
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save topology appearance {scene_path}")
    return {
        "schema": "real2sim.house_topology_visual_applied.v1",
        "topology_id": str(candidate["id"]),
        "appearance_style": str(plan["appearance_style"]),
        "lighting_scheme": str(plan["lighting_scheme"]),
        "decor_root_count": sum(map(len, decor_roots.values())),
        "decor_collision_geometry": decor_collisions,
        "requires_physics_replay": True,
        "decor_transform_readback": decor_readback,
        "maximum_decor_transform_error_m": max(
            (row["error_m"] for row in decor_readback), default=0.0),
        "decor_geometry_footprint_audit": {
            "passed": True,
            "measured_count": sum(row["measured"] for row in footprint_rows),
            "violation_count": 0,
            "rows": footprint_rows,
        },
        "decor_slots": dict(plan["decor_slots"]),
        "light_paths": lights,
        "room_light_intensities": [
            float(row["intensity"]) for row in plan["room_lights"]
        ],
        "minimum_decor_to_module_clearance_m": plan[
            "minimum_decor_to_module_clearance_m"],
        "physics_effect": "none",
    }


def smoke_all(
    baseline: Path | str,
    output: Path | str,
    *,
    catalog_path: Path | str | None = None,
    asset_root: Path | str = REPO,
    physics_overlay: bool = False,
) -> dict[str, Any]:
    """Author/save/reopen every overlay in a bounded candidate catalog."""

    from pxr import Sdf, Usd

    source = Path(baseline).resolve()
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    catalog = (
        house_topology.load_catalog()
        if catalog_path is None else load_overlay_catalog(catalog_path)
    )
    rows = []
    for candidate in catalog["candidates"]:
        scene = destination / f"{candidate['id']}.usda"
        layer = Sdf.Layer.CreateNew(str(scene))
        layer.subLayerPaths = [str(source)]
        if not layer.Save():
            raise RuntimeError(f"could not create smoke scene {scene}")
        applied = apply_topology_overlay(
            scene, candidate, catalog["interaction_modules"],
            asset_root=asset_root,
            physics_overlay=physics_overlay,
        )
        reopened = Usd.Stage.Open(str(scene), load=Usd.Stage.LoadNone)
        passed = bool(reopened and reopened.GetPrimAtPath(ROOT_PATH))
        rows.append({
            "topology_id": candidate["id"],
            "passed": passed,
            "floor_slab_count": applied["floor_slab_count"],
            "exterior_wall_count": applied["exterior_wall_count"],
            "render_only_proxy_count": len(applied["render_only_module_proxies"]),
            "composed_footprint_audit_passed": applied[
                "composed_module_footprint_audit"
            ]["passed"],
            "composed_footprint_measured_module_count": applied[
                "composed_module_footprint_audit"
            ]["measured_module_count"],
            "composed_footprint_violation_count": applied[
                "composed_module_footprint_audit"
            ]["violation_count"],
            "physics_overlay_mode": applied["physics_overlay_mode"],
            "disabled_canonical_collision_count": len(applied[
                "disabled_canonical_collision_paths"]),
        })
    return {
        "schema": "real2sim.house_topology_visual_smoke.v1",
        "candidate_status": str(catalog.get("status", house_topology.STATUS)),
        "catalog": (
            str(house_topology.DEFAULT_CATALOG_PATH.resolve())
            if catalog_path is None else str(Path(catalog_path).resolve())
        ),
        "evaluated_topologies": len(rows),
        "author_save_reopen_passed": sum(row["passed"] for row in rows),
        "all_passed": all(
            row["passed"]
            and row["composed_footprint_audit_passed"]
            and row["composed_footprint_violation_count"] == 0
            for row in rows
        ),
        "rows": rows,
        "live_collision": "not_run",
        "physics_replay": "not_run",
        "physics_overlay_authoring": bool(physics_overlay),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-all")
    smoke.add_argument("--baseline", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--catalog", type=Path)
    smoke.add_argument("--asset-root", type=Path, default=REPO)
    smoke.add_argument("--physics-overlay", action="store_true")
    apply = subparsers.add_parser("apply-overlay")
    apply.add_argument("--scene", type=Path, required=True)
    apply.add_argument("--topology-id", required=True)
    apply.add_argument(
        "--catalog", type=Path,
        help="optional final-delivery topology catalog used only by this canary",
    )
    apply.add_argument("--task-frame")
    apply.add_argument("--robot-path")
    apply.add_argument("--extra-prim-frames", type=Path)
    apply.add_argument("--asset-root", type=Path, default=REPO)
    apply.add_argument(
        "--physics-overlay", action="store_true",
        help=(
            "disable collisions on the hidden canonical shell so the newly "
            "generated collision shell can be exercised by an explicit canary"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "smoke-all":
        report = smoke_all(
            args.baseline,
            args.output,
            catalog_path=args.catalog,
            asset_root=args.asset_root,
            physics_overlay=args.physics_overlay,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["all_passed"] else 2
    if args.command == "apply-overlay":
        overlay_catalog = (
            None if args.catalog is None else load_overlay_catalog(args.catalog)
        )
        candidate, catalog = candidate_for_id(
            args.topology_id, overlay_catalog)
        extra_frames = None
        if args.extra_prim_frames is not None:
            extra_frames = json.loads(
                args.extra_prim_frames.read_text(encoding="utf-8"))
            if not isinstance(extra_frames, dict):
                raise ValueError("extra prim frames must be a JSON object")
        result = apply_topology_overlay(
            args.scene,
            candidate,
            catalog["interaction_modules"],
            task_frame=args.task_frame,
            robot_path=args.robot_path,
            extra_prim_frames=extra_frames,
            asset_root=args.asset_root,
            physics_overlay=args.physics_overlay,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["ok"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

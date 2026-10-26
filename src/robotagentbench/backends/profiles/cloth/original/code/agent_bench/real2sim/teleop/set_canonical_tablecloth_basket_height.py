#!/usr/bin/env python3
"""Install and validate the one canonical TableclothBasket asset.

The filename is retained because existing reproduce bundles pin it.  The old
implementation synchronized six placeholder cubes.  The current contract
instead installs the user-supplied ``indian_towel_basket.usdz`` as the single
render mesh *and* the single basket collision mesh, while retaining the
accepted riser and the exact previous world envelope.

Only the editable ``scene_room.usda`` is authored.  Rebuild the restore-only
baseline afterwards with ``build_scene_baseline.py --from-working``; never edit
the baseline directly.
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from tablecloth_basket_contract import (
    ASSET_MESH_RELATIVE_PATH,
    BASKET_COLLISION_PATH,
    BASKET_MODEL_PATH,
    BASKET_PATH,
    BASKET_RISER_PATH,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_DEMO = REPO / "arena_scene" / "v1_2" / "demo_sim"
DEFAULT_SCENE = SOURCE_DEMO / "scene_room.usda"

ROBOT_COLLISION_GROUP_PATH = "/World/CanonicalRobotCollisionGroup"
BASKET_COLLISION_GROUP_PATH = "/World/CanonicalTableclothBasketCollisionGroup"

ASSET_DIRECTORY_RELATIVE = Path("usdz/tablecloth_basket")
ASSET_USDZ_NAME = "indian_towel_basket.usdz"
ASSET_WRAPPER_NAME = "tablecloth_basket.usda"
ASSET_REFERENCE = f"./{ASSET_DIRECTORY_RELATIVE}/{ASSET_WRAPPER_NAME}"
ASSET_DEFAULT_PRIM = "/indian_towel_basket"
WRAPPER_DEFAULT_PRIM = "/TableclothBasketModel"
WRAPPER_GEOMETRY_PATH = f"{WRAPPER_DEFAULT_PRIM}/Geometry"
WRAPPER_COLLISION_PATH = (
    f"{WRAPPER_GEOMETRY_PATH}/{ASSET_MESH_RELATIVE_PATH}"
)
# The source asset is Y-up and numerically authored in centimetres.  USD layer
# composition does not rescale reference coordinates from metersPerUnit, so the
# wrapper explicitly rotates Y-up to Z-up and nonuniformly fits the accepted
# 398 x 318 x 190 mm basket envelope.  The source structure and these bounds
# are validated directly before any scene is changed.
SOURCE_MIN_XYZ = (
    -5.858242982779703,
    -0.1394683301908422,
    -4.5397322128811535,
)
SOURCE_MAX_XYZ = (
    5.858242355009696,
    4.306669663620314,
    4.5396945188990925,
)
BASKET_MODEL_SIZE_XYZ_M = (0.398, 0.318, 0.190)
BASKET_MODEL_SCALE_XYZ = (
    0.03396923126053196,
    0.04273371637688086,
    0.03502423769629873,
)
BASKET_MODEL_NORMALIZE_TRANSLATE_XYZ_M = (
    1.0662432267071844e-08,
    -6.601014937121264e-07,
    0.0059600000659326205,
)
BASKET_MODEL_ORIENT_WXYZ = (
    0.7071067811865476,
    0.7071067811865475,
    0.0,
    0.0,
)

# Preserve the accepted 2026-08-02/05 placement byte-for-byte at the basket
# root.  The model is placed where the old floor/walls began, above the same
# 260 mm riser.  The complete root therefore retains its previous world AABB.
BASKET_ROOT_TRANSLATE_XYZ_M = (0.0, 0.28091587972998144, 0.0)
BASKET_LOCAL_CENTER_XY_M = (-2.24, -1.156)
BASKET_WORLD_CENTER_XY_M = (-2.24, -0.8750841202700185)
BASKET_MODEL_BOTTOM_Z_M = 0.26
BASKET_MODEL_TRANSLATE_XYZ_M = (
    *BASKET_LOCAL_CENTER_XY_M,
    BASKET_MODEL_BOTTOM_Z_M,
)
BASKET_RIM_Z_M = 0.45
BASKET_EXPECTED_WORLD_MIN_M = (
    -2.439000000104308,
    -1.0340841203743265,
    0.0,
)
BASKET_EXPECTED_WORLD_MAX_M = (
    -2.0409999998956923,
    -0.7160841201657103,
    BASKET_RIM_Z_M,
)
RISER_SCALE_XYZ_M = (0.374, 0.294, 0.26)
RISER_TRANSLATE_XYZ_M = (-2.24, -1.156, 0.13)
LEGACY_GEOMETRY_NAMES = ("floor", "wxn", "wxp", "wyn", "wyp")
TOLERANCE_M = 2.0e-6

from managed_place_lock import (  # noqa: E402  (scene-root-scoped)
    PLACE_LOCK as _PLACE_LOCK,
    SCENE_TRANSACTION_LOCK as _SCENE_LOCK,
)

SCENE_LOCK = Path(_SCENE_LOCK)
PLACE_LOCK = Path(_PLACE_LOCK)


def _close(left: float, right: float, tolerance: float = TOLERANCE_M) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    )


def _vector(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(component) for component in value)


def _require_vector(
    actual: Sequence[float], expected: Sequence[float], *, label: str,
) -> None:
    if len(actual) != len(expected) or not all(
        _close(left, right) for left, right in zip(actual, expected)
    ):
        raise RuntimeError(
            f"{label} changed: actual={tuple(actual)}, expected={tuple(expected)}"
        )


def _bounds(
    stage: Usd.Stage, prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    result = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if result.IsEmpty():
        raise RuntimeError(f"{prim.GetPath()} has empty bounds")
    return (
        tuple(float(value) for value in result.GetMin()),
        tuple(float(value) for value in result.GetMax()),
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_export(stage: Usd.Stage, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        if not stage.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f"could not export {destination}")
        if destination.exists():
            temporary.chmod(destination.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_asset(source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"could not open basket USDZ: {source}")
    default = stage.GetDefaultPrim()
    if not default or str(default.GetPath()) != ASSET_DEFAULT_PRIM:
        raise RuntimeError(
            f"basket USDZ default prim changed: {default.GetPath() if default else None}"
        )
    meshes = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    ]
    if len(meshes) != 1:
        raise RuntimeError(
            f"basket USDZ must contain one render mesh, found {len(meshes)}"
        )
    expected_mesh = f"{ASSET_DEFAULT_PRIM}/{ASSET_MESH_RELATIVE_PATH}"
    if str(meshes[0].GetPath()) != expected_mesh:
        raise RuntimeError(
            f"basket mesh identity changed: {meshes[0].GetPath()}"
        )
    minimum, maximum = _bounds(stage, default)
    _require_vector(minimum, SOURCE_MIN_XYZ, label="basket source minimum")
    _require_vector(maximum, SOURCE_MAX_XYZ, label="basket source maximum")
    return {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "default_prim": str(default.GetPath()),
        "mesh_path": str(meshes[0].GetPath()),
        "mesh_count": 1,
        "bounds": {"min": list(minimum), "max": list(maximum)},
    }


def _write_wrapper(asset_directory: Path) -> Path:
    destination = asset_directory / ASSET_WRAPPER_NAME
    descriptor, name = tempfile.mkstemp(
        dir=asset_directory,
        prefix=f".{destination.name}.",
        suffix=".usda",
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        if stage is None:
            raise RuntimeError(f"could not create {temporary}")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        model = UsdGeom.Xform.Define(
            stage, WRAPPER_DEFAULT_PRIM
        ).GetPrim()
        stage.SetDefaultPrim(model)
        model.CreateAttribute(
            "real2sim:assetSchema", Sdf.ValueTypeNames.String, custom=True,
        ).Set("real2sim.tablecloth_basket_visual_physics.v1")
        model.CreateAttribute(
            "real2sim:visualCollisionPolicy",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("one source mesh used for both appearance and static physics")

        geometry = UsdGeom.Xform.Define(
            stage, WRAPPER_GEOMETRY_PATH
        ).GetPrim()
        geometry.GetReferences().AddReference(f"./{ASSET_USDZ_NAME}")
        xform = UsdGeom.Xformable(geometry)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(
            *BASKET_MODEL_NORMALIZE_TRANSLATE_XYZ_M
        ))
        xform.AddOrientOp().Set(Gf.Quatf(
            BASKET_MODEL_ORIENT_WXYZ[0],
            Gf.Vec3f(*BASKET_MODEL_ORIENT_WXYZ[1:]),
        ))
        xform.AddScaleOp().Set(Gf.Vec3f(*BASKET_MODEL_SCALE_XYZ))
        collision = stage.GetPrimAtPath(WRAPPER_COLLISION_PATH)
        if not collision or not collision.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"basket wrapper did not compose mesh {WRAPPER_COLLISION_PATH}"
            )
        UsdPhysics.CollisionAPI.Apply(
            collision
        ).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(
            collision
        ).CreateApproximationAttr().Set("none")
        stage.GetRootLayer().Save()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def install_asset(demo_root: Path, source: Path) -> dict[str, Any]:
    source_report = _validate_source_asset(source)
    directory = demo_root / ASSET_DIRECTORY_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    installed = directory / ASSET_USDZ_NAME
    _atomic_copy(source, installed)
    if not filecmp.cmp(source, installed, shallow=False):
        raise RuntimeError("installed basket USDZ differs from the source")
    wrapper = _write_wrapper(directory)
    wrapper_report = validate_wrapper(demo_root)
    return {
        "source": source_report,
        "installed_usdz": str(installed),
        "installed_usdz_bytes": installed.stat().st_size,
        "wrapper": str(wrapper),
        "wrapper_bytes": wrapper.stat().st_size,
        "wrapper_validation": wrapper_report,
    }


def validate_wrapper(demo_root: Path) -> dict[str, Any]:
    directory = demo_root / ASSET_DIRECTORY_RELATIVE
    installed = directory / ASSET_USDZ_NAME
    wrapper = directory / ASSET_WRAPPER_NAME
    if not installed.is_file():
        raise RuntimeError(f"canonical basket USDZ is missing: {installed}")
    stage = Usd.Stage.Open(str(wrapper))
    if stage is None:
        raise RuntimeError(f"could not open basket wrapper: {wrapper}")
    default = stage.GetDefaultPrim()
    if not default or str(default.GetPath()) != WRAPPER_DEFAULT_PRIM:
        raise RuntimeError("basket wrapper default prim changed")
    meshes = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    ]
    collisions = [
        prim
        for prim in Usd.PrimRange(default)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(
            prim
        ).GetCollisionEnabledAttr().Get() is not False
    ]
    if (
        len(meshes) != 1
        or len(collisions) != 1
        or meshes[0].GetPath() != collisions[0].GetPath()
        or str(meshes[0].GetPath()) != WRAPPER_COLLISION_PATH
    ):
        raise RuntimeError(
            "basket wrapper lost the one-mesh appearance/physics contract: "
            f"meshes={[str(item.GetPath()) for item in meshes]}, "
            f"collisions={[str(item.GetPath()) for item in collisions]}"
        )
    approximation = UsdPhysics.MeshCollisionAPI(
        collisions[0]
    ).GetApproximationAttr().Get()
    if approximation != "none":
        raise RuntimeError(
            f"basket collision must use the render mesh exactly, got {approximation}"
        )
    minimum, maximum = _bounds(stage, default)
    size = tuple(maximum[index] - minimum[index] for index in range(3))
    expected_min = (
        -BASKET_MODEL_SIZE_XYZ_M[0] / 2.0,
        -BASKET_MODEL_SIZE_XYZ_M[1] / 2.0,
        0.0,
    )
    expected_max = (
        BASKET_MODEL_SIZE_XYZ_M[0] / 2.0,
        BASKET_MODEL_SIZE_XYZ_M[1] / 2.0,
        BASKET_MODEL_SIZE_XYZ_M[2],
    )
    _require_vector(minimum, expected_min, label="normalized basket minimum")
    _require_vector(maximum, expected_max, label="normalized basket maximum")
    return {
        "path": str(wrapper),
        "bytes": wrapper.stat().st_size,
        "asset_path": str(installed),
        "asset_bytes": installed.stat().st_size,
        "bounds": {
            "min": list(minimum),
            "max": list(maximum),
            "size": list(size),
        },
        "render_mesh_paths": [str(meshes[0].GetPath())],
        "collision_mesh_paths": [str(collisions[0].GetPath())],
        "one_mesh_for_appearance_and_physics": True,
        "physics_approximation": str(approximation),
    }


def _configure_robot_filter(stage: Usd.Stage) -> None:
    expected = {
        ROBOT_COLLISION_GROUP_PATH: {
            "includes": ["/World/Robot"],
            "filtered": [],
        },
        BASKET_COLLISION_GROUP_PATH: {
            "includes": [BASKET_PATH],
            "filtered": [ROBOT_COLLISION_GROUP_PATH],
        },
    }
    for path, wanted in expected.items():
        group = UsdPhysics.CollisionGroup.Define(stage, path)
        collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
        collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
        collection.CreateIncludesRel().SetTargets([
            Sdf.Path(item) for item in wanted["includes"]
        ])
        group.CreateFilteredGroupsRel().SetTargets([
            Sdf.Path(item) for item in wanted["filtered"]
        ])


def _author_scene(stage: Usd.Stage, *, scene: Path) -> None:
    basket = stage.GetPrimAtPath(BASKET_PATH)
    if not basket or not basket.IsValid() or not basket.IsActive():
        raise RuntimeError(f"{scene}: missing active {BASKET_PATH}")
    root_translate = _vector(
        basket.GetAttribute("xformOp:translate").Get()
    )
    _require_vector(
        root_translate,
        BASKET_ROOT_TRANSLATE_XYZ_M,
        label=f"{scene} basket root translation",
    )

    riser = stage.GetPrimAtPath(BASKET_RISER_PATH)
    if not riser or not riser.IsA(UsdGeom.Cube):
        raise RuntimeError(f"{scene}: missing canonical basket riser")
    _require_vector(
        _vector(riser.GetAttribute("xformOp:scale").Get()),
        RISER_SCALE_XYZ_M,
        label=f"{scene} basket riser scale",
    )
    # Accept the pre-replacement local Y only as the known migration input.
    riser_translate = _vector(
        riser.GetAttribute("xformOp:translate").Get()
    )
    accepted_riser_translates = (
        RISER_TRANSLATE_XYZ_M,
        (-2.24, -1.065, 0.13),
    )
    if not any(
        len(riser_translate) == 3
        and all(_close(a, b) for a, b in zip(riser_translate, candidate))
        for candidate in accepted_riser_translates
    ):
        raise RuntimeError(
            f"{scene}: refusing unknown basket riser pose {riser_translate}"
        )
    riser.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(*RISER_TRANSLATE_XYZ_M)
    )
    UsdPhysics.CollisionAPI.Apply(riser).CreateCollisionEnabledAttr(True)

    for name in LEGACY_GEOMETRY_NAMES:
        stage.RemovePrim(f"{BASKET_PATH}/{name}")

    model = UsdGeom.Xform.Define(stage, BASKET_MODEL_PATH).GetPrim()
    references = model.GetReferences()
    references.ClearReferences()
    references.AddReference(ASSET_REFERENCE)
    xform = UsdGeom.Xformable(model)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*BASKET_MODEL_TRANSLATE_XYZ_M))
    model.CreateAttribute(
        "real2sim:assetSchema", Sdf.ValueTypeNames.String, custom=True,
    ).Set("real2sim.tablecloth_basket_visual_physics.v1")
    model.RemoveProperty("real2sim:source" + "Sha" + str(256))
    basket.CreateAttribute(
        "real2sim:canonicalBasketPolicy",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(
        "one visual/physics mesh above the retained riser; exact previous "
        "world size and pose"
    )
    _configure_robot_filter(stage)


def validate_scene(scene: Path, *, demo_root: Path) -> dict[str, Any]:
    validate_wrapper(demo_root)
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"could not open scene: {scene}")
    basket = stage.GetPrimAtPath(BASKET_PATH)
    if not basket or not basket.IsValid() or not basket.IsActive():
        raise RuntimeError(f"{scene}: missing active {BASKET_PATH}")
    for name in LEGACY_GEOMETRY_NAMES:
        if stage.GetPrimAtPath(f"{BASKET_PATH}/{name}").IsValid():
            raise RuntimeError(f"{scene}: legacy basket geometry remains: {name}")
    minimum, maximum = _bounds(stage, basket)
    _require_vector(
        minimum,
        BASKET_EXPECTED_WORLD_MIN_M,
        label=f"{scene} basket world minimum",
    )
    _require_vector(
        maximum,
        BASKET_EXPECTED_WORLD_MAX_M,
        label=f"{scene} basket world maximum",
    )
    collisions = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(basket)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(
            prim
        ).GetCollisionEnabledAttr().Get() is not False
    )
    expected_collisions = sorted((BASKET_RISER_PATH, BASKET_COLLISION_PATH))
    if collisions != expected_collisions:
        raise RuntimeError(
            f"{scene}: basket collision paths changed: {collisions}"
        )
    mesh_paths = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(basket)
        if prim.IsA(UsdGeom.Mesh)
    )
    if mesh_paths != [BASKET_COLLISION_PATH]:
        raise RuntimeError(
            f"{scene}: basket must compose one render mesh: {mesh_paths}"
        )

    def targets(path: str, relationship: str) -> list[str]:
        rel = stage.GetPrimAtPath(path).GetRelationship(relationship)
        return [str(item) for item in rel.GetTargets()] if rel else []

    robot_filter_ok = bool(
        targets(
            ROBOT_COLLISION_GROUP_PATH,
            "collection:colliders:includes",
        ) == ["/World/Robot"]
        and targets(
            ROBOT_COLLISION_GROUP_PATH,
            "physics:filteredGroups",
        ) == []
        and targets(
            BASKET_COLLISION_GROUP_PATH,
            "collection:colliders:includes",
        ) == [BASKET_PATH]
        and targets(
            BASKET_COLLISION_GROUP_PATH,
            "physics:filteredGroups",
        ) == [ROBOT_COLLISION_GROUP_PATH]
    )
    if not robot_filter_ok:
        raise RuntimeError(f"{scene}: robot/basket collision filter changed")
    return {
        "scene": str(scene.resolve()),
        "scene_bytes": scene.stat().st_size,
        "basket_world_bounds": {
            "min": list(minimum),
            "max": list(maximum),
        },
        "world_center_xy_m": list(BASKET_WORLD_CENTER_XY_M),
        "rim_z_m": BASKET_RIM_Z_M,
        "riser_height_m": RISER_SCALE_XYZ_M[2],
        "render_mesh_paths": mesh_paths,
        "collision_paths": collisions,
        "one_mesh_for_appearance_and_physics": True,
        "robot_basket_filter_only": robot_filter_ok,
    }


def update_scene(
    scene: Path, *, demo_root: Path | None = None, apply: bool,
) -> dict[str, Any]:
    root = (demo_root or scene.parent).resolve()
    scene = scene.resolve()
    if not apply:
        return validate_scene(scene, demo_root=root)
    if scene.name == "scene_room_baseline.usda":
        raise RuntimeError(
            "refusing to author the restore-only baseline; edit scene_room.usda "
            "and rebuild with build_scene_baseline.py --from-working"
        )
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"could not open scene: {scene}")
    _author_scene(stage, scene=scene)
    _atomic_export(stage, scene)
    return validate_scene(scene, demo_root=root)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--demo-root", type=Path, default=SOURCE_DEMO)
    parser.add_argument("--source-usdz", type=Path)
    parser.add_argument("--scene", action="append", type=Path, dest="scenes")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    demo_root = args.demo_root.resolve()
    scenes = tuple(
        path.resolve() for path in (args.scenes or [demo_root / "scene_room.usda"])
    )
    if args.apply and args.source_usdz is None:
        parser.error("--apply requires --source-usdz")
    if any(path.parent != demo_root for path in scenes):
        parser.error("every --scene must be directly below --demo-root")

    PLACE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PLACE_LOCK.open("a+", encoding="ascii") as place_stream:
        with SCENE_LOCK.open("a+", encoding="ascii") as scene_stream:
            fcntl.flock(place_stream, fcntl.LOCK_EX)
            fcntl.flock(scene_stream, fcntl.LOCK_EX)
            installed = (
                install_asset(demo_root, args.source_usdz.resolve())
                if args.apply else None
            )
            results = [
                update_scene(path, demo_root=demo_root, apply=args.apply)
                for path in scenes
            ]
            wrapper = validate_wrapper(demo_root)

    payload = {
        "schema": "real2sim.canonical_tablecloth_basket.v2",
        "status": "applied" if args.apply else "checked",
        "basket_path": BASKET_PATH,
        "asset_reference": ASSET_REFERENCE,
        "wrapper": wrapper,
        "installation": installed,
        "world_center_xy_m": list(BASKET_WORLD_CENTER_XY_M),
        "world_bounds_m": {
            "min": list(BASKET_EXPECTED_WORLD_MIN_M),
            "max": list(BASKET_EXPECTED_WORLD_MAX_M),
        },
        "riser_height_m": RISER_SCALE_XYZ_M[2],
        "one_mesh_for_appearance_and_physics": True,
        "legacy_placeholder_geometry_removed": list(LEGACY_GEOMETRY_NAMES),
        "scenes": results,
    }
    if args.report:
        _write_report(args.report.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

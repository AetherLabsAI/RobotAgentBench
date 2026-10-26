"""Compose L1's exact A11 USD layers without writing the downloaded House."""
import os
from pathlib import Path

from .l1_shelf_support_geometry import (
    BOARD, BRACKETS, BOARD_SUPPORT_PATHS, validate_shelf_supports,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "82cc53652ef38470106c00bdc952f276a08afa74"
LAYERS = tuple(ROOT / "resources/layout/l1_shelf_supports" / name for name in (
    "shelf_support_collisions.usda", "shelf_support_alignment.usda",
))


def apply(stage, *, policy=None):
    """Before obstacles/physics reset, or before preview prop-only protection."""
    from pxr import Sdf, UsdGeom
    from arena_api.l1_runtime import is_scene_output_directory
    target = stage.GetEditTarget().GetLayer()
    if target != stage.GetSessionLayer():
        if (target != stage.GetRootLayer() or not target.realPath
                or not is_scene_output_directory(Path(target.realPath).resolve().parent)):
            raise ValueError("Shelf fix only writes session or approved output layers")
    # Refuse a different rack; do not blindly apply vertex arrays to other assets.
    alignment = Sdf.Layer.FindOrOpen(str(LAYERS[1]))
    for path in (BOARD, BRACKETS):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError("Expected House shelf mesh: " + path)
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        source = alignment.GetAttributeAtPath(path + ".points").default
        if len(points) != len(source):
            raise ValueError("L1 shelf vertex count mismatch: " + path)
    for path in BOARD_SUPPORT_PATHS[1:3]:
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Cube):
            raise ValueError("Expected legacy shelf support: " + path)
    # A prepared whole-house scene already composes these layers BELOW its
    # layout transforms. Adding them to a stronger session would reset the
    # two world-root support Cubes to their old canonical XY positions.
    used = {Path(layer.realPath).resolve() for layer in stage.GetUsedLayers() if layer.realPath}
    if all(path.resolve() in used for path in LAYERS):
        report = validate_shelf_supports(stage, policy=policy)
        return dict(source_commit=SOURCE_COMMIT, source_assets_modified=False,
                    geometry_check=report, physics_replay=False,
                    retained_prepared_layout_transforms=True)
    paths = [os.path.relpath(p, Path(target.realPath).parent) if target.realPath else str(p)
             for p in LAYERS]
    target.subLayerPaths = paths + [p for p in target.subLayerPaths if p not in paths]
    report = validate_shelf_supports(stage, policy=policy)
    return dict(source_commit=SOURCE_COMMIT, source_assets_modified=False,
                geometry_check=report, physics_replay=False)

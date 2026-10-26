"""Current fixed dual-trash-bin layout for the CoffeeTable interaction area.

The real room keeps the teal destination and burgundy/red distractor together
in the narrow passage between the CoffeeTable and fireplace.  The red bin is
the +X (robot-side) member of the pair.  This module is the single source used
by every current initializer that renders those bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_positions import (
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M,
    translate_xy,
    translate_xyz,
)


DUAL_BIN_LAYOUT_SCHEMA = "real2sim.coffee_table_dual_bin_layout.v2"

# User-supplied outside dimensions.  The source asset needs this exact scale
# to measure 0.19 m in X, 0.24 m in Y, and 0.35 m in Z.
BIN_OUTER_DIMENSIONS_XYZ_M = (0.19, 0.24, 0.35)
BIN_SCALE_XYZ = (
    1.16471122,
    1.02925987,
    1.2068965391368551,
)
_AUTHORED_TEAL_ROOT_XYZ_M = (
    -1.97672664303,
    -1.05997630124,
    0.018021000856525946,
)
_AUTHORED_RED_ROOT_Z_M = 0.018021000856526283
_AUTHORED_COLLIDER_CENTER_XY_M = (-1.98, -1.06)

# Keep the pair beside the CoffeeTable, without treating its Y coordinate as
# authority for the refined TableclothBasket.  The basket keeps its accepted
# 2026-08-02 pose; the bins keep their independently reviewed row and the
# family-7 away-from-table offset below.  Only the X edge gap couples them.
BASKET_OUTER_MAX_X_M = -2.0410000000000004
BASKET_CENTER_Y_M = -0.8750841202700185
BINS_PRE_OFFSET_CENTER_Y_M = (
    -1.065 + COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M[1]
)
BASKET_TO_BIN_EDGE_GAP_M = 0.010
# 2026-08-09 user-confirmed House v1.2 furniture change: translate the whole
# bin pair exactly 30 mm away from the CoffeeTable (-Y).  This is shared global
# furniture state, not a task-local release-tuning offset.  Every consumer,
# including Family 7 train/eval and the data producer, inherits the same row.
BIN_AWAY_FROM_TABLE_OFFSET_Y_M = -0.030
TEAL_ADJACENT_CENTER_X_M = (
    BASKET_OUTER_MAX_X_M
    + BASKET_TO_BIN_EDGE_GAP_M
    + BIN_OUTER_DIMENSIONS_XYZ_M[0] / 2.0
)
BIN_OFFSET_X_M = (
    TEAL_ADJACENT_CENTER_X_M
    - _AUTHORED_TEAL_ROOT_XYZ_M[0]
    - COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0]
)
BIN_OFFSET_Y_M = (
    BINS_PRE_OFFSET_CENTER_Y_M
    + BIN_AWAY_FROM_TABLE_OFFSET_Y_M
    - _AUTHORED_TEAL_ROOT_XYZ_M[1]
    - COFFEE_TABLE_LAYOUT_SHIFT_XY_M[1]
)
BIN_CENTER_SPACING_X_M = 0.200
DISTRACTOR_ROBOT_SIDE_OFFSET_X_M = BIN_CENTER_SPACING_X_M

TEAL_BODY_PATH = "/World/Assets/TrashBinTeal"
RED_BODY_PATH = "/World/Assets/TrashBinRed"
TEAL_COLLIDER_PATH = f"{TEAL_BODY_PATH}/geo/Cube_022"
TEAL_BAG_COLLIDER_PATH = (
    f"{TEAL_BODY_PATH}/geo/Trash_Bag/Trash_Bag_002"
)
TEAL_BOTTOM_LOD_PATH = (
    f"{TEAL_BODY_PATH}/geo/Real2SimBottomCollisionLOD"
)
RED_COLLIDER_PATH = f"{RED_BODY_PATH}/geo/Cube_022"
RED_BAG_COLLIDER_PATH = (
    f"{RED_BODY_PATH}/geo/Trash_Bag/Trash_Bag_002"
)
RED_BOTTOM_LOD_PATH = (
    f"{RED_BODY_PATH}/geo/Real2SimBottomCollisionLOD"
)

BOTTOM_LOD_SCHEMA = "task29.geometry_derived_bottom_lod.v1"
BOTTOM_LOD_THICKNESS_M = 0.006
BOTTOM_LOD_FACE_VERTEX_COUNT = 16

BIN_ORIENT_WXYZ = (0.00931206625, 0.0, 0.0, 0.99995666742)


def _shifted_xy(x: float, y: float) -> tuple[float, float]:
    return translate_xy(
        (x + BIN_OFFSET_X_M, y + BIN_OFFSET_Y_M),
        COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    )


TEAL_ROOT_TRANSLATE_XYZ_M = translate_xyz(
    (
        _AUTHORED_TEAL_ROOT_XYZ_M[0] + BIN_OFFSET_X_M,
        _AUTHORED_TEAL_ROOT_XYZ_M[1] + BIN_OFFSET_Y_M,
        _AUTHORED_TEAL_ROOT_XYZ_M[2],
    ),
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
)
RED_ROOT_TRANSLATE_XYZ_M = (
    TEAL_ROOT_TRANSLATE_XYZ_M[0] + DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
    TEAL_ROOT_TRANSLATE_XYZ_M[1],
    _AUTHORED_RED_ROOT_Z_M,
)
TEAL_CENTER_XY_M = TEAL_ROOT_TRANSLATE_XYZ_M[:2]
RED_CENTER_XY_M = RED_ROOT_TRANSLATE_XYZ_M[:2]
TEAL_COLLIDER_CENTER_XY_M = _shifted_xy(
    *_AUTHORED_COLLIDER_CENTER_XY_M)
RED_COLLIDER_CENTER_XY_M = (
    TEAL_COLLIDER_CENTER_XY_M[0]
    + DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
    TEAL_COLLIDER_CENTER_XY_M[1],
)


DUAL_BIN_LAYOUT = {
    "destination": {
        "name": "teal",
        "role": "fixed_fireplace_side_destination",
        "body_path": TEAL_BODY_PATH,
        "translate": TEAL_ROOT_TRANSLATE_XYZ_M,
        "orient_wxyz": BIN_ORIENT_WXYZ,
        "scale": BIN_SCALE_XYZ,
        "outer_dimensions_xyz_m": BIN_OUTER_DIMENSIONS_XYZ_M,
        "calibrated_center_xy": TEAL_COLLIDER_CENTER_XY_M,
        "center_tolerance_m": 0.001,
        "scoring_collider_path": TEAL_COLLIDER_PATH,
        "bottom_collision_lod": {
            "schema": BOTTOM_LOD_SCHEMA,
            "path": TEAL_BOTTOM_LOD_PATH,
            "source_collider_path": TEAL_COLLIDER_PATH,
            "source_face_vertex_count": BOTTOM_LOD_FACE_VERTEX_COUNT,
            "thickness_m": BOTTOM_LOD_THICKNESS_M,
        },
        "colliders": (
            (TEAL_COLLIDER_PATH, True),
            # The visible liner has a raised bottom. Its static triangle
            # collision surface must support objects at that same height.
            (TEAL_BAG_COLLIDER_PATH, True),
            (TEAL_BOTTOM_LOD_PATH, True),
        ),
    },
    "distractor": {
        "name": "red",
        "role": "fixed_adjacent_robot_side_distractor",
        "body_path": RED_BODY_PATH,
        "translate": RED_ROOT_TRANSLATE_XYZ_M,
        "orient_wxyz": BIN_ORIENT_WXYZ,
        "scale": BIN_SCALE_XYZ,
        "outer_dimensions_xyz_m": BIN_OUTER_DIMENSIONS_XYZ_M,
        "calibrated_center_xy": RED_COLLIDER_CENTER_XY_M,
        "center_tolerance_m": 0.001,
        "scoring_collider_path": RED_COLLIDER_PATH,
        "bottom_collision_lod": {
            "schema": BOTTOM_LOD_SCHEMA,
            "path": RED_BOTTOM_LOD_PATH,
            "source_collider_path": RED_COLLIDER_PATH,
            "source_face_vertex_count": BOTTOM_LOD_FACE_VERTEX_COUNT,
            "thickness_m": BOTTOM_LOD_THICKNESS_M,
        },
        "colliders": (
            (RED_COLLIDER_PATH, True),
            (RED_BAG_COLLIDER_PATH, True),
            (RED_BOTTOM_LOD_PATH, True),
        ),
    },
}


def validate_dual_bin_layout(layout=DUAL_BIN_LAYOUT) -> None:
    if not isinstance(layout, dict) or set(layout) != {
        "destination",
        "distractor",
    }:
        raise ValueError("dual-bin layout must contain destination/distractor")
    for key, expected_name in (
        ("destination", "teal"),
        ("distractor", "red"),
    ):
        entry = layout[key]
        if entry.get("name") != expected_name:
            raise ValueError(f"dual-bin {key} identity changed")
        if tuple(entry.get("scale", ())) != BIN_SCALE_XYZ:
            raise ValueError(f"dual-bin {key} scale changed")
        if (
            tuple(entry.get("outer_dimensions_xyz_m", ()))
            != BIN_OUTER_DIMENSIONS_XYZ_M
        ):
            raise ValueError(f"dual-bin {key} outside dimensions changed")
    dx = (
        float(layout["distractor"]["calibrated_center_xy"][0])
        - float(layout["destination"]["calibrated_center_xy"][0])
    )
    dy = (
        float(layout["distractor"]["calibrated_center_xy"][1])
        - float(layout["destination"]["calibrated_center_xy"][1])
    )
    if abs(dx - BIN_CENTER_SPACING_X_M) > 1.0e-9 or abs(dy) > 1.0e-9:
        raise ValueError("red bin is not adjacent on teal's robot-side +X")
    teal_x, teal_y = TEAL_CENTER_XY_M
    basket_gap = (
        teal_x
        - BIN_OUTER_DIMENSIONS_XYZ_M[0] / 2.0
        - BASKET_OUTER_MAX_X_M
    )
    if (
        abs(basket_gap - BASKET_TO_BIN_EDGE_GAP_M) > 1.0e-9
        or abs(
            teal_y
            - BINS_PRE_OFFSET_CENTER_Y_M
            - BIN_AWAY_FROM_TABLE_OFFSET_Y_M
        ) > 1.0e-9
    ):
        raise ValueError(
            "teal bin lost its independently reviewed row "
            "(pre-offset bin Y plus the away-from-table offset)")


validate_dual_bin_layout()


def validate_scene(scene: Path) -> dict[str, object]:
    """Fail closed when a runtime scene does not use this exact bin row."""
    from pxr import Usd, UsdGeom

    scene = scene.resolve()
    if not scene.is_file():
        raise RuntimeError(f"scene is missing: {scene}")
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"could not open scene: {scene}")
    bins: dict[str, object] = {}
    for name, path, expected in (
        ("teal", TEAL_BODY_PATH, TEAL_ROOT_TRANSLATE_XYZ_M),
        ("red", RED_BODY_PATH, RED_ROOT_TRANSLATE_XYZ_M),
    ):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsActive():
            raise RuntimeError(f"active {name} bin is missing: {path}")
        translations = [
            op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if len(translations) != 1:
            raise RuntimeError(
                f"{path} must have exactly one translate op, got "
                f"{[op.GetOpName() for op in translations]}"
            )
        actual = tuple(float(value) for value in translations[0].Get())
        if any(abs(left - right) > 1.0e-9
               for left, right in zip(actual, expected)):
            raise RuntimeError(
                f"{path} does not match House v1.2: "
                f"actual={actual}, expected={expected}"
            )
        bins[name] = {
            "path": path,
            "translate_xyz_m": list(actual),
            "expected_translate_xyz_m": list(expected),
            "matches": True,
        }
    return {
        "ok": True,
        "schema": DUAL_BIN_LAYOUT_SCHEMA,
        "scene": str(scene),
        "global_house_layout": True,
        "task_local_bin_offset": False,
        "bins": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_scene(args.scene)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "schema": DUAL_BIN_LAYOUT_SCHEMA,
            "scene": str(args.scene.resolve()),
            "reason": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


__all__ = [
    "BIN_CENTER_SPACING_X_M",
    "BIN_OFFSET_X_M",
    "BIN_OFFSET_Y_M",
    "BIN_ORIENT_WXYZ",
    "BIN_OUTER_DIMENSIONS_XYZ_M",
    "BIN_SCALE_XYZ",
    "BASKET_CENTER_Y_M",
    "BASKET_OUTER_MAX_X_M",
    "BASKET_TO_BIN_EDGE_GAP_M",
    "BINS_PRE_OFFSET_CENTER_Y_M",
    "BOTTOM_LOD_FACE_VERTEX_COUNT",
    "BOTTOM_LOD_SCHEMA",
    "BOTTOM_LOD_THICKNESS_M",
    "DISTRACTOR_ROBOT_SIDE_OFFSET_X_M",
    "DUAL_BIN_LAYOUT",
    "DUAL_BIN_LAYOUT_SCHEMA",
    "RED_BODY_PATH",
    "RED_BOTTOM_LOD_PATH",
    "RED_CENTER_XY_M",
    "RED_ROOT_TRANSLATE_XYZ_M",
    "TEAL_BODY_PATH",
    "TEAL_BOTTOM_LOD_PATH",
    "TEAL_CENTER_XY_M",
    "TEAL_COLLIDER_PATH",
    "TEAL_ROOT_TRANSLATE_XYZ_M",
    "validate_dual_bin_layout",
    "validate_scene",
]


if __name__ == "__main__":
    raise SystemExit(main())

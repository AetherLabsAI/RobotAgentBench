"""Fail-closed contract for family-7 tidy-table code-control production.

The user-facing labels deliberately distinguish real-data small-task IDs from
benchmark-family behaviors that have no robot-template small-task ID.
"""

from __future__ import annotations

from dataclasses import dataclass

from task29_full_reproduce import CLONE_PLACEMENTS, CUP_ROOT_PATHS
from family7_single_target_action import (
    FIRST_OBJECT_END_FRAME_EXCLUSIVE,
    RETURN_HOME_FRAMES,
    SINGLE_TARGET_FRAMES,
)


SCHEMA = "real2sim.family7_code_control_contract.v3"
SOURCE_EPISODE_UID = "06ab4bb5-83ea-499a-8310-611117220838"
SOURCE_FULL_FRAMES = 471
E2E_END_FRAME_EXCLUSIVE = SINGLE_TARGET_FRAMES
FPS = 20.0
TARGET_COUNT = 50

CUP_TARGET_PATH = CUP_ROOT_PATHS[0]
CAN_TARGET_PATH = "/World/Assets/EmptyCan"
CAN_ASSET_REFERENCE = "./newcans/cans/sprite_open_empty/asset.usda"
CAN_DIMENSIONS_M = (0.066, 0.066, 0.122)
CAN_MASS_KG = 0.015
ZBAR_SOURCE_PATH = "/World/Assets/Tier2Items/zbar_t"
SNACK_TARGET_PATH = "/World/Assets/Family7SnackBox"
PAPER_TARGET_PATH = "/World/Assets/PaperBall"
PAPER_SUPPORT_PATH = "/World/Assets/PaperPlate"
TEAL_BIN_PATH = "/World/Assets/TrashBinTeal"
RED_BIN_PATH = "/World/Assets/TrashBinRed"
TABLE_PATH = "/World/Assets/CoffeeTable"
REFERENCE_XY_M = tuple(
    float(value)
    for value in CLONE_PLACEMENTS[CUP_TARGET_PATH]["translate_xyz"][:2]
)


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    data_binding: str
    runtime_task_id: str
    target_path: str
    destination_name: str
    destination_path: str
    source_asset_path: str
    destination_geometry_acceptance: str = "visible_liner_interior"
    requires_final_rest: bool = True


MODES = {
    "paper": Mode(
        key="paper",
        label="small Task 30 (paper ball to red ordinary bin)",
        data_binding="small_task_30_real_data",
        runtime_task_id="30",
        target_path=PAPER_TARGET_PATH,
        destination_name="red",
        destination_path=RED_BIN_PATH,
        source_asset_path=PAPER_TARGET_PATH,
    ),
    "cup": Mode(
        key="cup",
        label="benchmark Task 7.10 (empty cup to red ordinary bin)",
        data_binding="small_task_29_real_data",
        runtime_task_id="family7_cup_e2e",
        target_path=CUP_TARGET_PATH,
        destination_name="red",
        destination_path=RED_BIN_PATH,
        source_asset_path="/World/Assets/PaperCup",
        # User 2026-09-16: entering the correct bin range is sufficient.
        destination_geometry_acceptance="visible_bin_envelope",
        requires_final_rest=False,
    ),
    "can": Mode(
        key="can",
        label=(
            "benchmark tidy_table 7.8 (empty can to teal recycle bin; "
            "no small-task ID/data)"
        ),
        data_binding="no_real_data_action_from_small_task_29",
        runtime_task_id="family7_can_e2e",
        target_path=CAN_TARGET_PATH,
        destination_name="teal",
        destination_path=TEAL_BIN_PATH,
        source_asset_path=CAN_TARGET_PATH,
    ),
    "snack": Mode(
        key="snack",
        label=(
            "benchmark tidy_table 7.3+7.4 (empty snack packaging to red "
            "ordinary bin; no small-task ID/data)"
        ),
        data_binding="no_real_data_action_from_small_task_29",
        runtime_task_id="family7_snack_box_e2e",
        target_path=SNACK_TARGET_PATH,
        destination_name="red",
        destination_path=RED_BIN_PATH,
        source_asset_path=ZBAR_SOURCE_PATH,
    ),
}


def mode(name: str) -> Mode:
    try:
        return MODES[name]
    except KeyError as exc:
        raise ValueError(
            f"family-7 mode must be one of {sorted(MODES)}, got {name!r}"
        ) from exc


__all__ = [
    "CAN_ASSET_REFERENCE",
    "CAN_DIMENSIONS_M",
    "CAN_MASS_KG",
    "CAN_TARGET_PATH",
    "CUP_TARGET_PATH",
    "E2E_END_FRAME_EXCLUSIVE",
    "FIRST_OBJECT_END_FRAME_EXCLUSIVE",
    "FPS",
    "MODES",
    "PAPER_TARGET_PATH",
    "PAPER_SUPPORT_PATH",
    "RED_BIN_PATH",
    "REFERENCE_XY_M",
    "RETURN_HOME_FRAMES",
    "SCHEMA",
    "SNACK_TARGET_PATH",
    "SOURCE_EPISODE_UID",
    "SOURCE_FULL_FRAMES",
    "TABLE_PATH",
    "TARGET_COUNT",
    "TEAL_BIN_PATH",
    "ZBAR_SOURCE_PATH",
    "mode",
]

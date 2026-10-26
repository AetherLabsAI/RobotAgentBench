"""Audited episode/layout contract for the full small Task 29 reproduce.

The selected real episode contains the complete behavior requested for this
task: an upright right-side cup picked by the right hand and handed to the
left, a second upright cup picked by the left hand, and a lying cup picked at
its rim by the left hand.  All three are released into the original teal bin.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from robot_positions import (
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_CURRENT_ROOT_TRANSLATE_XYZ_M,
    COFFEE_TABLE_CURRENT_TOP_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M,
    robot_position,
    translate_xy,
    translate_xyz,
)
from dual_trash_bin_layout import (
    BIN_CENTER_SPACING_X_M,
    BIN_OFFSET_X_M,
    BIN_OFFSET_Y_M,
    DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
    TEAL_CENTER_XY_M,
)


TASK_ID = "29"
EPISODE_UID = "06ab4bb5-83ea-499a-8310-611117220838"
FPS = 20.0
FRAME_COUNT = 471

# FK release-anchor station re-solve (2026-07-31, CLOSEOUT Q9 code-half).
# The accepted 2026-07-27 full-physics run (best.task29_full_metrics.npz)
# recorded base (-1.0349849850327641, -0.670) with the teal centre at
# (-1.8199849850327643, -1.100); its left fingertip-midpoint release
# anchors (same pad-anchor computation as align_grasp.py, applied to the
# real episode joints) were (-1.80782, -1.09979) @181,
# (-1.81473, -1.08935) @282 and (-1.76344, -1.06766) @362.  The replayed
# joint trajectory is base-rigid, so every pad anchor translates exactly
# with the base.  The dual-bin row was later re-anchored beside the
# TableclothBasket, but the shared table_pick station inherited only the
# +Y room-shell component of that reviewed move ("move the shared
# table_pick station with the bins"), so the unchanged trajectory released
# the cups over the red distractor.  Complete the directive with the
# residual teal-centre delta below and move the pickup staging rigidly
# with the base, which restores every accepted release-to-teal offset.
TASK29_ACCEPTED_STATION_XY_M = (-1.0349849850327641, -0.670)
TASK29_ACCEPTED_TEAL_CENTER_XY_M = (-1.8199849850327643, -1.100)
TASK29_ACCEPTED_RELEASE_PAD_ANCHORS_XY_M = (
    (-1.8078195399949464, -1.099790633294559),
    (-1.8147307003144237, -1.0893539530380227),
    (-1.7634400721388608, -1.067658889862213),
)
TASK29_RELEASE_ALIGNED_TEAL_CENTER_XY_M = translate_xy(
    TASK29_ACCEPTED_TEAL_CENTER_XY_M,
    (
        robot_position("table_pick").xy_m[0]
        - TASK29_ACCEPTED_STATION_XY_M[0],
        robot_position("table_pick").xy_m[1]
        - TASK29_ACCEPTED_STATION_XY_M[1],
    ),
)
# Keep the station solve expressed against the shared layout's optional
# away-from-table component.  House v1.2 sets that component to zero, so the
# station and release mapping now follow the authored bin row directly.  The
# subtraction remains part of the general derivation and prevents a future
# global furniture move from silently changing the pinned pickup cluster.
from dual_trash_bin_layout import (  # noqa: E402  (layout must import first)
    BIN_AWAY_FROM_TABLE_OFFSET_Y_M as _BIN_AWAY_Y_M,
)
TASK29_STATION_BIN_FOLLOW_SHIFT_XY_M = (
    TEAL_CENTER_XY_M[0] - TASK29_RELEASE_ALIGNED_TEAL_CENTER_XY_M[0],
    (
        TEAL_CENTER_XY_M[1]
        - _BIN_AWAY_Y_M
        - TASK29_RELEASE_ALIGNED_TEAL_CENTER_XY_M[1]
    ),
)

# Shared coffee-table-pick parking pose: the robot stands in the corridor
# between the Sofa and CoffeeTable and faces west across the CoffeeTable.
# The task-local base derives from the shared station plus the reviewed
# bin-follow delta above, mirroring the small Task 30 release-mapped base.
ROBOT_BASE_XY = translate_xy(
    robot_position("table_pick").xy_m,
    TASK29_STATION_BIN_FOLLOW_SHIFT_XY_M,
)
ROBOT_ORIENT_WXYZ = robot_position("table_pick").quat_wxyz
COFFEE_TABLE_SUPPORT_Z = COFFEE_TABLE_CURRENT_TOP_M
CUP_SUPPORT_CLEARANCE_M = 0.004
CUP_RADIUS_M = 0.044500001
CUP_LENGTH_M = 0.129999995
LYING_CUP_TILT_DEG = 80.0
LYING_CUP_YAW_DEG = 90.0

# The CoffeeTable remains at its shared current-contract pose.  The previous
# render-only +50 mm clone was explicitly rejected because furniture and
# collision must be the same object.
TASK29_COFFEE_TABLE_OFFSET_XY_M = (0.0, 0.0)
TASK29_COFFEE_TABLE_CENTER_XY_M = translate_xy(
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    TASK29_COFFEE_TABLE_OFFSET_XY_M,
)
TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M = translate_xyz(
    COFFEE_TABLE_CURRENT_ROOT_TRANSLATE_XYZ_M,
    TASK29_COFFEE_TABLE_OFFSET_XY_M,
)
TASK29_RETIRED_COFFEE_TABLE_VISUAL_PATH = "/World/Task29CoffeeTableVisual"

# The shared coffee-table-pick station moved 50 mm toward the fireplace and
# then followed the bin row (see the station re-solve above).  Keep the
# recorded arm/cup relationship rigid by applying the same total delta to the
# three pickup poses and grasp anchors; the physical CoffeeTable does not move.
TASK29_PICK_CLUSTER_SHIFT_XY_M = translate_xy(
    (
        COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0],
        COFFEE_TABLE_LAYOUT_SHIFT_XY_M[1] - 0.050,
    ),
    TASK29_STATION_BIN_FOLLOW_SHIFT_XY_M,
)

# The real frame-zero table contains the existing floral PaperPlate between
# the cups.  It is a visual-only distractor for small Task 29 so adding it does
# not perturb the previously accepted three-cup PhysX solve.  The small Task
# 30 initializer explicitly re-enables its rigid body/collision before using
# it as the paper-ball support (at its own release-mapped pose).  Lift the
# authored root 28 mm so its measured bottom is 1 mm above the 0.45 m top.
# The plate belongs to the robot-relative frame-zero tableau, so it inherits
# the same bin-follow delta as the base and the cups.
TASK29_PAPER_PLATE_PATH = "/World/Assets/PaperPlate"
TASK29_PAPER_PLATE_TRANSLATE_XYZ_M = translate_xyz(
    translate_xyz(
        (-2.100, -0.500, 0.448),
        COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M,
    ),
    TASK29_STATION_BIN_FOLLOW_SHIFT_XY_M,
)
TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M = (0.260, 0.260, 0.019)
TASK29_PAPER_PLATE_BOTTOM_M = 0.451
TASK29_PAPER_PLATE_TOP_M = 0.470

# Consume the fixed shared dual-bin layout.  It keeps the exact
# 0.19 x 0.24 x 0.35 m outside dimensions, keeps the basket/bin row beside the
# newly north-aligned CoffeeTable, and puts the red distractor immediately
# beside teal on robot-side +X with a 10 mm physical gap.
TASK29_BIN_OFFSET_X_M = BIN_OFFSET_X_M
TASK29_BIN_OFFSET_Y_M = BIN_OFFSET_Y_M
TASK29_BIN_SPACING_M = BIN_CENTER_SPACING_X_M
# Release alignment to teal is restored by TASK29_STATION_BIN_FOLLOW_SHIFT
# above (the bins themselves stay basket-adjacent).  The red distractor is
# adjacent on teal's +X side, which is the side closer to the robot.  The
# previous -X placement intersected the TableclothBasket.
TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M = (
    DISTRACTOR_ROBOT_SIDE_OFFSET_X_M
)

# The CoffeeTable is unchanged, so no task-local cup support compensation is
# permitted.
TASK29_CUP2_SETTLE_COMPENSATION_XY_M = (0.0, 0.0)
TASK29_BAG_MOUTH_LINER_TOP_Z_M = 0.435

# The render-hidden small Task 56 secondary context stays physically active
# for PhysX-topology determinism, but its authored PaperPlate footprint
# (centre (-2.200, -0.539), ~0.101 m radius at 0.78 scale) lies under the
# bin-follow-shifted lying cup2 root (-2.133, -0.486): the invisible plate
# would tilt cup2 off the tabletop at frame zero.  The real small Task 29
# frame zero has no second plate there, so exactly that member is excluded
# (deactivated) by the initializer, fail-closed.
TASK29_TASK56_CONTEXT_EXCLUDED_MEMBERS = ("PaperPlate",)

# Drive toward the ordinary closed hard stop and let the cup/pad collision stop
# the joints at the real body or rim surface.  The command itself is not the
# rendered opening: the measured joint position must remain above it under
# bilateral contact, which is the pure-physics proof that the cup stopped the
# fingers.
TASK29_GRIP_OPEN_LIMIT_RAD = 1.20
TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD = {
    "body": 0.300,
    "rim": 0.200,
}
TASK29_GRIP_CLOSE_STIFFNESS = 900.0
TASK29_GRIP_CLOSE_DAMPING = 120.0
# Retain the authored outer-joint effort cap; the commanded body/rim widths
# match the real gripper's nonzero openings and the physical pad/cup contacts
# must remain bilateral at those widths.
TASK29_GRIP_CLOSE_MAX_FORCE = 80.0
TASK29_RIM_GRIP_CLOSE_MAX_FORCE = 0.50
# The recorded left wrist passes 15 mm behind and 28 mm above the lying cup's
# physical rim centre in this calibrated room/station frame.  Apply a smooth
# differential-IK translation to the third-grasp arm drive only; the cup stays
# on the table and remains fully dynamic.
TASK29_RIM_GRASP_WORLD_OFFSET_XYZ_M = (0.015, 0.0, -0.028)
TASK29_RIM_GRASP_OFFSET_PLAN_FRAME = 300
TASK29_RIM_GRASP_OFFSET_FULL_FRAME = 328
TASK29_RIM_GRASP_OFFSET_RELEASE_FRAME = 362
TASK29_RIM_GRASP_OFFSET_CLEAR_FRAME = 376
TASK29_RIM_GRIP_CLOSE_RAMP_FRAMES = 10
TASK29_GRIP_PHASES = (
    {
        "name": "right_cup0_body",
        "hand": "r",
        "cup_ordinal": 0,
        "start_frame": 70,
        "end_frame": 125,
        "physical_target_rad": TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD["body"],
    },
    {
        "name": "left_cup0_handoff_body",
        "hand": "l",
        "cup_ordinal": 0,
        "start_frame": 124,
        "end_frame": 180,
        "physical_target_rad": TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD["body"],
    },
    {
        "name": "left_cup1_body",
        "hand": "l",
        "cup_ordinal": 1,
        "start_frame": 251,
        "end_frame": 281,
        "physical_target_rad": TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD["body"],
    },
    {
        "name": "left_cup2_rim",
        "hand": "l",
        "cup_ordinal": 2,
        "start_frame": 328,
        "end_frame": 361,
        "physical_target_rad": TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD["rim"],
    },
)

# All three cups remain dynamic for the complete playback and settling tail.
TASK29_SETTLED_CUP_TRANSITIONS = ()
TASK29_DESTINATION_ROOT_MAX_Z_M = 0.350

# The source PaperCup's release-clearance material (0.35/0.25) retained the
# first cup in the right hand but slipped during the real right-to-left
# overlap.  The 1.0/0.8 and 0.65/0.50 canaries retained every grasp but added
# excessive opening impulses, so use the bounded intermediate rung below. This
# task-local target material is active from frame zero on all cup colliders.
# The grippers, table and bins retain their authored material and all contacts
# remain ordinary rigid-body collision/friction.
TASK29_TARGET_STATIC_FRICTION = 0.80
TASK29_TARGET_DYNAMIC_FRICTION = 0.60
TASK29_TARGET_COMBINE_MODE = "multiply"
# Use the same paper material at the cup body and reinforced rolled rim.  The
# earlier low-friction rim override made the aligned rim squirt out sideways
# while one pad was loading.
TASK29_RIM_STATIC_FRICTION = TASK29_TARGET_STATIC_FRICTION
TASK29_RIM_DYNAMIC_FRICTION = TASK29_TARGET_DYNAMIC_FRICTION
TASK29_RIM_COMBINE_MODE = TASK29_TARGET_COMBINE_MODE

CUP_ROOT_PATHS = (
    "/World/Assets/Task29Cup0",
    "/World/Assets/Task29Cup1",
    "/World/Assets/Task29Cup2",
)
SOURCE_CUP_PATH = "/World/Assets/PaperCup"
DESTINATION_BIN_PATH = "/World/Assets/TrashBinTeal"
DISTRACTOR_BIN_PATH = "/World/Assets/TrashBinRed"

# Ordered (inclusive start, inclusive final closed frame).  The one-frame
# overlap at the first transfer is the physical right-to-left handoff.
LEFT_GRIP_RUNS = ((124, 180), (251, 281), (328, 361))
RIGHT_GRIP_RUNS = ((70, 125),)
HANDOFF_OVERLAP_FRAMES = (124, 125)
RELEASE_FRAMES = (181, 282, 362)

# Pad-midpoint calibration transformed rigidly from the reviewed trajectory
# into the shared Sofa/CoffeeTable-corridor coffee-table-pick base.
GRASP_ANCHORS_WORLD = (
    translate_xyz(
        (-1.866, -0.440, 0.556), TASK29_PICK_CLUSTER_SHIFT_XY_M
    ),  # cup0: right hand, upright body pinch
    translate_xyz(
        (-1.853, -0.672, 0.538), TASK29_PICK_CLUSTER_SHIFT_XY_M
    ),  # cup1: left hand, upright body pinch
    translate_xyz(
        (-2.037, -0.744, 0.530), TASK29_PICK_CLUSTER_SHIFT_XY_M
    ),  # cup2: left hand, lying rim pinch
)

# Clone all cups upright first so the existing three-cup clone audit can check
# source identity, mass, collision, and spacing.  The full initializer then
# rotates cup2 to its explicitly audited lying pose.
_UPRIGHT_ROOT_Z = COFFEE_TABLE_SUPPORT_Z + CUP_SUPPORT_CLEARANCE_M
CLONE_PLACEMENTS = {
    CUP_ROOT_PATHS[0]: {
        "translate_xyz": translate_xyz(
            (-1.866, -0.440, _UPRIGHT_ROOT_Z),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (0.0, 0.0, 0.0),
    },
    CUP_ROOT_PATHS[1]: {
        "translate_xyz": translate_xyz(
            (-1.853, -0.672, _UPRIGHT_ROOT_Z),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (0.0, 0.0, 0.0),
    },
    CUP_ROOT_PATHS[2]: {
        "translate_xyz": translate_xyz(
            (-2.162, -0.752, _UPRIGHT_ROOT_Z),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (0.0, 0.0, 0.0),
    },
}

# Local +Z is the cup axis and points from the closed base toward the rim.
# A near-horizontal +80-degree roll plus +90-degree world yaw keeps the mouth
# plane compatible with the recorded one-pad-inside/one-pad-outside pinch.
# The cup and its audited ring-centre grasp anchor share the same 60 mm
# passage-ward translation as the common coffee-table robot pose.
FINAL_POSES = {
    CUP_ROOT_PATHS[0]: {
        "semantic_pose": "upright_body_grasp",
        "translate_xyz": translate_xyz(
            (-1.866, -0.440, _UPRIGHT_ROOT_Z),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (0.0, 0.0, 0.0),
        "initial_hand": "right",
    },
    CUP_ROOT_PATHS[1]: {
        "semantic_pose": "upright_body_grasp",
        "translate_xyz": translate_xyz(
            (-1.853, -0.672, _UPRIGHT_ROOT_Z),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (0.0, 0.0, 0.0),
        "initial_hand": "left",
    },
    CUP_ROOT_PATHS[2]: {
        "semantic_pose": "lying_rim_grasp",
        "translate_xyz": translate_xyz(
            (
                -2.162 + TASK29_CUP2_SETTLE_COMPENSATION_XY_M[0],
                -0.752 + TASK29_CUP2_SETTLE_COMPENSATION_XY_M[1],
                COFFEE_TABLE_SUPPORT_Z
                + CUP_RADIUS_M * math.sin(
                    math.radians(LYING_CUP_TILT_DEG)),
            ),
            TASK29_PICK_CLUSTER_SHIFT_XY_M,
        ),
        "rotate_xyz": (
            LYING_CUP_TILT_DEG,
            0.0,
            LYING_CUP_YAW_DEG,
        ),
        "initial_hand": "left",
    },
}


def _grip_runs(grip: np.ndarray) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start = None
    for frame, value in enumerate(grip):
        if value >= 0.5 and start is None:
            start = frame
        elif value < 0.5 and start is not None:
            runs.append((start, frame - 1))
            start = None
    if start is not None:
        runs.append((start, len(grip) - 1))
    return tuple(runs)


def validate_selected_trajectory(path: str | Path) -> dict[str, object]:
    """Bind initialization/playback to the reviewed full-task trajectory."""

    trajectory_path = Path(path)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {
            "fps", "arm", "lgrip", "rgrip", "waist", "head", "base_xyyaw"
        }
        if set(trajectory.files) != required:
            raise ValueError(
                "small Task 29 trajectory fields changed: "
                f"{sorted(trajectory.files)!r}"
            )
        fps = float(np.asarray(trajectory["fps"]))
        arm = np.asarray(trajectory["arm"])
        left = np.asarray(trajectory["lgrip"])
        right = np.asarray(trajectory["rgrip"])
        waist = np.asarray(trajectory["waist"])
        head = np.asarray(trajectory["head"])
        base = np.asarray(trajectory["base_xyyaw"])
    if (
        not math.isclose(fps, FPS, abs_tol=1e-6, rel_tol=0.0)
        or arm.shape != (FRAME_COUNT, 14)
        or left.shape != (FRAME_COUNT,)
        or right.shape != (FRAME_COUNT,)
        or waist.shape != (FRAME_COUNT, 2)
        or head.shape != (FRAME_COUNT, 2)
        or base.shape != (FRAME_COUNT, 3)
        or not all(
            np.isfinite(value).all()
            for value in (arm, left, right, waist, head, base)
        )
    ):
        raise ValueError("small Task 29 selected trajectory shape/value drift")
    left_runs = _grip_runs(left)
    right_runs = _grip_runs(right)
    if left_runs != LEFT_GRIP_RUNS or right_runs != RIGHT_GRIP_RUNS:
        raise ValueError(
            "small Task 29 grip sequence changed: "
            f"left={left_runs!r}, right={right_runs!r}"
        )
    if not np.all(left[list(HANDOFF_OVERLAP_FRAMES)] >= 0.5):
        raise ValueError("small Task 29 left hand does not close for handoff")
    if not np.all(right[list(HANDOFF_OVERLAP_FRAMES)] >= 0.5):
        raise ValueError("small Task 29 right hand does not overlap handoff")
    return {
        "task_id": TASK_ID,
        "episode_uid": EPISODE_UID,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "left_grip_runs": [list(run) for run in left_runs],
        "right_grip_runs": [list(run) for run in right_runs],
        "handoff_overlap_frames": list(HANDOFF_OVERLAP_FRAMES),
        "release_frames": list(RELEASE_FRAMES),
    }


def validate_final_pose_contract(
    poses: Mapping[str, Mapping[str, object]],
) -> None:
    if tuple(poses) != CUP_ROOT_PATHS:
        raise ValueError("small Task 29 final cup roots/order changed")
    expected_semantics = (
        "upright_body_grasp",
        "upright_body_grasp",
        "lying_rim_grasp",
    )
    for path, semantic in zip(CUP_ROOT_PATHS, expected_semantics):
        pose = poses[path]
        if pose.get("semantic_pose") != semantic:
            raise ValueError(f"small Task 29 semantic pose changed for {path}")
        for field in ("translate_xyz", "rotate_xyz"):
            values = np.asarray(pose.get(field), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {field} for {path}")
    rotations = tuple(
        tuple(float(value) for value in poses[path]["rotate_xyz"])
        for path in CUP_ROOT_PATHS
    )
    if rotations != (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (LYING_CUP_TILT_DEG, 0.0, LYING_CUP_YAW_DEG),
    ):
        raise ValueError("small Task 29 upright/lying orientation contract changed")


validate_final_pose_contract(FINAL_POSES)


__all__ = [
    "CLONE_PLACEMENTS",
    "CUP_ROOT_PATHS",
    "DESTINATION_BIN_PATH",
    "DISTRACTOR_BIN_PATH",
    "EPISODE_UID",
    "FINAL_POSES",
    "FRAME_COUNT",
    "GRASP_ANCHORS_WORLD",
    "LEFT_GRIP_RUNS",
    "RELEASE_FRAMES",
    "RIGHT_GRIP_RUNS",
    "ROBOT_BASE_XY",
    "ROBOT_ORIENT_WXYZ",
    "SOURCE_CUP_PATH",
    "TASK29_ACCEPTED_RELEASE_PAD_ANCHORS_XY_M",
    "TASK29_ACCEPTED_STATION_XY_M",
    "TASK29_ACCEPTED_TEAL_CENTER_XY_M",
    "TASK29_RELEASE_ALIGNED_TEAL_CENTER_XY_M",
    "TASK29_STATION_BIN_FOLLOW_SHIFT_XY_M",
    "TASK29_TASK56_CONTEXT_EXCLUDED_MEMBERS",
    "TASK29_BIN_OFFSET_X_M",
    "TASK29_BIN_OFFSET_Y_M",
    "TASK29_BAG_MOUTH_LINER_TOP_Z_M",
    "TASK29_BIN_SPACING_M",
    "TASK29_COFFEE_TABLE_CENTER_XY_M",
    "TASK29_COFFEE_TABLE_OFFSET_XY_M",
    "TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M",
    "TASK29_RETIRED_COFFEE_TABLE_VISUAL_PATH",
    "TASK29_CUP2_SETTLE_COMPENSATION_XY_M",
    "TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M",
    "TASK29_GRIP_PHASES",
    "TASK29_GRIP_OPEN_LIMIT_RAD",
    "TASK29_GRIP_CLOSE_DAMPING",
    "TASK29_GRIP_CLOSE_MAX_FORCE",
    "TASK29_GRIP_CLOSE_STIFFNESS",
    "TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD",
    "TASK29_RIM_GRASP_WORLD_OFFSET_XYZ_M",
    "TASK29_RIM_GRIP_CLOSE_MAX_FORCE",
    "TASK29_RIM_GRASP_OFFSET_PLAN_FRAME",
    "TASK29_RIM_GRASP_OFFSET_FULL_FRAME",
    "TASK29_RIM_GRASP_OFFSET_RELEASE_FRAME",
    "TASK29_RIM_GRASP_OFFSET_CLEAR_FRAME",
    "TASK29_RIM_GRIP_CLOSE_RAMP_FRAMES",
    "TASK29_PICK_CLUSTER_SHIFT_XY_M",
    "TASK29_PAPER_PLATE_BOTTOM_M",
    "TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M",
    "TASK29_PAPER_PLATE_PATH",
    "TASK29_PAPER_PLATE_TOP_M",
    "TASK29_PAPER_PLATE_TRANSLATE_XYZ_M",
    "TASK29_RIM_COMBINE_MODE",
    "TASK29_RIM_DYNAMIC_FRICTION",
    "TASK29_RIM_STATIC_FRICTION",
    "TASK29_TARGET_COMBINE_MODE",
    "TASK29_TARGET_DYNAMIC_FRICTION",
    "TASK29_TARGET_STATIC_FRICTION",
    "TASK29_SETTLED_CUP_TRANSITIONS",
    "TASK29_DESTINATION_ROOT_MAX_Z_M",
    "TASK_ID",
    "validate_final_pose_contract",
    "validate_selected_trajectory",
]

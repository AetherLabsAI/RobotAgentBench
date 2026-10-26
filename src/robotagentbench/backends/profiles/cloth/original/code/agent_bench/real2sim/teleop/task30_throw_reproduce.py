"""Reviewed spatial contract for small Task 30 (crumpled paper -> red bin).

The CoffeeTable and both bins remain in the canonical global layout.  The
task-specific movable PaperPlate and the robot station are placed from two
measured trajectory anchors:

* right-pad frame 92 is the first paper grasp midpoint;
* left-pad frame 276 is the recorded release midpoint.

This maps the recorded release to the canonical red-bin center and maps the
first grasp back to the PaperPlate, without a furniture offset or an
in-episode pose write.
"""

from __future__ import annotations

from dual_trash_bin_layout import RED_CENTER_XY_M
from robot_positions import robot_position
from task29_full_reproduce import (
    TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M,
    TASK29_PAPER_PLATE_BOTTOM_M,
    TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M,
    TASK29_PAPER_PLATE_PATH,
    TASK29_PAPER_PLATE_TOP_M,
    TASK29_PAPER_PLATE_TRANSLATE_XYZ_M,
)


TASK30_RECORDED_RIGHT_PAD_FRAME92_XY_M = (
    -1.63172827,
    -0.48093712,
)
TASK30_RECORDED_LEFT_PAD_FRAME276_XY_M = (
    -1.639265917929396,
    -0.8886880257087983,
)
_TABLE_PICK_BASE_XY_M = robot_position("table_pick").xy_m
# Express the release mapping against the shared layout's optional
# away-from-table component.  House v1.2 sets that component to zero, making
# this solve land on the authored red-bin row without moving furniture or
# retaining a Family-7-only bin offset.
from dual_trash_bin_layout import (  # noqa: E402  (layout imports first)
    BIN_AWAY_FROM_TABLE_OFFSET_Y_M as _BIN_AWAY_Y_M,
)
_TASK30_RELEASE_MAPPED_BASE_XY_M = (
    (
        _TABLE_PICK_BASE_XY_M[0]
        + RED_CENTER_XY_M[0]
        - TASK30_RECORDED_LEFT_PAD_FRAME276_XY_M[0]
    ),
    (
        _TABLE_PICK_BASE_XY_M[1]
        + RED_CENTER_XY_M[1]
        - _BIN_AWAY_Y_M
        - TASK30_RECORDED_LEFT_PAD_FRAME276_XY_M[1]
    ),
)
_TASK30_BASE_SHIFT_XY_M = (
    _TASK30_RELEASE_MAPPED_BASE_XY_M[0]
    - _TABLE_PICK_BASE_XY_M[0],
    _TASK30_RELEASE_MAPPED_BASE_XY_M[1]
    - _TABLE_PICK_BASE_XY_M[1],
)
# A whole-station landing shift was rejected by a fresh physical canary because
# it destabilized the weak first pinch.  Keep the reviewed station unchanged;
# the bounded left-wrist release retreat below corrects the east-rim crossing
# only after the two-hand transfer.
TASK30_LANDING_STATION_SHIFT_XY_M = (0.0, 0.0)
TASK30_PAPER_PLATE_TRANSLATE_XYZ_M = (
    (
        TASK30_RECORDED_RIGHT_PAD_FRAME92_XY_M[0]
        + _TASK30_BASE_SHIFT_XY_M[0]
        + TASK30_LANDING_STATION_SHIFT_XY_M[0]
    ),
    (
        TASK30_RECORDED_RIGHT_PAD_FRAME92_XY_M[1]
        + _TASK30_BASE_SHIFT_XY_M[1]
        + TASK30_LANDING_STATION_SHIFT_XY_M[1]
    ),
    TASK29_PAPER_PLATE_TRANSLATE_XYZ_M[2],
)
# At the reviewed paper scale, the recorded right closure is driven to a
# bounded 0.10 rad approach target.  This small robot-only station correction
# aligns its approach midpoint to the naturally settled paper root.  The
# corresponding frame-276 release remains well inside the canonical red bin.
#
# FK re-solve 2026-07-31 (CLOSEOUT Q9 code-half): with the staging constants
# unchanged, the frame-92 right fingertip-midpoint FK is bit-stable across
# runtimes at (-1.6890067825666026, -0.40833470779180137) — the same
# pad-anchor computation align_grasp.py uses, replayed from the real episode
# joints — but the naturally settled paper root moved between the
# grasp-proven 2026-07-28 trace (run d26vowmo) and the 2026-07-30
# current-runtime trace (both under
# Task_30_trash_paper/diagnostics/current_runtime_shared_camera0_
# physical_best/…pick_contact_trace.npz), so the previously calibrated tune
# missed the paper by exactly that settle delta and the pinch closed in
# free air.  Carrying the measured settle delta keeps the proven
# pad-to-root approach geometry rigid; the paper seed, the PaperPlate and
# both bins do not move.
TASK30_GRASP_FINE_TUNE_2026_07_28_XY_M = (
    0.03917219173569375,
    -0.031028664344838863,
)
TASK30_SETTLED_ROOT_2026_07_28_XY_M = (
    -1.760883092880249,
    -0.36997684836387634,
)
TASK30_SETTLED_ROOT_CURRENT_XY_M = (
    -1.7660341262817383,
    -0.4057859480381012,
)
TASK30_GRASP_FINE_TUNE_XY_M = (
    (
        TASK30_GRASP_FINE_TUNE_2026_07_28_XY_M[0]
        + TASK30_SETTLED_ROOT_CURRENT_XY_M[0]
        - TASK30_SETTLED_ROOT_2026_07_28_XY_M[0]
    ),
    (
        TASK30_GRASP_FINE_TUNE_2026_07_28_XY_M[1]
        + TASK30_SETTLED_ROOT_CURRENT_XY_M[1]
        - TASK30_SETTLED_ROOT_2026_07_28_XY_M[1]
    ),
)
TASK30_BASE_XY_M = (
    (
        _TASK30_RELEASE_MAPPED_BASE_XY_M[0]
        + TASK30_GRASP_FINE_TUNE_XY_M[0]
        + TASK30_LANDING_STATION_SHIFT_XY_M[0]
    ),
    (
        _TASK30_RELEASE_MAPPED_BASE_XY_M[1]
        + TASK30_GRASP_FINE_TUNE_XY_M[1]
        + TASK30_LANDING_STATION_SHIFT_XY_M[1]
    ),
)

TASK30_GENIE_PAPER_SOURCE_SIZE_XYZ_M = (
    0.10359204,
    0.05704446,
    0.03874579,
)
# 2026-08-05: the uniform 1.50x proxy measured 155 x 86 x 58 mm — far larger
# than the real crumpled tissue and, being long and flat, it draped across
# the open jaw instead of falling out of it (and bridged the 190 mm bin
# mouth).  The non-uniform scale below keeps the same reviewed mesh but
# brings it to a compact ~88 x 71 x 60 mm crumple: smaller than the jaw
# opening on every axis, so an ordinary release drops it.
# Values must be exactly representable in float32: the initializer writes a
# Gf.Vec3f and re-reads it under a strict equality guard (0.85 / 1.55 fail
# that readback; eighths and quarters do not).
TASK30_GENIE_PAPER_SCALE_XYZ = (0.875, 1.25, 1.5)
TASK30_GENIE_PAPER_SIZE_XYZ_M = (
    (
        TASK30_GENIE_PAPER_SOURCE_SIZE_XYZ_M[2]
        * TASK30_GENIE_PAPER_SCALE_XYZ[2]
    ),
    (
        TASK30_GENIE_PAPER_SOURCE_SIZE_XYZ_M[1]
        * TASK30_GENIE_PAPER_SCALE_XYZ[1]
    ),
    (
        TASK30_GENIE_PAPER_SOURCE_SIZE_XYZ_M[0]
        * TASK30_GENIE_PAPER_SCALE_XYZ[0]
    ),
)
TASK30_SUPPORT_CLEARANCE_M = 0.004
TASK30_BITE_Z_M = 0.012
# The 1.50x irregular mesh measures about 24 mm across the observed pickup
# contact normal.  Approach at 0.10 rad so the open pads surround the paper,
# while a finite rubber contact patch resists rolling out of the pinch.
TASK30_GRIP_PHYSICAL_CLOSED_TARGET_RAD = 0.10
TASK30_LEFT_HANDOFF_GRIP_TARGET_RAD = 0.45
TASK30_LEFT_HANDOFF_CLOSE_LEAD_FRAMES = 2
TASK30_LEFT_HANDOFF_HOLD_START_FRAME = 219
TASK30_PAD_TORSIONAL_PATCH_RADIUS_M = 0.006
TASK30_PAD_MIN_TORSIONAL_PATCH_RADIUS_M = 0.003
# Every frame-pinned assist below (pickup pulse bridge, handoff windows,
# left-follow/release choreography) was measured on this exact reviewed
# episode.  Applied to any other recording the same frame numbers fire at
# arbitrary points of a different motion — the 2026-08-04 probe batch showed
# a foreign relay episode force-held past its real release and left-hand
# carries snapped half-open when they crossed frame 219.  The runtime
# therefore engages the choreography only for this episode; other episodes
# replay their own recorded grip signals unassisted.
TASK30_CALIBRATED_EPISODE_UID = "0d91b669-2d09-4e16-8934-27f0f6149d63"
# The recorded normalized signal briefly reopens between two pickup pulses,
# while REAL3 shows the paper remaining in the physical right-hand pinch.
# Bridge only that interior pulse gap; the later handoff release stays
# recorded and is still required to produce contact-free evidence.
TASK30_PICKUP_HOLD_START_FRAME = 108
TASK30_PICKUP_HOLD_END_FRAME = 136
# The recorded right-hand open edge starts at frame 233.  Fresh 1.50x evidence
# shows the left hand becoming bilateral at frame 220 while the right inner
# pad spikes to 97.6 N.  Synchronize the right opening to that proven handoff
# contact so it unloads instead of squeezing the paper out of the left hand.
TASK30_HANDOFF_START_FRAME = 220
TASK30_HANDOFF_END_FRAME = 276
TASK30_RIGHT_RELEASE_LEAD_FRAMES = 13
# The recorded left gripper first touches the irregular mesh near its trailing
# edge.  Use the +75 mm capture pose that produced bilateral contact at frame
# 220, then back the gripper off by 20 mm along Y on the following frame so its
# pads remain centred on the paper while the right hand clears.  Both stages
# are robot-relative differential-IK corrections, not furniture or object
# pose offsets.
TASK30_LEFT_HANDOFF_OFFSET_XYZ_M = (-0.012, 0.070, -0.010)
TASK30_LEFT_OFFSET_PLAN_FRAME = 210
TASK30_LEFT_OFFSET_FULL_FRAME = 220
TASK30_LEFT_OFFSET_RELEASE_FRAME = 252
TASK30_LEFT_OFFSET_CLEAR_FRAME = 276
TASK30_LEFT_HANDOFF_FOLLOW_OFFSET_XYZ_M = (0.004, -0.020, 0.0)
TASK30_LEFT_FOLLOW_START_FRAME = 220
TASK30_LEFT_FOLLOW_FULL_FRAME = 221
TASK30_LEFT_FOLLOW_RELEASE_FRAME = 252
TASK30_LEFT_FOLLOW_CLEAR_FRAME = 276
# Opening the simulated right linkage moves its inner pad toward +X through the
# paper.  Fresh frame-220--238 pad traces show that a -Y retreat leaves that
# contact normal loaded and launches the paper toward +X.  Retract the whole
# right end effector toward -X at the recorded open edge so ordinary collision
# clears the object while the left gripper takes over.
TASK30_RIGHT_RELEASE_RETREAT_XYZ_M = (-0.080, 0.0, 0.0)
TASK30_RIGHT_RETREAT_PLAN_FRAME = 220
TASK30_RIGHT_RETREAT_FULL_FRAME = 223
TASK30_RIGHT_RETREAT_RELEASE_FRAME = 260
TASK30_RIGHT_RETREAT_CLEAR_FRAME = 280
# At the red-bin release, the left linkage's contacting pad opens toward +X.
# The fresh baseline put the paper root 2 mm beyond the east rim at frame 276.
# Start the bounded -X wrist retreat while the paper is already at the bin Y
# coordinate, carry it toward the bin center, then clear after the gripper
# opens instead of sweeping the paper over the east rim.
TASK30_LEFT_RELEASE_RETREAT_XYZ_M = (-0.080, 0.0, 0.0)
TASK30_LEFT_RELEASE_RETREAT_PLAN_FRAME = 276
TASK30_LEFT_RELEASE_RETREAT_FULL_FRAME = 284
TASK30_LEFT_RELEASE_RETREAT_RELEASE_FRAME = 300
TASK30_LEFT_RELEASE_RETREAT_CLEAR_FRAME = 320
# Lower the gripped paper inside the red-bin mouth before the recorded open
# edge.  The shared 6 mm bottom LOD showed that a 0.49 m root release reaches
# the bottom with enough impact to eject the irregular rigid proxy; this
# bounded wrist motion reduces that drop without moving either bin.
TASK30_LEFT_RELEASE_LIFT_XYZ_M = (0.0, 0.0, -0.080)
TASK30_LEFT_RELEASE_LIFT_PLAN_FRAME = 260
TASK30_LEFT_RELEASE_LIFT_FULL_FRAME = 272
TASK30_LEFT_RELEASE_LIFT_RELEASE_FRAME = 300
TASK30_LEFT_RELEASE_LIFT_CLEAR_FRAME = 320
# The tall irregular convex body settles to the plate before frame zero.
# Counter-seed only the pre-settle dynamic placement so the naturally settled
# frame-zero paper, not its authored pivot, meets the recorded right-pad
# midpoint at the plate center.
TASK30_PRESETTLE_COUNTER_OFFSET_XY_M = (
    0.0836453740265026,
    -0.0164997933309041,
)
TASK30_PAPER_PICK_ANCHOR_XYZ_M = (
    (
        TASK30_PAPER_PLATE_TRANSLATE_XYZ_M[0]
        + TASK30_PRESETTLE_COUNTER_OFFSET_XY_M[0]
    ),
    (
        TASK30_PAPER_PLATE_TRANSLATE_XYZ_M[1]
        + TASK30_PRESETTLE_COUNTER_OFFSET_XY_M[1]
    ),
    (
        TASK29_PAPER_PLATE_TOP_M
        + TASK30_SUPPORT_CLEARANCE_M
        + TASK30_GENIE_PAPER_SIZE_XYZ_M[2]
        - TASK30_BITE_Z_M
    ),
)

TASK30_TABLE_PLATE_CONTEXT = {
    "coffee_table_translate_xyz": (
        TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M
    ),
    "paper_plate_path": TASK29_PAPER_PLATE_PATH,
    "paper_plate_translate_xyz": (
        TASK30_PAPER_PLATE_TRANSLATE_XYZ_M
    ),
    "paper_plate_expected_size_xyz_m": (
        TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M
    ),
    "paper_plate_bottom_m": TASK29_PAPER_PLATE_BOTTOM_M,
    "paper_plate_top_m": TASK29_PAPER_PLATE_TOP_M,
}

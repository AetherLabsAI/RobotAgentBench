"""Audited two-hand choreography for benchmark Task 6.15.

The selected small Task 92 recording supplies the left-hand wallet
grasp/carry/place motion.  The benchmark additionally requires the right hand
to operate the same physical CoffeeTable drawer.  The drawer remains a
dynamic prismatic body and is never pose-written during playback.
"""

from __future__ import annotations

import numpy as np


SCHEMA = "real2sim.task6_drawer_place_action.v15"
DRAWER_PATH = "/World/Assets/CoffeeTable/black_box_1"
INITIAL_LIP_M = 0.020

# User-selected compact chest-front standby pose. The existing G1 camera-front
# probe solves the 14 arm joints from the fixed-dual prior; the values below
# preserve that measured pose as a Git-owned runtime reference.
FAMILY4_STANDARD_ARM_RAD = np.asarray((
    -1.436433364062376, 0.6753098664746975, -0.01587773127866186, -1.514742628447755, 0.2227344956156173, 1.537142672911116, -0.1894840697614276,
    1.444565663875355, -0.6796827400435493, 0.0175364406025143, 1.515869821939533, -0.2323560762495787, -1.53741944632962, 0.191259764080589,
), dtype=float)
FAMILY4_APPROACH_FRAMES = 100
# Occupied standby stays below the head and outside the torso. The empty
# camera-front standby would put the held wallet through the head collider.
DRAWER_OCCUPIED_LEFT_HOME_RAD = np.asarray((
    -.7852873727869147, .3711920669534041, .3203078669869054,
    -1.3856214559273845, .2227344956156173, 1.537142672911116, -.1894840697614276))


def drawer_home_arm_rad(*, pick: bool) -> np.ndarray:
    home = FAMILY4_STANDARD_ARM_RAD.copy()
    if pick:
        home[:7] = DRAWER_OCCUPIED_LEFT_HOME_RAD
    return home
DRAWER_RIGHT_REFERENCE_ARM_RAD = np.asarray((
    .4448823034763336, -.8037457466125488, -.4459410607814789,
    1.3607800006866455, -.9729213714599609, -.49694451689720154, -1.6296788454055786))
# Object-relative wrist transforms from successful physical grasps. These
# geometry references remain runnable without their historical recordings.
DRAWER_LEFT_GRASP_POSES = {
    "4.10": ([.05623474229845601, .00004745023184313642, .01609611119354389],
             [.5610732021776212, -.16892696854192985, -.7905738867786397, -.17791422270605228]),
    "4.4": ([0., 0., .025],
            [.7237871086121005, -.6718307233896551, -.10013288848512492, -.12144589396226617]),
}


def family4_approach_frames(arm: np.ndarray) -> int:
    """Recognize the recorded rest-pose approach, retaining old replay clocks."""
    values = np.asarray(arm, dtype=float)
    if values.ndim != 2 or values.shape[1] != 14 or not len(values):
        raise ValueError("Family-4 arm must have shape (T,14)")
    return (
        FAMILY4_APPROACH_FRAMES
        if np.max(np.abs(values[0] - FAMILY4_STANDARD_ARM_RAD)) < 0.01
        else 0
    )


def orientation_error_xyzw(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Shortest world rotation from actual to target, as a rotation vector."""
    target = np.asarray(target, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if (target.shape != (4,) or actual.shape != (4,)
            or not np.isfinite(target).all() or not np.isfinite(actual).all()
            or np.linalg.norm(target) < 1e-9 or np.linalg.norm(actual) < 1e-9):
        raise ValueError("orientations must be finite nonzero XYZW quaternions")
    target = target / np.linalg.norm(target)
    actual = actual / np.linalg.norm(actual)
    xyz = (actual[3] * target[:3] - target[3] * actual[:3]
           - np.cross(target[:3], actual[:3]))
    w = float(target[3] * actual[3] + np.dot(target[:3], actual[:3]))
    if w < 0:
        xyz, w = -xyz, -w
    length = float(np.linalg.norm(xyz))
    return xyz * (2.0 * np.arctan2(length, w) / length) if length > 1e-12 else 2.0 * xyz

# Frame-0 right pad midpoint is approximately
# (-1.465, -0.415, 0.331) m at the contracted robot station.
RIGHT_SEAT_OFFSET_XYZ_M = np.asarray(
    (-0.080, -0.100, -0.010), dtype=float)
RIGHT_PULL_OFFSET_XYZ_M = np.asarray(
    (+0.140, -0.100, -0.010), dtype=float)

SEAT_START_FRAME = 0
SEAT_FULL_FRAME = 14
GRIP_CLOSE_START_FRAME = 14
GRIP_CLOSE_FULL_FRAME = 30
PULL_START_FRAME = 34
# The former 48-frame pull lost one pad's force while the drawer still moved.
# Keep the same endpoints and physics, but slow the robot-relative pull until
# the visible front wall remains bilaterally loaded throughout the motion.
PULL_FULL_FRAME = 110
FAMILY4_PUSH_START_FRAME = 1300
FAMILY4_PUSH_FULL_FRAME = 1400
FAMILY4_RIGHT_RELEASE_START_FRAME = 1410
FAMILY4_RIGHT_RELEASE_FULL_FRAME = 1430
FAMILY4_RIGHT_RETREAT_START_FRAME = 1440
FAMILY4_RIGHT_CLEAR_FULL_FRAME = 1480
FAMILY4_RIGHT_RETURN_X_FULL_FRAME = 1520
FAMILY4_RIGHT_RETREAT_FULL_FRAME = 1560
# Once the fingers are open, first lift and move away from the handle while
# retaining the pulled X coordinate.  Returning X at the same time as opening
# made the palm push the dynamic drawer closed by about 65 mm.
FAMILY4_RIGHT_CLEAR_OFFSET_XYZ_M = np.asarray(
    (+0.125, 0.000, +0.050), dtype=float)
FAMILY4_RIGHT_RETURN_X_OFFSET_XYZ_M = np.asarray(
    (0.000, 0.000, +0.050), dtype=float)
# A physical rail detent is engaged only after the right hand has completed
# the measured contact pull.  It targets that live joint displacement; it
# never writes the drawer or target pose.  The existing damping/max-force
# values remain unchanged, while stiffness supplies the missing static hold
# that a zero-stiffness viscous rail cannot provide at near-zero velocity.
FAMILY4_DRAWER_DETENT_LATCH_FRAME = 120
FAMILY4_DRAWER_DETENT_STIFFNESS_N_PER_M = 8000.0
FAMILY4_DRAWER_DETENT_DAMPING_N_S_PER_M = 120.0
FAMILY4_DRAWER_DETENT_MAX_FORCE_N = 60.0
PUSH_START_FRAME = 700
PUSH_FULL_FRAME = 850
GRIP_OPEN_START_FRAME = 880
GRIP_OPEN_FULL_FRAME = 900
OUTPUT_FRAMES = 360
DRAWER_RECOVERY_OUTPUT_FRAMES = 1300
# The canonical full-room drawer reaches its physical pull peak about 16
# frames after the old wallet release command and can continue creeping until
# the right-hand return begins.  Hold the already-placed wallet and its
# left-arm pose for 40 more frames so release follows that measured peak; the
# right-hand drawer choreography remains unchanged.
TASK615_WALLET_RELEASE_DELAY_FRAMES = 40
# The thin bill used by benchmark Task 6.5 is released just before its rigid
# body reaches the drawer, then falls freely.  The left hand clears first, the
# still-clamped right hand closes the drawer, and both arms finish at their
# frame-zero standard poses.  Furniture and target poses are never written.
TASK65_SETTLE_OUTPUT_FRAMES = 500
TASK65_BILL_RELEASE_DELAY_FRAMES = 0
TASK65_BILL_RETREAT_AFTER_RELEASE_HOLD_FRAMES = 8
TASK65_DRAWER_PUSH_DELAY_FRAMES = 0
TASK65_BILL_RELEASE_COMMAND_FRAME = 166
TASK65_BILL_RETREAT_COMMAND_FRAME = 174
TASK65_BILL_RETREAT_FULL_FRAME = 234
TASK65_DRAWER_PUSH_START_FRAME = 242
TASK65_DRAWER_PUSH_FULL_FRAME = 292
TASK65_RIGHT_GRIP_OPEN_START_FRAME = 294
TASK65_RIGHT_GRIP_OPEN_FULL_FRAME = 308
TASK65_RIGHT_RETURN_START_FRAME = 310
TASK65_RIGHT_RETURN_FULL_FRAME = 370
# The 11 mm bill is initially clamped with a reviewed 14.5 mrad backoff from
# first physical contact.  If one pad loses target contact during the carry,
# recover that backoff in bounded 1 mrad steps; never close beyond the
# originally observed contact angle.
TASK65_BILL_GRIP_FOLLOW_STEP_RAD = 0.001
TASK65_BILL_GRIP_MAX_FOLLOW_RAD = 0.014
TASK65_BILL_GRIP_RELEASE_ADVANCE_FRAMES = 0

TASK65_BASKET_LEVEL_SETTLE_FRAMES = 80

MIN_PULL_TRAVEL_M = 0.120
MAX_FINAL_RETURN_ERROR_M = 0.035
MAX_LATERAL_DRIFT_M = 0.012
# Raw contact counts are diagnostic only: PhysX can retain a zero-impulse
# manifold for a frame, which must never be mistaken for a grasp.
MIN_RIGHT_PAD_FORCE_N = 0.50
MAX_RIGHT_PAD_FORCE_COSINE = -0.25
MIN_DRAWER_PULL_MOTION_STEP_M = 1.0e-5
MIN_CLAMPED_PULL_MOTION_RATIO = 0.95
MAX_UNCLAMPED_PULL_MOTION_M = 0.005
MAX_FRONT_WALL_CONTACT_X_ERROR_M = 0.020
FRONT_WALL_CONTACT_AABB_MARGIN_M = 0.005
MIN_RIGHT_PAD_CONTACT_FRAMES = 20
MAX_FINAL_ARM_HOME_ERROR_RAD = 1.0e-3
# Full closure drives the two rounded pads through the drawer's thin front
# wall.  A 90% action command allowed the lower pad to unload during the
# middle of the pull in the accepted 6.15 population.  The reviewed 92%
# command keeps both pads force-bearing through pull and push without changing
# any collision geometry, solver setting, material, or other physics state.
RIGHT_GRIP_HOLD = 0.92
# Benchmark Task 6.5's accepted r65 population already maintains bilateral
# force at 90%; keep that task-owned policy byte-compatible while the wallet
# task takes the separately reviewed 92% clamp.
TASK65_RIGHT_GRIP_HOLD = 0.90


def smooth_scale(frame: int, start: int, end: int) -> float:
    """Return a C1-smooth zero-to-one phase scale."""
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    value = (float(frame) - float(start)) / float(end - start)
    return value * value * (3.0 - 2.0 * value)


def arm_recovery_phase(frame: int, *, pick: bool, hand: str, task: str = ""):
    """Arm-only withdrawal waypoints, followed by compact standby.

    Coordinates are offsets from the measured wrist at recovery start, in
    the shared drawer station world frame. They are commands to physics,
    never object poses. The live controller must verify each endpoint.
    """
    if hand not in {"l", "r"}:
        raise ValueError("recovery hand must be l or r")
    start = ((1000 if hand == "l" else 1540) if pick else
             (470 if hand == "l" else 910) if task.endswith("6_5") else
             (342 if hand == "l" else 910))
    if frame < start:
        return None
    # First move out of the table edge, then above the basket wall, then
    # withdraw on the arm's own side. Raise/fold only after this clearance.
    waypoints = (
        [(0, (0., 0., 0.)), (50, (.06, 0., 0.)),
         (140, (.06, 0., .13)), (240, (.10, -.10, .13))]
        if hand == "l" else
        [(0, (0., 0., 0.)), (60, (.10, 0., 0.)),
         (130, (.16, .08, .12))]
    )
    if pick and hand == "l" and task in {"4.4", "benchmark_4_4"}:
        # Lift the long case above the supporting right fingertips before
        # withdrawing toward the robot; otherwise its nose brushes them.
        waypoints = [(0, (0., 0., 0.)), (80, (0., 0., .10)),
                     (140, (.06, 0., .13)), (240, (.10, -.10, .13))]
    elapsed = frame - start
    last_time, last_xyz = waypoints[-1]
    if elapsed <= last_time:
        for (a, xyz_a), (b, xyz_b) in zip(waypoints, waypoints[1:]):
            if elapsed <= b:
                weight = smooth_scale(elapsed, a, b)
                return ("cartesian", (1. - weight) * np.asarray(xyz_a) + weight * np.asarray(xyz_b))
    if not pick and hand == "l" and task.endswith("6_5") and elapsed > 440:
        return ("standby", smooth_scale(elapsed, 440, 600))
    return ("home", smooth_scale(elapsed, last_time + 40, last_time + 200))


def right_hand_offset(frame: int) -> np.ndarray:
    """World translation requested from the source frame-0 right arm."""
    if frame <= SEAT_FULL_FRAME:
        return smooth_scale(frame, SEAT_START_FRAME, SEAT_FULL_FRAME) * RIGHT_SEAT_OFFSET_XYZ_M
    if frame <= PULL_FULL_FRAME:
        scale = smooth_scale(frame, PULL_START_FRAME, PULL_FULL_FRAME)
        return (
            (1.0 - scale) * RIGHT_SEAT_OFFSET_XYZ_M
            + scale * RIGHT_PULL_OFFSET_XYZ_M
        )
    if frame < 600:
        return RIGHT_PULL_OFFSET_XYZ_M.copy()
    if frame < PUSH_START_FRAME:
        return RIGHT_PULL_OFFSET_XYZ_M + np.asarray([0., 0., .008 * smooth_scale(frame, 600, 660)])
    reinsert = RIGHT_SEAT_OFFSET_XYZ_M + np.asarray([-.020, 0., 0.])
    scale = smooth_scale(frame, PUSH_START_FRAME, PUSH_FULL_FRAME)
    offset = (1. - scale) * RIGHT_PULL_OFFSET_XYZ_M + scale * reinsert
    offset[2] += .008 * (1. - smooth_scale(frame, PUSH_FULL_FRAME, GRIP_OPEN_START_FRAME - 10))
    return offset


def open_only_right_hand_offset(frame: int) -> np.ndarray:
    """Hold the basket throughout the pick and reinsert before releasing.

    The historical function name is retained for action normalization. The
    right hand now supplies the support previously replaced by a rail detent.
    """
    frame = int(frame)
    if frame <= FAMILY4_PUSH_START_FRAME:
        return right_hand_offset(min(frame, PULL_FULL_FRAME))
    if frame <= FAMILY4_PUSH_FULL_FRAME:
        scale = smooth_scale(frame, FAMILY4_PUSH_START_FRAME, FAMILY4_PUSH_FULL_FRAME)
        return (1.0 - scale) * RIGHT_PULL_OFFSET_XYZ_M + scale * RIGHT_SEAT_OFFSET_XYZ_M
    if frame <= FAMILY4_RIGHT_RETREAT_START_FRAME:
        return RIGHT_SEAT_OFFSET_XYZ_M.copy()
    if frame <= FAMILY4_RIGHT_CLEAR_FULL_FRAME:
        scale = smooth_scale(
            frame,
            FAMILY4_RIGHT_RETREAT_START_FRAME,
            FAMILY4_RIGHT_CLEAR_FULL_FRAME,
        )
        return (
            (1.0 - scale) * RIGHT_SEAT_OFFSET_XYZ_M
            + scale * FAMILY4_RIGHT_CLEAR_OFFSET_XYZ_M
        )
    if frame <= FAMILY4_RIGHT_RETURN_X_FULL_FRAME:
        scale = smooth_scale(
            frame,
            FAMILY4_RIGHT_CLEAR_FULL_FRAME,
            FAMILY4_RIGHT_RETURN_X_FULL_FRAME,
        )
        return (
            (1.0 - scale) * FAMILY4_RIGHT_CLEAR_OFFSET_XYZ_M
            + scale * FAMILY4_RIGHT_RETURN_X_OFFSET_XYZ_M
        )
    scale = smooth_scale(
        frame,
        FAMILY4_RIGHT_RETURN_X_FULL_FRAME,
        FAMILY4_RIGHT_RETREAT_FULL_FRAME,
    )
    return (1.0 - scale) * FAMILY4_RIGHT_RETURN_X_OFFSET_XYZ_M


def task65_right_hand_offset(frame: int) -> np.ndarray:
    """Reinsert only after the placing arm has cleared the basket."""
    return right_hand_offset(frame)


def task65_right_grip(frame: int) -> float:
    """Keep the bill task's measured clamp force through reinsertion."""
    hold = TASK65_RIGHT_GRIP_HOLD + (RIGHT_GRIP_HOLD - TASK65_RIGHT_GRIP_HOLD) * smooth_scale(frame, 500, 560)
    return right_grip(frame) * (hold / RIGHT_GRIP_HOLD)


def right_grip(frame: int) -> float:
    """Normalized right gripper closure for clamp/hold/release."""
    hold = RIGHT_GRIP_HOLD + .04 * smooth_scale(frame, 600, 650)
    if frame <= GRIP_CLOSE_START_FRAME:
        return 0.0
    if frame <= GRIP_CLOSE_FULL_FRAME:
        return hold * smooth_scale(
            frame, GRIP_CLOSE_START_FRAME, GRIP_CLOSE_FULL_FRAME)
    if frame < GRIP_OPEN_START_FRAME:
        return hold
    return hold * (
        1.0 - smooth_scale(
            frame, GRIP_OPEN_START_FRAME, GRIP_OPEN_FULL_FRAME))


def open_only_right_grip(frame: int) -> float:
    """Clamp through the proven pull, then release before arm retreat."""
    frame = int(frame)
    if frame <= FAMILY4_RIGHT_RELEASE_START_FRAME:
        return right_grip(min(frame, PULL_FULL_FRAME))
    return RIGHT_GRIP_HOLD * (
        1.0 - smooth_scale(
            frame,
            FAMILY4_RIGHT_RELEASE_START_FRAME,
            FAMILY4_RIGHT_RELEASE_FULL_FRAME,
        )
    )

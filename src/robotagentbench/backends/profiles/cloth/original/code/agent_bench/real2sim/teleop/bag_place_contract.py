"""Shared current contract for the four held-object-to-tote small tasks.

These recordings begin with the target already held.  They share the same
robot station, deformable tote, initial tote contents, and fixed upper-body
posture.  Per-episode arm and gripper trajectories remain authoritative.
"""
from __future__ import annotations

import math


BAG_PLACE_TASK_IDS = ("71", "72", "73", "74")
BAG_PLACE_TASK_ID_SET = frozenset(BAG_PLACE_TASK_IDS)
BAG_PLACE_STATION = "kitchen_table_place"
BAG_PLACE_SCENE_ROOT = "/World/Task73"
BAG_PLACE_TOTE_PATH = f"{BAG_PLACE_SCENE_ROOT}/Tote"
BAG_PLACE_EXISTING_BOTTLE_PATH = (
    f"{BAG_PLACE_SCENE_ROOT}/ExistingBottle"
)
BAG_PLACE_SCENE_FILENAME = "scene_room.usda"
BAG_PLACE_MAX_APPROACH_OFFSET_M = 0.100
# The glasses case, wide bottle, and wallet can remain shelved between the
# stock gripper linkages at the composed 45-degree stop.  The long umbrella
# instead releases cleanly at the stock stop; opening it farther changes its
# drop dynamics enough to slide over the tote wall.
BAG_PLACE_GRIP_OPEN_LIMIT_RAD = 1.20
BAG_PLACE_WIDE_OPEN_TASK_ID_SET = frozenset(("72", "73", "74"))

BAG_PLACE_TARGETS = {
    "71": {
        "root": "/World/Assets",
        "prim": "Umbrella",
        "label": "折叠雨伞→软皮托特包",
        # The real hand holds the folded umbrella below its centre of mass;
        # keep the shaft/canopy extending upward from the right gripper.
        "seat_long_m": 0.000,
        # The accepted REAL exemplar starts in the left hand.  A small
        # counter-clockwise rotation in that pad plane gives both pads
        # opposing purchase on the asymmetric folded canopy.
        "grasp_long_plane_rotation_deg": 10.0,
        "staging_xyz_m": (2.62, 0.67, 5.0),
    },
    "72": {
        "root": "/World/Assets",
        "prim": "GlassesCase",
        "label": "墨镜盒→软皮托特包",
        # The real right hand grips the long case close to one end.  Seating
        # its centre between the pads leaves the released case outside the
        # near tote wall at the fixed dining-table-head station.
        "seat_long_m": -0.050,
        # Preserve the reviewed REAL grasp: the 170 mm case enters the tote
        # long-axis vertical.  A horizontal reorientation makes the case
        # bridge the soft mouth and is not faithful to the source episode.
        # At the fixed dining-table-head station the recorded wrist centre
        # stops just outside the near interior wall.  Ramp the robot-relative
        # approach 50 mm farther into the opening before release; furniture
        # and robot base remain in their shared canonical poses.
        "approach_offset_xyz_m": (0.0, -0.050, 0.0),
        "staging_xyz_m": (2.82, 0.67, 5.0),
    },
    "73": {
        "root": BAG_PLACE_SCENE_ROOT,
        "prim": "Bottle",
        "label": "运动水瓶→软皮托特包",
        "staging_xyz_m": (2.82, 0.67, 5.0),
        "up_axis": 1,
        # The recorded wrist stops at the near side of the soft mouth.  Move
        # only the active arm's final robot-relative approach deeper into the
        # shared tote so the tall bottle is supported before it opens.
        "approach_offset_xyz_m": (0.0, -0.040, -0.060),
    },
    "74": {
        "root": "/World/Assets",
        "prim": "Wallet",
        "label": "钱包→软皮托特包",
        "staging_xyz_m": (3.02, 0.67, 5.0),
    },
}

# User-reviewed common robot posture.  HEAD_ADJ remains the shared camera
# adjustment (zero); this value is the physical head pitch joint target.
BAG_PLACE_HEAD_YAW_RAD = 0.0
BAG_PLACE_HEAD_PITCH_DEG = 25.0
BAG_PLACE_HEAD_PITCH_RAD = math.radians(BAG_PLACE_HEAD_PITCH_DEG)
BAG_PLACE_WAIST_PITCH_DEG = 0.0
BAG_PLACE_WAIST_PITCH_RAD = 0.0
BAG_PLACE_WAIST_LIFT_M = 0.3


def bag_place_robot_posture_contract() -> dict:
    """Return JSON-safe posture metadata used by host and Isaac runtime."""
    return {
        "head_yaw_deg": 0.0,
        "head_pitch_deg": BAG_PLACE_HEAD_PITCH_DEG,
        "waist_pitch_deg": BAG_PLACE_WAIST_PITCH_DEG,
        "waist_lift_m": BAG_PLACE_WAIST_LIFT_M,
        "policy": (
            "fixed for the complete bag-place playback; arm and gripper "
            "remain episode-relative"
        ),
    }


def bag_place_target(task_id: str) -> dict:
    """Return a defensive copy of one supported target descriptor."""
    task = str(task_id)
    try:
        return dict(BAG_PLACE_TARGETS[task])
    except KeyError as exc:
        raise KeyError(f"small Task {task} is not a bag-place task") from exc

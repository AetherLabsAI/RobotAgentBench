"""Canonical reusable robot stations for the current simulator furniture layout.

The coordinates in this module are room-world poses.  Furniture-relative
episode calibrations may keep a small reviewed offset from a station, but new
tasks should start from one of the named stations below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import group_shift


# Furniture layout transactions applied on 2026-07-26 and 2026-07-28.
#
# CoffeeTable kept its 0.8 x 1.2 m footprint and moved rigidly in +X.
# The reviewed room-shell correction then moved the complete table +Y so its
# north edge aligns with the Sofa north edge.  Both components belong to the
# one current global table transform and every table-relative asset/station
# must inherit both.
COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M = (
    0.0,
    0.28091587972998145,
)
COFFEE_TABLE_LAYOUT_SHIFT_XY_M = (
    0.1450150149672358,
    COFFEE_TABLE_NORTH_LAYOUT_SHIFT_XY_M[1],
)
COFFEE_TABLE_AUTHORED_TOP_M = 0.422941977
COFFEE_TABLE_CURRENT_CENTER_XY_M = (
    -1.9500000008940699,
    -0.12104358305158935,
)
COFFEE_TABLE_CURRENT_ROOT_TRANSLATE_XYZ_M = (
    -2.1073394989357204 + COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0],
    COFFEE_TABLE_CURRENT_CENTER_XY_M[1],
    0.0,
)
COFFEE_TABLE_CURRENT_SIZE_XYZ_M = (0.8, 1.2, 0.45)
COFFEE_TABLE_CURRENT_AABB_MIN_XYZ_M = (
    -2.3500000008940696,
    -0.7210435940515894,
    0.0,
)
COFFEE_TABLE_CURRENT_AABB_MAX_XYZ_M = (
    -1.55000000089407,
    0.478956428051033,
    0.45,
)
COFFEE_TABLE_CURRENT_BOTTOM_M = 0.0
COFFEE_TABLE_CURRENT_TOP_M = 0.45
COFFEE_TABLE_TOP_SHIFT_M = (
    COFFEE_TABLE_CURRENT_TOP_M - COFFEE_TABLE_AUTHORED_TOP_M
)
# The ShelvingUnit moved north with the room-shell inset.  Its contents and
# support overlays use the same parent-space delta; shelf robot stations and
# world-space trajectory anchors use this value exactly once.
SHELF_LAYOUT_SHIFT_XY_M = (0.0, 0.6306552699470328)
SHELF_CURRENT_CENTER_XY_M = (
    -2.3690211510018,
    2.252956604618633,
)
# The TV cabinet was reconstructed from the small Task 58 RGB-D recordings.
# Its long axis is world Y.  The television dimensions below describe the
# active 16:9 display area; the stand and bezel remain part of the asset.
TV_CABINET_CURRENT_CENTER_XY_M = (
    -3.7097067168016444,
    -0.29850629754082375,
)
TV_CABINET_CURRENT_SIZE_XYZ_M = (0.30, 1.68, 0.45)
TV_CABINET_CURRENT_BOTTOM_M = 0.0
TV_CABINET_CURRENT_TOP_M = 0.45
TV_STANDARD_DIAGONAL_IN = 55.0
TV_STANDARD_ASPECT_RATIO = (16.0, 9.0)
TV_STANDARD_SCREEN_SIZE_WH_M = (
    1.2175910253629953,
    0.6848949517666849,
)
SOFA_INTERACTION_FACE_SHIFT_XY_M = (0.5096436061872428, 0.0)
SOFA_AUTHORED_SEAT_REFERENCE_TOP_M = 0.415
SOFA_CURRENT_SEAT_TOP_M = 0.452
SOFA_CURRENT_WORLD_SIZE_XYZ_M = (
    0.8,
    2.7,
    0.78
    * SOFA_CURRENT_SEAT_TOP_M
    / SOFA_AUTHORED_SEAT_REFERENCE_TOP_M,
)
SOFA_CURRENT_SEAT_COLLISION_CENTER_XY_M = (
    -0.52 + SOFA_INTERACTION_FACE_SHIFT_XY_M[0],
    -1.065,
)
SOFA_CURRENT_SEAT_COLLISION_SIZE_XYZ_M = (0.68, 2.31, 0.02)
KITCHEN_TABLE_INTERACTION_SHIFT_XY_M = (0.87, 0.37)


@dataclass(frozen=True)
class RobotPosition:
    """One reusable planar robot pose in the simulator world frame."""

    key: str
    label: str
    xy_m: tuple[float, float]
    yaw_deg: float
    anchor: str

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        half = math.radians(self.yaw_deg) / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))


def translate_xy(
    xy: Iterable[float],
    delta_xy: Iterable[float],
) -> tuple[float, float]:
    x, y = (float(value) for value in xy)
    dx, dy = (float(value) for value in delta_xy)
    return (x + dx, y + dy)


def translate_xyz(
    xyz: Iterable[float],
    delta_xy: Iterable[float],
    *,
    delta_z: float = 0.0,
) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in xyz)
    shifted = translate_xy((x, y), delta_xy)
    return (shifted[0], shifted[1], z + float(delta_z))


def translate_interval(
    bounds: Iterable[float],
    delta: float,
) -> tuple[float, float]:
    lower, upper = (float(value) for value in bounds)
    offset = float(delta)
    return (lower + offset, upper + offset)


def translate_zone_xy(
    zone: Iterable[float],
    delta_xy: Iterable[float],
) -> tuple[float, float, float, float, float]:
    cx, cy, hx, hy, surface_z = (float(value) for value in zone)
    shifted = translate_xy((cx, cy), delta_xy)
    return (shifted[0], shifted[1], hx, hy, surface_z)


ROBOT_POSITIONS = {
    "shelf_pick": RobotPosition(
        key="shelf_pick",
        label="shelf pick",
        # 2026-08-02: y was 0.700, which stood the robot too far back. Measured
        # two independent ways. From the grasp itself: at the moment task 37's
        # jaw is around the box, the pad midpoint sits at world y 2.0405 and the
        # box centre at 2.0706, so the hand is 30.1 mm short and the open jaw
        # shoves the box away before it closes. From the recordings: head RGB-D
        # puts the real camera 0.394 m from the board's front edge in that
        # episode and 0.366 m in task 40's, against 0.411-0.418 m in the sim
        # render. +0.030 lands the jaw centre on the box centre.
        xy_m=translate_xy(
            (-2.369, 0.730),
            SHELF_LAYOUT_SHIFT_XY_M,
        ),
        yaw_deg=90.0,
        anchor="Tier2 shelf interaction face",
    ),
    "shelf_place_table": RobotPosition(
        key="shelf_place_table",
        label="shelf place table",
        xy_m=translate_xy(
            (-2.080, 0.645),
            COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
        ),
        yaw_deg=-90.0,
        anchor="CoffeeTable north interaction edge",
    ),
    "table_pick": RobotPosition(
        key="table_pick",
        label="table pick",
        xy_m=translate_xy(
            (-1.180, -0.670),
            COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
        ),
        yaw_deg=180.0,
        anchor="CoffeeTable east interaction edge, 50 mm toward fireplace",
    ),
    "family7_paper_bin": RobotPosition(
        key="family7_paper_bin",
        label="Family 7 paper-to-bin",
        # Stable pre-perturbation root used by the accepted 7.2 cores.  This
        # task needs a slightly different handoff line from the other tidy
        # tasks, so it is a named station rather than an unexplained offset
        # from table_pick.
        xy_m=(-1.097697908769164, -0.35131797885030236),
        yaw_deg=180.0,
        anchor="CoffeeTable-to-ordinary-bin paper handoff line",
    ),
    "family7_tidy_bin": RobotPosition(
        key="family7_tidy_bin",
        label="Family 7 tidy-to-bin",
        # Shared stable pre-perturbation root used by accepted 7.4, 7.8 and
        # 7.10 cores.  Per-trajectory nonzero base noise is applied later.
        xy_m=(-1.1510000000000002, -0.35408412027001845),
        yaw_deg=180.0,
        anchor="CoffeeTable-to-waste-bin tidy handoff line",
    ),
    "tablecloth": RobotPosition(
        key="tablecloth",
        label="tablecloth table and basket",
        xy_m=(-2.9193, -0.48654358305158935),
        yaw_deg=0.0,
        anchor="CoffeeTable west edge and TableclothBasket shared workspace",
    ),
    "table_place": RobotPosition(
        key="table_place",
        label="table place",
        xy_m=translate_xy(
            (-1.350, -1.250),
            COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
        ),
        yaw_deg=134.0,
        anchor="CoffeeTable southeast interaction corner",
    ),
    "sofa_pick": RobotPosition(
        key="sofa_pick",
        label="sofa pick",
        xy_m=translate_xy(
            (-1.478, -1.050),
            SOFA_INTERACTION_FACE_SHIFT_XY_M,
        ),
        yaw_deg=0.0,
        anchor="Sofa west interaction face",
    ),
    "sidetable_pick": RobotPosition(
        key="sidetable_pick",
        label="sidetable pick",
        # 2026-08-03: was (-2.100, -0.850) facing -90, which is not beside the
        # side table at all -- x -2.100 sits 130 mm off the CoffeeTable's south
        # edge, while the CoffeeTable/TvTable corridor is x in [-3.56, -2.35].
        # This pose is the one small Task 90 carried privately: it stands in
        # that corridor, 180 mm off the table's west edge, with the robot, the
        # table and the fireplace collinear -- the geometry the real recording
        # shows. It is also the only side-table station with a passing grasp on
        # record (245.5 mm held 3.6 s); the retired one was never validated.
        xy_m=(-2.780, -1.380),
        yaw_deg=-45.0,
        anchor="Side-table interaction corner in the CoffeeTable/TvTable corridor",
    ),
    # --- 2026-08-03: three poses that are genuinely their own, not group
    # members. Opening a drawer is not placing on a table: the hand has to hook
    # its lip. The two stations below are different approach poses for the
    # same canonical black_box_1 drawer. The TV cabinet is its own piece of
    # furniture. Folding any of these into a
    # shared station is what made small Task 28 land 144 mm outside its zone.
    "coffee_table_drawer_bill": RobotPosition(
        key="coffee_table_drawer_bill",
        label="coffee table drawer (bill, black_box_1)",
        # Solved from the REAL head frame: the three table corners in the
        # recording fix this pose (station_solve.py, 14.9 px RMS). The retired
        # shared station stood ~26 cm too far forward and ~6 deg off, which
        # made the replayed pull slap the drawer front instead of hooking it.
        xy_m=(-1.1925, -1.1752),
        yaw_deg=127.82,
        anchor="CoffeeTable black_box_1 drawer face",
    ),
    "coffee_table_drawer_wallet": RobotPosition(
        key="coffee_table_drawer_wallet",
        label="coffee table drawer (wallet, black_box_1)",
        # Task 92 uses a different approach pose to the same physical drawer:
        # the right hand pinches the front and pulls it out.
        xy_m=(-0.9125, -0.6236),
        yaw_deg=179.7629,
        anchor="CoffeeTable black_box_1 drawer face",
    ),
    "coffee_table_drawer_bill_two_hand": RobotPosition(
        key="coffee_table_drawer_bill_two_hand",
        label="coffee table black_box_1 drawer (bill, two-hand)",
        # The task-owned bill producer uses the validated two-hand drawer
        # choreography rather than small Task 28's side-insertion attempt.
        # Its first physical baseline landed the bill 1 mm outside the strict
        # Y zone. A bounded 5 mm inward correction keeps the bill inside the
        # zone without the upper-lip contact observed at 20 mm.
        xy_m=(-0.9125, -0.6286),
        yaw_deg=179.7629,
        anchor="CoffeeTable black_box_1 drawer face, 5 mm inward bill seat",
    ),
    "tv_cabinet_pick": RobotPosition(
        key="tv_cabinet_pick",
        label="tv cabinet pick",
        # Was a bare literal inside init_pick.FAMILIES, so the one piece of
        # furniture with its own station had no entry in the station table.
        xy_m=(-3.000, -0.300),
        yaw_deg=180.0,
        anchor="TvTable north interaction face",
    ),
    "kitchen_table_place": RobotPosition(
        key="kitchen_table_place",
        label="kitchen table place",
        xy_m=translate_xy(
            (1.950, 0.300),
            KITCHEN_TABLE_INTERACTION_SHIFT_XY_M,
        ),
        yaw_deg=-90.0,
        anchor="Kitchen counter north interaction edge",
    ),
    "coat_rack_hang": RobotPosition(
        key="coat_rack_hang",
        label="coat rack hang",
        xy_m=(-0.716270, 1.827494),
        yaw_deg=0.0,
        anchor="FK-aligned canonical CoatRack branch 6 interaction face",
    ),
}


def robot_position(key: str) -> RobotPosition:
    try:
        station = ROBOT_POSITIONS[str(key)]
    except KeyError as exc:
        raise KeyError(f"unknown robot position {key!r}") from exc
    if group_shift.is_identity():
        return station
    # A drawer station exists to put the hand on one specific drawer's face;
    # the drawer is furniture and stays put, so its station does too.  The
    # drawer task entry points fail fast instead (init_pick's drawer_pick
    # guard, init_place's task-28 destination guard).
    if station.key == "coat_rack_hang":
        # The destination is a fixed canonical CoatRack branch.  A group shift
        # that moved only the robot would silently retarget a different world
        # point; moving the rack would violate the one-global-scene contract.
        # Second-axis base perturbation must instead be handled by the task
        # planner and an explicitly retargeted action.
        return station
    if station.key.startswith("coffee_table_drawer"):
        return station
    return replace(station, xy_m=group_shift.apply_xy(station.xy_m))

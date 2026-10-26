"""Canonical global coat-rack geometry and small Task 70 hang station.

The rack is structure, not an episode object.  Its floor pose is chosen once
against the current NorthInsetReturn wall: the rotated tripod remains 23 mm
clear of the wall and 4 mm inside the north room shell.  Tasks adapt the robot
station and its robot-relative action; they never move this rack.
"""

from __future__ import annotations

import math


SCHEMA = "real2sim.coat_rack_global_contract.v1"
ASSET_ID = "agent_bench.coat_rack.photo_dimensioned.v1"
PRIM_PATH = "/World/Assets/CoatRack"
ASSET_REFERENCE = "./usdz/coat_rack/coat_rack.usda"

# Current room-shell NorthInsetReturn wall: west face x=0.410 m, south/north
# limits y=[1.7592, 2.5056568] m.  The values below are the composed USD
# BBoxCache bounds, which conservatively rotate each inclined Cylinder's local
# extent box.  Consequently the reported lower Z is 6.15 mm below the true
# rounded foot contact at Z=0; the authored visual and collision geometry is
# not translated into the floor.  The conservative bound still leaves the
# rack 22.55 mm clear of the wall and 4.04 mm inside the north shell.
TRANSLATE_XYZ_M = (0.0, 2.020, 0.0)
YAW_DEG = 205.0
AABB_MIN_XYZ_M = (
    -0.5011395488544037,
    1.6266256797888303,
    -0.006150419852009696,
)
AABB_MAX_XYZ_M = (
    0.38744704775572686,
    2.501615533261932,
    1.777999996471405,
)

HANG_BRANCH_INDEX_TOP_DOWN = 6
HANG_BRANCH_SHAFT_PATH = f"{PRIM_PATH}/Branch06Shaft"
HANG_BRANCH_TIP_PATH = f"{PRIM_PATH}/Branch06Tip"
BRANCH_JUNCTION_HEIGHT_M = 42.5 * 0.0254
BRANCH_EXPOSED_LENGTH_M = 4.75 * 0.0254
BRANCH_RADIUS_M = 0.75 * 0.0254 / 2.0
BRANCH_ELEVATION_DEG = 35.0
BRANCH_LOCAL_AZIMUTH_DEG = 335.0


def _branch_world_geometry() -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    azimuth = math.radians(BRANCH_LOCAL_AZIMUTH_DEG + YAW_DEG)
    elevation = math.radians(BRANCH_ELEVATION_DEG)
    radial = (math.cos(azimuth), math.sin(azimuth))
    direction = (
        math.cos(elevation) * radial[0],
        math.cos(elevation) * radial[1],
        math.sin(elevation),
    )
    trunk_radius = 1.5 * 0.0254 / 2.0
    junction = (
        TRANSLATE_XYZ_M[0] + radial[0] * trunk_radius,
        TRANSLATE_XYZ_M[1] + radial[1] * trunk_radius,
        BRANCH_JUNCTION_HEIGHT_M,
    )
    tip_center = tuple(
        junction[index]
        + direction[index] * (BRANCH_EXPOSED_LENGTH_M - BRANCH_RADIUS_M)
        for index in range(3)
    )
    return junction, tip_center


HANG_BRANCH_JUNCTION_XYZ_M, HANG_BRANCH_TIP_CENTER_XYZ_M = (
    _branch_world_geometry()
)

# The base faces +X toward the global rack; its left hand is therefore on the
# rack's +Y side, matching the REAL3 small Task 70 view.  The 30 cm torso
# posture is compensated in the robot-relative left-arm trajectory, never by
# moving this station or the global rack.
ROBOT_STATION_XY_M = (-0.716270, 1.827494)
ROBOT_STATION_YAW_DEG = 0.0


__all__ = [name for name in globals() if name.isupper()]

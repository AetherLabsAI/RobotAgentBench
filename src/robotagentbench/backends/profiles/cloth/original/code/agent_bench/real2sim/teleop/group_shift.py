# -*- coding: utf-8 -*-
"""Per-episode rigid GROUP SHIFT of the movable task set over the fixed room.

The reproduce pipeline replays JOINT-space trajectories from a fixed-base
robot, so one rigid world-XY translation applied to everything the episode
owns -- the robot station, the manipulable objects, the acceptance zones and
the plausibility bands -- leaves the interaction itself untouched.  That is
the lever this module provides: set one delta per process and every
world-coordinate truth source moves together, which is what makes
station/object position randomization possible without touching furniture.

What never moves: furniture (AGENTS.md: adapt a task by moving the robot
station, never the furniture), support heights (Z), and anything bound to a
fixed scene prim.  Destinations that ARE fixed prims (tray, drawers, tote,
throw bins) and drawer stations are incompatible with a shift; their owners
call :func:`require_identity` and fail fast instead of scoring nonsense.

Activation -- set BEFORE any teleop module is imported (one value per
process; the queue passes it per episode):

    REAL2SIM_GROUP_SHIFT_XY_M="0.08,-0.05"    # metres, world XY
    REAL2SIM_GROUP_SHIFT_MAX_M="0.15"         # optional cap override

Default is identity: with the variable unset every constant, station and
zone is bit-identical to the un-shifted pipeline.  Stdlib only, like
machine_paths -- the container runtime imports this next to
robot_positions.py.
"""
from __future__ import annotations

import os

_ENV_SHIFT = "REAL2SIM_GROUP_SHIFT_XY_M"
_ENV_CAP = "REAL2SIM_GROUP_SHIFT_MAX_M"
_DEFAULT_CAP_M = 0.15


def _parse() -> tuple[float, float]:
    raw = os.environ.get(_ENV_SHIFT, "").strip()
    if not raw:
        return (0.0, 0.0)
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"{_ENV_SHIFT} must be 'dx,dy' in metres; got {raw!r}")
    try:
        dx, dy = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_SHIFT} must be 'dx,dy' in metres; got {raw!r}") from exc
    cap = float(os.environ.get(_ENV_CAP, _DEFAULT_CAP_M))
    if max(abs(dx), abs(dy)) > cap:
        raise ValueError(
            f"{_ENV_SHIFT}={raw!r} exceeds the {cap} m per-axis cap; a "
            "larger shift walks placements off their support surface -- "
            f"raise {_ENV_CAP} only with a reviewed reason")
    return (dx, dy)


GROUP_SHIFT_XY_M = _parse()


def is_identity() -> bool:
    return GROUP_SHIFT_XY_M == (0.0, 0.0)


def require_identity(context: str) -> None:
    """Fail fast where a shift cannot be honoured (fixed-prim destinations)."""
    if not is_identity():
        raise RuntimeError(
            f"{context} is bound to fixed scene prims and does not support "
            f"{_ENV_SHIFT}={GROUP_SHIFT_XY_M}; run it without a group shift")


def apply_xy(xy) -> tuple[float, float]:
    x, y = (float(value) for value in xy)
    return (x + GROUP_SHIFT_XY_M[0], y + GROUP_SHIFT_XY_M[1])


def apply_xyz(xyz) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in xyz)
    shifted = apply_xy((x, y))
    return (shifted[0], shifted[1], z)


def apply_zone(zone) -> tuple[float, float, float, float, float]:
    """Shift a (cx, cy, half_x, half_y, surface_z) zone; extent and Z stay."""
    cx, cy, hx, hy, surface_z = (float(value) for value in zone)
    shifted = apply_xy((cx, cy))
    return (shifted[0], shifted[1], hx, hy, surface_z)


def apply_interval(bounds, axis: str) -> tuple[float, float]:
    """Shift a 1-D (lower, upper) world interval along ``axis`` ('x'|'y')."""
    lower, upper = (float(value) for value in bounds)
    delta = GROUP_SHIFT_XY_M[{"x": 0, "y": 1}[axis]]
    return (lower + delta, upper + delta)


def apply_box_xyzxyz(box) -> tuple[float, float, float, float, float, float]:
    """Shift a (xmin, xmax, ymin, ymax, zmin, zmax) world acceptance box."""
    xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in box)
    dx, dy = GROUP_SHIFT_XY_M
    return (xmin + dx, xmax + dx, ymin + dy, ymax + dy, zmin, zmax)

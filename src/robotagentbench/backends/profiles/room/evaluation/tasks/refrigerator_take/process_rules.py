"""Approved 9.1 rules only: no USD, physics control, runtime or registration.

Frame facts must come from a trusted Isaac reader. Boolean facts are not model
inputs or proof by themselves. The physical reader/pair classifier is pending.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Real

SCHEMA = "refrigerator_take.rules.v1"
STAGE_POINTS = (("open", 3), ("grasp", 1), ("take_out", 3), ("close", 1))
OPEN_DEG = 60.0
CLOSED_DEG = 5.0
HOLD_SECONDS = 5.0
HOLD_FAILURE_PENALTY = 3
COLLISION_CAP = 1
MAXIMUM_SCORE = 8


def finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("Expected finite measured number")
    return float(value)


@dataclass(frozen=True)
class Frame:
    episode_id: str
    target_prim: str
    step_index: int
    time_s: float
    target_door_angle_deg: float  # opening-positive, relative to closed pose
    door_angles_deg: tuple[float, ...]  # every door required to be closed
    target_held: bool  # measured grasp/carry; pad + link contact is allowed
    target_fully_out: bool  # full geometry, not centre-only or height-only
    non_operational_collision: bool  # accumulated over the control interval
    hold_conditions_throughout_interval: bool  # held + out + doors closed
    measurements_complete: bool


@dataclass(frozen=True)
class Result:
    score: int
    maximum_score: int
    stages: tuple[str, ...]
    collision_deduction: int
    hold_failure_deduction: int
    continuous_hold_s: float
    success: bool
    finalized: bool


class ProcessScorer:
    """One instance per accepted episode; public lifecycle handles termination.

Observe consecutive control frames including accepted reset frame zero.
The interval flag includes substep interruptions; missing frames raise instead
of counting unobserved time. Simultaneously verified stages can advance in
dependency order, but reset cannot already be open/held.
"""
    def __init__(self, episode_id, target_prim, *, initial_state_accepted):
        if initial_state_accepted is not True:
            raise ValueError("Physical initial-state acceptance required")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Missing episode identity")
        if not isinstance(target_prim, str) or not target_prim.startswith("/"):
            raise ValueError("Missing target binding")
        self.episode_id, self.target_prim = episode_id, target_prim
        self._last = None
        self._stages = []
        self._collision = False
        self._hold_start = None
        self._hold_s = 0.0
        self._success = False
        self._finalized = False

    def observe(self, frame):
        if not isinstance(frame, Frame):
            raise ValueError("Expected trusted measured Frame")
        if (frame.episode_id, frame.target_prim) != (self.episode_id, self.target_prim):
            raise ValueError("Cross-episode or wrong-target evidence")
        for name in ("target_held", "target_fully_out", "non_operational_collision",
                     "hold_conditions_throughout_interval", "measurements_complete"):
            if type(getattr(frame, name)) is not bool:
                raise ValueError("Missing boolean fact: " + name)
        if not frame.measurements_complete:
            raise ValueError("Incomplete physical evidence is not a task failure")
        if type(frame.step_index) is not int or frame.step_index < 0:
            raise ValueError("Invalid step")
        now = finite(frame.time_s)
        angle = finite(frame.target_door_angle_deg)
        if not isinstance(frame.door_angles_deg, tuple) or not frame.door_angles_deg:
            raise ValueError("Missing refrigerator door measurements")
        closed = all(abs(finite(a)) <= CLOSED_DEG for a in frame.door_angles_deg)
        if self._last is not None and frame == self._last:
            return self.result()
        if self._finalized:
            raise ValueError("Cannot advance finalized scorer")
        if self._last is None:
            if frame.step_index != 0 or now != 0 or not closed or frame.target_held:
                raise ValueError("Expected accepted closed, unheld reset at frame zero")
            if abs(angle) > CLOSED_DEG:
                raise ValueError("Target door is not closed at reset")
            self._last = frame
            return self.result()  # initialization contacts never incur task penalties
        if (frame.step_index != self._last.step_index + 1 or now <= self._last.time_s
                or len(frame.door_angles_deg) != len(self._last.door_angles_deg)):
            raise ValueError("Missing/stale/conflicting frame or door bindings")
        self._collision |= frame.non_operational_collision
        checks = (angle >= OPEN_DEG, frame.target_held,
                  frame.target_held and frame.target_fully_out, closed)
        for (stage, _), passed in zip(STAGE_POINTS, checks):
            if stage in self._stages:
                continue
            if not passed:
                break
            self._stages.append(stage)
        hold_now = (len(self._stages) == len(STAGE_POINTS) and closed
                    and frame.target_held and frame.target_fully_out)
        if not hold_now:
            self._hold_start = None
            self._hold_s = 0.0
        else:
            if self._hold_start is None or not frame.hold_conditions_throughout_interval:
                self._hold_start = now
            self._hold_s = now - self._hold_start
        self._success = hold_now and self._hold_s >= HOLD_SECONDS
        self._last = frame
        return self.result()

    def result(self):
        hold_penalty = HOLD_FAILURE_PENALTY if self._finalized and not self._success else 0
        collision_penalty = COLLISION_CAP if self._collision else 0
        earned = sum(points for stage, points in STAGE_POINTS if stage in self._stages)
        return Result(max(0, earned - collision_penalty - hold_penalty), MAXIMUM_SCORE,
                      tuple(self._stages), collision_penalty, hold_penalty,
                      self._hold_s, self._success, self._finalized)

    def finalize(self):
        if self._last is None:
            raise ValueError("No accepted evidence to finalize")
        self._finalized = True
        return self.result()

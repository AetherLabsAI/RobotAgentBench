"""Task-local v1 standard selected under the user's delegation.

These define this task, not simulator overrides. The initialization-only
registration is unchanged and does not automatically become a scored task.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PutStandard:
    version: str = "microwave_put.beef_score.v2_open_gt40"
    payload_variant: str = "beef_box"
    door_open_deg: float = 40.0
    door_closed_abs_deg: float = 5.0
    stable_hold_s: float = 2.0
    episode_limit_s: float = 200.0
    opened_points: int = 3
    placed_and_released_points: int = 6
    closed_points: int = 1
    inner_wall_collision_penalty: int = 1
    minimum_score: int = 0

    @property
    def stage_points(self):
        return {"opened": self.opened_points,
                "placed_and_released": self.placed_and_released_points,
                "closed_after_placement": self.closed_points}

    @property
    def maximum_score(self):
        return sum(self.stage_points.values())


V1 = PutStandard()


def door_state(opening_angle_deg):
    """Measured hinge angle: closed=0, normal opening direction is positive."""
    if (type(opening_angle_deg) not in (int, float)
            or not math.isfinite(opening_angle_deg)):
        raise ValueError("Finite measured door angle required")
    return (opening_angle_deg > V1.door_open_deg,
            abs(opening_angle_deg) <= V1.door_closed_abs_deg)


def control_step_budget(control_dt_s):
    """Convert the new task limit using the EXISTING controller interval."""
    if type(control_dt_s) not in (int, float) or not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("Existing positive control interval required")
    steps = V1.episode_limit_s / control_dt_s
    if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8) or steps < 1:
        raise ValueError("The task duration must fit the existing control grid")
    return int(round(steps))

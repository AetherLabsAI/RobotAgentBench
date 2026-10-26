"""Shared placement/drop detection, the 5-x settle state machine and scoring.

2026-09-17 user confirmed: destination +3, held-object drop -0.5, payload
drop -1, each deduction once. Reuse dining_place's detector and scorer as-is.

2026-09-23 the user approved the 5-1 … 5-14 definition tables, which add an
explicit three-state machine on top of that:

  (1) placing        - not yet validly placed; carrying, lowering, retrying
  (2) placed         - gripper released AND the tray floor really bears the
                       object AND its whole horizontal projection is inside the
                       floor contour
  (3) settled        - (2) holds continuously and the object stays still for 1 s
                       -> success, episode ends immediately

Rollback: losing any part of (2) returns to (1) and clears the settle window;
moving beyond the stillness tolerance while still in (2) only clears the window.
Placement points are decided by the FINAL state - never latched from history.
Stillness tolerance (user 2026-09-23): in the tray frame, every pair of frames
inside the 1 s window within 5 mm and 5 deg.
Failure (user 2026-09-23, same approach as 4-x): the target touching the floor,
or leaving the operable region, during (1) or (2) ends the episode immediately;
failure is checked before success on the same physical step.

30-1 … 30-3 keep the earlier score-only behaviour: they pass settle=None.
"""
from dataclasses import dataclass
import math
from evaluation.tasks.dining_place.physical_detection import (
    DetectionConfig, PhysicalDetector, PhysicalFacts)
from .orientation_rules import placement_orientation

NOT_UPRIGHT_PENALTY = 1.0
SETTLE_SECONDS = 1.0
SETTLE_POSITION_TOLERANCE_M = .005
SETTLE_ANGLE_TOLERANCE_RAD = math.radians(5)
STATE_PLACING, STATE_PLACED, STATE_SETTLED = 1, 2, 3
# Stage 1 budget from each API's own five displayed episodes (hold 0.50 + carry
# 3.00 + lower 5.75 + release 1.50 = 10.75 s, median x2). Recorded only: the user
# has not decided what a stage overrun should do, so it never ends an episode.
STAGE_PLACE_BUDGET_S = 21.50


@dataclass(frozen=True)
class HeldFacts:
    step: int
    time_s: float
    height_m: float
    grasped: bool
    supported_by_destination: bool
    in_destination_region: bool
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    # Pose in the tray frame drives the stillness window; world pose would also
    # move if the tray moved. Floor contact and the operable region are the two
    # fatal facts (4-x approach, user 2026-09-23).
    position_in_destination_m: tuple[float, float, float] | None = None
    quaternion_in_destination_wxyz: tuple[float, float, float, float] | None = None
    on_floor: bool = False
    outside_workspace: bool = False


def _angle_between(a, b):
    """Relative rotation angle between two wxyz quaternions."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(max(-1., min(1., dot)))


class SettleWindow:
    """Strict reading of the table: EVERY pair of frames in the 1 s window must be
    within tolerance, so each new frame is compared against all frames still in it."""
    def __init__(self, seconds=SETTLE_SECONDS, position_m=SETTLE_POSITION_TOLERANCE_M,
                 angle_rad=SETTLE_ANGLE_TOLERANCE_RAD):
        self.seconds, self.position_m, self.angle_rad = seconds, position_m, angle_rad
        self.frames = []

    def clear(self):
        self.frames = []

    def observe(self, time_s, position, quaternion):
        if position is None or quaternion is None:
            raise ValueError('Stillness needs the pose in the destination frame')
        for _, p, q in self.frames:
            moved = math.dist(p, position) > self.position_m
            turned = _angle_between(q, quaternion) > self.angle_rad
            if moved or turned:
                self.frames = []
                break
        self.frames.append((time_s, tuple(position), tuple(quaternion)))
        return self.elapsed_s

    @property
    def elapsed_s(self):
        return 0. if not self.frames else self.frames[-1][0] - self.frames[0][0]

    @property
    def satisfied(self):
        return bool(self.frames) and self.elapsed_s + 1e-9 >= self.seconds


@dataclass(frozen=True)
class PayloadFacts:
    height_m: float
    height_in_held_body_m: float
    on_held_body: bool
    inside_held_body: bool
    above_held_body: bool
    grasped: bool
    on_other_support: bool


class PlacementProcess:
    def __init__(self, episode_id, *, task_id, carries_payload, settle=None):
        if type(carries_payload) is not bool:
            raise TypeError('Explicit payload mode required')
        # 5-1 … 5-14 run the settle state machine the user approved on 2026-09-23;
        # 30-1 … 30-3 are out of that scope and keep the earlier score-only mode.
        if settle is None:
            settle = str(task_id).startswith('5-')
        if type(settle) is not bool:
            raise TypeError('Explicit settle mode required')
        self.task_id, self.carries_payload = task_id, carries_payload
        self.detector = PhysicalDetector(episode_id)
        self.orientation = None
        # 5-x run the settle state machine; 30-x keep the earlier score-only mode.
        self.settle_enabled = settle
        self.window = SettleWindow() if settle else None
        self.state = STATE_PLACING
        self.placed_now = False
        self.started_s = None
        self.first_placed_s = None
        self.stage_place_elapsed_s = None
        self.failure = None
        self.terminal = False
        self.success = False
        self.rollbacks = 0
        self.post_terminal_samples = 0
        self.events = []

    def observe(self, held, payload=None):
        if not isinstance(held, HeldFacts):
            raise TypeError('Measured held-object facts required')
        if self.carries_payload != isinstance(payload, PayloadFacts):
            raise ValueError('Missing or unexpected payload facts')
        if not self.carries_payload and payload is not None:
            raise ValueError('Single-object placement has no payload')
        # Missing payload is NOT claimed supported/present. All disabled-role
        # facts are false; the reused detector cannot start a payload fall.
        p = payload or PayloadFacts(0., 0., False, False, False, False, False)
        orientation = placement_orientation(self.task_id, held.quaternion_wxyz)
        # Posture is a quality deduction, not region/support eligibility.
        # Preserve raw support so a sideways object does not look airborne.
        row = self.detector.observe(PhysicalFacts(held.step, held.time_s,
            held.height_m, p.height_m, p.height_in_held_body_m,
            held.grasped, held.supported_by_destination, held.in_destination_region,
            p.on_held_body, p.inside_held_body, p.above_held_body, p.grasped, p.on_other_support))
        self.orientation = orientation
        if self.settle_enabled:
            self._advance(held, row)
        return dict(state=self.state, settle_elapsed_s=(self.window.elapsed_s if self.window else None),
                    failure=self.failure, success=self.success,
                    destination_support_confirmed=row['table_support_confirmed'],
                    placement_orientation=orientation,
                    held_drop_event=row['plate_drop_event'],
                    **({'payload_drop_event': row['payload_drop_event']} if self.carries_payload else {}))

    def _event(self, held, kind, **detail):
        self.events.append(dict(kind=kind, step=held.step, time_s=held.time_s, **detail))

    def _advance(self, held, row):
        """Three-state machine. Failure is evaluated before success on the same step."""
        if self.terminal:
            # 公共 worker 一个控制步固定推进 SUBSTEPS 个物理步且不会提前退出，
            # 所以终局（成功或失败）之后还会喂进这一区间剩余的子步。终局那一刻的
            # 结论已经锁定：这些子步只计数，不再改变状态、计时或分数。
            self.post_terminal_samples += 1
            return
        if self.started_s is None:
            self.started_s = held.time_s
        # (2) needs all three at once: released, really borne by the destination,
        # whole projection inside the contour.
        placed = (not held.grasped and bool(row['table_support_confirmed'])
                  and held.in_destination_region)
        self.placed_now = placed
        if held.on_floor or held.outside_workspace:
            self.failure = 'target_on_floor' if held.on_floor else 'target_outside_workspace'
            self.terminal = True
            if self.window:
                self.window.clear()
            self._event(held, 'terminal_failure', reason=self.failure, state=self.state)
            return
        if placed:
            if self.state != STATE_PLACED:
                self.state = STATE_PLACED
                self.window.clear()
                if self.first_placed_s is None:
                    self.first_placed_s = held.time_s
                    self.stage_place_elapsed_s = held.time_s - self.started_s
                    self._event(held, 'placed', stage_elapsed_s=self.stage_place_elapsed_s,
                                stage_budget_s=STAGE_PLACE_BUDGET_S,
                                over_budget=self.stage_place_elapsed_s > STAGE_PLACE_BUDGET_S)
            before = self.window.elapsed_s
            self.window.observe(held.time_s, held.position_in_destination_m,
                                held.quaternion_in_destination_wxyz)
            if self.window.elapsed_s < before:
                self._event(held, 'stillness_restarted')
            if self.window.satisfied:
                self.state = STATE_SETTLED
                self.success = True
                self.terminal = True
                self._event(held, 'settled', settle_s=self.window.elapsed_s)
        else:
            if self.state == STATE_PLACED:
                self.rollbacks += 1
                self._event(held, 'rollback_to_placing', grasped=held.grasped,
                            support=bool(row['table_support_confirmed']),
                            in_region=held.in_destination_region)
            self.state = STATE_PLACING
            if self.window:
                self.window.clear()

    def summary(self, *, final=False):
        s = (self.detector.scorer.finalize() if final else self.detector.scorer.summary()).to_dict()
        # The final state decides the placement points; a historical placement is
        # never latched. For 5-x that is this machine's own (2) condition.
        reached = self.placed_now if self.settle_enabled else s['target_reached']
        target_points = s['target_points'] if reached else 0.
        posture_penalty = (NOT_UPRIGHT_PENALTY if reached and
            self.orientation['required'] and not self.orientation['satisfied'] else 0.)
        raw = target_points - s['plate_drop_penalty'] - (
            s['payload_drop_penalty'] if self.carries_payload else 0.) - posture_penalty
        return dict(schema='arena.coffee_table_place.process.v3' if self.settle_enabled
                    else 'arena.coffee_table_place.process.v2', task_id=self.task_id,
            episode_id=s['episode_id'], final_sample_index=s['final_sample_index'],
            state=self.state, settle_required_s=SETTLE_SECONDS,
            settle_elapsed_s=(self.window.elapsed_s if self.window else None),
            rollbacks=self.rollbacks, failure=self.failure,
            post_terminal_samples=self.post_terminal_samples,
            stage_place_elapsed_s=self.stage_place_elapsed_s,
            stage_place_budget_s=STAGE_PLACE_BUDGET_S if self.settle_enabled else None,
            target_reached=reached, target_points=target_points,
            held_drop_penalty=s['plate_drop_penalty'],
            payload_drop_penalty=s['payload_drop_penalty'] if self.carries_payload else None,
            not_upright_penalty=posture_penalty,
            raw_score=raw, score=max(s['minimum_score'],raw), minimum_score=s['minimum_score'],
            maximum_score=s['maximum_score'], carries_payload=self.carries_payload,
            placement_orientation=self.orientation,
            task_success=(self.success if self.settle_enabled else None),
            automatic_termination=bool(self.settle_enabled and self.terminal))

"""Approved shelf-pick arithmetic on trusted, contiguous physical evidence.

No USD, controller, parameter overrides or model-provided verdicts. A batch is
one control observation and includes every intervening physical sample. The
API 4-1 … 4-14 use the user-approved v4 rules (2026-09-23): 60 s total budget,
three explicit stages, and a -1 charge with a rollback to stage 1 every time a
correctly grasped target drops back onto its original shelf board.
Target identity stays committed; no fixed lift-height threshold is used.

Stage 1 (grasp) and stage 2 (separation) budgets are measured and reported only.
The user has NOT decided how a stage overrun should fail or settle, so this
module never ends an episode on a stage budget.
"""
from dataclasses import dataclass
import math
from .shelf_collision import SHELF, RULE_VERSION

TASK_ID = '4-1'
MAXIMUM_SCORE = 4
GRASP_POINTS = 2
LIFT_POINTS = 2
HOLD_S = 3.0
HOLD_PENALTY = 1
COLLISION_PENALTY = 0.5
COLLISION_CAP = 1.0
WRONG_GRASP_PENALTY = 2
DROP_BACK_PENALTY = 1
TIME_LIMIT_S = 60.0
STAGE_GRASP, STAGE_SEPARATE, STAGE_HOLD = 1, 2, 3
# Stage 1 / stage 2 budgets (s) per API: each API's own 5 displayed source
# episodes, median of (reach+insert+close) and of (first bilateral contact ->
# end of lift), each x2. Recorded against these; never a failure condition.
STAGE_BUDGET_S = {
    '4-1': (26.4, 7.3), '4-2': (25.6, 8.2), '4-3': (27.0, 8.1), '4-4': (23.1, 6.8),
    '4-5': (28.0, 7.4), '4-6': (23.9, 7.2), '4-7': (24.6, 7.3), '4-8': (25.1, 7.8),
    '4-9': (26.5, 6.5), '4-10': (25.7, 7.7), '4-11': (22.5, 7.4), '4-12': (24.1, 8.2),
    '4-13': (22.2, 8.1), '4-14': (23.0, 5.4),
}


@dataclass(frozen=True)
class PhysicalFrame:
    episode_id: str
    sample_index: int
    time_s: float
    target_grasped: bool
    target_separated_from_support: bool
    lift_m: float
    neighbor_contacts: frozenset[str]
    wrong_drinks_grasped: frozenset[str]
    target_on_floor: bool
    target_outside_workspace: bool
    complete: bool = True
    shelf_contact: bool = False
    active_target: str = ''   # instance the target_* facts describe; '' means the single fixed target


class ProcessScorer:
    def __init__(self, episode_id, physics_dt_s, task_id=TASK_ID):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Missing episode identity')
        if isinstance(physics_dt_s, bool) or not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError('Invalid physical sampling interval')
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError('Missing task identity')
        self.episode_id = episode_id
        self.task_id = task_id
        self.rule_version = RULE_VERSION
        self.required_hold_s = HOLD_S
        self.dt = physics_dt_s
        self.last = None
        self.grasp_awarded = False
        self.committed_target = None
        self.lift_awarded = False
        self.hold_start = None
        self.hold_elapsed = 0.0
        self.stage = STAGE_GRASP
        self.stage_budget_s = STAGE_BUDGET_S.get(task_id)
        self.stage_started_s = None          # episode t0, set by the first frame
        self.stage_grasp_elapsed_s = None    # t0 -> first correct grasp
        self.stage_separate_elapsed_s = None # first correct grasp -> first separation
        self.grasp_time_s = None
        self.left_support_since_arm = False  # target actually left the board since the last grasp
        self.drop_back_armed = False
        self.drop_back_count = 0
        self.drop_back_deduction = 0
        self.contact_started = {}
        self.collision_deduction = 0
        self.neighbor_deduction = 0
        self.shelf_deduction = 0
        self.wrong_deduction = 0
        self.hold_deduction = 0
        self.carrying = False
        self.hold_confirmed = False
        self.failure = None
        self.terminal = False
        self.finalized = False
        self.reason = None
        self.events = []

    def _validate_batch(self, frames):
        if not isinstance(frames, tuple) or not frames:
            raise ValueError('Nonempty physical evidence batch required')
        previous = self.last
        committed = self.committed_target
        for f in frames:
            if not isinstance(f, PhysicalFrame) or f.episode_id != self.episode_id:
                raise ValueError('Wrong episode or physical frame')
            expected = 0 if previous is None else previous.sample_index + 1
            if type(f.sample_index) is not int or f.sample_index != expected:
                raise ValueError('Missing, duplicate or out-of-order physical sample')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (f.time_s, f.lift_m)) or f.time_s < 0:
                raise ValueError('Nonfinite physical measurement')
            if previous is not None and abs(f.time_s - previous.time_s - self.dt) > 1e-6:
                raise ValueError('Physical evidence gap or changed sampling interval')
            for name in ('target_grasped', 'target_separated_from_support',
                         'target_on_floor', 'target_outside_workspace', 'complete', 'shelf_contact'):
                if type(getattr(f, name)) is not bool:
                    raise ValueError('Missing physical boolean: ' + name)
            if not f.complete:
                raise ValueError('Incomplete evidence is not a zero-score episode')
            if not isinstance(f.active_target, str) or (f.active_target and not f.active_target.startswith('/')):
                raise ValueError('Expected a resolved active target identity')
            for values in (f.neighbor_contacts, f.wrong_drinks_grasped):
                if not isinstance(values, frozenset) or any(not isinstance(v, str) or not v.startswith('/') for v in values):
                    raise ValueError('Expected resolved object-instance identities')
            if f.active_target and f.active_target in f.neighbor_contacts | f.wrong_drinks_grasped:
                raise ValueError('Active target cannot also be a neighbor or wrong object')
            if SHELF in f.neighbor_contacts:
                raise ValueError('Shelf must use its separate collision field')
            if committed is not None and f.active_target != committed:
                raise ValueError('Active target changed after the grasp was awarded')
            if f.target_grasped and committed is None:
                committed = f.active_target
            previous = f

    def _event(self, f, kind, points, **details):
        self.events.append(dict(kind=kind, sample_index=f.sample_index,
                                time_s=f.time_s, points=points, **details))

    def observe_batch(self, frames):
        if self.finalized or self.terminal:
            raise RuntimeError('Terminal episode; reset requires a new scorer')
        self._validate_batch(frames)  # reject the whole batch before state changes
        for f in frames:
            self.last = f
            if self.failure is not None:
                continue  # remaining substeps cannot earn rewards after failure
            if self.stage_started_s is None:
                self.stage_started_s = f.time_s
            if f.target_grasped and not self.grasp_awarded:
                self.grasp_awarded = True
                self.committed_target = f.active_target
                self.grasp_time_s = f.time_s
                self.stage_grasp_elapsed_s = f.time_s - self.stage_started_s
                self._event(f, 'correct_grasp', GRASP_POINTS, object_path=f.active_target or None,
                            stage_elapsed_s=self.stage_grasp_elapsed_s,
                            stage_budget_s=None if not self.stage_budget_s else self.stage_budget_s[0],
                            over_budget=None if not self.stage_budget_s else self.stage_grasp_elapsed_s > self.stage_budget_s[0])
            if f.target_grasped:
                # A correctly grasped target arms one drop-back charge; it can only
                # be charged again after the target is grasped again.
                self.drop_back_armed = self.grasp_awarded
                # Reported state only: grasped but resting on the board is stage 2
                # (waiting to leave it); the scored effect is that the hold timer
                # below is not running.
                if self.stage != STAGE_HOLD or not f.target_separated_from_support:
                    self.stage = STAGE_SEPARATE
            contacts = f.neighbor_contacts | (frozenset({SHELF}) if f.shelf_contact else frozenset())
            for obj in set(self.contact_started) - contacts:
                del self.contact_started[obj]
            for obj in sorted(contacts):
                is_new_contact = obj not in self.contact_started
                if obj not in self.contact_started:
                    self.contact_started[obj] = f.time_s
                # One onset per billing object, not per second or contact point.
                # All shelf and neighbor collisions share one episode-wide cap.
                if is_new_contact:
                    charge = min(COLLISION_PENALTY, max(0., COLLISION_CAP - self.collision_deduction))
                    self.collision_deduction += charge
                    if obj == SHELF: self.shelf_deduction += charge
                    else: self.neighbor_deduction += charge
                    self._event(f, 'shelf_contact' if obj == SHELF else 'neighbor_contact',
                                -charge, object_path=obj, charged_at_s=f.time_s,
                                cap_reached=self.collision_deduction >= COLLISION_CAP)
            if f.wrong_drinks_grasped:
                self.wrong_deduction = WRONG_GRASP_PENALTY
                self._event(f, 'wrong_drink_grasp', -WRONG_GRASP_PENALTY,
                            object_paths=sorted(f.wrong_drinks_grasped))
                self.failure = 'wrong_drink_grasp'
            elif f.target_on_floor:
                self.failure = 'target_on_floor'
            elif f.target_outside_workspace:
                self.failure = 'target_outside_workspace'
            if self.failure:
                self.hold_start = None
                self.hold_elapsed = 0.0
                self.carrying = False
                self.drop_back_armed = False
                self._event(f, 'terminal_failure', 0, reason=self.failure)
                continue
            carrying = self.grasp_awarded and f.target_grasped and f.target_separated_from_support
            if carrying:
                if not self.left_support_since_arm:
                    self.left_support_since_arm = True
                    if self.stage_separate_elapsed_s is None and self.grasp_time_s is not None:
                        self.stage_separate_elapsed_s = f.time_s - self.grasp_time_s
                        self._event(f, 'separated_from_support', 0,
                                    stage_elapsed_s=self.stage_separate_elapsed_s,
                                    stage_budget_s=None if not self.stage_budget_s else self.stage_budget_s[1],
                                    over_budget=None if not self.stage_budget_s else self.stage_separate_elapsed_s > self.stage_budget_s[1])
                self.stage = STAGE_HOLD
            # Drop back onto the original board: the user-approved predicate is
            # "released by the gripper AND supported by the original board again",
            # and it only applies to a target that actually left that board since
            # its last grasp. One charge per drop back; staying on the board does
            # not charge again, and the next charge needs a new grasp first.
            elif (self.drop_back_armed and self.left_support_since_arm
                  and not f.target_grasped and not f.target_separated_from_support):
                self.drop_back_count += 1
                self.drop_back_deduction += DROP_BACK_PENALTY
                self.drop_back_armed = False
                self.left_support_since_arm = False
                self.stage = STAGE_GRASP
                self._event(f, 'dropped_back_to_support', -DROP_BACK_PENALTY,
                            occurrence=self.drop_back_count, object_path=self.committed_target)
            self.carrying = carrying
            if carrying:
                if self.hold_start is None:
                    self.hold_start = f.time_s
                self.hold_elapsed = f.time_s - self.hold_start
            else:
                self.hold_start = None
                self.hold_elapsed = 0.0
        # Stage rewards and completion are separate: v3 gives lift credit at
        # an actually carried endpoint, but waits for 3 s before auto-success.
        self.hold_confirmed = (not self.failure and self.hold_start is not None
                               and self.hold_elapsed + 1e-9 >= self.required_hold_s)
        if not self.failure and self.carrying and not self.lift_awarded:
            self.lift_awarded = True
            self._event(self.last, 'lifted', LIFT_POINTS)
        if self.hold_confirmed:
            self._event(self.last, 'stable_hold', 0, required_hold_s=self.required_hold_s)
        self.terminal = self.failure is not None or self.hold_confirmed
        self.reason = self.failure or ('success' if self.hold_confirmed else None)
        return self.summary()

    def summary(self):
        earned = GRASP_POINTS * self.grasp_awarded + LIFT_POINTS * self.lift_awarded
        raw = (earned - self.collision_deduction - self.drop_back_deduction
               - self.wrong_deduction - self.hold_deduction)
        return dict(schema=self.rule_version, episode_id=self.episode_id,
                    task_id=self.task_id, score=max(0, raw), raw_score=raw,
                    maximum_score=MAXIMUM_SCORE, earned_points=earned,
                    grasp_awarded=self.grasp_awarded, lift_awarded=self.lift_awarded,
                    committed_target=self.committed_target,
                    collision_deduction=self.collision_deduction,
                    neighbor_collision_deduction=self.neighbor_deduction,
                    shelf_collision_deduction=self.shelf_deduction,
                    wrong_grasp_deduction=self.wrong_deduction,
                    drop_back_deduction=self.drop_back_deduction, drop_back_count=self.drop_back_count,
                    stage=self.stage, stage_budget_s=None if not self.stage_budget_s else list(self.stage_budget_s),
                    stage_grasp_elapsed_s=self.stage_grasp_elapsed_s,
                    stage_separate_elapsed_s=self.stage_separate_elapsed_s,
                    time_limit_s=TIME_LIMIT_S,
                    hold_deduction=self.hold_deduction, required_hold_s=self.required_hold_s,
                    currently_carried=self.carrying, hold_confirmed=self.hold_confirmed,
                    hold_elapsed_s=self.hold_elapsed, failure=self.failure,
                    success=self.hold_confirmed and self.failure is None,
                    reason=self.reason, finalized=self.finalized)

    def finalize(self, reason):
        if self.last is None:
            raise ValueError('Cannot settle absent evidence')
        if self.finalized:
            if reason != self.reason:
                raise ValueError('Contradictory repeated finalization')
            return self.summary()
        if self.reason is not None and reason != self.reason:
            raise ValueError('Finalization disagrees with measured terminal event')
        if self.reason is None and reason not in ('time_limit', 'client_closed'):
            raise ValueError('No evidence for requested terminal result')
        self.reason = reason
        if not self.hold_confirmed:
            self.hold_deduction = HOLD_PENALTY
            self._event(self.last, 'insufficient_stable_hold', -self.hold_deduction,
                        required_hold_s=self.required_hold_s, hold_elapsed_s=self.hold_elapsed)
        self.finalized = True
        return self.summary()

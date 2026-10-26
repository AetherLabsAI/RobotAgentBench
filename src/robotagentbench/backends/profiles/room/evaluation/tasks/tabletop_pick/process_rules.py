"""Approved tabletop-pick rules. Consumes trusted facts, never controls physics.

Each control observation supplies contiguous lightweight physical samples, not
expanded contact points. The detector must bind facts to the reset target and
measure actual holding/support; a close command is not a holding measurement.
"""
from dataclasses import dataclass
import math

RULE_VERSION = 'arena.tabletop_pick.score.v1'
TARGETS = {'7-2': '/World/Assets/Wallet', '7-1': '/World/Assets/GlassesCase'}
MAXIMUM_SCORE = 3
GRASP_POINTS = 1
LIFT_POINTS = 2
HOLD_S = 3.0
HOLD_PENALTY = 1
TIME_LIMIT_S = 180.0


@dataclass(frozen=True)
class PhysicalFrame:
    episode_id: str
    sample_index: int
    time_s: float  # simulation clock; first accepted sample is episode origin
    target_path: str
    target_held: bool
    target_separated_from_table: bool
    target_on_floor: bool
    complete: bool = True


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class ProcessScorer:
    def __init__(self, episode_id, physics_dt_s, task_id):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Missing episode identity')
        if not _finite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError('Invalid physical sampling interval')
        if task_id not in TARGETS:
            raise ValueError('Unsupported tabletop-pick task')
        self.episode_id, self.task_id = episode_id, task_id
        self.target = TARGETS[task_id]
        self.dt = physics_dt_s
        self.last = None
        self.origin = None
        self.grasp_awarded = self.lift_awarded = False
        self.hold_start = None
        self.hold_elapsed = 0.0
        self.carrying = False
        self.hold_deduction = 0
        self.reason = None
        self.finalized = False
        self.events = []

    def _validate(self, frames):
        if not isinstance(frames, tuple) or not frames:
            raise ValueError('Nonempty physical evidence tuple required')
        previous = self.last
        for frame in frames:
            if not isinstance(frame, PhysicalFrame) or frame.episode_id != self.episode_id:
                raise ValueError('Wrong episode or physical frame')
            if frame.target_path != self.target:
                raise ValueError('Evidence is not bound to the reset target')
            expected = 0 if previous is None else previous.sample_index + 1
            if type(frame.sample_index) is not int or frame.sample_index != expected:
                raise ValueError('Missing, duplicate or out-of-order physical sample')
            if not _finite(frame.time_s) or frame.time_s < 0:
                raise ValueError('Invalid simulation time')
            if previous is not None and abs(frame.time_s - previous.time_s - self.dt) > 1e-6:
                raise ValueError('Physical evidence gap or changed sampling interval')
            for name in ('target_held', 'target_separated_from_table', 'target_on_floor', 'complete'):
                if type(getattr(frame, name)) is not bool:
                    raise ValueError('Missing physical boolean: ' + name)
            if not frame.complete:
                raise ValueError('Incomplete evidence is not a zero-score episode')
            previous = frame

    def _event(self, frame, kind, points):
        self.events.append(dict(kind=kind, points=points, sample_index=frame.sample_index,
                                time_s=frame.time_s, target_path=self.target))

    def observe_batch(self, frames):
        if self.finalized or self.reason is not None:
            raise RuntimeError('Terminal episode; create a new scorer for reset')
        self._validate(frames)  # atomic rejection, including later malformed samples
        if self.origin is None:
            self.origin = frames[0].time_s
        for frame in frames:
            self.last = frame
            # A terminal physical event stops scoring even if the worker has
            # already collected further samples within this control interval.
            if self.reason is not None:
                continue
            elapsed = frame.time_s - self.origin
            if elapsed > TIME_LIMIT_S + 1e-9:
                self.reason = 'time_limit'
                continue  # late evidence cannot earn rewards or extend holding
            if frame.target_on_floor:
                self.carrying = False
                self.hold_start = None
                self.hold_elapsed = 0.0
                self.reason = 'target_on_floor'
                self._event(frame, 'terminal_failure', 0)
                continue
            if frame.target_held and not self.grasp_awarded:
                self.grasp_awarded = True
                self._event(frame, 'correct_grasp', GRASP_POINTS)
            self.carrying = frame.target_held and frame.target_separated_from_table
            if self.carrying:
                if not self.lift_awarded:
                    self.lift_awarded = True
                    self._event(frame, 'lifted', LIFT_POINTS)
                if self.hold_start is None:
                    self.hold_start = frame.time_s
                self.hold_elapsed = frame.time_s - self.hold_start
            else:
                self.hold_start = None
                self.hold_elapsed = 0.0
        # Success is checked at this control observation, never from a historic
        # substep whose grasp was subsequently lost. A floor event wins.
        if self.reason is None:
            if self.carrying and self.hold_elapsed + 1e-9 >= HOLD_S:
                self._event(self.last, 'stable_hold', 0)
                self.reason = 'success'
            elif self.last.time_s - self.origin + 1e-9 >= TIME_LIMIT_S:
                self.reason = 'time_limit'
        return self.summary()

    def summary(self):
        earned = GRASP_POINTS * self.grasp_awarded + LIFT_POINTS * self.lift_awarded
        raw = earned - self.hold_deduction
        return dict(schema=RULE_VERSION, episode_id=self.episode_id, task_id=self.task_id,
                    target_path=self.target, maximum_score=MAXIMUM_SCORE,
                    score=max(0, raw), raw_score=raw, earned_points=earned,
                    grasp_awarded=self.grasp_awarded, lift_awarded=self.lift_awarded,
                    currently_carried=self.carrying, hold_elapsed_s=self.hold_elapsed,
                    required_hold_s=HOLD_S, hold_deduction=self.hold_deduction,
                    success=self.reason == 'success',
                    failure='target_on_floor' if self.reason == 'target_on_floor' else None,
                    reason=self.reason, finalized=self.finalized,
                    time_limit_s=TIME_LIMIT_S)

    def finalize(self, reason):
        if self.last is None:
            raise ValueError('Cannot settle without valid evidence')
        if self.finalized:
            if reason != self.reason:
                raise ValueError('Contradictory repeated finalization')
            return self.summary()
        if self.reason is not None and reason != self.reason:
            raise ValueError('Finalization disagrees with measured terminal event')
        if self.reason is None and reason not in ('client_closed', 'time_limit'):
            raise ValueError('No evidence for requested terminal result')
        # The common lifecycle can end a deliberately shorter diagnostic run
        # as time_limit; this does not redefine the task's 180 s maximum.
        self.reason = reason
        if reason != 'success':
            self.hold_deduction = HOLD_PENALTY
            self._event(self.last, 'insufficient_stable_hold', -HOLD_PENALTY)
        self.finalized = True
        return self.summary()

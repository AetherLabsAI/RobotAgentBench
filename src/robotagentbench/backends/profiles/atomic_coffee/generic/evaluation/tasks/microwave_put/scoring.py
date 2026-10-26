"""Pure task scoring on trusted, current-frame progress; no Isaac control.

Not connected to the initialization-only WorkerTask. The future physics
adapter must supply ordered progress, real wall impacts and substep hold
interruptions. Model statements and gripper commands are not valid evidence.
"""
from dataclasses import dataclass, replace

from arena_api.episode_clock import ControlClock
from arena_api.task_runtime import TaskEvidence, TaskScore, Finalization
from .process_rules import PutProgress, StageEvent, finite_number
from .standards import V1, control_step_budget
from .success import check_success
from .variants import variant_for


@dataclass(frozen=True)
class ScoringFrame:
    progress: PutProgress
    inner_wall_collision: bool
    final_condition_interrupted: bool


class PutScorer:
    """One episode, one collision deduction, monotonic milestone history.

The public lifecycle still owns termination; the clock here checks incoming
frame continuity with that same existing clock implementation. No worker
budget or control/physics frequency is changed by this class.
"""
    def __init__(self, *, episode_id, task_id, payload_variant, control_dt_s):
        self.variant = variant_for(payload_variant).name
        if any(not isinstance(v, str) or not v for v in (episode_id, task_id)):
            raise ValueError("Episode and task identity required")
        self.episode_id, self.task_id = episode_id, task_id
        self.clock = ControlClock(control_step_budget(control_dt_s), control_dt_s)
        self.previous = self.result = None
        self.events = ()
        self.collision_event = None
        self.hold_since = None
        self.finalized = None

    def _validate(self, evidence):
        if (not isinstance(evidence, TaskEvidence) or evidence.source != "isaac_physics"
                or evidence.episode_id != self.episode_id or evidence.task_id != self.task_id):
            raise ValueError("Current trusted task evidence required")
        frame = evidence.data
        if not isinstance(frame, ScoringFrame) or not isinstance(frame.progress, PutProgress):
            raise ValueError("Complete ScoringFrame required")
        if type(frame.inner_wall_collision) is not bool or type(frame.final_condition_interrupted) is not bool:
            raise ValueError("Explicit measured collision and interruption flags required")
        progress = frame.progress
        if ((progress.episode_id, progress.task_id, progress.step_index, progress.time_s)
                != (evidence.episode_id, evidence.task_id, evidence.step_index, evidence.time_s)):
            raise ValueError("Progress and evidence must be from the same frame")
        if (type(progress.step_index) is not int or progress.step_index < 0
                or not finite_number(progress.time_s) or progress.time_s < 0
                or type(progress.final_candidate) is not bool
                or not finite_number(progress.final_hold_s) or progress.final_hold_s < 0):
            raise ValueError("Invalid progress values")
        if not isinstance(progress.events, tuple) or any(not isinstance(e, StageEvent) for e in progress.events):
            raise ValueError("Typed immutable milestone history required")
        names = tuple(e.stage for e in progress.events)
        stages = tuple(V1.stage_points)
        if names != stages[:len(names)] or len(names) > len(stages):
            raise ValueError("Milestones must be a unique ordered prefix")
        for e in progress.events:
            if (type(e.step_index) is not int or not 1 <= e.step_index <= progress.step_index
                    or not finite_number(e.time_s) or not 0 <= e.time_s <= progress.time_s):
                raise ValueError("Invalid milestone timestamp")
        if any((b.step_index < a.step_index or b.time_s < a.time_s)
               for a, b in zip(progress.events, progress.events[1:])):
            raise ValueError("Milestones are out of time order")
        if len(progress.events) >= 2 and progress.events[1].step_index <= progress.events[0].step_index:
            raise ValueError("Opening must be observed before placement/release")
        if progress.events[:len(self.events)] != self.events:
            raise ValueError("Previously earned milestones cannot be removed or rewritten")
        if any(e.step_index != progress.step_index or e.time_s != progress.time_s
               for e in progress.events[len(self.events):]):
            raise ValueError("New milestones must be observed on the current frame")
        if progress.final_candidate and names != stages:
            raise ValueError("Final candidate requires every ordered milestone")
        return frame

    def evaluate(self, evidence):
        frame = self._validate(evidence)
        if self.previous and evidence.step_index == self.previous.step_index:
            if evidence != self.previous:
                raise ValueError("A repeated frame cannot change its evidence")
            return self.result
        if self.finalized is not None:
            raise RuntimeError("Finalized scoring cannot consume another frame")
        progress = frame.progress
        if self.previous is None:
            if progress.step_index != 0 or progress.events or progress.final_candidate:
                raise ValueError("Scoring starts at the accepted, uncompleted initial frame")
            self.clock.start(progress.time_s)
        else:
            if progress.step_index != self.previous.step_index + 1:
                raise ValueError("Missing or stale scoring frame")
            self.clock.observe(progress.time_s)

        # Initialization and settling do not count as task collisions.
        if progress.step_index > 0 and frame.inner_wall_collision and self.collision_event is None:
            self.collision_event = dict(step_index=progress.step_index, time_s=progress.time_s)
        if not progress.final_candidate or frame.final_condition_interrupted:
            self.hold_since = None
        if progress.final_candidate and self.hold_since is None:
            self.hold_since = progress.time_s
        hold = 0.0 if self.hold_since is None else progress.time_s - self.hold_since
        current = replace(progress, final_hold_s=hold)
        earned = {e.stage: V1.stage_points[e.stage] for e in current.events}
        deduction = V1.inner_wall_collision_penalty if self.collision_event else 0
        self.result = TaskScore(
            score=max(V1.minimum_score, sum(earned.values())-deduction),
            maximum_score=V1.maximum_score,
            details=dict(standard_version=V1.version, payload_variant=self.variant,
                earned=earned, inner_wall_collision_penalty=deduction,
                first_inner_wall_collision=self.collision_event,
                final_candidate=current.final_candidate, final_hold_s=hold,
                required_hold_s=V1.stable_hold_s,
                success=check_success(current, stable_hold_s=V1.stable_hold_s),
                task_limit_s=V1.episode_limit_s,
                note="Score and success are independent; these trusted predicates still require the Isaac adapter."))
        self.events, self.previous = current.events, evidence
        return self.result

    def check_success(self, evidence, score):
        if evidence != self.previous or score is not self.result:
            raise ValueError("Use the current cached score and evidence")
        return score.details["success"]

    def check_failure(self, evidence, score):
        self.check_success(evidence, score)
        return None  # Recoverable errors continue; public lifecycle handles timeout.

    def finalize(self, evidence, score, outcome):
        success = self.check_success(evidence, score)
        if not isinstance(outcome, Finalization) or outcome.success != success:
            raise ValueError("Finalization must agree with the current success predicate")
        if self.finalized is not None and outcome != self.finalized:
            raise ValueError("Cannot change an already finalized outcome")
        self.finalized = outcome
        return score  # No hidden hold/drop/timeout deduction in v1.

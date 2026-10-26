"""Common task orchestration. No L1, microwave, robot geometry or scoring rules."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .episode_clock import ControlClock
from .task_runtime import TaskEvidence, TaskScore, Finalization


class TaskLifecycle:
    def __init__(self, task, context, registration, control_dt_s):
        if context.task_id != registration.task_id or not registration.evaluator_id:
            raise ValueError('Task lifecycle registration mismatch')
        self.task, self.context, self.registration = task, context, registration
        self.clock = ControlClock(context.max_steps, control_dt_s)
        self.phase = 'new'
        self.closed = False
        self.evidence = self.score = self.snapshot = None
        self.reason = None
        self.success = False
        self.task_failed = False
        self.settle_steps = task.settle_steps
        if type(self.settle_steps) is not int or self.settle_steps < 0:
            raise ValueError('Invalid task settling budget')

    def _require(self, *phases):
        if self.closed or self.phase not in phases:
            raise RuntimeError('Invalid task lifecycle phase: ' + self.phase)

    def prepare_scene(self, source):
        self._require('new')
        self.phase = 'preparing'
        initial = Path(self.task.initialize(source)).resolve(strict=True)
        scene = Path(self.task.randomize(initial)).resolve(strict=True)
        self.phase = 'prepared'
        return scene

    def attach(self, stage):
        self._require('prepared')
        self.task.attach(stage)
        self.phase = 'attached'

    def before_physics_reset(self, world):
        self._require('attached')
        if getattr(self, '_before_reset_requested', False):
            raise RuntimeError('Pre-reset initialization may only be requested once')
        # A partially failed setup must not be silently retried.
        self._before_reset_requested = True
        if self.task.before_physics_reset(world) is not None:
            raise ValueError('before_physics_reset must return None')

    def initialize(self, world):
        self._require('attached')
        self.task.on_physics_ready(world)
        self.phase = 'settling'

    def robot_initialization(self):
        self._require('attached')
        if getattr(self, '_robot_reset_requested', False):
            raise RuntimeError('Robot initialization may only be requested once')
        self._robot_reset_requested = True
        return self.task.robot_initialization()

    def after_physics(self, helper_report):
        self._require('settling', 'running')
        self.task.after_physics(helper_report)

    def accept_initial(self, metadata, time_s):
        self._require('settling')
        if self.task.validate_initial(metadata) is not None:
            raise ValueError('validate_initial must raise on failure and return None on acceptance')
        self.clock.start(time_s)
        self.phase = 'running'
        self._evaluate(time_s)

    def after_control(self, time_s):
        self._require('running')
        self.clock.observe(time_s)
        self._evaluate(time_s)

    @staticmethod
    def _check_score(score):
        if not isinstance(score, TaskScore):
            raise ValueError('Task must return TaskScore')
        if (any(isinstance(v, bool) or not math.isfinite(v) for v in (score.score, score.maximum_score))
                or score.maximum_score < 0 or score.score > score.maximum_score
                or not isinstance(score.details, dict)):
            raise ValueError('Invalid task score')
        json.dumps(score.details, allow_nan=False)  # Reject NaN/non-wire task details.
        return score

    def _evaluate(self, time_s):
        # Read exactly ONCE per control frame. Both predicates share this same
        # object and score; neither is allowed to reread/advance the simulator.
        evidence = self.task.collect_evidence()
        if (not isinstance(evidence, TaskEvidence) or evidence.source != 'isaac_physics'
                or evidence.episode_id != self.context.episode_id
                or evidence.task_id != self.context.task_id
                or type(evidence.step_index) is not int or evidence.step_index != self.clock.steps
                or isinstance(evidence.time_s, bool) or not math.isfinite(evidence.time_s)
                or abs(evidence.time_s-time_s) > 1e-6):
            raise ValueError('Missing, stale or cross-episode task evidence')
        score = self._check_score(self.task.evaluate(evidence))
        failure = self.task.check_failure(evidence, score)
        success = self.task.check_success(evidence, score)
        if type(success) is not bool or (failure is not None and (
                not isinstance(failure, str) or not failure.strip() or
                failure in ('success', 'time_limit', 'client_closed', 'runtime_error'))):
            raise ValueError('Invalid task outcome contract')
        self.evidence, self.score = evidence, score
        # A declared terminal failure wins over success. Success wins over a
        # deadline reached on the same measured frame. Deductions never imply
        # failure: only check_failure can declare a task-specific early stop.
        self.task_failed = failure is not None
        self.success = success and not self.task_failed
        self.reason = failure or ('success' if success else 'time_limit' if self.clock.expired else None)
        if self.reason:
            self.phase = 'terminal'
        self._snapshot()

    def _snapshot(self, *, finalized=False, valid=True):
        if self.evidence is None:
            return None
        stopped = finalized or self.reason is not None or not valid
        self.snapshot = dict(task_id=self.context.task_id, episode_id=self.context.episode_id,
            step_index=self.evidence.step_index, time_s=self.evidence.time_s,
            evaluator_id=self.registration.evaluator_id, valid=valid,
            score=self.score.score if valid else None, maximum_score=self.score.maximum_score,
            success=(self.success if stopped else None) if valid else None,
            terminated=valid and (self.success or self.task_failed),
            truncated=stopped and not (valid and (self.success or self.task_failed)),
            reason=self.reason if valid else 'runtime_error', finalized=finalized,
            details={**deepcopy(self.score.details), 'time_limit': self.clock.summary(),
                     'initial_state_accepted': True,
                     'randomization_applied': dict(self.context.randomization),
                     'lifecycle_contract': 'arena.task_lifecycle.v1'})
        return self.snapshot

    def evaluation(self):
        return self.snapshot

    def close(self, completed):
        if self.closed:
            if not completed:
                self._snapshot(finalized=True, valid=False)
            return self.snapshot
        try:
            if self.evidence is not None:
                if completed:
                    self.reason = self.reason or 'client_closed'
                    outcome = Finalization(self.reason, self.success, self.clock.summary())
                    self.score = self._check_score(self.task.finalize(self.evidence, self.score, outcome))
                self._snapshot(finalized=True, valid=completed)
        except BaseException:
            self._snapshot(finalized=True, valid=False)
            raise
        finally:
            self.closed = True
            self.phase = 'closed'
            try:
                self.task.close()
            except BaseException:
                self._snapshot(finalized=True, valid=False)
                raise
        return self.snapshot

"""Task-local budgets and trusted evaluation hooks, isolated from model input.

No task predicates are implemented yet. A timeout is a declared failure,
never a numeric score or a claim of benchmark validity.
"""
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
from typing import Annotated, Callable, Literal

from pydantic import Field, JsonValue, model_validator

from . import bimanual as b, policy_execution as pe
from .schemas import EpisodeState, Number, WireModel, Randomization
from .task_registry import default_registry


class TaskDefinition(WireModel):
    task_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")]
    instruction: Annotated[str, Field(min_length=1, max_length=1000)]
    timeout_s: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
    timeout_basis: Literal["simulation_time"] = "simulation_time"
    prefix_steps: Annotated[int, Field(strict=True, ge=1, le=30)] = 10
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**32)] = 7
    environment_task_id: str = pe.TASK_ID
    setup_status: Literal["unvalidated_existing_scene", "task_runtime"] = "unvalidated_existing_scene"
    randomization: Randomization = Field(default_factory=Randomization)
    task_parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)
    evaluator_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")] = "timeout_only.v1"

    @property
    def timeout_steps(self):
        # Preserve wall-independent simulation seconds at the public control rate.
        return int((Decimal(str(self.timeout_s)) / Decimal(str(b.CONTROL_DT))).to_integral_value(rounding=ROUND_CEILING))

    @model_validator(mode="after")
    def supported_budget(self):
        if not self.instruction.strip():
            raise ValueError("Task instruction cannot be blank")
        registration = default_registry().get(self.environment_task_id)
        self.task_parameters = registration.resolve_parameters(self.task_parameters)
        self.instruction = registration.resolve_instruction(self.instruction, self.task_parameters)
        registration.validate(self.timeout_steps, self.randomization.model_dump())
        if math.ceil(self.timeout_steps / self.prefix_steps) > registration.max_inferences:
            raise ValueError('Timeout/prefix exceeds registered task inference budget')
        return self


class TaskRegistry(WireModel):
    schema_version: Literal["arena.tasks.v1"]
    tasks: Annotated[list[TaskDefinition], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_ids(self):
        names = [task.task_id for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate task IDs")
        return self


def load_tasks(path: Path):
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate configuration key: " + key)
            value[key] = item
        return value
    registry = TaskRegistry.model_validate(json.loads(path.read_text(), object_pairs_hook=unique_keys))
    return {task.task_id: task for task in registry.tasks}


class Checkpoint(WireModel):
    checkpoint_id: Annotated[str, Field(min_length=1, max_length=100)]
    status: Literal["pending", "passed", "failed", "not_evaluated"]
    score: Number | None = None
    reason: str = ""
    metrics: dict[str, Number] = Field(default_factory=dict)


class EvaluationUpdate(WireModel):
    verdict: Literal["continue", "success", "failure"] = "continue"
    reason: Annotated[str, Field(min_length=1)] = "criteria_not_defined"
    score: Number | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)


class EvaluationContext(WireModel):
    task_id: str
    phase: Literal["reset", "step"]
    elapsed_sim_s: Number
    remaining_sim_s: Number
    state: EpisodeState


# Trusted, evaluation-side Python only. Never received from the model or wire.
Evaluator = Callable[[EvaluationContext], EvaluationUpdate]

# Add future vetted criteria here. Config names never import arbitrary modules.
def server_evaluation(context):
    """Consume typed worker evidence; never reimplement a task's predicates here."""
    state = context.state
    value = state.evaluation
    expected = default_registry().get(state.task_id).evaluator_id
    if (value is None or not value.valid or value.evaluator_id != expected
            or value.task_id != state.task_id or value.episode_id != state.episode_id
            or value.step_index != state.observation.step_index
            or abs(value.time_s - state.observation.time_s) > 1e-6):
        raise ValueError('Missing/current task evaluation required from the environment')
    verdict = ('success' if value.success is True else 'failure') if value.terminated or value.truncated else 'continue'
    return EvaluationUpdate(verdict=verdict, reason=value.reason or 'task_in_progress', score=value.score)


EVALUATORS: dict[str, Evaluator | None] = {'timeout_only.v1': None, 'server.v1': server_evaluation}


class TaskMonitor:
    def __init__(self, task: TaskDefinition, *, evaluator: Evaluator | None = None,
                 record_event: Callable[[dict], None] | None = None):
        self.task = task
        self.evaluator = evaluator
        self.record_event = record_event
        self.initial_time = None
        self.episode_id = None
        self.last_step = -1
        self.elapsed = 0.
        self.verdict = "undetermined"
        self.reason = "criteria_not_defined"
        self.score = None
        self.checkpoints = {}
        self.event_count = 0
        self.server_valid = False
        self.last_observation_time = None

    def observe(self, state: EpisodeState):
        if self.verdict != "undetermined":
            raise RuntimeError("Evaluation already terminal; no more actions may execute")
        obs = state.observation
        if state.task_id != self.task.environment_task_id:
            raise ValueError('Evaluation received a different environment task')
        if obs.source != "isaac_physics" or obs.time_basis != "simulation_time":
            raise ValueError("Task timeout requires measured simulation time")
        if obs.step_index != self.last_step + 1:
            raise ValueError("Missing, stale or reordered evaluation observation")
        if self.initial_time is None:
            self.initial_time, self.episode_id = obs.time_s, obs.episode_id
        if obs.episode_id != self.episode_id or state.episode_id != self.episode_id:
            raise ValueError("Evaluation observation belongs to a different episode")
        elapsed = obs.time_s - self.initial_time
        # Validate each acknowledged interval, not float32 dt error accumulated
        # across 6000 controls (which slightly exceeds 10 microseconds).
        if elapsed < 0 or (self.last_observation_time is not None and
                abs(obs.time_s - self.last_observation_time - b.CONTROL_DT) > 1e-6):
            raise ValueError("Task clock disagrees with acknowledged physical steps")
        self.last_observation_time = obs.time_s
        self.last_step, self.elapsed = obs.step_index, elapsed
        context = EvaluationContext(task_id=self.task.task_id,
            phase="reset" if obs.step_index == 0 else "step",
            elapsed_sim_s=elapsed, remaining_sim_s=max(0., self.task.timeout_s - elapsed),
            state=state.model_copy(deep=True))
        update = (EvaluationUpdate.model_validate(self.evaluator(context))
                  if self.evaluator else EvaluationUpdate())
        self.server_valid = self.evaluator is server_evaluation and state.evaluation_valid
        for point in update.checkpoints:
            self.checkpoints[point.checkpoint_id] = point.model_dump(mode="json")
        if update.score is not None:
            self.score = update.score
        # Evaluate the last allowed observation first; without an explicit
        # terminal criterion, exhausting the task's time budget is a failure.
        if update.verdict != "continue":
            self.verdict, self.reason = update.verdict, update.reason
        elif obs.step_index >= self.task.timeout_steps:
            self.verdict, self.reason = "failure", "task_timeout"
        self.event_count += 1
        if self.record_event:
            self.record_event({"event_index": self.event_count - 1, "phase": context.phase,
                "step_index": obs.step_index, "elapsed_sim_s": elapsed,
                "remaining_sim_s": context.remaining_sim_s,
                "update": update.model_dump(mode="json"),
                "verdict": self.verdict, "reason": self.reason})
        return None if self.verdict == "undetermined" else self.reason

    def summary(self):
        return {"verdict": self.verdict,
                "success": None if self.verdict == "undetermined" else self.verdict == "success",
                "termination_reason": self.reason, "score": self.score,
                "scoring_status": "not_evaluated" if self.evaluator is None else "evaluator_reported",
                "evaluation_valid": self.server_valid, "timeout_basis": self.task.timeout_basis,
                "timeout_s": self.task.timeout_s, "timeout_steps": self.task.timeout_steps,
                "effective_timeout_s": self.task.timeout_steps * b.CONTROL_DT,
                "elapsed_sim_s": self.elapsed, "last_evaluated_step": self.last_step,
                "evaluation_events": self.event_count,
                "checkpoints": list(self.checkpoints.values())}

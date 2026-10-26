"""Trusted task extension boundary. No task geometry, predicates or Isaac imports.

Factories are installed by server-side code, NEVER named/imported by a request.
Each factory returns a fresh WorkerTask owned by exactly one episode process.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

from . import policy_execution as pe
from .physics_defaults import steps_for_seconds


@dataclass(frozen=True)
class WorkerContext:
    episode_id: str
    task_id: str
    output: Path
    seed: int
    max_steps: int
    randomization: dict
    task_parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RobotInitialization:
    """Trusted pre-episode reset values, native units (rad / metres).

    Applied once by the worker after World.reset, before settling/scoring.
    Not a network action and never called during an episode.
    """
    positions: dict[str, float]
    hold_targets: dict[str, float]


@dataclass(frozen=True)
class TaskEvidence:
    """One current measured frame, shared by score/success/failure hooks.

    data is task-owned, trusted, read-only evidence (not model input). Hooks
    must not reread Isaac or mutate it. No contact point expansion is required.
    """
    episode_id: str
    task_id: str
    step_index: int
    time_s: float
    data: object
    source: str = 'isaac_physics'


@dataclass(frozen=True)
class TaskScore:
    score: float
    maximum_score: float
    details: dict


@dataclass(frozen=True)
class Finalization:
    """Public lifecycle decides why to stop; task only settles its score."""
    reason: str
    success: bool
    time_limit: dict


class WorkerTask:
    """Explicit task hooks, orchestrated by TaskLifecycle on the worker thread.

    Scene authoring is allowed only in initialize/randomize, BEFORE World.reset.
    Runtime hooks observe physics; they must never execute scripted task actions.
    Raise on invalid evidence/infrastructure; return a failure reason only for
    a genuine task-defined terminal failure. Score deductions are independent.
    """
    settle_steps = steps_for_seconds(0.5)

    def initialize(self, source: Path) -> Path:
        return source

    def randomize(self, scene: Path) -> Path:
        return scene

    def attach(self, stage):
        pass

    def before_physics_reset(self, world):
        """One-time task physics setup after World construction, before reset.

        Default no-op. Only approved task initialization belongs here; never
        execute actions, advance physics or reset the world from this hook.
        """
        pass

    def on_physics_ready(self, world):
        pass

    def scene_transitions(self):
        """Optional trusted ClothTransitionSession, owned by the whole episode.

        Default None: no new per-step reads or state changes for existing tasks.
        Only approved cloth static-to-dynamic activation is supported, not scene
        replacement, reset, object assistance or task scoring. The worker calls
        the session after physics/readers, before the next step, after initial
        acceptance. A native backend and full-scene reader must be provided.
        """
        return None

    def robot_initialization(self) -> RobotInitialization | None:
        return None  # Existing tasks keep their native USD reset unchanged.

    def after_physics(self, helper_report: dict):
        pass

    def validate_initial(self, metadata: dict) -> None:
        raise NotImplementedError

    def collect_evidence(self) -> TaskEvidence:
        raise NotImplementedError

    def evaluate(self, evidence: TaskEvidence) -> TaskScore:
        raise NotImplementedError

    def check_success(self, evidence: TaskEvidence, score: TaskScore) -> bool:
        raise NotImplementedError

    def check_failure(self, evidence: TaskEvidence, score: TaskScore) -> str | None:
        return None  # No early terminal failures unless the task defines them.

    def finalize(self, evidence: TaskEvidence, score: TaskScore, outcome: Finalization) -> TaskScore:
        return score

    def close(self):
        """Resource cleanup, including failed initialization; called once."""
        pass


@dataclass(frozen=True)
class TaskRegistration:
    task_id: str
    description: str
    max_steps: int = pe.MAX_STEPS
    default_steps: int = pe.MAX_STEPS
    max_inferences: int = 100
    instruction: str = ""
    controller_id: str = pe.TASK_ID
    evaluator_id: str | None = None
    runtime_factory: Callable[[WorkerContext], WorkerTask] | None = None
    randomization_flags: tuple[str, ...] = ()
    validate_randomization: Callable[[dict], None] | None = None
    ready: bool = True
    parameter_schema: dict = field(default_factory=dict)
    normalize_parameters: Callable[[dict], dict] | None = None
    parameter_instruction: Callable[[dict], str] | None = None
    gripper_execution_mode: str = pe.DEFAULT_GRIPPER_MODE

    def resolve_parameters(self, parameters):
        if not isinstance(parameters, dict):
            raise ValueError('Task parameters must be an object')
        value = json.loads(json.dumps(parameters, allow_nan=False))
        if self.normalize_parameters is None:
            if value:
                raise ValueError('This task does not accept parameters')
            return {}
        result = self.normalize_parameters(value)
        if not isinstance(result, dict):
            raise ValueError('Task parameter normalizer must return an object')
        return json.loads(json.dumps(result, allow_nan=False))

    def resolve_instruction(self, instruction, parameters):
        if self.parameter_instruction and (not instruction or instruction == self.instruction):
            return self.parameter_instruction(parameters)
        return instruction or self.instruction

    def validate(self, steps, randomization):
        if type(steps) is not int or not 1 <= steps <= self.max_steps:
            raise ValueError(f"Task {self.task_id} accepts 1..{self.max_steps} control steps")
        if any(v and k not in self.randomization_flags for k, v in randomization.items()):
            raise ValueError("This task does not support the requested randomization")
        if self.validate_randomization:
            self.validate_randomization(randomization)


class RuntimeRegistry:
    def __init__(self):
        self._tasks = {}

    def register(self, task: TaskRegistration):
        if task.task_id in self._tasks:
            raise ValueError("Duplicate task registration: " + task.task_id)
        if not 1 <= task.default_steps <= task.max_steps <= 60000:
            raise ValueError("Invalid task step budget")
        if task.gripper_execution_mode != 'direct_target.v1':
            raise ValueError('Only unmodified native-target gripper execution is supported')
        self._tasks[task.task_id] = task

    def get(self, task_id):
        if task_id not in self._tasks:
            raise ValueError("Unregistered task ID: " + task_id)
        return self._tasks[task_id]

    def all(self):
        return tuple(self._tasks.values())

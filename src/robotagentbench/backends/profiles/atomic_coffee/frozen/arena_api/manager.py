"""Single-owner episode lifecycle and retry-safe stepping, in memory only."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
from uuid import UUID

from .isaac_backend import IsaacBackend, IsaacOptions
from .schemas import EpisodeConfig, EpisodeResult, EpisodeState, Observation, StepRequest
from .schemas import TaskEvaluation
from .task_registry import default_registry
from . import robot_profiles as rp


class ArenaError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Episode:
    config: EpisodeConfig
    backend: IsaacBackend
    state: EpisodeState
    initial_state: EpisodeState
    requests: dict[UUID, tuple[StepRequest, EpisodeState]] = field(default_factory=dict)


class EpisodeManager:
    """Exactly one active episode per process; no distributed/multi-worker state."""

    def __init__(self, *, retention: int = 8,
                 isaac_options: IsaacOptions | None = None, registry=None):
        if retention < 1:
            raise ValueError("retention must be positive")
        self.isaac_options = isaac_options
        self.retention = retention
        self.episodes: OrderedDict[UUID, Episode] = OrderedDict()
        self.active_id: UUID | None = None
        self.lock = threading.RLock()
        self.stopped = False
        self.registry = registry if registry is not None else default_registry()

    def _find(self, episode_id: UUID) -> Episode:
        if episode_id not in self.episodes:
            raise ArenaError(404, "episode_not_found", "Episode is unknown or no longer retained.")
        return self.episodes[episode_id]

    def create(self, episode_id: UUID, config: EpisodeConfig) -> EpisodeState:
        with self.lock:
            try:
                task = self.registry.get(config.task_id)
                config = config.model_copy(update={'max_steps': config.max_steps or task.default_steps})
                task.validate(config.max_steps, config.randomization.model_dump())
                parameters = task.resolve_parameters(config.task_parameters)
                config = config.model_copy(update={'task_parameters': parameters,
                    'instruction': task.resolve_instruction(config.instruction, parameters)})
            except ValueError as exc:
                code = ('randomization_not_supported' if any(config.randomization.model_dump().values())
                        else 'invalid_request')
                raise ArenaError(422, code, str(exc)) from exc
            if self.stopped:
                raise ArenaError(503, "service_stopped", "Service is shutting down.")
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                if episode.config != config:
                    raise ArenaError(409, "episode_id_reused", "Use a new UUID for a different reset.")
                if episode.state.status in ("closed", "error"):
                    raise ArenaError(409, "episode_closed", "Use a new UUID to reset a closed episode.")
                return episode.initial_state.model_copy(deep=True)
            if not task.ready:
                raise ArenaError(503, "backend_not_ready", "Task backend is not ready; no fallback.")
            if self.isaac_options is None:
                raise ArenaError(503, "backend_not_ready", "Start with --backend isaac and explicit worker configuration.")
            if self.active_id is not None:
                raise ArenaError(409, "environment_busy", "Close the active episode before resetting another.")
            while len(self.episodes) >= self.retention:
                self.episodes.popitem(last=False)
            backend = None
            try:
                # Existing execution-only backends retain the original constructor.
                kwargs = ({'task_registration': task, 'max_steps': config.max_steps,
                           'randomization': config.randomization.model_dump(),
                           'task_parameters': config.task_parameters} if task.runtime_factory else {})
                backend = IsaacBackend(self.isaac_options, episode_id, config.seed, config.task_id,
                                       config.instruction or task.instruction,
                                       robot_profile=config.robot_profile, **kwargs)
                fields = backend.observation_fields(0)
                state = EpisodeState(
                    episode_id=episode_id, task_id=config.task_id, seed=config.seed, status="running",
                    task_parameters=config.task_parameters, robot_profile=config.robot_profile,
                    backend="isaac_physics",
                    observation=Observation(episode_id=episode_id, step_index=0, **fields),
                )
                self._evaluation(state, backend, task)
                state.randomization_applied = config.randomization.model_copy(deep=True)
                episode = Episode(config.model_copy(deep=True), backend, state, state.model_copy(deep=True))
                if state.terminated or state.truncated:
                    self._close_backend(episode, task)
                    episode.initial_state = state.model_copy(deep=True)
            except Exception as exc:
                if backend:
                    try:
                        backend.close()
                    except Exception:
                        pass  # Preserve initialization failure; never return a usable episode.
                raise ArenaError(503, "backend_initialization_failed",
                                 "Worker initialization failed. Inspect this episode's worker.log; no fallback.") from exc
            self.episodes[episode_id] = episode
            self.active_id = episode_id if state.status == 'running' else None
            return state.model_copy(deep=True)

    def _evaluation(self, state, backend, task):
        if not task.evaluator_id:
            return
        value = TaskEvaluation.model_validate(backend.last_evaluation)
        if (value.task_id != state.task_id or value.episode_id != state.episode_id
                or value.step_index != state.observation.step_index
                or abs(value.time_s - state.observation.time_s) > 1e-6
                or value.evaluator_id != task.evaluator_id or not value.valid):
            raise ValueError('Missing, stale, cross-episode or invalid task evaluation')
        state.evaluation = value
        state.evaluation_valid, state.success = value.valid, value.success
        state.terminated, state.truncated = value.terminated, value.truncated
        if value.terminated or value.truncated:
            state.status = 'terminated' if value.terminated else 'truncated'
            state.stop_reason = value.reason

    def _close_backend(self, episode, task):
        episode.backend.close()
        if getattr(episode.backend, 'recording_error', None):
            raise RuntimeError('Background recording incomplete')
        self._evaluation(episode.state, episode.backend, task)

    def step(self, episode_id: UUID, request: StepRequest) -> EpisodeState:
        with self.lock:
            episode = self._find(episode_id)
            task = self.registry.get(episode.config.task_id)
            cached = episode.requests.get(request.request_id)
            if cached is not None:
                if cached[0] != request:
                    raise ArenaError(409, "request_id_reused", "The same request_id cannot identify different actions.")
                return cached[1].model_copy(deep=True)
            if episode.state.status != "running":
                raise ArenaError(409, "episode_not_running", "Episode is closed, errored or has reached its step limit.")
            index = episode.state.observation.step_index
            if request.expected_step_index != index:
                raise ArenaError(409, "stale_step", f"Expected current step_index {index}; no action was executed.")
            try:
                if request.action.unit != rp.get(episode.config.robot_profile).wire_unit:
                    raise ValueError("Action units do not match the reset robot profile")
                rp.values(request.action.targets, episode.config.robot_profile)
                episode.backend.validate(request.action.targets)
                if request.physics_hz is not None:
                    validator = getattr(episode.backend, 'validate_physics_hz', None)
                    if validator is None:
                        raise ValueError('Backend does not support physics-rate switching')
                    validator(request.physics_hz)
            except ValueError as exc:
                raise ArenaError(422, "invalid_action", str(exc)) from exc
            try:
                if request.physics_hz is None:
                    episode.backend.step(request.action.targets)
                else:
                    episode.backend.step(request.action.targets, physics_hz=request.physics_hz)
                next_index = index + 1
                state = episode.state.model_copy(deep=True)
                fields = episode.backend.observation_fields(next_index)
                state.observation = Observation(episode_id=episode_id, step_index=next_index, **fields)
                state.execution = getattr(episode.backend, "last_execution", None)
                self._evaluation(state, episode.backend, task)
                episode.state = state
                if state.terminated or state.truncated:
                    self._close_backend(episode, task)
                    self.active_id = None
                elif next_index >= episode.config.max_steps:
                    state.status, state.truncated = "truncated", True
                    state.stop_reason = "max_steps"
                    self._close_backend(episode, task)
                    self.active_id = None
                episode.state = state
            except Exception as exc:
                # If a backend might have advanced before an error, fail closed.
                # Never retry the physical action automatically.
                episode.state.status = "error"
                episode.state.stop_reason = "backend_error_state_unknown"
                episode.state.evaluation_valid = False
                episode.state.success = None
                episode.state.evaluation = None
                episode.state.reward = None
                episode.state.terminated = False
                try:
                    episode.backend.close()
                except Exception:
                    pass  # Keep the original execution failure; never resume this episode.
                finally:
                    self.active_id = None
                raise ArenaError(500, "backend_error", "Execution failed; state may be uncertain. Start a new episode.") from exc
            episode.requests[request.request_id] = (request.model_copy(deep=True), state.model_copy(deep=True))
            return state.model_copy(deep=True)

    def observe(self, episode_id: UUID) -> EpisodeState:
        with self.lock:
            return self._find(episode_id).state.model_copy(deep=True)

    def lifecycle(self, episode_id: UUID, operation: str, expected_step: int) -> EpisodeState:
        """Operator pause/resume/lease renewal, serialized with physical steps.

        Resumption is only for an intact live worker, never a terminated episode.
        Clients must keep renewing the existing idle lease; no timeout is disabled.
        """
        if operation not in ('pause', 'resume', 'keepalive'):
            raise ArenaError(422, 'invalid_lifecycle_operation', 'Unknown lifecycle operation.')
        with self.lock:
            episode = self._find(episode_id)
            state = episode.state
            if self.stopped or state.status not in ('running', 'paused'):
                raise ArenaError(409, 'episode_not_resumable', 'Only live episodes can be paused or resumed.')
            if type(expected_step) is not int or expected_step != state.observation.step_index:
                raise ArenaError(409, 'stale_step', 'Lifecycle request must identify the current step.')
            try:
                episode.backend.keepalive(expected_step)
            except Exception as exc:
                state.status, state.stop_reason = 'error', 'backend_keepalive_failed'
                state.evaluation_valid, state.success, state.evaluation = False, None, None
                state.reward, state.terminated, state.truncated = None, False, False
                try:
                    episode.backend.close()
                except Exception:
                    pass
                self.active_id = None
                raise ArenaError(503, 'backend_keepalive_failed',
                                 'Worker unavailable; live resumption is not safe.') from exc
            if operation == 'pause':
                state.status = 'paused'
            elif operation == 'resume':
                state.status = 'running'
            return state.model_copy(deep=True)

    def result(self, episode_id: UUID) -> EpisodeResult:
        with self.lock:
            episode = self._find(episode_id)
            fields = episode.backend.result_fields()
            if episode.state.status == 'error':
                fields.update(evaluation_valid=False, success=None, score=None,
                              scoring_status='invalid', evaluation=None,
                              termination_reason=episode.state.stop_reason)
            return EpisodeResult(episode_id=episode_id, task_id=episode.config.task_id,
                                 status=episode.state.status, steps=episode.state.observation.step_index,
                                 seed=episode.config.seed,
                                 task_parameters=episode.config.task_parameters,
                                 robot_profile=episode.config.robot_profile, **fields)

    @property
    def physics_ready(self):
        # Liveness, not a claim of complete physical fidelity or task readiness.
        episode = self.episodes.get(self.active_id)
        return bool(episode and episode.backend.ready)

    def frame_path(self, episode_id, index, camera):
        with self.lock:
            episode = self._find(episode_id)
            path = episode.backend.frame_path(index, camera)
            if path is None:
                raise ArenaError(404, "frame_not_found", "No recorded frame for this episode, index and camera.")
            return path

    def frame_content(self, episode_id, index, camera):
        """Return immutable current RGB or a completed historical PNG path."""
        with self.lock:
            episode = self._find(episode_id)
            getter = getattr(episode.backend, 'frame_rgb', None)
            rgb = getter(index, camera) if getter else None
            if rgb is not None:
                return rgb
            return self.frame_path(episode_id, index, camera)

    def close(self, episode_id: UUID) -> EpisodeState:
        with self.lock:
            episode = self._find(episode_id)
            if episode.state.status == 'error':
                episode.backend.close()
                if getattr(episode.backend, 'recording_error', None):
                    raise ArenaError(500, 'backend_close_failed', 'Recording failure persists; inspect episode logs.')
                return episode.state.model_copy(deep=True)
            try:
                self._close_backend(episode, self.registry.get(episode.config.task_id))
            except Exception as exc:
                episode.state.status = 'error'
                episode.state.stop_reason = 'backend_close_failed'
                episode.state.evaluation_valid = False
                episode.state.success = None
                episode.state.evaluation = None
                episode.state.reward = None
                episode.state.terminated = False
                if self.active_id == episode_id:
                    self.active_id = None
                raise ArenaError(500, 'backend_close_failed',
                                 'Worker shutdown or recording failed; inspect this episode logs.') from exc
            if episode.state.status in ("running", "paused"):
                episode.state.truncated = True
                episode.state.stop_reason = "client_closed"
            episode.state.status = "closed"
            if self.active_id == episode_id:
                self.active_id = None
            return episode.state.model_copy(deep=True)

    def shutdown(self) -> None:
        with self.lock:
            self.stopped = True
            for episode_id in self.episodes:
                try:
                    self.close(episode_id)
                except ArenaError:
                    # close() already retained an error result. Continue
                    # resource cleanup without relabelling it as completed.
                    import logging
                    logging.getLogger(__name__).exception('Episode shutdown failed: %s', episode_id)

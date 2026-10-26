"""HTTP-side adapter. Isaac lives ONLY in an isolated main-thread worker."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import subprocess

from .worker_wire import receive, send
from .worker_process import shutdown_worker
from . import policy_execution as pe, robot_profiles as rp

CAMERAS = ("head", "left", "right")


def advance_fixed_steps(world, apply_action, count, physics_dt, after_step=None):
    """Rendering ticks can batch physics substeps. Keep physics-only calls exact."""
    before = float(world.current_time)
    for _ in range(count):
        apply_action()
        world.step(render=False)
        if after_step is not None:
            after_step()
    elapsed = float(world.current_time) - before
    if abs(elapsed - count * physics_dt) > 1e-6:
        raise RuntimeError(f"Unexpected physical time advance: {elapsed}, expected {count * physics_dt}")
    return elapsed


def render_until_ready(world, read_frames, ready, max_updates):
    """Warm up render products without introducing hidden physics steps."""
    before = float(world.current_time)
    for index in range(max_updates):
        world.render()
        if abs(float(world.current_time) - before) > 1e-10:
            raise RuntimeError("Rendering unexpectedly advanced physics")
        if index < 3:
            continue  # identical four render updates; no unused GPU readbacks
        frames = read_frames()
        if ready(frames):
            return frames
    raise RuntimeError("Camera render products did not become ready within bounded render updates")


def contact_offset_issue(contact, rest):
    """PhysX -inf is the documented automatic-default sentinel, not penetration.

    https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_physx_schema_physx_collision_a_p_i.html
    Preserve explicit zero errors; never export non-finite diagnostic JSON.
    """
    def display(value):
        if value is None or value == -math.inf:
            return "automatic"
        return value if math.isfinite(value) else str(value)
    if contact is None or contact == -math.inf:
        return None
    effective_rest = 0.0 if rest is None or rest == -math.inf else rest
    if not math.isfinite(contact) or not math.isfinite(effective_rest) or contact <= max(0, effective_rest):
        return {"kind": "invalid_contact_offset", "contact_offset": display(contact),
                "rest_offset": display(rest)}
    return None


def split_joint_state(names, units, positions, velocities):
    """Keep the linear torso axis out of radian fields; never infer units by index."""
    if len(names) != len(positions) or len(names) != len(velocities) or set(names) != set(units):
        raise ValueError("Incomplete robot state/unit mapping")
    if any(unit not in ("rad", "m") for unit in units.values()):
        raise ValueError("Unsupported robot joint unit")
    fields = {"joint_positions_rad": {}, "joint_velocities_rad_s": {},
              "joint_positions_m": {}, "joint_velocities_m_s": {}}
    for name, position, velocity in zip(names, positions, velocities):
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError("Non-finite robot state")
        unit = units[name]
        fields["joint_positions_" + unit][name] = float(position)
        fields["joint_velocities_" + unit + "_s"][name] = float(velocity)
    return fields


def unscored_step_budget(max_steps=None):
    """Per-run unscored execution capacity; leave the global default unchanged."""
    if max_steps is None:
        return pe.MAX_STEPS
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("Execution max_steps must be a positive integer")
    return max_steps


@dataclass(frozen=True)
class IsaacOptions:
    python: Path
    scene: Path
    output: Path
    gpu: int
    timeout_s: float = 180.0
    async_observations: bool = True
    initialization_timeout_s: float = 300.0
    diagnostic_id: str | None = None

    def __post_init__(self):
        from .worker_diagnostics import validate_diagnostic
        validate_diagnostic(self.diagnostic_id)
        for value in (self.timeout_s, self.initialization_timeout_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Worker wall-clock timeouts must be finite and positive')


class IsaacBackend:
    kind = "isaac_physics"

    def __init__(self, options, episode_id, seed, task_id=pe.TASK_ID, instruction="", *,
                 task_registration=None, max_steps=None, randomization=None, task_parameters=None,
                 robot_profile=rp.DEFAULT):
        if task_registration is None and task_id != pe.TASK_ID:
            raise ValueError("Only the current physical policy controller is supported")
        step_budget = unscored_step_budget(max_steps) if task_registration is None else max_steps
        self.robot_profile = rp.get(robot_profile).id
        self.last_execution = None
        self.last_evaluation = None
        self.task_registration = task_registration
        self.closed = False
        self.process = None
        self._worker_process_group = None
        self.sock = None
        self.metadata = {}
        self.output = options.output / str(episode_id)
        self.output.mkdir(parents=True, exist_ok=False)
        self.episode_id = episode_id
        self.last_target = {}
        from .mobile_base import ZERO
        self.last_base_velocity = dict(ZERO)
        self.fields = {}
        self.rgb_frames = {}
        self.last_rgb_index = -1
        self.recording_error = None
        self.async_observations = options.async_observations
        parent, child = socket.socketpair()
        self.sock = parent
        # Cold GPU/Kit loading and task initialization are not episode time.
        parent.settimeout(options.initialization_timeout_s)
        try:
            with (self.output / "worker.log").open("xb") as log:
                command = [
                    str(options.python), "-m", "arena_api.isaac_worker", "--fd", str(child.fileno()),
                    "--scene", str(options.scene), "--output", str(self.output),
                    "--gpu", str(options.gpu), "--seed", str(seed),
                    "--task-id", task_registration.controller_id if task_registration else task_id,
                    "--instruction", instruction, "--robot-profile", self.robot_profile,
                ]
                if task_registration:
                    command += ['--runtime-task', task_id, '--episode-id', str(episode_id),
                                '--max-steps', str(max_steps), '--randomization-json', json.dumps(randomization),
                                '--task-parameters-json', json.dumps(task_parameters or {}, allow_nan=False),
                                '--efficient-replay']
                elif max_steps is not None:
                    command += ['--max-steps', str(step_budget)]
                if options.diagnostic_id is not None:
                    command += ['--diagnostic-id', options.diagnostic_id]
                if options.async_observations:
                    command.append('--async-observations')
                self.process = subprocess.Popen(command, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1],
                    start_new_session=os.name == "posix")
                if os.name == "posix":
                    self._worker_process_group = self.process.pid
            child.close()
            response = receive(parent, image_sink=self._store_images)
            if not response.get("ok"):
                raise RuntimeError("Isaac initialization failed; inspect worker.log")
            self.metadata = response["metadata"]
            expected_spec = pe.spec(self.robot_profile)
            if task_registration is None and max_steps is not None:
                expected_spec = dict(expected_spec, max_episode_steps=step_budget)
                if self.metadata.get("max_episode_steps") != step_budget:
                    raise RuntimeError("Worker execution budget handshake mismatch")
            if self.metadata.get("action_spec") != expected_spec:
                raise RuntimeError("Worker robot profile handshake mismatch")
            self.last_evaluation = response.get('task_evaluation')
            self._update(response["observation"])
            self.last_target = dict(self.metadata["initial_action_targets_rad" if self.robot_profile == rp.JOINT16 else "initial_action_targets_native"])
            parent.settimeout(options.timeout_s)
        except BaseException as exc:
            child.close()
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Worker cleanup also failed: {cleanup_error}")
            raise

    @property
    def ready(self):
        return not self.closed and self.process is not None and self.process.poll() is None

    def _update(self, fields):
        if self.robot_profile == rp.MOBILE22:
            from .schemas import BaseState
            BaseState.model_validate(fields.get('base_state'))
        self.fields = fields
        self.positions = fields["joint_positions_rad"]
        index = fields.pop("frame_index")
        if self.async_observations and (index not in self.rgb_frames or self.rgb_frames[index][0] != fields['time_s']):
            raise RuntimeError('Missing current-frame memory images')
        self.fields["images"] = {camera: f"/v1/episodes/{self.episode_id}/frames/{index}/{camera}.png"
                                 for camera in CAMERAS}

    def _store_images(self, index, time_s, images):
        from .image_recording import check_rgb
        check_rgb(images)
        if index != self.last_rgb_index+1:
            raise RuntimeError('Missing or reordered image observation')
        self.last_rgb_index = index
        self.rgb_frames[index] = (time_s, dict(images))
        while len(self.rgb_frames) > 2:
            del self.rgb_frames[min(self.rgb_frames)]

    def frame_rgb(self, index, camera):
        if camera not in CAMERAS or index not in self.rgb_frames:
            return None
        return self.rgb_frames[index][1][camera]

    def observation_fields(self, index):
        return self.fields

    def validate(self, targets):
        pe.validate(targets, self.metadata["action_limits_rad" if self.robot_profile == rp.JOINT16
                                           else "action_limits_native"], self.robot_profile)

        for inner, outer in rp.FOLLOWERS.items():
            low, high = self.metadata["native_follower_limits"][inner]
            if not low <= -targets[outer] <= high:
                raise ValueError("Paired inner target exceeds existing USD limit: " + inner)

    def validate_physics_hz(self, hz):
        from .physics_rate import validate_hz
        validate_hz(hz)
        capability = self.metadata.get('physics_rate_switch', {})
        if not capability.get('supported'):
            raise ValueError(capability.get('reason', 'Worker does not support physics-rate switching'))

    def validate_base_velocity(self, velocity):
        from .mobile_base import validate, validate_profile
        validate_profile(self.robot_profile, velocity)
        return validate(velocity, self.last_base_velocity)

    def step(self, targets, *, physics_hz=None, base_velocity=None):
        self.validate(targets)
        from .mobile_base import validate_profile
        validate_profile(self.robot_profile, base_velocity)
        if not self.ready:
            raise RuntimeError("Isaac worker unavailable")
        command = {"op": "step", "targets": targets}
        if base_velocity is not None:
            command['base_velocity'] = self.validate_base_velocity(base_velocity)
        if physics_hz is not None:
            self.validate_physics_hz(physics_hz)
            command['physics_hz'] = physics_hz
        send(self.sock, command)
        response = receive(self.sock, image_sink=self._store_images)
        if not response.get("ok"):
            raise RuntimeError("Isaac step failed; state uncertain, inspect worker.log")
        self._update(response["observation"])
        self.last_evaluation = response.get('task_evaluation')
        self.last_execution = response.get("execution")
        if self.last_execution and 'physics_rate' in self.last_execution:
            self.metadata['latest_physics_rate'] = self.last_execution['physics_rate']
            self.metadata['physics_rate_modified_during_episode'] = self.last_execution['physics_rate']['revision'] > 0
        self.last_target = dict(targets)
        if base_velocity is not None:
            self.last_base_velocity = dict(base_velocity)
        return dict(self.positions)

    def keepalive(self, expected_step):
        """Ping the idle worker; do not advance, render, score or reset targets."""
        if not self.ready:
            raise RuntimeError('Isaac worker unavailable')
        send(self.sock, {'op': 'keepalive'})
        response = receive(self.sock)
        if (response.get('ok') is not True or response.get('op') != 'keepalive'
                or response.get('frame_index') != expected_step
                or response.get('time_s') != self.fields['time_s']):
            raise RuntimeError('Worker keepalive state mismatch')

    def frame_path(self, index, camera):
        if camera not in CAMERAS or index < 0:
            return None
        path = self.output / f"step{index:04d}_{camera}.png"
        return path if path.is_file() else None

    def result_fields(self):
        result = {"backend": self.kind,
                "reason": "Physical policy execution; task reset/scoring unavailable. See physics issues.",
                "metrics": self.metadata,
                "artifacts": list(self.fields.get("images", {}).values())}
        if self.task_registration and self.last_evaluation:
            e = self.last_evaluation
            result.update(evaluation=e, evaluation_valid=e['valid'], success=e['success'],
                          score=e['score'], maximum_score=e['maximum_score'],
                          scoring_status='finalized' if e['finalized'] else 'scoring',
                          reason=e['reason'] or 'Task evaluation in progress', termination_reason=e['reason'])
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sock:
            try:
                self.sock.settimeout(1)
                send(self.sock, {"op": "close"})
            except (OSError, ValueError):
                pass
        shutdown = None
        try:
            if self.process:
                shutdown = shutdown_worker(
                    self.process, process_group=getattr(self, '_worker_process_group', None))
                self.metadata['worker_shutdown'] = shutdown
                (self.output / 'worker_shutdown.json').write_text(
                    json.dumps(shutdown, indent=2, allow_nan=False) + '\n')
        except Exception as exc:
            self.recording_error = f'Worker shutdown failed: {exc}'
            raise RuntimeError(self.recording_error) from exc
        finally:
            if self.sock:
                self.sock.close()
        if self.async_observations and self.last_rgb_index >= 0:
            # The final image may still be encoding after the last step reply.
            # Worker exit alone is not proof that all recording finished.
            try:
                closed = json.loads((self.output/'closed.json').read_text())
                report = json.loads((self.output/'image_recording.json').read_text())
                expected = self.last_rgb_index+1
                if (closed.get('steps') != self.last_rgb_index
                        or report != closed.get('image_recording') or report.get('completed') is not True
                        or report.get('submitted_frames') != expected
                        or report.get('completed_frames') != expected or report.get('dropped_frames') != 0):
                    raise ValueError('Missing or incomplete background image recording')
                self.metadata['image_recording'] = report
            except (OSError, ValueError, TypeError) as exc:
                self.recording_error = 'Background image recording did not finish successfully'
                self.metadata['image_recording'] = dict(completed=False, error=self.recording_error)
                raise RuntimeError(self.recording_error) from exc
        if shutdown is not None and not shutdown['completed']:
            # A complete PNG record is not proof of a clean worker shutdown,
            # but an exit failure must not be mislabeled as dropped images.
            self.recording_error = (
                f"Worker shutdown failed: {shutdown.get('reason', 'unknown')}; "
                f"returncode={shutdown['returncode']}; forced={shutdown['forced']}")
            raise RuntimeError(self.recording_error)
        if getattr(self, 'task_registration', None) and self.last_evaluation is not None:
            from .schemas import TaskEvaluation
            try:
                closed = json.loads((self.output / 'closed.json').read_text())
                final = TaskEvaluation.model_validate_json((self.output / 'task_evaluation.json').read_text())
                if (self.process.returncode != 0 or closed['steps'] != final.step_index
                        or not final.valid or not final.finalized
                        or str(final.episode_id) != str(self.episode_id)):
                    raise ValueError('Incomplete task shutdown evidence')
                self.last_evaluation = final.model_dump(mode='json')
            except (OSError, ValueError, KeyError) as exc:
                self.recording_error = 'Task evaluation finalization failed'
                raise RuntimeError(self.recording_error) from exc

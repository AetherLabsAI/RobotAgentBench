"""Lifecycle-only adapter from an admitted MultiTaskRoom to existing controls.

No World, task factory, reset, scorer, physics setter, or robot-state substitute
lives here. The native owner must supply a freshly accepted room and actual
NativeDriver callbacks. Incomplete/failed transport is never a valid zero.
"""
from copy import deepcopy
import math
from types import MappingProxyType
from weakref import WeakSet
from pathlib import Path


_CLAIMED_ROOMS = WeakSet()


class OpenEndControlSession:
    """Present the SessionHost lifecycle without replacing room semantics."""

    def __init__(self, room, *, episode_id, instruction):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Native owner must supply the episode identity')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('A public scenario prompt is required')
        if (not room.started or room.closed or room.faulted
                or room.pending is not None or room.frame != 0
                or room.dispatch.ended):
            raise ValueError('Attach only to a freshly accepted live MultiTaskRoom')
        if (room.clock is None or room.clock.hz not in (60, 240)
                or not math.isfinite(room.clock.current_time)):
            raise ValueError('Existing initialized room clock required')
        if room in _CLAIMED_ROOMS:
            raise ValueError('A room cannot receive a second control session')
        self.room = room
        self.episode_id = episode_id
        # Compatibility metadata only: never a synthetic task definition/goal.
        self.plan = MappingProxyType({'instruction': instruction})
        self._closed = False
        self._error = None
        self._last_native_frame = None
        _CLAIMED_ROOMS.add(room)

    @property
    def clock(self):
        return self.room.clock

    @property
    def control_index(self):
        return self.room.frame

    @property
    def pending(self):
        return self.room.pending

    @property
    def started(self):
        return self.room.started

    @property
    def closed(self):
        return self._closed or self.room.closed

    @property
    def paused(self):
        # The owner does not step while awaiting tools. Do not pause task ports
        # or reset their persistent process state merely to take an observation.
        return False

    @property
    def faulted(self):
        return self._error is not None or self.room.faulted

    def _fault(self, exc):
        if self._error is None:
            self._error = {'type': type(exc).__name__, 'message': str(exc)}

    def begin_control(self, robot, *, command_vw, base_velocity):
        if self.closed or self.faulted:
            raise RuntimeError('Control session is closed or faulted')
        try:
            interval = self.room.begin_control(
                robot, command_vw=command_vw, base_velocity=base_velocity)
            # The existing SessionHost acknowledges whole executed 20-Hz frames.
            # Never acknowledge a zero-physics terminal boundary as one action.
            if (type(interval.get('substeps')) is not int
                    or interval['substeps'] <= 0
                    or interval.get('physics_hz') != self.clock.hz
                    or interval['substeps'] * 20 != self.clock.hz):
                raise RuntimeError('Room did not admit one physical 20-Hz interval')
            return interval
        except BaseException as exc:
            self._fault(exc)
            raise

    def after_physics(self, helper_report):
        if self.closed or self.faulted:
            raise RuntimeError('No reader updates after session failure/close')
        try:
            return self.room.after_physics(helper_report)
        except BaseException as exc:
            self._fault(exc)
            raise

    def finish_control(self, *, completed_substeps):
        if self.closed or self.faulted:
            raise RuntimeError('No frame completion after session failure/close')
        try:
            # Native room alone selects tasks, scores, revokes/restores finals,
            # advances its frame, expires its watchdog and settles terminal data.
            self._last_native_frame = self.room.finish_control(
                completed_substeps=completed_substeps)
            return self.result()
        except BaseException as exc:
            self._fault(exc)
            raise

    def result(self):
        """Private owner result. Existing public observation filters still apply."""
        native_terminal = getattr(self.room, 'terminal_result', None)
        if self.faulted:
            status, valid = 'error', False
        elif self.room.dispatch.ended:
            status = 'finished' if native_terminal is not None else 'error'
            valid = native_terminal is not None
        elif self.closed:
            status, valid = 'incomplete', False
        else:
            status, valid = 'running', False
        # Do not derive an aggregate success, change scores or terminate because
        # all ports temporarily report success. Native OpenEnd owns those rules.
        return {
            'status': status, 'evaluation_valid': valid,
            'episode_id': self.episode_id, 'step': self.control_index,
            'termination_reason': self.room.dispatch.termination_reason,
            'native_outcome': deepcopy(native_terminal),
            'last_native_frame': deepcopy(self._last_native_frame),
            'error': deepcopy(self._error),
        }

    def close(self):
        if self._closed:
            return
        if self.pending is not None:
            self._fault(RuntimeError('Closed with an incomplete physical interval'))
        try:
            # No forced terminal-score settlement on abandonment/system errors.
            # A completed room has already settled its own final_result.json.
            self.room.close()
        except BaseException as exc:
            self._fault(exc)
            raise
        finally:
            self._closed = True


def bind_native_control(room, *, episode_id, instruction, adapter,
                        mobile_controller, measured_joints,
                        apply_native_targets, simulate_fetch, capture_png):
    """Called by the native owner after acceptance; no bootstrap/parameter edits."""
    from short_tasks.agent_probe_v1.cap_v1.native_tools.native_driver import NativeDriver
    from short_tasks.agent_probe_v1.cap_v1.native_tools.host import SessionHost

    session = OpenEndControlSession(room, episode_id=episode_id, instruction=instruction)
    try:
        driver = NativeDriver(
            session, adapter, mobile_controller, measured_joints=measured_joints,
            apply_native_targets=apply_native_targets, simulate_fetch=simulate_fetch,
            capture_png=capture_png)
        return SessionHost(session, driver)
    except BaseException as exc:
        session._fault(exc)
        raise


def attach_coding_host(host, held_targets, output):
    """The unchanged broker provides move_to, base/turn/look/wait and sandbox.

    Its existing workspace allowlist remains authoritative. This function does
    not bypass it or silently fall back to the raw-action host.
    """
    from short_tasks.agent_probe_v1.cap_v1.native_tools.broker import CodingHost
    if not isinstance(host.session, OpenEndControlSession):
        raise ValueError('An OpenEnd control session is required')
    target = Path(output).resolve()
    root = Path(__file__).resolve().parent
    if target == root or not target.is_relative_to(root):
        raise ValueError('Keep this adapter workspace inside its task directory')
    return CodingHost(host, held_targets, output)

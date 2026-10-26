"""Trusted bridge into the existing ShortTaskSession, not another scorer/World.

The scene owner supplies a driver; no module names/paths from the model are loaded.
Driver contract: limits, validate_targets(targets) (read-only), observe() (read-only),
base_state() (read-only and lightweight; no camera renders),
apply_action(targets, base_velocity) (native drives only), advance(dt) (one simulate/
fetch, advance the SAME session clock, return actual helper evidence).
The owner remains responsible for native bootstrap, camera capture and World cleanup.
"""
from copy import deepcopy
import json
import math
import sys
from uuid import UUID

from arena_api import mobile_base
from .contract import MAX_TOOL_STEPS, public_base, public_observation, specification


class InputRejected(ValueError):
    pass


class SessionHost:
    def __init__(self, session, driver):
        if (not session.started or session.faulted or session.closed or session.paused
                or session.pending is not None or session.control_index != 0):
            raise ValueError('Attach once to a freshly admitted ShortTaskSession')
        self.session, self.driver = session, driver
        self.contract = specification(driver.limits)
        self.previous_base = dict(mobile_base.ZERO)
        self.closed = self.faulted = False
        self.cached = None
        self.seen_ids = set()
        first = self.observe()
        if first['instruction'] != session.plan['instruction']:
            raise ValueError('Driver instruction differs from the resolved Short Task')
        if max(abs(x) for field in ('linear_velocity_world_m_s', 'angular_velocity_world_rad_s')
               for x in first['base_state'][field]) > .005:
            raise ValueError('New mobile bridge requires a settled base, not a moving resumed episode')

    def observe(self):
        try:
            before = self.session.clock.current_time
            row = public_observation(self.driver.observe())
            if (row['episode_id'] != self.session.episode_id or row['step'] != self.session.control_index
                    or abs(row['time_s']-before) > 1e-6 or self.session.clock.current_time != before):
                raise RuntimeError('Observation advanced physics or supplied stale/cross-episode data')
            return row
        except BaseException as exc:
            self.faulted = True
            self.session._fault(exc)
            raise

    def ready(self):
        return dict(op='ready', contract=deepcopy(self.contract), observation=self.observe())

    def _validate(self, request):
        if (not isinstance(request, dict) or set(request) !=
                {'op', 'request_id', 'episode_id', 'expected_step', 'actions'}
                or request['op'] != 'execute_actions'):
            raise InputRejected('Malformed action envelope')
        try:
            UUID(request['request_id'])
        except (ValueError, TypeError, AttributeError):
            raise InputRejected('A valid request UUID is required') from None
        if (request['episode_id'] != self.session.episode_id
                or type(request['expected_step']) is not int
                or request['expected_step'] != self.session.control_index):
            raise InputRejected('Stale step or wrong episode; no action executed')
        rows = request['actions']
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TOOL_STEPS:
            raise InputRejected('Provide 1..600 explicit 22D rows')
        previous, checked = self.previous_base, []
        try:
            for row in rows:
                targets, base = mobile_base.split_action(row)
                previous = mobile_base.validate(base, previous)
                # Check complete native limits/followers via the caller's existing adapter.
                for name, target in targets.items():
                    if not self.driver.limits[name][0] <= target <= self.driver.limits[name][1]:
                        raise ValueError('Joint target outside native limits')
                self.driver.validate_targets(targets)
                checked.append((targets, base))
        except ValueError:
            # Driver exception may include scene paths/truth; never echo it to the policy.
            raise InputRejected('Invalid 22D targets, native limits or base velocity transition') from None
        return checked

    def execute(self, request):
        if self.faulted:
            raise RuntimeError('Execution outcome uncertain/invalid; bridge cannot resume')
        if not isinstance(request, dict):
            raise InputRejected('Expected action envelope')
        try:
            fingerprint = json.dumps(request, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError):
            raise InputRejected('Actions must contain finite JSON values') from None
        if self.cached and request.get('request_id') == self.cached[0]:
            if fingerprint != self.cached[1]:
                raise InputRejected('Request UUID reused with different payload')
            return deepcopy(self.cached[2])
        if request.get('request_id') in self.seen_ids:
            raise InputRejected('Expired request UUID; no action re-executed')
        if self.closed or self.session.result()['status'] != 'running':
            raise InputRejected('Episode no longer accepts actions')
        checked = self._validate(request)  # Entire chunk before any native writes.
        self.seen_ids.add(request['request_id'])
        executed = 0
        try:
            for targets, base in checked:
                now = self.session.clock.current_time
                actual = public_base(self.driver.base_state())
                if self.session.clock.current_time != now:
                    raise RuntimeError('Base measurement advanced physics')
                interval = self.session.begin_control(
                    dict(xy=actual['position_world_m'][:2],
                         yaw=math.degrees(actual['yaw_world_unwrapped_rad'])),
                    command_vw=tuple(base[k] for k in mobile_base.NAMES),
                    base_velocity=actual['linear_velocity_world_m_s']+actual['angular_velocity_world_rad_s'])
                self.driver.apply_action(targets, base)
                for _ in range(interval['substeps']):
                    report = self.driver.advance(1/interval['physics_hz'])
                    self.session.after_physics(report)
                result = self.session.finish_control(completed_substeps=interval['substeps'])
                self.previous_base = dict(base)
                executed += 1
                if result['status'] != 'running':
                    break
            reply = dict(op='result', actions_executed=executed,
                episode_finished=self.session.result()['status'] != 'running', observation=self.observe())
            self.cached = (request['request_id'], fingerprint, deepcopy(reply))
            return reply
        except BaseException as exc:
            # Do not retry an uncertain native write or settle a partial frame as a valid score.
            self.faulted = True
            self.session._fault(exc)
            raise

    def exchange(self, row):
        if not isinstance(row, dict):
            raise InputRejected('Expected an object')
        if row == {'op': 'close'}:
            self.close()
            return dict(op='closed', result=self.session.result())  # Private, never fed to GPT.
        if self.closed or self.faulted:
            raise RuntimeError('Bridge is closed or faulted')
        if row == {'op': 'observe'}:
            return dict(op='result', actions_executed=0, observation=self.observe(),
                        episode_finished=self.session.result()['status'] != 'running')
        return self.execute(row)

    def close(self):
        if not self.closed:
            self.closed = True
            self.session.close()  # Caller still owns and must close its World.


def serve(host, input_stream=None, output_stream=None):
    """One JSON line per RPC; use only with an externally authorized native owner.

    No shell, reset, dynamic factory, asset path or scoring input is model-exposed.
    On error, do not print privileged native exception details to the model pipe.
    """
    src, dst = input_stream or sys.stdin, output_stream or sys.stdout
    def emit(row):
        dst.write(json.dumps(row, allow_nan=False)+'\n')
        dst.flush()
    try:
        emit(host.ready())
        while True:
            line = src.readline(1024*1024+1)
            if not line:
                break
            if len(line) > 1024*1024:
                raise RuntimeError('Oversized RPC')
            try:
                row = json.loads(line)
                reply = host.exchange(row)
            except (InputRejected, json.JSONDecodeError):
                reply = dict(op='result', status='rejected', actions_executed=0,
                             error='Invalid arguments or stale request; no action from this request executed',
                             observation=host.observe(),
                             episode_finished=host.session.result()['status'] != 'running')
            emit(reply)
            if reply['op'] == 'closed':
                break
    finally:
        host.close()

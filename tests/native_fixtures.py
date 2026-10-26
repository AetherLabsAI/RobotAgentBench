"""Deterministic CPU doubles; these are not simulated physical evidence."""
import base64
from copy import deepcopy
from types import SimpleNamespace
import struct
import zlib

from robotagentbench.interfaces import mobile_base, robot_profiles as rp

REQUEST_ID = '00000000-0000-0000-0000-000000000001'


def png(width=640, height=480):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data)))
    data = b'\x89PNG\r\n\x1a\n'
    data += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress((b'\x00' + b'\x00' * (width * 3)) * height))
    data += chunk(b'IEND', b'')
    return base64.b64encode(data).decode()


PNG = png()


def base_state():
    return dict(position_world_m=[0., 0., 0.], orientation_world_xyzw=[0., 0., 0., 1.],
                yaw_world_unwrapped_rad=0., linear_velocity_world_m_s=[0., 0., 0.],
                angular_velocity_world_rad_s=[0., 0., 0.])


class SessionDouble:
    def __init__(self, *, hz=60, terminal_after=None):
        self.clock = SimpleNamespace(hz=hz, current_time=0.)
        self.started, self.closed, self.paused, self.faulted = True, False, False, False
        self.pending = None
        self.control_index = 0
        self.episode_id = 'test-episode'
        self.plan = {'instruction': 'Test instruction'}
        self.terminal_after = terminal_after
        self.trace = []
        self.close_count = 0

    def result(self):
        return {'status': 'error' if self.faulted else (
            'finished' if self.terminal_after is not None and self.control_index >= self.terminal_after
            else 'running'), 'opaque_task_result': {'preserved': True}}

    def _fault(self, error):
        self.faulted = True
        self.trace.append(('fault', type(error).__name__))

    def begin_control(self, robot, *, command_vw, base_velocity):
        self.trace.append(('begin', command_vw))
        self.pending = {'substeps': self.clock.hz // 20, 'physics_hz': self.clock.hz}
        return dict(self.pending)

    def before_physics(self, dt):
        self.trace.append(('before', dt))

    def after_physics(self, report):
        self.trace.append(('after', deepcopy(report)))

    def finish_control(self, *, completed_substeps):
        assert completed_substeps == self.pending['substeps']
        self.trace.append(('finish', completed_substeps))
        self.pending = None
        self.control_index += 1
        return self.result()

    def close(self):
        self.closed = True
        self.close_count += 1


def make_host(host_class, driver_class, **session_options):
    session = SessionDouble(**session_options)
    native = rp.NativeAdapter(rp.MOBILE22, list(rp.NATIVE_JOINTS), rp.NATIVE_UNITS,
                              list(rp.NATIVE_JOINTS), {name: 0. for name in rp.NATIVE_JOINTS},
                              {name: (-10., 10.) for name in rp.NATIVE_JOINTS})

    class Mobile:
        previous = dict(mobile_base.ZERO)

        def observe(self):
            return base_state()

        def apply(self, velocity):
            self.previous = dict(velocity)
            session.trace.append(('base', dict(velocity)))

    def targets(value):
        session.trace.append(('targets', value))

    def advance(dt):
        session.trace.append(('simulate', dt))
        session.clock.current_time += dt
        return {'measured_test_event': session.clock.current_time}

    def capture():
        session.trace.append(('capture', session.control_index))
        return dict.fromkeys(('head', 'left', 'right'), PNG)

    driver = driver_class(session, native, Mobile(),
                          measured_joints=lambda: dict.fromkeys(rp.NATIVE_JOINTS, 0.),
                          apply_native_targets=targets, simulate_fetch=advance, capture_png=capture)
    return host_class(session, driver), session


def request(*, count=1, step=0, request_id=REQUEST_ID):
    return dict(op='execute_actions', request_id=request_id, episode_id='test-episode',
                expected_step=step, actions=[[0.] * 22 for _ in range(count)])

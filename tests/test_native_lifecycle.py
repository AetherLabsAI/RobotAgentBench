"""Lifecycle delegation checks; task scoring remains outside these tests."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from robotagentbench.runtime.room_session import OpenEndControlSession
from robotagentbench.simulation.bootstrap import camera_resolutions, valid_camera_frame
from robotagentbench.simulation.clock import NativeSceneClock


class RoomDouble:
    def __init__(self):
        self.started, self.closed, self.faulted = True, False, False
        self.pending, self.frame = None, 0
        self.clock = SimpleNamespace(hz=60, current_time=4.)
        self.dispatch = SimpleNamespace(ended=False, termination_reason=None)
        self.terminal_result = None
        self.trace = []

    def begin_control(self, robot, **kwargs):
        self.trace.append(('begin', robot, kwargs))
        self.pending = {'substeps': 3, 'physics_hz': 60}
        return dict(self.pending)

    def after_physics(self, report):
        self.trace.append(('after', report))

    def finish_control(self, *, completed_substeps):
        self.trace.append(('finish', completed_substeps))
        self.frame += 1
        self.pending = None
        self.dispatch.ended = True
        self.dispatch.termination_reason = 'native_test_termination'
        self.terminal_result = {'opaque_score': {'process': 7, 'final': 3},
                                'native_reason': 'unchanged task-owned outcome'}
        return {'native_frame': self.frame}

    def close(self):
        self.trace.append(('close',))
        self.closed = True


def attach(room):
    return OpenEndControlSession(room, episode_id='test-episode', instruction='Test request')


class NativeLifecycleTests(unittest.TestCase):
    def test_native_room_alone_owns_progress_termination_and_score(self):
        room = RoomDouble()
        session = attach(room)
        session.begin_control({'xy': [0, 0]}, command_vw=(0., 0.), base_velocity=(0.,) * 6)
        session.after_physics({'native_test_contact': True})
        result = session.finish_control(completed_substeps=3)
        self.assertEqual(result['native_outcome'], room.terminal_result)
        self.assertTrue(result['evaluation_valid'])
        self.assertEqual(result['termination_reason'], room.dispatch.termination_reason)
        result['native_outcome']['opaque_score'].clear()
        self.assertEqual(room.terminal_result['opaque_score'], {'process': 7, 'final': 3})
        self.assertEqual([row[0] for row in room.trace], ['begin', 'after', 'finish'])

    def test_observation_and_result_reads_do_not_step_or_finalize_room(self):
        room = RoomDouble()
        session = attach(room)
        before = room.clock.current_time
        for _ in range(3):
            self.assertEqual(session.result()['status'], 'running')
            self.assertIsNone(session.result()['native_outcome'])
        self.assertEqual(room.trace, [])
        self.assertEqual(room.clock.current_time, before)
        self.assertFalse(room.dispatch.ended)

    def test_abandoned_room_is_incomplete_not_zero_and_close_is_idempotent(self):
        room = RoomDouble()
        session = attach(room)
        session.close()
        session.close()
        outcome = session.result()
        self.assertEqual(outcome['status'], 'incomplete')
        self.assertFalse(outcome['evaluation_valid'])
        self.assertIsNone(outcome['native_outcome'])
        self.assertEqual(room.trace, [('close',)])

    def test_close_with_partial_frame_is_invalid_without_forced_settlement(self):
        room = RoomDouble()
        session = attach(room)
        session.begin_control({}, command_vw=(0., 0.), base_velocity=(0.,) * 6)
        session.close()
        self.assertEqual(session.result()['status'], 'error')
        self.assertFalse(session.result()['evaluation_valid'])
        self.assertIsNone(room.terminal_result)
        self.assertNotIn('finish', [row[0] for row in room.trace])

    def test_room_can_only_be_claimed_once(self):
        room = RoomDouble()
        session = attach(room)
        with self.assertRaises(ValueError):
            attach(room)
        self.assertEqual(session.result()['status'], 'running')

    def test_invalid_control_interval_faults_instead_of_acknowledging_action(self):
        room = RoomDouble()
        room.begin_control = lambda *args, **kwargs: {'substeps': 0, 'physics_hz': 60}
        session = attach(room)
        with self.assertRaises(RuntimeError):
            session.begin_control({}, command_vw=(0., 0.), base_velocity=(0.,) * 6)
        self.assertFalse(session.result()['evaluation_valid'])
        self.assertEqual(session.result()['status'], 'error')

    def test_ended_room_without_native_terminal_receipt_is_invalid(self):
        room = RoomDouble()
        session = attach(room)
        room.dispatch.ended = True
        room.dispatch.termination_reason = 'test_transport_error'
        self.assertEqual(session.result()['status'], 'error')
        self.assertFalse(session.result()['evaluation_valid'])
        self.assertIsNone(session.result()['native_outcome'])

    def test_session_does_not_replace_task_timeout(self):
        room = RoomDouble()
        session = attach(room)
        room.dispatch.ended = True
        room.dispatch.termination_reason = 'native_no_progress_timeout'
        room.terminal_result = {'task_owned_partial_result': [4, 9]}
        before = deepcopy(room.terminal_result)
        result = session.result()
        self.assertEqual(result['termination_reason'], 'native_no_progress_timeout')
        self.assertEqual(result['native_outcome'], before)
        self.assertEqual(room.trace, [])

    def test_full_camera_profiles_are_not_resized_or_task_dependent(self):
        expected = {'head': (1280, 800), 'left': (848, 480), 'right': (848, 480)}
        for name in ('prepare_coffee', 'openend', 'select_trash'):
            self.assertEqual(camera_resolutions(name), expected)
        returned = camera_resolutions('openend')
        returned.clear()
        self.assertEqual(camera_resolutions('openend'), expected)
        self.assertTrue(valid_camera_frame('head', SimpleNamespace(shape=(800, 1280, 4), dtype='uint8')))
        self.assertFalse(valid_camera_frame('head', SimpleNamespace(shape=(480, 640, 4), dtype='uint8')))

    def test_clock_uses_measured_time_and_only_updates_session_layer(self):
        class Attribute:
            value = 60

            def Get(self):
                return self.value

            def Set(self, value):
                self.value = value

        attribute = Attribute()
        layer = object()
        stage = SimpleNamespace(
            GetPrimAtPath=lambda path: SimpleNamespace(GetAttribute=lambda name: attribute),
            GetSessionLayer=lambda: layer)
        calls = []
        simulation = SimpleNamespace(flush_changes=lambda: calls.append('flush'))
        clock = NativeSceneClock(stage, simulation, lambda: 3.75, lambda: b'unchanged')
        self.assertEqual(clock.current_time, 3.75)
        self.assertEqual(clock.get_physics_dt(), 1 / 60)

        class EditContext:
            def __init__(self, target, target_layer):
                self.assertions = (target is stage, target_layer is layer)
                assert all(self.assertions)

            def __enter__(self):
                calls.append('session_layer_enter')

            def __exit__(self, *args):
                calls.append('session_layer_exit')

        with patch.dict('sys.modules', {'pxr': SimpleNamespace(Usd=SimpleNamespace(EditContext=EditContext))}):
            clock.set_hz(240)
        self.assertEqual(clock.current_time, 3.75)
        self.assertEqual(clock.hz, 240)
        self.assertEqual(calls, ['session_layer_enter', 'session_layer_exit', 'flush'])
        with self.assertRaises(ValueError):
            clock.set_hz(120)


if __name__ == '__main__':
    unittest.main()

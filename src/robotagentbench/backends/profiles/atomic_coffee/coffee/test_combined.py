"""Combined lifecycle wiring tests with fake native actors; never physics proof."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from .combined_session import CombinedNativeSession
from .variants import BAG_MOVE, MACHINE_PUSH


class FakeActor:
    def __init__(self, stage, initial):
        self.completed = self.retired = False
        self.completion_evidence = None
        self.pending_event = dict(start=False, reason='not_eligible')
        self.commands = []
        self.observations = []
        self.bind = Mock()
        self.arm = Mock()
        self.close = Mock()

    def observe(self, **kwargs):
        self.observations.append(kwargs)
        return self.pending_event

    def command_before_step(self, now):
        self.commands.append(now)
        return {'phase': 'fake_native_command'}


class CombinedSessionTests(unittest.TestCase):
    def make(self):
        bag = ModuleType(__package__ + '.native_bag')
        bag.NativeBagRelocation = FakeActor
        push = ModuleType(__package__ + '.native_push')
        push.NativePalmPush = FakeActor
        with patch.dict(sys.modules, {bag.__name__: bag, push.__name__: push}):
            session = CombinedNativeSession(None, dict(variant='bag_then_machine', relocation_pair={}))
        reader = SimpleNamespace(world=SimpleNamespace(current_time=1.), last_time=1., latest={'time_s': 1.})
        session.bind('one-native-view', reader)
        session.arm(robot_native_poses={'robot': 'native'})
        return session, reader

    def tick(self, session, reader, now):
        reader.world.current_time = reader.last_time = now
        reader.latest = {'time_s': now}
        return session.observe(robot_native_poses={'robot': 'native'})

    def bag_finished(self, session, reader):
        session.bag.pending_event = dict(start=True, reason='proximity')
        self.tick(session, reader, 1.)
        session.bag.pending_event = dict(start=False, reason='done')
        session.bag.completed = session.bag.retired = True
        session.bag.completion_evidence = {'supported': True, 'in_training_domain': True}
        self.tick(session, reader, 2.)

    def test_both_actors_share_reader_and_view(self):
        session, reader = self.make()
        for actor in (session.bag, session.push):
            actor.bind.assert_called_once_with('one-native-view', reader)
        session.bag.arm.assert_called_once_with(robot_native_poses={'robot': 'native'})
        session.push.arm.assert_called_once_with()

    def test_only_bag_observed_before_completion(self):
        session, reader = self.make()
        self.tick(session, reader, 1.)
        self.assertEqual(len(session.bag.observations), 1)
        self.assertEqual(session.push.observations, [])
        self.assertFalse(session.robot_process_scoring_enabled)

    def test_commands_never_freeze_robot_or_run_two_humans(self):
        session, reader = self.make()
        self.assertEqual(session.command_before_step(1.)['phase'], 'absent')
        session.bag.pending_event = dict(start=True, reason='proximity')
        self.tick(session, reader, 1.)
        session.command_before_step(1.)
        self.assertEqual(session.bag.commands, [1.])
        self.assertEqual(session.push.commands, [])

    def test_after_bag_native_completion_push_still_waits_for_own_trigger(self):
        session, reader = self.make()
        self.bag_finished(session, reader)
        self.assertTrue(session.robot_process_scoring_enabled)
        self.assertEqual(session.sequence.eligible_component, MACHINE_PUSH)
        self.assertEqual(session.push.observations, [])
        self.tick(session, reader, 3.)
        self.assertIsNone(session.sequence.active)
        self.assertEqual(session.command_before_step(3.)['phase'], 'absent')
        session.push.pending_event = dict(start=True, reason='current_hold_mouth_and_tilt')
        self.tick(session, reader, 4.)
        session.command_before_step(4.)
        self.assertEqual(session.push.commands, [4.])
        self.assertEqual(session.sequence.active, MACHINE_PUSH)

    def test_unproven_bag_completion_cannot_unlock(self):
        session, reader = self.make()
        session.bag.pending_event = dict(start=True, reason='proximity')
        self.tick(session, reader, 1.)
        session.bag.completed = session.bag.retired = True
        session.bag.pending_event = dict(start=False, reason='done')
        with self.assertRaisesRegex(RuntimeError, 'Missing native'):
            self.tick(session, reader, 2.)
        self.assertIsNotNone(session.sequence.invalid_reason)
        self.assertFalse(session.robot_process_scoring_enabled)
        self.assertEqual(session.push.observations, [])
        with self.assertRaisesRegex(RuntimeError, 'cannot resume'):
            session.command_before_step(2.)
        with self.assertRaisesRegex(RuntimeError, 'cannot resume'):
            self.tick(session, reader, 3.)

    def test_duplicate_or_stale_reader_rejected_before_actor(self):
        session, reader = self.make()
        self.tick(session, reader, 1.)
        with self.assertRaisesRegex(RuntimeError, 'fresh native'):
            self.tick(session, reader, 1.)
        reader.world.current_time = 2.
        with self.assertRaisesRegex(RuntimeError, 'fresh native'):
            session.observe(robot_native_poses={})
        self.assertEqual(len(session.bag.observations), 1)

    def test_machine_appearance_is_not_second_trigger(self):
        session, reader = self.make()
        self.bag_finished(session, reader)
        session.push.pending_event = dict(start=True, reason='queued')
        self.tick(session, reader, 3.)
        session.push.pending_event = dict(start=False, reason='human_appeared')
        self.tick(session, reader, 4.)
        self.assertEqual(sum(e['kind'] == 'started' and e['component'] == MACHINE_PUSH for e in session.events), 1)
        session.push.completed = session.push.retired = True
        self.tick(session, reader, 5.)
        self.assertEqual(session.sequence.completed, [BAG_MOVE, MACHINE_PUSH])
        self.tick(session, reader, 6.)
        self.assertEqual(len(session.push.observations), 3)

    def test_cleanup_attempts_both_after_failure(self):
        session, reader = self.make()
        session.bag.close.side_effect = RuntimeError('bag cleanup')
        with self.assertRaisesRegex(RuntimeError, 'bag cleanup'):
            session.close()
        session.push.close.assert_called_once_with()

    def test_builder_rejects_source_replay_before_writing(self):
        from .build_combined import build
        with self.assertRaisesRegex(ValueError, 'paired/random'):
            build(Path('unused'), seed=2, role='source')

    def test_cli_keeps_machine_default_but_dispatches_combined_explicitly(self):
        from . import __main__ as entry
        with patch(__package__ + '.build_combined.build', return_value={'combined': True}) as build:
            self.assertEqual(entry.prepare(Path('unused'), 2, 'random', variant='bag_then_machine'), {'combined': True})
            build.assert_called_once_with(Path('unused'), seed=2, role='random')
        with self.assertRaisesRegex(ValueError, 'separate'):
            entry.prepare(Path('unused'), 2, 'random', variant='bag_only')


if __name__ == '__main__':
    unittest.main()

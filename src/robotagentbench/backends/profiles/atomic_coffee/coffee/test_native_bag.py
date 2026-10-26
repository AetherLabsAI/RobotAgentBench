"""Synthetic actor evidence tests; no native physics or successful grasp claim."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from evaluation.tasks.coffee_machine.specs import BAG
from evaluation.tasks.coffee_machine.facts import COUNTER
from .native_bag import NativeBagRelocation


class BagActorEvidenceTests(unittest.TestCase):
    def fixture(self):
        actor = object.__new__(NativeBagRelocation)
        actor.target_path = BAG
        actor.grasp = SimpleNamespace(active=False, ever_acquired=True)
        actor.released = True
        actor.trigger = SimpleNamespace(locked_direction=dict(
            start_origin_m=[.2, .2, 0.], reference_position_m=[0., 0., 0.]))
        actor.reservation = {'training_domain': {'origin_bounds_world_xy_m': [[0., 0.], [1., 1.]]}}
        actor.reader = SimpleNamespace(world=SimpleNamespace(current_time=1.), last_time=1., beans_in_bag=lambda: 100)
        row = dict(time_s=1., poses={BAG: {'position_m': [.3, .2, 0.]}},
                   contacts={BAG: {'supports': [COUNTER]}})
        actor.reader.latest = row
        return actor, row

    def test_native_release_support_domain_and_away_motion_required(self):
        actor, row = self.fixture()
        result = actor._validate_completion(row)
        self.assertTrue(result['supported'])
        self.assertTrue(result['in_training_domain'])
        self.assertTrue(result['direction']['allowed'])
        self.assertEqual(result['beans_in_bag'], 100)

    def test_timer_or_release_without_actual_acquisition_rejected(self):
        for acquired, active, released in ((False, False, True), (True, True, True), (True, False, False)):
            actor, row = self.fixture()
            actor.grasp.ever_acquired, actor.grasp.active, actor.released = acquired, active, released
            with self.assertRaises(RuntimeError):
                actor._validate_completion(row)

    def test_bad_final_support_domain_or_direction_rejected(self):
        for position, supports in (([.3, .2, 0.], []), ([1.01, .2, 0.], [COUNTER]), ([.1, .2, 0.], [COUNTER])):
            actor, row = self.fixture()
            row['poses'][BAG]['position_m'] = position
            row['contacts'][BAG]['supports'] = supports
            with self.assertRaisesRegex(RuntimeError, 'support/domain/away'):
                actor._validate_completion(row)

    def test_stale_reader_frame_is_not_completion_evidence(self):
        actor, row = self.fixture()
        self.assertIs(actor._current(), row)
        actor.reader.world.current_time = 2.
        with self.assertRaisesRegex(RuntimeError, 'Fresh validated'):
            actor._current()

    def test_closed_hand_without_contact_cannot_carry(self):
        actor, row = self.fixture()
        actor.start, actor.retired = 0., False
        actor.grasp = SimpleNamespace(active=False, ever_acquired=False, acquire=Mock())
        actor.motion = SimpleNamespace(sample=lambda elapsed: SimpleNamespace(phase='carry'))
        actor.pairs = {}
        actor.human = SimpleNamespace(links={'hand.R': '/Human/Hand'})
        actor.digits = ['thumb', 'index']
        actor.closure = SimpleNamespace(ready=lambda digits: True)
        actor.last_closure = {'thumb': 1., 'index': 1.}
        with self.assertRaisesRegex(RuntimeError, 'No carrying'):
            actor.command_before_step(1.)
        actor.grasp.acquire.assert_not_called()

    def test_cleanup_releases_human_constraint_even_before_activation(self):
        actor, row = self.fixture()
        actor.grasp.close = Mock()
        actor.human = SimpleNamespace(appearance_state='waiting', retire_after_exit=Mock())
        actor.close()
        actor.grasp.close.assert_called_once_with()
        actor.human.retire_after_exit.assert_not_called()


if __name__ == '__main__':
    unittest.main()

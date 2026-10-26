"""CPU-only preset/order tests; not combined native execution evidence."""
import contextlib
import io
import json
import unittest
from . import __main__ as entry
from .variants import (
    BAG_MOVE, MACHINE_PUSH, VARIANT_NAMES, InterferenceSequence,
    describe_variant, get_variant,
)


class VariantTests(unittest.TestCase):
    def test_exactly_three_presets_one_base_scorer(self):
        self.assertEqual(VARIANT_NAMES, ('bag_only', 'machine_only', 'bag_then_machine'))
        for name in VARIANT_NAMES:
            spec = get_variant(name)
            self.assertEqual(spec.process_score_instances, 1)
            self.assertEqual(spec.source_atomic_api, '1-2')
            self.assertFalse(spec.runtime_registered)
            self.assertFalse(spec.native_combination_verified)

    def test_unknown_variant_does_not_fallback(self):
        for name in (None, '', 'both', 'prepare_coffee', 1):
            with self.assertRaises(ValueError):
                get_variant(name)

    def test_machine_only_preserves_immediate_machine_eligibility(self):
        seq = InterferenceSequence('machine_only')
        self.assertEqual(seq.eligible_component, MACHINE_PUSH)
        self.assertTrue(seq.robot_process_scoring_enabled)
        with self.assertRaises(ValueError):
            seq.begin(BAG_MOVE, simulation_time_s=1.)

    def test_bag_only_never_enables_machine(self):
        seq = InterferenceSequence('bag_only')
        self.assertFalse(seq.robot_process_scoring_enabled)
        seq.begin(BAG_MOVE, simulation_time_s=1.)
        seq.complete(BAG_MOVE, simulation_time_s=2., native_completion_confirmed=True, actor_retired=True)
        self.assertIsNone(seq.eligible_component)
        self.assertTrue(seq.robot_process_scoring_enabled)

    def test_combination_requires_bag_then_machine_not_simultaneous(self):
        seq = InterferenceSequence('bag_then_machine')
        with self.assertRaises(ValueError):
            seq.begin(MACHINE_PUSH, simulation_time_s=1.)
        seq.begin(BAG_MOVE, simulation_time_s=1.)
        self.assertIsNone(seq.eligible_component)
        with self.assertRaises(ValueError):
            seq.begin(MACHINE_PUSH, simulation_time_s=1.)
        seq.complete(BAG_MOVE, simulation_time_s=2., native_completion_confirmed=True, actor_retired=True)
        self.assertEqual(seq.eligible_component, MACHINE_PUSH)
        self.assertTrue(seq.robot_process_scoring_enabled)
        seq.begin(MACHINE_PUSH, simulation_time_s=3.)
        seq.complete(MACHINE_PUSH, simulation_time_s=4., native_completion_confirmed=True, actor_retired=True)
        self.assertEqual(seq.completed, [BAG_MOVE, MACHINE_PUSH])
        self.assertIsNone(seq.eligible_component)

    def test_release_timer_or_retirement_alone_does_not_unlock(self):
        for complete, retired in ((False, False), (True, False), (False, True)):
            seq = InterferenceSequence('bag_then_machine')
            seq.begin(BAG_MOVE, simulation_time_s=1.)
            self.assertFalse(seq.complete(BAG_MOVE, simulation_time_s=2.,
                native_completion_confirmed=complete, actor_retired=retired))
            self.assertFalse(seq.robot_process_scoring_enabled)
            self.assertIsNone(seq.eligible_component)

    def test_repeat_cannot_reactivate_or_duplicate_completion(self):
        seq = InterferenceSequence('bag_then_machine')
        with self.assertRaises(ValueError):
            seq.complete(None, simulation_time_s=0., native_completion_confirmed=True, actor_retired=True)
        seq.begin(BAG_MOVE, simulation_time_s=1.)
        seq.complete(BAG_MOVE, simulation_time_s=2., native_completion_confirmed=True, actor_retired=True)
        with self.assertRaises(ValueError):
            seq.begin(BAG_MOVE, simulation_time_s=3.)
        with self.assertRaises(ValueError):
            seq.complete(BAG_MOVE, simulation_time_s=3., native_completion_confirmed=True, actor_retired=True)
        self.assertEqual(seq.completed, [BAG_MOVE])

    def test_invalid_route_not_silently_skipped_or_retried(self):
        seq = InterferenceSequence('bag_then_machine')
        seq.invalidate(BAG_MOVE, simulation_time_s=1., reason='Reserved route blocked')
        self.assertIsNone(seq.eligible_component)
        self.assertFalse(seq.robot_process_scoring_enabled)
        for component in (BAG_MOVE, MACHINE_PUSH):
            with self.assertRaises(ValueError):
                seq.begin(component, simulation_time_s=2.)

    def test_bad_time_and_nonboolean_facts_rejected(self):
        seq = InterferenceSequence('bag_only')
        for now in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                seq.begin(BAG_MOVE, simulation_time_s=now)
        seq.begin(BAG_MOVE, simulation_time_s=2.)
        with self.assertRaises(ValueError):
            seq.complete(BAG_MOVE, simulation_time_s=1., native_completion_confirmed=True, actor_retired=True)
        with self.assertRaises(ValueError):
            seq.complete(BAG_MOVE, simulation_time_s=3., native_completion_confirmed=1, actor_retired=True)

    def test_sequences_do_not_leak_between_episodes(self):
        first, second = (InterferenceSequence('bag_then_machine') for _ in range(2))
        first.begin(BAG_MOVE, simulation_time_s=1.)
        self.assertEqual(second.eligible_component, BAG_MOVE)
        self.assertEqual(second.completed, [])

    def test_cli_list_describes_only_presets_not_runnable_registration(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(['variants']), 0)
        presets = json.loads(output.getvalue())
        self.assertEqual([item['name'] for item in presets], list(VARIANT_NAMES))
        for item in presets:
            self.assertIn('native_owner_and_scoring_pending', item['status'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(['describe', '--variant', 'bag_then_machine']), 0)
        self.assertEqual(json.loads(output.getvalue())['component_order'], [BAG_MOVE, MACHINE_PUSH])

    def test_descriptors_are_fresh_copies(self):
        value = describe_variant('bag_only')
        value['component_definitions'].clear()
        self.assertEqual(len(describe_variant('bag_only')['component_definitions']), 2)


if __name__ == '__main__':
    unittest.main()

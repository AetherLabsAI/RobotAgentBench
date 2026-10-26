"""Synthetic rule checks, not physical recovery validation."""
import unittest
from .common import HERE, owned, validate_native_start
from .rules import shelf_scorer, coffee_scorer


def accepted_checks():
    return dict(target_supported=True, target_upright=True, empty_gripper=True,
        no_robot_target_contact=True, robot_reset_accepted=True, original_initial_accepted=True)


class Boundaries(unittest.TestCase):
    def test_outputs_cannot_escape(self):
        for path in (HERE, HERE.parent, '/tmp/recovery'):
            with self.assertRaises(ValueError):
                owned(path)

    def test_complete_native_receipt_required(self):
        with self.assertRaises(ValueError):
            validate_native_start({})

    def test_every_initial_check_is_required(self):
        for key in accepted_checks():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_start(dict(accepted_checks(), **{key: False}))

    def test_boolean_checks_not_truthy_strings(self):
        with self.assertRaises(ValueError):
            validate_native_start(dict(accepted_checks(), empty_gripper='true'))

    def test_no_progress_inherited(self):
        result = validate_native_start(accepted_checks())
        self.assertEqual(result['initial_score'], 0)
        self.assertFalse(result['source_progress_inherited'])


class ShelfRules(unittest.TestCase):
    def frame(self, index, *, held=False, separated=False):
        from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
        return PhysicalFrame('recovery-test', index, index/240, held, separated, 0.,
            frozenset(), frozenset(), False, False, active_target='/World/Assets/ShelfItems/gatorade_t')

    def test_empty_hand_and_idle_cannot_score(self):
        scorer = shelf_scorer('recovery-test', 1/240, accepted_checks())
        for index in range(241):
            scorer.observe_batch((self.frame(index),))
        self.assertEqual(scorer.summary()['score'], 0)
        self.assertFalse(scorer.summary()['success'])

    def test_pick_requires_hold_but_no_place_stage(self):
        scorer = shelf_scorer('recovery-test', 1/240, accepted_checks())
        scorer.observe_batch((self.frame(0),))
        scorer.observe_batch((self.frame(1, held=True, separated=True),))
        self.assertEqual(scorer.summary()['score'], 4)
        self.assertFalse(scorer.summary()['success'])
        for index in range(2, 722):
            result = scorer.observe_batch((self.frame(index, held=True, separated=True),))
        self.assertTrue(result['success'])
        self.assertEqual(result['score'], 4)

    def test_new_episode_has_no_source_awards(self):
        scorer = shelf_scorer('recovery-test', 1/240, accepted_checks())
        self.assertEqual(scorer.summary()['earned_points'], 0)
        self.assertIsNone(scorer.committed_target)


class CoffeeRules(unittest.TestCase):
    def frame(self, step, *, held=False, beans=0):
        from evaluation.tasks.coffee_machine.facts import Facts, ObjectFacts, CupPlacement, COUNTER, TRAY
        return Facts(step, step*.05,
            ObjectFacts((0., 0., 0.), 0., False, False, frozenset({TRAY})),
            ObjectFacts((0., 0., 0.), 0., held, held, frozenset() if held else frozenset({COUNTER})),
            CupPlacement(0., True, False), False, beans_in_hopper=beans, beans_total=100)

    def test_empty_hand_cannot_score(self):
        scorer = coffee_scorer(accepted_checks())
        for step in range(21):
            scorer.observe(self.frame(step))
        self.assertEqual(scorer.score, 0)
        self.assertFalse(scorer.success)

    def test_original_pour_and_return_endpoint(self):
        scorer = coffee_scorer(accepted_checks())
        scorer.observe(self.frame(0))
        scorer.observe(self.frame(1, held=True))
        self.assertEqual(scorer.score, 2)
        self.assertFalse(scorer.success)
        scorer.observe(self.frame(2, held=True, beans=100))
        self.assertEqual(scorer.score, 5)
        for step in range(3, 24):
            scorer.observe(self.frame(step, beans=100))
        self.assertTrue(scorer.success)
        self.assertEqual(scorer.score, 6)

    def test_new_process_has_no_source_awards(self):
        scorer = coffee_scorer(accepted_checks())
        self.assertEqual(scorer.score, 0)
        self.assertEqual(scorer.stage, 1)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['shelf_drink', 'coffee_bag'], required=True)
    args = p.parse_args()
    suite = unittest.TestSuite()
    for group in (Boundaries, ShelfRules if args.variant == 'shelf_drink' else CoffeeRules):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(group))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

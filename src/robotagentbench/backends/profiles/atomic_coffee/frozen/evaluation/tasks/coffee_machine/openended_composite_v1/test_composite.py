"""Synthetic evidence only. These tests never move a simulator or prove physics."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from .composite import Binding, Frame, CoffeeComposite, PHASES, UnresolvedSequence
from evaluation.tasks.coffee_machine.facts import (
    Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts, COUNTER, TRAY,
)
from evaluation.tasks.coffee_machine.process_rules import CONTROL_DT_S


def binding(label='coffee-one'):
    return Binding('room-episode', label, tuple(sorted(
        (role, '/scene/'+label+'/'+role) for role in
        ('machine', 'cup', 'bag', 'beans', 'button', 'cup_support', 'counter'))))


def obj(position, support=COUNTER, held=False):
    return ObjectFacts(position=position, tilt_deg=0., pads_bilateral=held,
                       robot_contact=held, supports=frozenset() if held else frozenset({support}))


class Run:
    def __init__(self):
        self.c = CoffeeComposite(binding())
        self.n = 0
        self.trace = []
        self.f = Facts(0, 0., obj((0., 0., 0.)), obj((0., 0.3, 0.)),
                       CupPlacement(.3, False, True), False, 0, 100,
                       LiquidFacts(0, 0, 0, 0, mode='visual_cpu_v1'),
                       ButtonFacts(0., False, False))
        self.feed()

    def feed(self, count=1, **changes):
        self.f = replace(self.f, **changes)
        for _ in range(count):
            self.f = replace(self.f, step=self.n, time_s=self.n*CONTROL_DT_S)
            result = self.c.observe(Frame(self.c.binding, self.f))
            self.trace.append(dict(step=self.n, phase=result['phase'],
                                   score=result['score'], success=result['success']))
            self.n += 1
        return result

    def place(self):
        self.feed(cup=obj((0., 0., .1), held=True))
        self.feed(21, cup=obj((.3, 0., 0.), TRAY), placement=CupPlacement(.004, True, False))
        assert self.c.phase == '1-2'

    def pour(self, beans=100):
        self.feed()  # enter next evaluator without resetting scene
        self.feed(bag=obj((0., .3, .1), held=True))
        self.feed(21, bag=obj((.1, .3, 0.)), beans_in_hopper=beans)
        assert self.c.phase == '1-3'

    def brew(self, units=8000):
        self.feed()
        self.feed(button=ButtonFacts(-.003, True, True, ('started',), (True,)))
        self.feed(liquid=LiquidFacts(units, units, 0, 0, mode='visual_cpu_v1'),
                  button=ButtonFacts(0., False, True))
        self.feed(button=ButtonFacts(-.003, True, False, ('stopped_by_button',), (True,)))
        self.feed(21, button=ButtonFacts(0., False, False))
        assert self.c.phase == '1-4'

    def retrieve(self):
        self.feed()
        self.feed(cup=obj((.3, 0., .1), held=True))
        self.feed(11, cup=obj((.6, 0., .1), held=True), placement=CupPlacement(.3, False, True))
        assert self.c.success


class CompositionTests(unittest.TestCase):
    def test_continuous_four_phases_and_actual_fill(self):
        r = Run(); r.place(); r.pour(); r.brew(); r.retrieve()
        s = r.c.summary()
        self.assertEqual((s['score'], s['maximum_score'], s['completed_phases']), (21., 21, 4))
        self.assertEqual([x['api'] for x in s['completed']], list(PHASES))
        self.assertEqual(s['completed'][3]['result']['final_counts']['denominator_volume_units'], 8000)
        self.assertTrue(all(not x['scene_reset'] for x in s['transitions']))
        self.assertFalse(s['physics_verified']); self.assertFalse(s['runtime_registered'])

    def test_inherited_pose_changes_not_false_deductions(self):
        r = Run(); r.place(); r.pour(); r.brew(); r.retrieve()
        self.assertEqual([x['result']['deduction_total'] for x in r.c.completed], [0, 0, 0, 0])

    def test_current_phase_displacement_still_deducted_once(self):
        r = Run(); r.place(); r.feed()
        r.feed(3, cup=replace(r.f.cup, position=(.4, 0., 0.)))
        ds = r.c.summary()['active']['deductions']
        self.assertEqual([d['kind'] for d in ds], ['cup_displaced'])

    def test_phase_boundary_does_not_hide_first_frame_disturbance(self):
        r = Run(); r.place()
        r.feed(cup=replace(r.f.cup, position=(.4, 0., 0.)))
        self.assertEqual(r.c.summary()['active']['deductions'][0]['kind'], 'cup_displaced')

    def test_original_bean_settlement_preserved(self):
        r = Run(); r.place(); r.pour(75); r.brew(); r.retrieve()
        self.assertEqual(r.c.summary()['score'], 19.)
        self.assertEqual(r.c.completed[1]['result']['deduction_total'], 2)

    def test_insufficient_beans_fails_without_brew(self):
        r = Run(); r.place(); r.feed(); r.feed(bag=obj((0., .3, .1), held=True))
        r.feed(21, bag=obj((.1, .3, 0.)), beans_in_hopper=69)
        self.assertEqual(r.c.failure['reason'], 'beans_below_70')
        self.assertFalse(r.c.success)

    def test_early_button_is_not_credited_or_silently_ignored(self):
        r = Run(); before = r.c.summary()
        with self.assertRaises(UnresolvedSequence):
            r.feed(button=ButtonFacts(-.003, True, True, ('started',), (True,)))
        self.assertEqual(r.c.summary(), before)

    def test_wrong_instance_or_episode_rejected(self):
        r = Run(); f = replace(r.f, step=1, time_s=.05)
        for b in (binding('coffee-two'), replace(r.c.binding, episode_id='other-room')):
            with self.assertRaises(ValueError): r.c.observe(Frame(b, f))
        self.assertEqual(r.c.last.step, 0)

    def test_missing_or_duplicate_binding_rejected(self):
        with self.assertRaises(ValueError): Binding('e', 't', ())
        b = binding()
        with self.assertRaises(ValueError): Binding('e', 't', tuple((k, '/same') for k, _ in b.objects))

    def test_duplicate_skipped_time_and_invalid_facts_transactional(self):
        r = Run(); before = r.c.summary()
        for f in (r.f, replace(r.f, step=2, time_s=.1),
                  replace(r.f, step=1, time_s=.1),
                  replace(r.f, step=1, time_s=.05, cup=replace(r.f.cup, tilt_deg=float('nan')))):
            with self.assertRaises(ValueError): r.c.observe(Frame(r.c.binding, f))
        self.assertEqual(r.c.summary(), before)
        r.feed()  # valid retry uses the same next step

    def test_terminal_replay_cannot_double_score(self):
        r = Run(); r.place(); r.pour(); r.brew(); r.retrieve()
        before = r.c.summary()
        with self.assertRaises(ValueError): r.feed()
        self.assertEqual(r.c.summary(), before)

    def test_ground_contact_preserves_original_failure(self):
        r = Run()
        r.feed(cup=replace(r.f.cup, ground_contact=True))
        self.assertEqual(r.c.failure['reason'], 'cup_on_ground')
        self.assertFalse(r.c.success)

    def test_gripper_command_is_not_grasp(self):
        r = Run(); r.feed(10)
        self.assertEqual(r.c.summary()['score'], 0)

    def test_hold_interruption_does_not_accumulate(self):
        r = Run(); r.feed(cup=obj((0., 0., .1), held=True))
        r.feed(12, cup=obj((.3, 0., 0.), TRAY), placement=CupPlacement(.004, True, False))
        r.feed(cup=obj((.3, 0., .1), held=True))
        r.feed(20, cup=obj((.3, 0., 0.), TRAY))
        self.assertEqual(r.c.phase, '1-1')
        r.feed(); self.assertEqual(r.c.phase, '1-2')
        self.assertEqual(r.c.summary()['score'], 5.)

    def test_phase_budget_not_disabled(self):
        r = Run()
        while not r.c.ended: r.feed()
        self.assertEqual(r.c.failure['reason'], 'stage1_budget_exhausted')

    def test_device_counters_cannot_reset_across_phases(self):
        r = Run(); r.place(); r.pour(); r.brew()
        before = r.c.summary()
        with self.assertRaises(ValueError):
            r.feed(liquid=LiquidFacts(0, 0, 0, 0, mode='visual_cpu_v1'))
        self.assertEqual(r.c.summary(), before)

    def test_two_coffee_instances_have_independent_state(self):
        r = Run(); other = CoffeeComposite(binding('coffee-two')); r.place()
        self.assertEqual(other.summary()['score'], 0)
        self.assertEqual(other.phase, '1-1')

    def test_snapshot_cannot_mutate_evaluator(self):
        r = Run(); r.place(); s = r.c.summary()
        s['completed'][0]['result']['score'] = 500
        self.assertEqual(r.c.summary()['score'], 5.)

    def test_no_scene_initialization_interface(self):
        r = Run()
        for name in ('initialize', 'reset', 'randomize', 'robot_initialization', 'step_physics'):
            self.assertFalse(hasattr(r.c, name))

    def test_post_brew_cup_failure_cannot_report_success(self):
        r = Run(); r.place(); r.pour(); r.brew()
        r.feed(cup=replace(r.f.cup, ground_contact=True))
        self.assertFalse(r.c.success)
        self.assertEqual(r.c.failure['reason'], 'cup_on_ground')

    def test_visual_tilt_rule_preserved(self):
        r = Run(); r.place(); r.pour(); r.brew()
        r.feed(cup=replace(r.f.cup, tilt_deg=15.01))
        self.assertEqual(r.c.failure['reason'], 'cup_tilt_exceeded')
        self.assertFalse(r.c.success)

    def test_restart_during_retrieve_not_silently_ignored(self):
        r = Run(); r.place(); r.pour(); r.brew()
        with self.assertRaises(UnresolvedSequence):
            r.feed(button=ButtonFacts(-.003, True, True, ('started',), (True,)))
        self.assertFalse(r.c.success)

    def test_missing_device_evidence_never_means_success(self):
        r = Run(); before = r.c.summary()
        with self.assertRaises(ValueError): r.feed(button=None)
        self.assertEqual(r.c.summary(), before)

    def test_prefilled_start_rejected_without_scene_repair(self):
        r = Run(); c = CoffeeComposite(binding())
        filled = replace(r.f, liquid=LiquidFacts(8000, 8000, 0, 0, mode='visual_cpu_v1'))
        with self.assertRaises(ValueError): c.observe(Frame(c.binding, filled))
        self.assertEqual(c.summary()['score'], 0)
        self.assertIsNone(c.last)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CompositionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = Path(__file__).resolve().parent
    report = dict(kind='synthetic_cpu_rule_tests', tests=result.testsRun,
                  passed=result.wasSuccessful(), failures=len(result.failures), errors=len(result.errors),
                  physics_verified=False, runtime_registered=False)
    if result.wasSuccessful():
        demo = Run(); demo.place(); demo.pour(); demo.brew(); demo.retrieve()
        report['synthetic_demo'] = demo.c.summary()
        (root/'synthetic_trace.json').write_text(json.dumps(demo.trace, indent=2)+'\n')
    (root/'test_report.json').write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)

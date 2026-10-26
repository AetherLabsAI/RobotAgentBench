"""Rules of 1-1 … 1-4 on hand-built Facts. Every expected number is a README sentence, see apis/<api>/README.md."""
import json
import math
import re
import unittest
from pathlib import Path

from evaluation.tasks.coffee_machine.facts import (
    Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts, ML_PER_PARTICLE, COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER)
from evaluation.tasks.coffee_machine.process_rules import Process, SPECS, RULE_VERSION, CONTROL_DT_S

DT = CONTROL_DT_S
APIS = Path(__file__).resolve().parents[1] / 'apis'

# object states
ON_COUNTER = dict(supports=frozenset({COUNTER}))
ON_TRAY = dict(supports=frozenset({TRAY}))
ON_OTHER = dict(supports=frozenset({OTHER}))
HELD = dict(pads_bilateral=True, robot_contact=True, supports=frozenset())
GRIPPED_ON_COUNTER = dict(pads_bilateral=True, robot_contact=True, supports=frozenset({COUNTER}))
GRIPPED_ON_TRAY = dict(pads_bilateral=True, robot_contact=True, supports=frozenset({TRAY}))
FALLING = dict(supports=frozenset())
POURING = dict(HELD, tilt_deg=110.)
# cup relative to the machine
AWAY = dict(xy_error_m=0.30, outlet_over_opening=False, outside_bay=True)
AT_ANCHOR = dict(xy_error_m=0.004, outlet_over_opening=True, outside_bay=False)
IN_BAY = dict(xy_error_m=0.05, outlet_over_opening=False, outside_bay=False)
# button
IDLE = dict(displacement_m=-0.00043, robot_on_button=False, dispensing=False, events=(), event_by_robot=())
RUNNING = dict(IDLE, dispensing=True)
FULL = 10304        # 1-4 pre-filled particles


def press(event, by_robot=True):
    return dict(displacement_m=-0.0031, robot_on_button=by_robot, dispensing=event == 'started', events=(event,), event_by_robot=(by_robot,))


def liquid(emitted, airborne=0, spilled=0, in_cup=None, initial=0):
    return dict(emitted=emitted, airborne=airborne, spilled=spilled, initial_in_cup=initial,
                in_cup=emitted - airborne - spilled if in_cup is None else in_cup)


def thing(state):
    return ObjectFacts(**{**dict(position=(0., 0., 0.), tilt_deg=0., pads_bilateral=False, robot_contact=False), **state})


def facts(step, *, cup=ON_COUNTER, bag=ON_COUNTER, placement=AWAY, liquid=None, button=None, time_s=None, **rest):
    return Facts(step=step, time_s=step * DT if time_s is None else time_s, cup=thing(cup), bag=thing(bag),
                 placement=CupPlacement(**placement), robot_machine_contact=rest.pop('robot_machine_contact', False),
                 liquid=liquid and LiquidFacts(**liquid), button=button and ButtonFacts(**button), **rest)


SCENES = {
    '1-1': dict(),
    '1-2': dict(cup=ON_TRAY, placement=AT_ANCHOR, beans_total=100),
    '1-3': dict(cup=ON_TRAY, placement=AT_ANCHOR, beans_in_hopper=100, beans_total=100, button=IDLE, liquid=liquid(0)),
    '1-4': dict(cup=ON_TRAY, placement=AT_ANCHOR, liquid=liquid(FULL, initial=FULL)),
}


class Run:
    """Feeds consecutive control frames. `scene` is what every frame looks like unless a call changes it."""
    def __init__(self, api, **scene):
        self.p, self.step, self.scene = Process(api), 0, dict(SCENES[api], **scene)

    def feed(self, count=1, **change):
        for _ in range(count):
            self.p.observe(facts(self.step, **dict(self.scene, **change)))
            self.step += 1
        return self.p

    def keep(self, **scene):
        self.scene.update(scene)
        return self

    def used(self, stage):
        return self.p.summary()['budgets'][str(stage)]['used_s']

    def kinds(self):
        return [d['kind'] for d in self.p.summary()['deductions']]


class Rules(unittest.TestCase):
    def state(self, r, stage, score, success=False, failure=None):
        p = r.p
        self.assertEqual((p.stage, p.score, p.success, p.failure), (stage, score, success, failure))

    def succeeds_after_one_second(self, r, stage=3, score=5, **frame):
        """20 frames span 0.95 s: not yet. The 21st spans exactly 1.0 s."""
        r.feed(20, **frame)
        self.state(r, stage, score)
        self.assertAlmostEqual(r.p.summary()['hold']['window_s'], 0.95)
        r.feed(1, **frame)
        self.state(r, stage, score, success=True)
        self.assertAlmostEqual(r.p.summary()['hold']['window_s'], 1.0)


# ---------------------------------------------------------------- shared contract
def readme_row(api, name):
    for line in (APIS / api / 'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] == name: return cells[1]
    raise AssertionError('row not found: ' + name)


class Contract(Rules):
    def test_specs_repeat_the_readme_rows(self):
        self.assertEqual(sorted(SPECS), ['1-1', '1-2', '1-3', '1-4'])
        for api, spec in SPECS.items():
            with self.subTest(api=api):
                self.assertEqual(spec.instruction, readme_row(api, '发给 VLA 的指令'))
                self.assertEqual(spec.time_limit_s, float(re.search(r'总时限 (\d+) s', readme_row(api, '时限与终止')).group(1)))
                budgets = re.findall(r'阶段([①②③])(?:放回操作)? (\d+(?:\.\d+)?) s', readme_row(api, '阶段时限'))
                self.assertEqual(dict(spec.stage_budget_s), {'①②③'.index(s) + 1: float(v) for s, v in budgets})
                self.assertEqual(spec.maximum_score, int(re.search(r'满分 (\d+)', readme_row(api, '评分规则')).group(1)))
                self.assertEqual(spec.maximum_score, sum(spec.stage_points.values()))
                hold = float(re.search(r'连续保持 (\d+(?:\.\d+)?) s', readme_row(api, '阶段时限')).group(1))
                self.assertEqual((spec.thresholds['hold_s'], spec.thresholds['hold_span_m']), (hold, 0.002))
                json.dumps(spec.as_dict())
                with self.assertRaises(Exception): spec.thresholds['hold_s'] = 2.
                with self.assertRaises(Exception): spec.time_limit_s = 1.
        self.assertEqual({a: dict(s.stage_points) for a, s in SPECS.items()},
                         {'1-1': {1: 2, 2: 3}, '1-2': {1: 2, 2: 3, 3: 1}, '1-3': {1: 2, 2: 3}, '1-4': {1: 2, 2: 3}})
        self.assertEqual(RULE_VERSION, 'arena.coffee_machine.process.v4')

    def test_liquid_bounds_in_particles(self):
        t = SPECS['1-3'].thresholds
        self.assertEqual(math.ceil(t['cup_liquid_min_ml'] / ML_PER_PARTICLE), 6104)
        self.assertEqual(math.floor(t['cup_liquid_max_ml'] / ML_PER_PARTICLE), 14648)

    def test_unknown_api(self):
        for api in ('1-5', '1.1', None):
            with self.assertRaises(ValueError): Process(api)

    def test_frames_must_be_consecutive_and_on_the_clock(self):
        for api in SPECS:
            with self.subTest(api=api):
                with self.assertRaises(ValueError): Process(api).observe(facts(1, **SCENES[api]))      # frame 0 missing
                r = Run(api); r.feed(3)
                for step in (2, 4, 0, -1, 3.0, True):                                                  # repeated, skipped, not an index
                    with self.assertRaises(ValueError): r.p.observe(facts(step, **dict(r.scene, time_s=3 * DT)))
                with self.assertRaises(ValueError): r.p.observe(facts(3, **dict(r.scene, time_s=3 * DT + 0.01)))
                with self.assertRaises(ValueError): r.p.observe('frame')
                r.feed(1)                                                                              # rejected frames changed nothing
                self.assertEqual(r.p.summary()['frames'], 4)

    def test_non_finite_or_mistyped_facts(self):
        nan, inf = float('nan'), float('inf')
        bad = [dict(time_s=nan), dict(cup=dict(ON_COUNTER, tilt_deg=nan)), dict(cup=dict(ON_COUNTER, tilt_deg=inf)),
               dict(cup=dict(ON_COUNTER, position=(0., inf, 0.))), dict(cup=dict(ON_COUNTER, position=(0., 0.))),
               dict(bag=dict(ON_COUNTER, displacement_m=nan)), dict(bag=dict(ON_COUNTER, tilt_deg=-1.)),
               dict(cup=dict(ON_COUNTER, tilt_deg=True)), dict(cup=dict(ON_COUNTER, robot_contact=1)),
               dict(cup=dict(supports=frozenset({'shelf'}))), dict(cup=dict(supports=(COUNTER,))),
               dict(placement=dict(AWAY, xy_error_m=nan)), dict(placement=dict(AWAY, outside_bay=None)),
               dict(robot_machine_contact=0), dict(beans_in_hopper=-1), dict(beans_in_hopper=1.5)]
        for api in SPECS:
            for change in bad:
                with self.subTest(api=api, change=change), self.assertRaises(ValueError): Run(api).feed(1, **change)

    def test_contradictory_contacts(self):
        for change in (dict(cup=dict(ON_COUNTER, pads_bilateral=True)),                     # pads touch, robot does not
                       dict(cup=dict(supports=frozenset({GROUND}))),                         # ground supports, no ground contact
                       dict(bag=dict(supports=frozenset({MACHINE_BODY})))):                  # machine supports, no machine contact
            with self.subTest(change=change), self.assertRaises(ValueError): Run('1-1').feed(1, **change)

    def test_missing_liquid_or_button_is_not_a_pass(self):
        for api, change in (('1-3', dict(liquid=None)), ('1-3', dict(button=None)), ('1-4', dict(liquid=None)),
                            ('1-4', dict(liquid=liquid(FULL, initial=0))), ('1-3', dict(liquid=dict(liquid(5), in_cup=-1)))):
            with self.subTest(api=api, change=change), self.assertRaises(ValueError): Run(api).feed(1, **change)
        r = Run('1-4'); r.feed(2)
        with self.assertRaises(ValueError): r.feed(1, liquid=liquid(FULL, initial=FULL - 1))   # the denominator is fixed
        r = Run('1-4'); r.feed(2, liquid=liquid(FULL, spilled=9, initial=FULL))
        with self.assertRaises(ValueError): r.feed(1, liquid=liquid(FULL, spilled=8, initial=FULL))   # spilled is cumulative
        r = Run('1-3'); r.feed(2, liquid=liquid(50))
        with self.assertRaises(ValueError): r.feed(1, liquid=liquid(49))                       # emitted is cumulative
        with self.assertRaises(ValueError): Run('1-2').feed(1, beans_in_hopper=101)
        for button in (dict(IDLE, events=('started',)), dict(IDLE, events=('pressed',), event_by_robot=(True,)),
                       dict(IDLE, events=('started',), event_by_robot=(1,)), dict(IDLE, displacement_m=float('nan')),
                       dict(IDLE, events=['started'], event_by_robot=[True])):
            with self.subTest(button=button), self.assertRaises(ValueError): Run('1-3').feed(1, button=button)

    def test_finalize(self):
        for api in SPECS:
            with self.subTest(api=api):
                with self.assertRaises(ValueError): Process(api).finalize('time_limit')       # nothing observed
                r = Run(api); r.feed(5)
                for reason in ('', None, 'success'):                                           # success was never measured
                    with self.assertRaises(ValueError): r.p.finalize(reason)
                r.p.finalize('client_closed'); r.p.finalize('client_closed')
                with self.assertRaises(ValueError): r.p.finalize('time_limit')
                with self.assertRaises(RuntimeError): r.feed(1)
                s = r.p.summary()
                self.assertEqual((s['finalized'], s['termination'], s['success'], s['score']), (True, 'client_closed', False, 0.))

    def test_summary_is_json_safe_and_complete(self):
        for api in SPECS:
            with self.subTest(api=api):
                r = Run(api); r.feed(3, robot_machine_contact=True)
                s = json.loads(json.dumps(r.p.summary()))
                self.assertEqual((s['schema'], s['api'], s['stage']), (RULE_VERSION, api, 1))
                for key in ('stage_points', 'deductions', 'budgets', 'hold', 'success', 'failure', 'final_counts', 'score'): self.assertIn(key, s)
                self.assertEqual(s['deductions'], [dict(kind='robot_touched_machine', amount=1, step=0, time_s=0.)])
                self.assertEqual(s['budgets']['1'], dict(budget_s=SPECS[api].stage_budget_s[1], used_s=0.1, exhausted=False))
                self.assertEqual(s['stage_points']['1'], dict(points=2, awarded=False, step=None, time_s=None))

    def test_frames_after_the_end_are_counted_not_judged(self):
        r = Run('1-1'); r.feed(3); r.feed(1, cup=dict(ON_COUNTER, ground_contact=True))
        before = r.p.summary()
        r.feed(4, cup=HELD, robot_machine_contact=True)
        after = r.p.summary()
        self.assertEqual(after['post_terminal_frames'], 4)
        for key in ('score', 'stage', 'failure', 'deductions', 'budgets', 'end_step', 'stage_points'): self.assertEqual(after[key], before[key])
        with self.assertRaises(ValueError): r.p.observe(facts(r.step + 1))                     # order is still checked


# ---------------------------------------------------------------- 1-1 放杯子
class PlaceCup(Rules):
    def grasped(self, reach=20, carry=40):
        r = Run('1-1'); r.feed(reach); r.feed(carry, cup=HELD)
        return r.keep(placement=AT_ANCHOR)

    def test_nominal_full_score(self):
        r = Run('1-1')
        r.feed(20); self.state(r, 1, 0.)
        r.feed(1, cup=GRIPPED_ON_COUNTER); self.state(r, 1, 0.)            # pads closed, still borne by the counter
        r.feed(1, cup=HELD); self.state(r, 2, 2.)
        r.feed(60, cup=HELD); self.state(r, 2, 2.)
        r.keep(placement=AT_ANCHOR)
        r.feed(4, cup=GRIPPED_ON_TRAY); self.state(r, 2, 2.)               # set down, not released
        r.feed(1, cup=ON_TRAY); self.state(r, 3, 5.)
        r.feed(19, cup=ON_TRAY); self.state(r, 3, 5.)                       # 0.95 s
        r.feed(1, cup=ON_TRAY); self.state(r, 3, 5., success=True)          # 1.0 s
        s = r.p.summary()
        self.assertEqual((s['stage_points']['1']['time_s'], s['stage_points']['2']['time_s'], s['end_time_s']), (21 * DT, 86 * DT, 106 * DT))
        self.assertEqual((s['budgets']['1']['used_s'], s['budgets']['2']['used_s']), (1.05, 3.25))
        self.assertEqual((s['deductions'], s['regressions'], s['hold']['window_s']), ([], 0, 1.0))
        r.p.finalize('success')
        self.assertEqual(r.p.summary()['score'], 5.)

    def test_placed_thresholds(self):
        cases = [(dict(cup=dict(ON_TRAY, tilt_deg=11.5)), True), (dict(cup=dict(ON_TRAY, tilt_deg=11.4)), True),
                 (dict(cup=dict(ON_TRAY, tilt_deg=11.51)), False),
                 (dict(cup=ON_TRAY, placement=dict(AT_ANCHOR, xy_error_m=0.012)), True),
                 (dict(cup=ON_TRAY, placement=dict(AT_ANCHOR, xy_error_m=0.0119)), True),
                 (dict(cup=ON_TRAY, placement=dict(AT_ANCHOR, xy_error_m=0.0121)), False),
                 (dict(cup=ON_TRAY, placement=dict(AT_ANCHOR, outlet_over_opening=False)), False),
                 (dict(cup=dict(ON_TRAY, robot_contact=True)), False),          # a finger still on the cup
                 (dict(cup=GRIPPED_ON_TRAY), False), (dict(cup=ON_COUNTER), False), (dict(cup=ON_OTHER), False), (dict(cup=FALLING), False)]
        for frame, placed in cases:
            with self.subTest(frame=frame):
                r = self.grasped(); r.feed(25, **frame)
                self.assertEqual((r.p.score, r.p.success, r.p.failure), (5. if placed else 2., placed, None))

    def test_placing_needs_the_grasp_first(self):
        r = Run('1-1', placement=AT_ANCHOR); r.feed(30, cup=ON_TRAY)          # pushed there, never grasped
        self.state(r, 1, 0.)

    def test_hold_window(self):
        r = self.grasped(); r.keep(cup=ON_TRAY)
        r.feed(20); r.feed(1, cup=dict(ON_TRAY, robot_contact=True)); self.state(r, 2, 5.)      # interrupted at 0.95 s
        self.assertEqual(r.p.summary()['hold']['window_s'], 0.)
        self.succeeds_after_one_second(r)                                                       # a whole new second
        self.assertEqual(r.p.summary()['stage_points']['2']['step'], 60)                        # paid once, at the first time

    def test_hold_span(self):
        at = lambda x: dict(ON_TRAY, position=(x, 0., 0.93831))
        r = self.grasped(); r.feed(10, cup=at(0.)); r.feed(11, cup=at(0.002)); self.state(r, 3, 5., success=True)   # exactly 2 mm
        r = self.grasped(); r.feed(10, cup=at(0.)); r.feed(11, cup=at(0.0021)); self.state(r, 3, 5.)               # above: restart
        self.assertEqual((r.p.summary()['hold']['window_s'], r.p.summary()['hold']['restarts']), (0.5, 1))
        r.feed(10, cup=at(0.0021)); self.state(r, 3, 5., success=True)
        r = self.grasped()                                                                      # any two frames, not only neighbours
        for i in range(60): r.feed(1, cup=at(0.00015 * i))
        self.state(r, 3, 5.); self.assertGreaterEqual(r.p.summary()['hold']['restarts'], 3)

    def test_cup_on_ground_fails_at_once(self):
        for state in (dict(HELD, ground_contact=True), dict(supports=frozenset({GROUND}), ground_contact=True), dict(FALLING, ground_contact=True)):
            with self.subTest(state=state):
                r = self.grasped(); r.feed(1, cup=state); self.state(r, 2, 2., failure='cup_on_ground')
        r = Run('1-1'); r.feed(1, cup=dict(ON_COUNTER, ground_contact=True)); self.state(r, 1, 0., failure='cup_on_ground')
        r = self.grasped(); r.feed(30, cup=FALLING); self.state(r, 2, 2.)     # falling is not landing

    def test_cup_toppled_after_half_a_second(self):
        lying = dict(ON_COUNTER, tilt_deg=80.)
        r = self.grasped(); r.feed(10, cup=lying); self.state(r, 1, 2.)       # released for 0.45 s
        r.feed(1, cup=lying); self.state(r, 1, 2., failure='cup_toppled')     # 0.5 s
        for tilt, fails in ((41., False), (40.9, False), (41.01, True)):
            with self.subTest(tilt=tilt):
                r = self.grasped(); r.feed(30, cup=dict(ON_TRAY, tilt_deg=tilt))
                self.assertEqual(r.p.failure, 'cup_toppled' if fails else None)
        for state in (dict(HELD, tilt_deg=80.), dict(GRIPPED_ON_COUNTER, tilt_deg=80.), dict(lying, robot_contact=True),
                      dict(ON_OTHER, tilt_deg=80.), dict(FALLING, tilt_deg=80.),
                      dict(supports=frozenset({MACHINE_BODY}), machine_body_contact=True, tilt_deg=80.)):
            with self.subTest(state=state):
                r = self.grasped(); r.feed(30, cup=state); self.assertIsNone(r.p.failure)

    def test_half_second_counts_from_the_last_release(self):
        lying = dict(ON_COUNTER, tilt_deg=80.)
        r = self.grasped(); r.feed(8, cup=lying); r.feed(1, cup=dict(lying, robot_contact=True)); r.feed(10, cup=lying)
        self.assertIsNone(r.p.failure)
        r.feed(1, cup=lying); self.assertEqual(r.p.failure, 'cup_toppled')
        r = self.grasped(); r.feed(30, cup=dict(FALLING, tilt_deg=80.)); self.assertIsNone(r.p.failure)   # no airborne countdown
        r.feed(1, cup=lying); self.assertEqual(r.p.failure, 'cup_toppled')                                 # released long ago
        r = Run('1-1'); r.feed(10, cup=lying); self.assertIsNone(r.p.failure)                              # never touched: from t = 0
        r.feed(1, cup=lying); self.assertEqual(r.p.failure, 'cup_toppled')

    def test_deductions_once_each(self):
        r = self.grasped()
        r.feed(3, cup=dict(HELD, machine_body_contact=True)); self.assertEqual(r.p.score, 1.)
        r.feed(3, cup=HELD); r.feed(3, cup=dict(HELD, machine_body_contact=True)); self.assertEqual(r.p.score, 1.)
        r.feed(2, cup=HELD, robot_machine_contact=True); r.feed(2, cup=HELD); r.feed(2, cup=HELD, robot_machine_contact=True)
        self.assertEqual(r.p.score, 0.)
        r.feed(2, cup=HELD, bag=dict(ON_COUNTER, displacement_m=0.05)); r.feed(2, cup=HELD, bag=dict(ON_COUNTER, tilt_deg=80.))
        self.assertEqual(r.kinds(), ['cup_touched_machine', 'robot_touched_machine', 'bag_disturbed'])
        self.assertEqual((r.p.score, r.p.summary()['raw_score']), (0., -1))                     # never below 0
        self.assertEqual(r.p.summary()['deductions'][0], dict(kind='cup_touched_machine', amount=1, step=60, time_s=3.0))
        self.succeeds_after_one_second(r, score=2., cup=ON_TRAY)                                # 5 - 3

    def test_cup_leaning_on_the_bay_wall_is_charged_even_when_placed(self):
        r = self.grasped()
        self.succeeds_after_one_second(r, score=4., cup=dict(supports=frozenset({TRAY, MACHINE_BODY}), machine_body_contact=True))

    def test_bag_thresholds(self):
        for bag, charged in ((dict(displacement_m=0.02), False), (dict(displacement_m=0.0201), True), (dict(tilt_deg=41.), False),
                             (dict(tilt_deg=41.1), True), (dict(displacement_m=0.03, tilt_deg=60.), True)):
            with self.subTest(bag=bag):
                r = Run('1-1'); r.feed(5, bag=dict(ON_COUNTER, **bag))
                self.assertEqual(r.kinds(), ['bag_disturbed'] if charged else [])
        r = Run('1-1'); r.feed(5, cup=dict(ON_COUNTER, displacement_m=0.5)); self.assertEqual(r.kinds(), [])   # the cup is the target

    def test_failure_beats_success_and_charges_nothing_else(self):
        r = self.grasped(); r.feed(20, cup=ON_TRAY)
        r.feed(1, cup=dict(ON_TRAY, ground_contact=True), robot_machine_contact=True)           # would have been the 1.0 s frame
        self.state(r, 3, 5., failure='cup_on_ground'); self.assertEqual(r.kinds(), [])

    def test_regression_still_gripped(self):
        r = self.grasped(); r.feed(10, cup=ON_TRAY); self.state(r, 3, 5.)
        for again in (GRIPPED_ON_TRAY, HELD, dict(ON_TRAY, robot_contact=True)):
            r.feed(3, cup=again); self.state(r, 2, 5.)
            r.feed(20, cup=ON_TRAY); self.state(r, 3, 5.)                                       # the old 0.5 s do not count
        r.feed(1, cup=ON_TRAY); self.state(r, 3, 5., success=True)
        s = r.p.summary()
        self.assertEqual((s['regressions'], s['stage_points']['2']['step']), (3, 60))

    def test_regression_wrong_place(self):
        r = self.grasped()
        r.feed(3, cup=GRIPPED_ON_COUNTER, placement=AWAY); self.state(r, 2, 2.)                 # set down, not released
        r.feed(5, cup=ON_COUNTER, placement=AWAY); self.state(r, 1, 2.)                         # upright in the wrong place
        r.feed(3, cup=GRIPPED_ON_COUNTER, placement=AWAY); self.state(r, 1, 2.)                 # closing the gripper is not a grasp
        r.feed(5, cup=HELD); self.state(r, 2, 2.)
        self.assertEqual(r.p.summary()['stage_points']['1']['step'], 20)
        self.succeeds_after_one_second(r, cup=ON_TRAY)

    def test_regression_from_the_hold_to_a_new_grasp(self):
        r = self.grasped(); r.feed(10, cup=ON_TRAY)
        r.feed(5, cup=ON_TRAY, placement=dict(AT_ANCHOR, xy_error_m=0.03)); self.state(r, 1, 5.)   # pushed away, left standing
        r.feed(5, cup=dict(ON_TRAY, tilt_deg=25.)); self.state(r, 1, 5.)                           # 11.5 … 41: recoverable
        r.feed(30, cup=FALLING); self.state(r, 1, 5.)
        self.succeeds_after_one_second(r, cup=ON_TRAY)                                          # back in place without a grasp

    def test_free_flight_keeps_the_stage(self):
        r = self.grasped(); r.feed(2, cup=FALLING); self.state(r, 2, 2.)
        r.feed(1, cup=ON_TRAY); self.state(r, 3, 5.); self.assertEqual(r.p.summary()['regressions'], 0)
        r.feed(1, cup=FALLING); self.state(r, 3, 5.); self.assertEqual(r.p.summary()['hold']['window_s'], 0.)
        self.succeeds_after_one_second(r, cup=ON_TRAY)

    def test_stage_one_budget_and_its_boundary(self):
        r = Run('1-1'); r.feed(598); self.state(r, 1, 0.); self.assertEqual(r.used(1), 29.85)
        r.feed(1); self.state(r, 1, 0., failure='stage1_budget_exhausted')
        s = r.p.summary(); self.assertEqual((s['budgets']['1'], s['end_time_s']), (dict(budget_s=29.9, used_s=29.9, exhausted=True), 598 * DT))
        r = Run('1-1'); r.feed(598); r.feed(1, cup=HELD); self.state(r, 2, 2.)                  # achieved exactly at 29.9 s
        self.assertEqual(r.used(1), 29.9)
        r = Run('1-1'); r.feed(597); r.feed(1, cup=HELD); r.feed(100, cup=HELD); self.state(r, 2, 2.)

    def test_stage_two_budget_and_its_boundary(self):
        r = self.grasped(carry=1); r.feed(705, cup=HELD); self.state(r, 2, 2.); self.assertEqual(r.used(2), 35.25)
        r.feed(1, cup=HELD); self.state(r, 2, 2., failure='stage2_budget_exhausted')
        r = self.grasped(carry=1); r.feed(705, cup=HELD)
        self.succeeds_after_one_second(r, cup=ON_TRAY)                                          # placed exactly at 35.3 s, then held
        self.assertEqual((r.used(1), r.used(2)), (1.0, 35.3))                                   # the hold cost nothing

    def test_regression_does_not_reset_budgets(self):
        r = Run('1-1', placement=AT_ANCHOR); r.feed(592); r.feed(6, cup=HELD); r.feed(10, cup=ON_TRAY)
        self.assertEqual((r.used(1), r.used(2)), (29.6, 0.3))
        r.feed(4, cup=HELD); self.assertEqual((r.p.stage, r.used(2)), (2, 0.45))               # 0.3 + the 3 intervals back in ②
        r.feed(6, cup=ON_COUNTER, placement=AWAY); self.state(r, 1, 5.)
        self.assertEqual((r.used(1), r.used(2)), (29.85, 0.5))
        r.feed(1, cup=ON_COUNTER, placement=AWAY); self.state(r, 1, 5., failure='stage1_budget_exhausted')

    def test_returning_to_a_spent_stage_ends_the_episode(self):
        r = Run('1-1', placement=AT_ANCHOR); r.feed(598); r.feed(1, cup=HELD); r.feed(5, cup=ON_TRAY)
        r.feed(1, cup=ON_TRAY, placement=AWAY); self.state(r, 1, 5., failure='stage1_budget_exhausted')

    def test_time_limit_keeps_points_and_deductions(self):
        r = self.grasped(); r.feed(5, cup=HELD, robot_machine_contact=True); r.feed(15, cup=ON_TRAY)
        r.p.finalize('time_limit'); self.state(r, 3, 4.)
        s = r.p.summary(); self.assertEqual((s['termination'], s['ended'], s['final_counts']), ('time_limit', False, {}))


# ---------------------------------------------------------------- 1-2 倾倒咖啡豆
class PourBeans(Rules):
    def poured(self, beans, reach=20):
        r = Run('1-2'); r.feed(reach); r.feed(20, bag=HELD); r.feed(40, bag=POURING, beans_in_hopper=beans)
        return r.keep(beans_in_hopper=beans, bag=ON_COUNTER)

    def test_nominal_full_score(self):
        r = Run('1-2')
        r.feed(20); self.state(r, 1, 0.)
        r.feed(1, bag=GRIPPED_ON_COUNTER); self.state(r, 1, 0.)
        r.feed(1, bag=HELD); self.state(r, 2, 2.)
        r.feed(30, bag=POURING, beans_in_hopper=69); self.state(r, 2, 2.)
        r.feed(1, bag=POURING, beans_in_hopper=70); self.state(r, 3, 5.)
        r.keep(beans_in_hopper=95)
        r.feed(30, bag=POURING); r.feed(3, bag=GRIPPED_ON_COUNTER); self.state(r, 3, 5.)
        r.feed(1, bag=ON_COUNTER); self.state(r, 3, 6.)                                         # +1 when put back, before the hold
        r.feed(19, bag=ON_COUNTER); self.state(r, 3, 6.)
        r.feed(1, bag=ON_COUNTER); self.state(r, 3, 6., success=True)
        s = r.p.summary()
        self.assertEqual([s['stage_points'][k]['time_s'] for k in '123'], [21 * DT, 52 * DT, 86 * DT])
        self.assertEqual([s['budgets'][k]['used_s'] for k in '123'], [1.05, 1.55, 1.7])
        self.assertEqual((s['deductions'], s['final_counts']), ([], dict(beans_in_hopper=95, beans_total=100, beans_band='90_or_more')))

    def test_bean_bands_at_the_automatic_settlement(self):
        for beans, minus, band in ((100, 0, '90_or_more'), (90, 0, '90_or_more'), (89, 1, '80_89'), (80, 1, '80_89'), (79, 2, '70_79'), (70, 2, '70_79')):
            with self.subTest(beans=beans):
                r = self.poured(beans); r.feed(20); self.state(r, 3, 6.)
                self.assertEqual(r.kinds(), [])                                                 # charged when the window ends
                r.feed(1); self.state(r, 3, 6. - minus, success=True)
                self.assertEqual(r.kinds(), ['beans_' + band] if minus else [])
                self.assertEqual(r.p.summary()['final_counts']['beans_band'], band)
                r.p.finalize('success'); self.assertEqual(r.p.score, 6. - minus)                # settled once

    def test_below_seventy_fails_at_the_automatic_settlement(self):
        r = self.poured(69); r.feed(20); self.state(r, 1, 2.)                                   # bag down, too few: a new grasp is due
        r.feed(1); self.state(r, 1, 2., failure='beans_below_70')
        self.assertEqual((r.kinds(), r.p.summary()['final_counts']['beans_band']), ([], 'below_70'))
        r = Run('1-2'); r.feed(200, beans_in_hopper=0); self.state(r, 1, 0.)                    # 不能在尚未倒豆时失败

    def test_settlement_uses_the_current_count_not_the_maximum(self):
        r = self.poured(92); r.feed(10); r.feed(11, beans_in_hopper=85); self.state(r, 3, 5., success=True)
        self.assertEqual(r.kinds(), ['beans_80_89'])
        r = self.poured(92); r.feed(10); r.feed(11, beans_in_hopper=69); self.state(r, 1, 6., failure='beans_below_70')
        r = self.poured(69); r.feed(10); r.feed(11, beans_in_hopper=70); self.state(r, 3, 4., success=True)   # a late bean

    def test_settlement_on_time_limit(self):
        for beans, score, failure, kinds in ((95, 5., None, []), (85, 4., None, ['beans_80_89']), (75, 3., None, ['beans_70_79']),
                                             (50, 2., 'beans_below_70', [])):
            with self.subTest(beans=beans):
                r = self.poured(beans); r.feed(30, bag=HELD)                                    # still carrying the bag
                r.p.finalize('time_limit')
                self.assertEqual((r.p.score, r.p.success, r.p.failure, r.kinds()), (score, False, failure, kinds))
        r = Run('1-2'); r.feed(10); r.p.finalize('client_closed'); self.state(r, 1, 0., failure='beans_below_70')

    def test_settlement_on_failure_and_stage_timeout(self):
        r = self.poured(75); r.feed(1, bag=HELD, cup=dict(ON_TRAY, ground_contact=True))
        self.state(r, 3, 3., failure='cup_on_ground'); self.assertEqual(r.kinds(), ['beans_70_79'])
        r.p.finalize('cup_on_ground'); self.assertEqual((r.p.score, r.kinds()), (3., ['beans_70_79']))
        r = self.poured(85); r.feed(618, bag=HELD); self.state(r, 3, 5.)                        # ③ since step 40: 32.85 s used
        self.assertEqual(r.used(3), 32.85)
        r.feed(1, bag=HELD); self.state(r, 3, 4., failure='stage3_budget_exhausted')
        r = self.poured(40); r.feed(1, bag=dict(HELD, ground_contact=True)); self.state(r, 2, 2., failure='bag_on_ground')
        self.assertEqual((r.kinds(), r.p.summary()['final_counts']['beans_band']), ([], 'below_70'))

    def test_ground_and_toppled_cup(self):
        r = Run('1-2'); r.feed(1, bag=dict(ON_COUNTER, ground_contact=True)); self.state(r, 1, 0., failure='bag_on_ground')
        r = Run('1-2'); r.feed(1, cup=dict(ON_TRAY, ground_contact=True)); self.state(r, 1, 0., failure='cup_on_ground')
        for cup, fails in ((dict(ON_TRAY, tilt_deg=41.01), True), (dict(ON_COUNTER, tilt_deg=41.01), True), (dict(ON_TRAY, tilt_deg=41.), False),
                           (dict(ON_TRAY, tilt_deg=80., robot_contact=True), True),             # no release condition here
                           (dict(ON_OTHER, tilt_deg=80.), False), (dict(FALLING, tilt_deg=80.), False)):
            with self.subTest(cup=cup):
                r = Run('1-2'); r.feed(3); r.feed(1, cup=cup)                                   # first frame, no 0.5 s tolerance
                self.assertEqual(r.p.failure, 'cup_toppled' if fails else None)

    def test_deductions_once_each(self):
        r = self.poured(95).keep(bag=HELD)
        r.feed(2, robot_machine_contact=True); r.feed(2); r.feed(2, robot_machine_contact=True)
        r.feed(2, cup=dict(ON_TRAY, displacement_m=0.02)); self.assertEqual(r.kinds(), ['robot_touched_machine'])
        r.feed(2, cup=dict(ON_TRAY, displacement_m=0.0201)); r.feed(2, cup=dict(ON_TRAY, displacement_m=0.04))
        r.feed(2, cup=dict(ON_TRAY, machine_body_contact=True))                                 # not a 1-2 deduction
        self.assertEqual((r.kinds(), r.p.score), (['robot_touched_machine', 'cup_displaced'], 3.))
        r.feed(21, bag=ON_COUNTER); self.state(r, 3, 4., success=True)

    def test_put_back_thresholds(self):
        cases = [(dict(ON_COUNTER, tilt_deg=10.), True), (dict(ON_COUNTER, tilt_deg=10.01), False), (ON_OTHER, False), (ON_TRAY, False),
                 (dict(ON_COUNTER, robot_contact=True), False), (GRIPPED_ON_COUNTER, False),
                 (dict(supports=frozenset({MACHINE_BODY}), machine_body_contact=True), False)]
        for bag, back in cases:
            with self.subTest(bag=bag):
                r = self.poured(95); r.feed(25, bag=bag)
                self.assertEqual((r.p.score, r.p.success, r.p.failure), (6. if back else 5., back, None))

    def test_hold_window(self):
        r = self.poured(95); r.feed(20); r.feed(1, bag=dict(ON_COUNTER, robot_contact=True)); self.state(r, 3, 6.)
        r.feed(20); self.state(r, 3, 6.)
        r.feed(1); self.state(r, 3, 6., success=True)
        r = self.poured(95); r.feed(10); r.feed(11, bag=dict(ON_COUNTER, position=(0., 0.0021, 0.))); self.state(r, 3, 6.)
        r = self.poured(95); r.feed(10); r.feed(11, bag=dict(ON_COUNTER, position=(0., 0.002, 0.))); self.state(r, 3, 6., success=True)

    def test_regrasp_before_the_window_ends(self):
        r = self.poured(40); r.feed(20); self.state(r, 1, 2.)                                   # 袋已放下需重抓时回①
        r.feed(2, bag=GRIPPED_ON_COUNTER); self.state(r, 1, 2.)
        r.feed(10, bag=HELD); self.state(r, 2, 2.)                                              # 仍持袋且不足 70 颗回②
        r.keep(beans_in_hopper=93)
        r.feed(10, bag=POURING); self.state(r, 3, 5.)
        r.feed(21); self.state(r, 3, 6., success=True)
        s = r.p.summary(); self.assertEqual((s['stage_points']['1']['step'], s['regressions']), (20, 1))

    def test_regression_after_the_put_back(self):
        r = self.poured(95); r.feed(10); self.state(r, 3, 6.)
        before = r.used(3)
        r.feed(10, bag=HELD); self.state(r, 3, 6.); self.assertEqual(r.used(3), round(before + 0.45, 6))   # 进仓 ≥70 颗: ③, may pour on
        r.feed(5, bag=HELD, beans_in_hopper=69); self.state(r, 2, 6.)                           # beans left the hopper
        r.feed(5, bag=dict(ON_COUNTER, tilt_deg=30.)); self.state(r, 1, 6.)                     # leaning: a new grasp is due
        r.feed(5, bag=HELD); self.state(r, 3, 6.)
        r.feed(2, bag=FALLING); self.state(r, 3, 6.)
        r.feed(21); self.state(r, 3, 6., success=True)
        self.assertEqual([r.p.summary()['stage_points'][k]['step'] for k in '123'], [20, 40, 80])

    def test_budgets_and_boundaries(self):
        r = Run('1-2'); r.feed(566); self.state(r, 1, 0.)
        r.feed(1); self.state(r, 1, 0., failure='stage1_budget_exhausted')
        self.assertEqual(r.p.summary()['final_counts']['beans_band'], 'below_70')
        r = Run('1-2'); r.feed(566); r.feed(1, bag=HELD); self.state(r, 2, 2.)
        r.feed(581, bag=POURING, beans_in_hopper=60); self.state(r, 2, 2.)
        r.feed(1, bag=POURING, beans_in_hopper=60); self.state(r, 2, 2., failure='stage2_budget_exhausted')
        r = Run('1-2'); r.feed(1, bag=HELD); r.feed(581, bag=POURING); r.feed(1, bag=POURING, beans_in_hopper=90); self.state(r, 3, 5.)
        self.assertEqual(r.used(2), 29.1)
        r.keep(beans_in_hopper=90)
        r.feed(657, bag=HELD); r.feed(1); self.state(r, 3, 6.)                                  # put back exactly at 32.9 s
        r.feed(20); self.state(r, 3, 6., success=True)
        self.assertEqual([r.used(k) for k in (1, 2, 3)], [0., 29.1, 32.9])                      # the hold cost nothing


# ---------------------------------------------------------------- 1-3 按按钮
def brew(r, particles, wait=40):
    """Valid start after `wait` frames, 10 dispensing frames, valid stop, 4 frames until the last drops have landed."""
    before = r.scene['liquid']['emitted']
    r.feed(wait)
    r.feed(1, button=press('started'))
    for i in range(1, 11):
        out = before + (particles - before) * i // 11
        r.feed(1, button=RUNNING, liquid=liquid(out, airborne=min(40, out)))
    r.feed(1, button=press('stopped_by_button'), liquid=liquid(particles, airborne=40))
    r.feed(4, liquid=liquid(particles, airborne=10))
    return r.keep(liquid=liquid(particles))


class PressButton(Rules):
    def test_nominal_full_score(self):
        r = Run('1-3'); r.feed(40); self.state(r, 1, 0.)
        r.feed(1, button=press('started')); self.state(r, 2, 2.)
        r.feed(10, button=dict(RUNNING, displacement_m=-0.004, robot_on_button=True)); self.state(r, 2, 2.)   # 按住只触发一次
        r.feed(100, button=RUNNING, liquid=liquid(6000, airborne=40))
        r.feed(1, button=press('stopped_by_button'), liquid=liquid(12000, airborne=40)); self.state(r, 3, 2.)
        r.feed(4, liquid=liquid(12000, airborne=10)); self.state(r, 3, 2.)
        r.keep(liquid=liquid(12000))
        r.feed(20); self.state(r, 3, 2.)                                                        # the +3 waits for the settlement
        r.feed(1); self.state(r, 3, 5., success=True)
        s = r.p.summary()
        self.assertEqual((s['stage_points']['1']['time_s'], s['stage_points']['2']['time_s'], s['end_time_s']), (40 * DT, 176 * DT, 176 * DT))
        self.assertEqual((s['budgets']['1']['used_s'], s['budgets']['2']['used_s'], s['deductions']), (2.0, 5.55, []))
        self.assertEqual((s['final_counts']['valid_button_events'], s['final_counts']['in_cup_particles']), (2, 12000))
        self.assertAlmostEqual(s['final_counts']['in_cup_ml'], 49.152)

    def test_cup_amount_bounds(self):
        for particles, ok in ((6103, False), (6104, True), (14648, True)):
            with self.subTest(particles=particles):
                r = brew(Run('1-3'), particles); r.feed(60)
                self.assertEqual((r.p.score, r.p.success, r.p.failure), (5. if ok else 2., ok, None))
        r = brew(Run('1-3'), 14648); r.feed(1, liquid=liquid(14649)); self.state(r, 3, 2., failure='cup_overfilled')
        r = Run('1-3'); r.feed(1, button=press('started'))
        r.feed(1, button=RUNNING, liquid=liquid(14700, airborne=52)); self.state(r, 2, 2.)      # 14648 in the cup
        r.feed(1, button=RUNNING, liquid=liquid(14700, airborne=51)); self.state(r, 2, 2., failure='cup_overfilled')

    def test_only_robot_presses_are_valid(self):
        r = Run('1-3'); r.feed(5)
        r.feed(1, button=press('started', by_robot=False)); self.state(r, 1, 0.)                # something else pressed it
        r.feed(20, button=RUNNING, liquid=liquid(7000)); self.state(r, 1, 0.)
        r.feed(1, button=press('stopped_by_button', by_robot=False), liquid=liquid(7000))
        r.keep(liquid=liquid(7000)); r.feed(30); self.state(r, 1, 0.)
        self.assertEqual(r.p.summary()['final_counts']['valid_button_events'], 0)
        r = Run('1-3'); r.feed(1, button=press('started'))
        r.feed(20, button=RUNNING, liquid=liquid(7000))
        r.feed(1, button=press('stopped_by_button', by_robot=False), liquid=liquid(7000))       # stopped, but not by the robot
        r.keep(liquid=liquid(7000)); r.feed(40); self.state(r, 2, 2.)

    def test_stopped_too_early_then_again(self):
        r = brew(Run('1-3'), 5000); r.feed(60); self.state(r, 3, 2.)                            # 20.5 ml: no success, no failure
        first = r.p.summary()['stage_points']['1']['step']
        r.feed(1, button=press('started')); self.state(r, 2, 1.)                                # third valid event
        r.feed(10, button=RUNNING, liquid=liquid(8000, airborne=20)); self.state(r, 2, 1.)
        r.feed(1, button=press('stopped_by_button'), liquid=liquid(9000, airborne=20)); self.state(r, 3, 0.)   # fourth
        r.keep(liquid=liquid(9000))
        r.feed(20); self.state(r, 3, 0.)
        r.feed(1); self.state(r, 3, 3., success=True)                                           # 2 + 3 - 2
        s = r.p.summary()
        self.assertEqual((r.kinds(), s['stage_points']['1']['step'], s['regressions']), (['extra_button_event'] * 2, first, 1))

    def test_every_valid_event_from_the_third_costs_one(self):
        r = brew(Run('1-3'), 12000)
        for n in range(3):
            r.feed(1, button=press('started')); r.feed(1, button=press('stopped_by_button'))
            self.assertEqual(len(r.kinds()), 2 * n + 2)
        self.state(r, 3, 0.)
        r.feed(22); self.state(r, 3, 0., success=True)                                          # 2 + 3 - 6
        self.assertEqual((r.p.summary()['raw_score'], r.p.summary()['final_counts']['valid_button_events']), (-1, 8))

    def test_invalid_presses_are_not_counted(self):
        r = brew(Run('1-3'), 12000)
        r.feed(1, button=dict(press('blocked_no_beans'), dispensing=False), beans_in_hopper=0)
        r.feed(1, button=dict(IDLE, displacement_m=-0.002, robot_on_button=True))               # pressed, not far enough for the machine
        self.assertEqual((r.kinds(), r.p.summary()['final_counts']['valid_button_events']), ([], 2))
        r.feed(22); self.state(r, 3, 5., success=True)
        r = Run('1-3', beans_in_hopper=0); r.feed(1, button=dict(press('blocked_no_beans'), dispensing=False)); self.state(r, 1, 0.)

    def test_running_dry_is_not_a_stop(self):
        r = Run('1-3'); r.feed(1, button=press('started'))
        r.feed(10, button=RUNNING, liquid=liquid(7000, airborne=30))
        r.feed(1, button=dict(IDLE, events=('stopped_no_beans',), event_by_robot=(False,)), liquid=liquid(8000))
        r.keep(liquid=liquid(8000)); r.feed(60); self.state(r, 2, 2.)                           # everything settled, never stopped
        self.assertEqual(r.p.summary()['final_counts']['valid_button_events'], 1)
        r.feed(1, button=dict(IDLE, events=('stopped_no_beans',), event_by_robot=(True,))); self.state(r, 2, 2.)
        r.feed(502); self.state(r, 2, 2., failure='stage2_budget_exhausted')

    def test_spill_ratio(self):
        r = Run('1-3'); r.feed(1, button=press('started'))
        r.feed(2, button=RUNNING, liquid=liquid(10000, spilled=500)); self.assertIsNone(r.p.failure)    # exactly 5 %
        r.feed(1, button=RUNNING, liquid=liquid(10000, spilled=501)); self.state(r, 2, 2., failure='liquid_spilled')
        r = Run('1-3'); r.feed(3, liquid=dict(liquid(0), spilled=7, in_cup=0)); self.assertIsNone(r.p.failure)   # nothing emitted: no ratio
        r = Run('1-3'); r.feed(1, button=press('started'))
        r.feed(5, button=RUNNING, liquid=liquid(10000, airborne=6000)); self.assertIsNone(r.p.failure)  # airborne is not spilled
        r = brew(Run('1-3'), 10000); r.feed(21, liquid=liquid(10000, spilled=500)); self.state(r, 3, 5., success=True)   # 95 % kept
        r = brew(Run('1-3'), 10000); r.feed(60, liquid=liquid(10000, spilled=500, in_cup=9499)); self.state(r, 3, 2.)

    def test_every_settlement_condition(self):
        blocked = [dict(liquid=liquid(12000, airborne=1)), dict(button=dict(IDLE, displacement_m=-0.0011)), dict(button=RUNNING),
                   dict(cup=dict(ON_TRAY, tilt_deg=11.51)), dict(cup=ON_COUNTER), dict(cup=HELD), dict(cup=ON_OTHER)]
        allowed = [dict(button=dict(IDLE, displacement_m=-0.001)), dict(button=dict(IDLE, displacement_m=0.0002)),
                   dict(button=dict(IDLE, robot_on_button=True)), dict(cup=dict(ON_TRAY, tilt_deg=11.5)),
                   dict(cup=dict(ON_TRAY, robot_contact=True)), dict(bag=dict(ON_COUNTER, tilt_deg=80.))]
        for frames, ok in ((blocked, False), (allowed, True)):
            for frame in frames:
                with self.subTest(frame=frame):
                    r = brew(Run('1-3'), 12000); r.feed(21, **frame)
                    self.assertEqual((r.p.stage, r.p.score, r.p.success, r.p.failure), (3, 5. if ok else 2., ok, None))
                    r.feed(20); r.feed(1)
                    self.assertTrue(r.p.success)                                                # a whole second after the interruption
        r = brew(Run('1-3'), 9000)
        for i in range(30): r.feed(1, liquid=liquid(9000 + i))                                  # still dripping: new liquid every frame
        self.state(r, 3, 2.)
        r = brew(Run('1-3'), 12000); r.feed(10); r.feed(11, cup=dict(ON_TRAY, position=(0.0021, 0., 0.))); self.state(r, 3, 2.)

    def test_cup_failures(self):
        r = brew(Run('1-3'), 12000); r.feed(1, cup=dict(ON_TRAY, ground_contact=True)); self.state(r, 3, 2., failure='cup_on_ground')
        r = Run('1-3'); r.feed(3); r.feed(3, cup=dict(ON_TRAY, robot_contact=True, tilt_deg=60.)); self.assertIsNone(r.p.failure)
        r.feed(10, cup=dict(ON_COUNTER, tilt_deg=60.)); self.assertIsNone(r.p.failure)
        r.feed(1, cup=dict(ON_COUNTER, tilt_deg=60.)); self.state(r, 1, 0., failure='cup_toppled')
        for cup, fails in ((dict(ON_TRAY, tilt_deg=41.), False), (dict(ON_TRAY, tilt_deg=41.01), True), (dict(ON_OTHER, tilt_deg=60.), True),
                           (dict(supports=frozenset({MACHINE_BODY}), machine_body_contact=True, tilt_deg=60.), True),
                           (dict(FALLING, tilt_deg=60.), False)):                               # 1-3 says 承托面: any surface
            with self.subTest(cup=cup):
                r = Run('1-3'); r.feed(30, cup=cup); self.assertEqual(r.p.failure, 'cup_toppled' if fails else None)

    def test_deductions_once_each(self):
        r = Run('1-3')
        r.feed(2, robot_machine_contact=True); r.feed(2); r.feed(2, robot_machine_contact=True)
        r.feed(2, cup=dict(ON_TRAY, displacement_m=0.02), bag=dict(ON_COUNTER, displacement_m=0.02, tilt_deg=80.))
        self.assertEqual(r.kinds(), ['robot_touched_machine'])                                  # 2 cm exactly; a fallen bag is not in the 1-3 table
        r.feed(2, cup=dict(ON_TRAY, displacement_m=0.021)); r.feed(2, bag=dict(ON_COUNTER, displacement_m=0.021))
        r.feed(2, cup=dict(ON_TRAY, displacement_m=0.05, machine_body_contact=True), bag=dict(ON_COUNTER, displacement_m=0.05))
        self.assertEqual(r.kinds(), ['robot_touched_machine', 'cup_displaced', 'bag_displaced'])
        brew(r, 12000, wait=0); r.feed(21); self.state(r, 3, 2., success=True)                  # 5 - 3

    def test_budgets_and_boundaries(self):
        r = Run('1-3'); r.feed(316); self.state(r, 1, 0.)
        r.feed(1); self.state(r, 1, 0., failure='stage1_budget_exhausted')
        r = Run('1-3'); r.feed(316); r.feed(1, button=press('started')); self.state(r, 2, 2.)   # started exactly at 15.8 s
        r.keep(button=RUNNING, liquid=liquid(3000))
        r.feed(573); self.state(r, 2, 2.); self.assertEqual(r.used(2), 28.65)
        r.feed(1); self.state(r, 2, 2., failure='stage2_budget_exhausted')
        r = Run('1-3'); r.feed(1, button=press('started')); r.feed(573, button=RUNNING, liquid=liquid(3000))
        r.feed(1, button=press('stopped_by_button'), liquid=liquid(3000)); self.state(r, 3, 2.) # stopped exactly at 28.7 s
        r.keep(liquid=liquid(3000)); r.feed(500); self.state(r, 3, 2.)                          # ③ has no budget of its own
        self.assertEqual((r.used(1), r.used(2)), (0., 28.7))
        r.feed(1, button=press('started')); self.state(r, 2, 1., failure='stage2_budget_exhausted')   # back in ② with nothing left

    def test_restart_continues_the_stage_two_budget(self):
        r = brew(Run('1-3'), 5000); used = r.used(2); r.feed(100)
        self.assertEqual(r.used(2), used)
        r.feed(1, button=press('started')); r.feed(10, button=RUNNING, liquid=liquid(5500)); self.assertEqual(r.used(2), round(used + 0.5, 6))

    def test_stop_points_need_the_settlement(self):
        r = brew(Run('1-3'), 12000); r.feed(20); r.p.finalize('time_limit'); self.state(r, 3, 2.)
        s = r.p.summary(); self.assertEqual((s['stage_points']['2']['awarded'], s['hold']['window_s']), (False, 0.95))


# ---------------------------------------------------------------- 1-4 取杯子
class RetrieveCup(Rules):
    def lifted(self, reach=20, carry=20):
        r = Run('1-4'); r.feed(reach); r.feed(carry, cup=HELD, placement=IN_BAY)
        return r.keep(cup=HELD, placement=AWAY)

    def test_nominal_full_score(self):
        r = Run('1-4'); r.feed(30); self.state(r, 1, 0.)
        r.feed(5, cup=GRIPPED_ON_TRAY); self.state(r, 1, 0.)                                    # gripped, the tray still carries it
        r.feed(1, cup=HELD, placement=IN_BAY); self.state(r, 2, 2.)
        r.feed(40, cup=HELD, placement=IN_BAY); self.state(r, 2, 2.)
        r.feed(1, cup=HELD, placement=AWAY); self.state(r, 3, 5.)
        r.feed(9, cup=HELD, placement=AWAY); self.state(r, 3, 5.)
        r.feed(1, cup=HELD, placement=AWAY); self.state(r, 3, 5., success=True)
        s = r.p.summary()
        self.assertEqual((s['stage_points']['1']['time_s'], s['stage_points']['2']['time_s'], s['end_time_s']), (35 * DT, 76 * DT, 86 * DT))
        self.assertEqual((s['budgets']['1']['used_s'], s['budgets']['2']['used_s'], s['deductions']), (1.75, 2.05, []))
        self.assertEqual((s['final_counts']['denominator_particles'], s['final_counts']['kept_ratio']), (FULL, 1.0))

    def test_stage_thresholds(self):
        for tilt, ok in ((15., True), (15.01, False)):
            with self.subTest(tilt=tilt):
                r = Run('1-4'); r.feed(5); r.feed(5, cup=dict(HELD, tilt_deg=tilt), placement=IN_BAY)
                self.state(r, 2 if ok else 1, 2. if ok else 0.)
                r = self.lifted(); r.feed(11, cup=dict(HELD, tilt_deg=tilt)); self.state(r, 3 if ok else 2, 5. if ok else 2., success=ok)
        r = self.lifted(); r.feed(30, placement=IN_BAY); self.state(r, 2, 2.)                   # not beyond the exit plane
        for cup in (GRIPPED_ON_COUNTER, ON_COUNTER, dict(robot_contact=True, supports=frozenset()), FALLING):
            with self.subTest(cup=cup):
                r = self.lifted(); r.feed(30, cup=cup); self.assertEqual((r.p.score, r.p.success), (2., False))

    def test_spill_ratio(self):
        r = self.lifted(); r.feed(11, liquid=liquid(FULL, spilled=515, initial=FULL)); self.state(r, 3, 5., success=True)   # 4.998 %
        r = self.lifted(); r.feed(1, liquid=liquid(FULL, spilled=516, initial=FULL)); self.state(r, 2, 2., failure='liquid_spilled')
        r = Run('1-4'); r.feed(1, liquid=liquid(FULL, spilled=516, initial=FULL)); self.state(r, 1, 0., failure='liquid_spilled')

    def test_airborne_liquid_is_not_spilled_but_not_kept(self):
        r = self.lifted(); r.feed(40, liquid=liquid(FULL, airborne=3000, initial=FULL)); self.state(r, 3, 5.)
        self.assertEqual(r.p.summary()['hold']['window_s'], 0.)
        r.feed(10); self.state(r, 3, 5.)                                                        # it fell back into the cup
        r.feed(1); self.state(r, 3, 5., success=True)

    def test_kept_ratio(self):
        r = self.lifted(); r.feed(11, liquid=liquid(FULL, airborne=515, initial=FULL)); self.state(r, 3, 5., success=True)   # 9789 of 10304
        r = self.lifted(); r.feed(40, liquid=liquid(FULL, airborne=516, initial=FULL)); self.state(r, 3, 5.)
        r = self.lifted(); r.feed(10); r.feed(1, liquid=liquid(FULL, airborne=516, initial=FULL)); self.state(r, 3, 5.)      # interrupted
        r.feed(10); self.state(r, 3, 5.)
        r.feed(1); self.state(r, 3, 5., success=True)

    def test_hold_span(self):
        at = lambda z: dict(HELD, position=(0., 0., z))
        r = self.lifted(); r.feed(5, cup=at(0.)); r.feed(6, cup=at(0.002)); self.state(r, 3, 5., success=True)
        r = self.lifted(); r.feed(5, cup=at(0.)); r.feed(6, cup=at(0.0021)); self.state(r, 3, 5.)
        r = self.lifted()
        for i in range(60): r.feed(1, cup=at(0.0005 * i))                                       # carried on at 1 cm/s
        self.state(r, 3, 5.)

    def test_cup_failures(self):
        r = self.lifted(); r.feed(1, cup=dict(HELD, ground_contact=True)); self.state(r, 2, 2., failure='cup_on_ground')
        lying = dict(ON_COUNTER, tilt_deg=70.)
        r = self.lifted(); r.feed(10, cup=lying); self.state(r, 1, 2.)
        r.feed(1, cup=lying); self.state(r, 1, 2., failure='cup_toppled')
        for cup, fails in ((dict(ON_TRAY, tilt_deg=41.01), True), (dict(ON_TRAY, tilt_deg=41.), False), (dict(ON_OTHER, tilt_deg=70.), False),
                           (dict(HELD, tilt_deg=70.), False), (dict(lying, robot_contact=True), False)):
            with self.subTest(cup=cup):
                r = self.lifted(); r.feed(30, cup=cup); self.assertEqual(r.p.failure, 'cup_toppled' if fails else None)

    def test_deductions_once_each(self):
        r = self.lifted(carry=1).keep(placement=IN_BAY)
        r.feed(2, cup=dict(HELD, machine_body_contact=True)); r.feed(2); r.feed(2, cup=dict(HELD, machine_body_contact=True))
        r.feed(2, robot_machine_contact=True); r.feed(2); r.feed(2, robot_machine_contact=True)
        r.feed(2, bag=dict(ON_COUNTER, displacement_m=0.02, tilt_deg=41.)); self.assertEqual(len(r.kinds()), 2)
        r.feed(2, bag=dict(ON_COUNTER, tilt_deg=41.5)); r.feed(2, bag=dict(ON_COUNTER, displacement_m=0.3))
        r.feed(2, cup=dict(HELD, displacement_m=0.3))                                           # the cup is the target
        self.assertEqual((r.kinds(), r.p.score), (['cup_touched_machine', 'robot_touched_machine', 'bag_disturbed'], 0.))
        r.feed(11, placement=AWAY); self.state(r, 3, 2., success=True)

    def test_regression_back_on_a_surface(self):
        r = self.lifted()
        r.feed(3, cup=GRIPPED_ON_TRAY, placement=IN_BAY); self.state(r, 1, 2.)                  # the tray carries it again
        r.feed(5, cup=ON_TRAY, placement=IN_BAY); self.state(r, 1, 2.)
        r.feed(3, cup=dict(HELD, tilt_deg=20.), placement=IN_BAY); self.state(r, 1, 2.)         # ① asks for <= 15 deg again
        r.feed(3, placement=IN_BAY); self.state(r, 2, 2.)
        r.feed(5); self.state(r, 3, 5.)
        r.feed(5, cup=ON_COUNTER); self.state(r, 1, 5.)                                         # put down outside the bay
        r.feed(10); self.state(r, 3, 5.)                                                        # picked up there: ① and ② at once
        r.feed(1); self.state(r, 3, 5., success=True)
        s = r.p.summary()
        self.assertEqual((s['stage_points']['1']['step'], s['stage_points']['2']['step'], s['regressions']), (20, 54, 2))

    def test_regression_still_held(self):
        r = self.lifted(); r.feed(10); self.state(r, 3, 5.)
        r.feed(3, placement=IN_BAY); self.state(r, 2, 5.)                                       # moved back into the bay
        r.feed(10); self.state(r, 3, 5.)
        r.feed(3, cup=dict(HELD, tilt_deg=30.)); self.state(r, 2, 5.)                           # held, tilted
        r.feed(10); r.feed(3, cup=dict(robot_contact=True, supports=frozenset())); self.state(r, 2, 5.)   # on one finger
        r.feed(2, cup=FALLING); self.state(r, 2, 5.)
        r.feed(10); self.state(r, 3, 5.)
        r.feed(1); self.state(r, 3, 5., success=True)

    def test_budgets_and_boundaries(self):
        r = Run('1-4'); r.feed(1038); self.state(r, 1, 0.)
        r.feed(1); self.state(r, 1, 0., failure='stage1_budget_exhausted')
        r = Run('1-4'); r.feed(1038); r.feed(1, cup=HELD, placement=IN_BAY); self.state(r, 2, 2.)        # lifted exactly at 51.9 s
        r.keep(cup=HELD, placement=IN_BAY)
        r.feed(691); self.state(r, 2, 2.)
        r.feed(1); self.state(r, 2, 2., failure='stage2_budget_exhausted')
        r = self.lifted(carry=1); r.feed(691, placement=IN_BAY); r.feed(11); self.state(r, 3, 5., success=True)   # out exactly at 34.6 s
        self.assertEqual((r.used(1), r.used(2)), (1.0, 34.6))
        r = self.lifted(carry=1); r.feed(691, placement=IN_BAY); r.feed(10); r.feed(1, placement=IN_BAY)
        self.state(r, 2, 5., failure='stage2_budget_exhausted')                                 # regression cannot refill ②

    def test_time_limit_keeps_points_and_deductions(self):
        r = self.lifted(); r.feed(9, robot_machine_contact=True); r.p.finalize('time_limit'); self.state(r, 3, 4.)
        self.assertEqual(r.p.summary()['final_counts']['spilled_particles'], 0)


if __name__ == '__main__':
    unittest.main()

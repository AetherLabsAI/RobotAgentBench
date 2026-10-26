"""Process rules of the coffee machine family: 1-1 放杯子, 1-2 倾倒咖啡豆, 1-3 按按钮, 1-4 取杯子.

Arithmetic on Facts only (facts.py): no Isaac, USD, controller, parameter overrides or model-provided verdicts. One
Facts per 20 Hz control frame, in order. The single source of every rule and constant is the definition table and the
实现绑定 section of apis/<api>/README.md: change the table first, then this file.

Order inside one frame (失败优先于成功，成功优先于超时；恰在预算边界达成阶段先转阶段):
  1. the interval since the previous frame is charged to the operation stage that was active during it;
  2. failure conditions: a failing frame ends the episode and neither charges a deduction nor awards a stage;
  3. once-per-episode deductions;
  4. stage conditions: points (historical, once, never revoked), regression, the 1 s stability window, success;
  5. the budget of the operation stage that is still unachieved after 4.
The total time limit is enforced by the public framework; finalize() performs the settlement for it.

Stages (stage = the one currently being worked on):
  1-1  ① grasped +2   ② on the tray, released, <= 11.5 deg, <= 1.2 cm from the anchor, outlet over the opening +3
       ③ ② for 1 s = success
  1-2  ① grasped +2   ② 70 beans in the hopper +3   ③ bag on the counter, released, <= 10 deg +1, then 1 s.
       The bean count is settled when that window ends, or when the episode ends for any other reason.
  1-3  ① valid start +2   ② valid stop   ③ stopped, no new liquid, button back, nothing airborne, 25-60 ml and
       >= 95 % kept, cup on the tray <= 11.5 deg, for 1 s = success; the +3 of ② is paid at that moment
  1-4  ① grasped off the tray, <= 15 deg +2   ② still grasped, whole cup beyond the bay exit, <= 15 deg +3
       ③ ② and >= 95 % of the initial liquid for 0.5 s = success (README 第五版, 2026-09-30)

Regression never pays points again and never resets a used budget. Named by the tables:
  1-1  still gripped -> ②; put down upright in the wrong place -> ①
  1-2  bag put down and to be grasped again -> ①; still holding with fewer than 70 beans -> ②; 70 or more -> ③
  1-3  restarted after a stop -> ②
  1-4  back on the tray or the counter -> ①; still held but not meeting ② -> ②
Not named by the tables and filled in here so that the machine is total (listed in the hand-over report):
  the robot still touching the object without a bilateral grasp counts as "still with the robot" (1-1, 1-2: the stage
  that precedes the release; 1-4: ②, unless something outside the robot supports the cup, then ①); an object that is
  released and rests on anything without meeting the stage condition needs a new grasp (①), whatever its tilt; an
  object in free flight keeps its stage; once ① has been paid, ② may be reached from ① without a new grasp.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
from .facts import (Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts, ML_PER_PARTICLE,
                    COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER)

RULE_VERSION = 'arena.coffee_machine.process.v4'   # v4 2026-10-06: stage budgets and total limits = ceil(2 x the longest of the data producer's 50 episodes) (README 第六版/第五版/第四版); keeps the visual_cpu_v1 liquid mode of v3_visual.  # v2 2026-09-30: 1-4 hold 0.5 s, stage 2 budget 24.3 s, tilt 15 deg (README 第五版)
CONTROL_DT_S = 0.05                 # one Facts per 20 Hz control frame
CLOCK_TOLERANCE_S = 1e-6            # accepted disagreement between time_s and step * CONTROL_DT_S
EPS = 1e-9                          # only for durations built from integer frame counts
SURFACES = frozenset({COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER})
BUTTON_EVENTS = ('started', 'stopped_by_button', 'stopped_no_beans', 'blocked_no_beans')
VALID_BUTTON_EVENTS = ('started', 'stopped_by_button')     # 自动耗尽不算按停; a blocked press starts nothing


@dataclass(frozen=True)
class Spec:
    api: str
    title: str
    instruction: str                    # README row 发给 VLA 的指令, verbatim
    target: str                         # object whose origin drives the stability window
    time_limit_s: float                 # README row 时限与终止; enforced by the public framework
    stage_budget_s: MappingProxyType    # operation stage -> budget; no entry = no operation budget
    stage_points: MappingProxyType      # stage -> points
    maximum_score: int
    deductions: tuple                   # once-per-episode deductions that exist for this API
    thresholds: MappingProxyType

    def as_dict(self):
        return dict(api=self.api, title=self.title, instruction=self.instruction, target=self.target,
                    time_limit_s=self.time_limit_s, maximum_score=self.maximum_score,
                    stage_budget_s={str(k): v for k, v in self.stage_budget_s.items()},
                    stage_points={str(k): v for k, v in self.stage_points.items()}, deductions=list(self.deductions),
                    thresholds={k: list(v) if isinstance(v, tuple) else v for k, v in self.thresholds.items()})


def _spec(api, title, instruction, target, time_limit_s, budgets, points, deductions, **thresholds):
    common = dict(hold_s=1.0, hold_span_m=0.002, topple_tilt_deg=41.0, topple_supports=(COUNTER, TRAY),
                  release_settle_s=0.5, ground_failure=('cup',), disturb_displacement_m=0.02)
    return Spec(api, title, instruction, target, float(time_limit_s), MappingProxyType(dict(budgets)),
                MappingProxyType(dict(points)), sum(points.values()), tuple(deductions),
                MappingProxyType(dict(common, **thresholds)))


SPECS = {
    '1-1': _spec('1-1', '放杯子', '把咖啡杯放到咖啡机出液口下方的接水盘上，松手并放稳。', 'cup', 72,
                 {1: 29.9, 2: 35.3}, {1: 2, 2: 3}, ('cup_touched_machine', 'robot_touched_machine', 'bag_disturbed'),
                 cup_upright_tilt_deg=11.5, place_xy_error_m=0.012, bag_topple_tilt_deg=41.0),
    '1-2': _spec('1-2', '倾倒咖啡豆', '拿起咖啡豆袋，把咖啡豆倒进咖啡机的豆仓，倒完后把豆袋放回台面并立稳。', 'bag', 93,
                 {1: 28.3, 2: 29.1, 3: 32.9}, {1: 2, 2: 3, 3: 1}, ('robot_touched_machine', 'cup_displaced'),
                 bag_upright_tilt_deg=10.0, release_settle_s=None, ground_failure=('cup', 'bag'),
                 beans_required=70, beans_minus_two_below=80, beans_minus_one_below=90),
    '1-3': _spec('1-3', '按按钮', '按一下咖啡机的按钮开始出咖啡，接取 25–60 ml 咖啡后再按一下按钮停止。', 'cup', 67,
                 {1: 15.8, 2: 28.7}, {1: 2, 2: 3}, ('robot_touched_machine', 'cup_displaced', 'bag_displaced'),
                 cup_upright_tilt_deg=11.5, topple_supports=tuple(sorted(SURFACES)), button_released_depth_m=0.001,
                 cup_liquid_min_ml=25.0, cup_liquid_max_ml=60.0, keep_percent=95, spill_percent=5,
                 spill_base='emitted', free_button_events=2, extra_button_event_deduction=1),
    '1-4': _spec('1-4', '取杯子', '把接好咖啡的杯子从咖啡机里取出来，端稳，不要洒。', 'cup', 83,
                 {1: 51.9, 2: 34.6}, {1: 2, 2: 3}, ('cup_touched_machine', 'robot_touched_machine', 'bag_disturbed'),
                 cup_upright_tilt_deg=15.0, hold_s=0.5, keep_percent=95, spill_percent=5, spill_base='initial_in_cup',
                 bag_topple_tilt_deg=41.0),
}


def _number(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Non-finite or non-numeric fact: ' + name)
    if low is not None and value < low: raise ValueError('Fact out of range: ' + name)


def _count(name, value):
    if type(value) is not int or value < 0: raise ValueError('Non-negative integer count required: ' + name)


def _flag(name, value):
    if type(value) is not bool: raise ValueError('Boolean fact required: ' + name)


def _object(name, o):
    if not isinstance(o, ObjectFacts): raise ValueError('Object facts required: ' + name)
    if not isinstance(o.position, (tuple, list)) or len(o.position) != 3: raise ValueError('Position (x, y, z) required: ' + name)
    for i, v in enumerate(o.position): _number('%s.position[%d]' % (name, i), v)
    _number(name + '.tilt_deg', o.tilt_deg, 0.)
    if o.tilt_deg > 180.: raise ValueError('Fact out of range: ' + name + '.tilt_deg')
    _number(name + '.displacement_m', o.displacement_m, 0.)
    for key in ('pads_bilateral', 'robot_contact', 'ground_contact', 'machine_body_contact'): _flag(name + '.' + key, getattr(o, key))
    if not isinstance(o.supports, (frozenset, set)) or not o.supports <= SURFACES: raise ValueError('Unknown support surface: ' + name)
    # the three contradictions below would let a drop, a release or a machine contact go unnoticed
    if o.pads_bilateral and not o.robot_contact: raise ValueError('Finger pads touch but the robot does not: ' + name)
    if GROUND in o.supports and not o.ground_contact: raise ValueError('Ground supports without ground contact: ' + name)
    if MACHINE_BODY in o.supports and not o.machine_body_contact: raise ValueError('Machine supports without machine contact: ' + name)


class Process:
    """Rule engine of one episode of one API. observe() once per control frame, finalize() once at the end."""

    def __init__(self, api):
        if api not in SPECS: raise ValueError('Unknown coffee_machine API: %r' % (api,))
        self.api, self.spec = api, SPECS[api]
        self._advance = {'1-1': self._place_cup, '1-2': self._pour_beans, '1-3': self._press_button, '1-4': self._retrieve_cup}[api]
        self._stage = 1
        self._awards = {}               # stage -> dict(points, step, time_s)
        self._deductions = []           # dict(kind, amount, step, time_s)
        self._once = set()
        self._used = {s: 0 for s in self.spec.stage_budget_s}   # control intervals charged to each operation stage
        self._charging = 1              # stage that pays for the next interval; None while a condition is being held
        self._window = []               # (step, position) of the current stability window
        self._window_restarts = 0
        self._regressions = 0
        self._events = []
        self._released_step = None      # first frame of the current stretch without robot contact on the cup
        self._success, self._failure = False, None
        self._end = None                # dict(step, time_s) of the frame that ended the episode
        self._settled = False
        self._finalized, self._termination = False, None
        self._facts = None              # last frame judged (never a frame after the end)
        self._last_step = self._last_time_s = None
        self._frames = self._post_terminal_frames = 0
        self._initial_in_cup = None     # 1-4 denominator, fixed at the first frame
        self._valid_events = 0          # 1-3
        self._stopped_by_robot = False  # 1-3: the machine is stopped and a valid stop was the last thing that changed it
        self._previous_emitted = self._previous_spilled = None
        self._beans_band = None         # 1-2 settlement

    # ---- public state ----
    @property
    def stage(self):
        return self._stage

    @property
    def score(self):
        return max(0., float(self._points() - self._deducted()))

    @property
    def success(self):
        return self._success

    @property
    def failure(self):
        return self._failure

    def _points(self):
        return sum(a['points'] for a in self._awards.values())

    def _deducted(self):
        return sum(d['amount'] for d in self._deductions)

    # ---- evidence integrity ----
    def _validate(self, f):
        if not isinstance(f, Facts): raise ValueError('Facts of one control frame required')
        expected = 0 if self._last_step is None else self._last_step + 1
        if type(f.step) is not int or f.step != expected:
            raise ValueError('Skipped, repeated or out-of-order control frame: expected step %d, got %r' % (expected, f.step))
        _number('time_s', f.time_s)
        if abs(f.time_s - f.step * CONTROL_DT_S) > CLOCK_TOLERANCE_S: raise ValueError('time_s is not on the 20 Hz control clock')
        _object('cup', f.cup); _object('bag', f.bag)
        if not isinstance(f.placement, CupPlacement): raise ValueError('Cup placement facts required')
        _number('placement.xy_error_m', f.placement.xy_error_m, 0.)
        _flag('placement.outlet_over_opening', f.placement.outlet_over_opening); _flag('placement.outside_bay', f.placement.outside_bay)
        _flag('robot_machine_contact', f.robot_machine_contact)
        _flag('contacts_complete', f.contacts_complete)
        _count('beans_in_hopper', f.beans_in_hopper); _count('beans_total', f.beans_total)
        if 'beans_required' in self.spec.thresholds and f.beans_in_hopper > f.beans_total: raise ValueError('More beans in the hopper than exist')
        needs_liquid, needs_button = 'spill_base' in self.spec.thresholds, 'button_released_depth_m' in self.spec.thresholds
        if f.liquid is None and needs_liquid: raise ValueError('Missing liquid facts is an invalid episode, not a pass')
        if f.button is None and needs_button: raise ValueError('Missing button facts is an invalid episode, not a pass')
        if f.liquid is not None:
            if not isinstance(f.liquid, LiquidFacts): raise ValueError('Liquid facts required')
            if f.liquid.mode not in ('pbd', 'visual_cpu_v1'): raise ValueError('Unknown liquid evidence mode')
            for key in ('emitted', 'in_cup', 'airborne', 'spilled', 'initial_in_cup'): _count('liquid.' + key, getattr(f.liquid, key))
            if needs_liquid:
                if self._previous_spilled is not None and f.liquid.spilled < self._previous_spilled: raise ValueError('Spilled particles are cumulative')
                if self.spec.thresholds['spill_base'] == 'emitted':
                    if self._previous_emitted is not None and f.liquid.emitted < self._previous_emitted: raise ValueError('Emitted particles are cumulative')
                else:
                    fixed = f.liquid.initial_in_cup if self._initial_in_cup is None else self._initial_in_cup
                    if fixed <= 0 or f.liquid.initial_in_cup != fixed: raise ValueError('Initial cup content must be positive and fixed')
        if f.button is not None:
            b = f.button
            if not isinstance(b, ButtonFacts): raise ValueError('Button facts required')
            _number('button.displacement_m', b.displacement_m)
            _flag('button.robot_on_button', b.robot_on_button); _flag('button.dispensing', b.dispensing)
            if not isinstance(b.events, tuple) or not isinstance(b.event_by_robot, tuple) or len(b.events) != len(b.event_by_robot):
                raise ValueError('One robot flag per button event required')
            if any(e not in BUTTON_EVENTS for e in b.events): raise ValueError('Unknown button event')
            for v in b.event_by_robot: _flag('button.event_by_robot', v)

    # ---- bookkeeping ----
    def _award(self, f, stage):
        if stage in self._awards: return
        points = self.spec.stage_points[stage]
        self._awards[stage] = dict(points=points, step=f.step, time_s=float(f.time_s))
        self._events.append(dict(kind='stage_points', stage=stage, points=points, step=f.step, time_s=float(f.time_s)))

    def _deduct(self, f, kind, amount=1, once=True):
        if once:
            if kind in self._once: return
            self._once.add(kind)
        self._deductions.append(dict(kind=kind, amount=amount, step=f.step, time_s=float(f.time_s)))
        self._events.append(dict(kind='deduction', name=kind, points=-amount, step=f.step, time_s=float(f.time_s)))

    def _enter(self, f, stage):
        if stage == self._stage: return
        back = stage < self._stage
        self._regressions += back
        self._events.append(dict(kind='regression' if back else 'stage', from_stage=self._stage, to_stage=stage,
                                 step=f.step, time_s=float(f.time_s)))
        self._stage = stage

    def _charge(self, holding=False):
        """The next interval is paid by the current stage unless it has no budget or its condition is being held."""
        self._charging = None if holding or self._stage not in self.spec.stage_budget_s else self._stage

    def _hold(self, f, holding, position):
        """1 s window: every frame meets the condition and no two origins in it are further apart than 2 mm."""
        t = self.spec.thresholds
        if not holding:
            self._window = []
            return False
        if any(math.dist(p, position) > t['hold_span_m'] for _, p in self._window):
            self._window = []; self._window_restarts += 1
            self._events.append(dict(kind='window_restarted', step=f.step, time_s=float(f.time_s)))
        self._window.append((f.step, tuple(position)))
        return self._window_s() >= t['hold_s'] - EPS

    def _window_s(self):
        return (self._window[-1][0] - self._window[0][0]) * CONTROL_DT_S if self._window else 0.

    def _finish(self, f, *, success=False, failure=None):
        self._success, self._failure = success, failure
        self._end = dict(step=f.step, time_s=float(f.time_s))
        self._events.append(dict(kind='success' if success else 'terminal_failure', reason=failure, stage=self._stage,
                                 step=f.step, time_s=float(f.time_s)))
        self._settle(f)

    def _settle(self, f):
        """End-of-episode settlement, once. 1-2: the bean count of the last judged frame, never the historical maximum."""
        if self._settled: return
        self._settled = True
        t = self.spec.thresholds
        if 'beans_required' not in t: return
        n = f.beans_in_hopper
        if n < t['beans_required']:
            self._beans_band = 'below_%d' % t['beans_required']     # failure, no bean deduction on top
            if self._failure is None:
                self._success, self._failure = False, 'beans_below_%d' % t['beans_required']
                self._events.append(dict(kind='terminal_failure', reason=self._failure, stage=self._stage, step=f.step, time_s=float(f.time_s)))
        elif n < t['beans_minus_two_below']:
            self._beans_band = '%d_%d' % (t['beans_required'], t['beans_minus_two_below'] - 1)
            self._deduct(f, 'beans_' + self._beans_band, 2)
        elif n < t['beans_minus_one_below']:
            self._beans_band = '%d_%d' % (t['beans_minus_two_below'], t['beans_minus_one_below'] - 1)
            self._deduct(f, 'beans_' + self._beans_band, 1)
        else:
            self._beans_band = '%d_or_more' % t['beans_minus_one_below']

    # ---- one control frame ----
    def observe(self, facts):
        if self._finalized: raise RuntimeError('Episode already finalized; a new episode needs a new Process')
        self._validate(facts)
        self._last_step, self._last_time_s = facts.step, float(facts.time_s)
        self._frames += 1
        if facts.liquid is not None:
            if self._initial_in_cup is None: self._initial_in_cup = facts.liquid.initial_in_cup
            emitted_before = self._previous_emitted
            self._previous_emitted, self._previous_spilled = facts.liquid.emitted, facts.liquid.spilled
        else:
            emitted_before = None
        if self._end is not None:
            self._post_terminal_frames += 1     # the verdict is locked; later frames are only counted
            return
        self._facts = facts
        if facts.step > 0 and self._charging is not None: self._used[self._charging] += 1
        if not facts.contacts_complete:
            # Record the real control time, retain earned stages, and break
            # continuous stability. Unknown is neither release nor no collision.
            self._window.clear()
            self._released_step=None
            self._charge_deductions(facts)  # measured positive events remain chargeable
            self._events.append(dict(kind='contact_evidence_gap',step=facts.step,time_s=float(facts.time_s),points=0))
            return
        if facts.cup.robot_contact: self._released_step = None
        elif self._released_step is None: self._released_step = facts.step
        reason = self._failure_now(facts)
        if reason is not None: return self._finish(facts, failure=reason)
        self._charge_deductions(facts)
        verdict = self._advance(facts, emitted_before)
        if verdict == 'success': return self._finish(facts, success=True)
        if verdict is not None: return self._finish(facts, failure=verdict)
        stage = self._charging
        if stage is not None and self._used[stage] * CONTROL_DT_S >= self.spec.stage_budget_s[stage] - EPS:
            return self._finish(facts, failure='stage%d_budget_exhausted' % stage)

    def _failure_now(self, f):
        t, cup = self.spec.thresholds, f.cup
        if 'cup' in t['ground_failure'] and cup.ground_contact: return 'cup_on_ground'
        if 'bag' in t['ground_failure'] and f.bag.ground_contact: return 'bag_on_ground'
        if cup.tilt_deg > t['topple_tilt_deg'] and cup.supports & frozenset(t['topple_supports']):
            settle = t['release_settle_s']      # None: no release condition and no tolerance (1-2)
            if settle is None or (self._released_step is not None and (f.step - self._released_step) * CONTROL_DT_S >= settle - EPS):
                return 'cup_toppled'
        if self.api == '1-4' and f.liquid.mode == 'visual_cpu_v1' and cup.tilt_deg > t['cup_upright_tilt_deg']:
            return 'cup_tilt_exceeded'
        base = t.get('spill_base')
        if base is not None:
            whole = f.liquid.emitted if base == 'emitted' else self._initial_in_cup
            if whole > 0 and f.liquid.spilled * 100 > t['spill_percent'] * whole: return 'liquid_spilled'
        if 'cup_liquid_max_ml' in t and f.liquid.in_cup * ML_PER_PARTICLE > t['cup_liquid_max_ml']: return 'cup_overfilled'
        return None

    def _charge_deductions(self, f):
        t, far = self.spec.thresholds, self.spec.thresholds['disturb_displacement_m']
        happened = dict(cup_touched_machine=f.cup.machine_body_contact, robot_touched_machine=f.robot_machine_contact,
                        bag_disturbed=f.bag.tilt_deg > t.get('bag_topple_tilt_deg', math.inf) or f.bag.displacement_m > far,
                        bag_displaced=f.bag.displacement_m > far, cup_displaced=f.cup.displacement_m > far)
        for kind in self.spec.deductions:
            if happened[kind]: self._deduct(f, kind)

    # ---- 1-1 放杯子 ----
    def _place_cup(self, f, _):
        t, cup, p = self.spec.thresholds, f.cup, f.placement
        held = cup.grasped
        placed = (TRAY in cup.supports and cup.released and cup.tilt_deg <= t['cup_upright_tilt_deg']
                  and p.xy_error_m <= t['place_xy_error_m'] and p.outlet_over_opening)
        if self._stage == 1 and held: self._award(f, 1)
        if 1 in self._awards:
            if placed: self._award(f, 2); self._enter(f, 3)
            elif held: self._enter(f, 2)                # 仍被夹持则回阶段②
            elif self._stage == 1: pass                 # a new grasp is still outstanding
            elif cup.robot_contact: self._enter(f, 2)   # set down or touched but not released yet
            elif cup.supports: self._enter(f, 1)        # released somewhere else: 需重抓
        self._charge()
        return 'success' if self._hold(f, 1 in self._awards and placed, cup.position) else None

    # ---- 1-2 倾倒咖啡豆 ----
    def _pour_beans(self, f, _):
        t, bag = self.spec.thresholds, f.bag
        held, enough = bag.grasped, f.beans_in_hopper >= t['beans_required']
        back = COUNTER in bag.supports and bag.released and bag.tilt_deg <= t['bag_upright_tilt_deg']
        if self._stage == 1 and held: self._award(f, 1)
        if 1 not in self._awards:
            self._charge()
            return None
        if enough: self._award(f, 2)
        if enough and back: self._award(f, 3)
        if back: self._enter(f, 3 if enough else 1)             # too few beans: 袋已放下需重抓时回①
        elif held: self._enter(f, 3 if enough else 2)           # 仍持袋且进仓不足 70 颗回②; 70 or more: ③ or keep pouring
        elif self._stage == 1: pass
        elif bag.released and bag.supports: self._enter(f, 1)   # put down but not upright on the counter
        else: self._enter(f, 3 if enough else 2)                # robot still touching it, or in free flight
        self._charge(holding=back and self._stage == 3)
        # ①完成后豆袋松手并放稳 1 s: settled with the count of this frame, whatever the stage
        if self._hold(f, back, bag.position): return 'success' if enough else 'beans_below_%d' % t['beans_required']
        return None

    # ---- 1-3 按按钮 ----
    def _press_button(self, f, emitted_before):
        t, b, liquid, cup = self.spec.thresholds, f.button, f.liquid, f.cup
        for name, by_robot in zip(b.events, b.event_by_robot):
            if name == 'started': self._stopped_by_robot = False
            if not by_robot or name not in VALID_BUTTON_EVENTS: continue
            self._valid_events += 1
            if self._valid_events > t['free_button_events']:
                self._deduct(f, 'extra_button_event', t['extra_button_event_deduction'], once=False)
            if name == 'started': self._award(f, 1)
            else: self._stopped_by_robot = True
        if 1 in self._awards: self._enter(f, 3 if self._stopped_by_robot else 2)
        self._charge()
        settled = (self._stage == 3 and not b.dispensing and emitted_before is not None and liquid.emitted == emitted_before
                   and -b.displacement_m <= t['button_released_depth_m'] and liquid.airborne == 0
                   and t['cup_liquid_min_ml'] <= liquid.in_cup * ML_PER_PARTICLE <= t['cup_liquid_max_ml']
                   and liquid.in_cup * 100 >= t['keep_percent'] * liquid.emitted
                   and TRAY in cup.supports and cup.tilt_deg <= t['cup_upright_tilt_deg'])
        if not self._hold(f, settled, cup.position): return None
        self._award(f, 2)       # 有效按停且最终液量与稳定条件达成 +3
        return 'success'

    # ---- 1-4 取杯子 ----
    def _retrieve_cup(self, f, _):
        t, cup = self.spec.thresholds, f.cup
        lifted = cup.grasped and cup.tilt_deg <= t['cup_upright_tilt_deg']
        out = lifted and f.placement.outside_bay
        if self._stage == 1 and lifted: self._award(f, 1)
        if 1 in self._awards:
            if out: self._award(f, 2); self._enter(f, 3)
            elif self._stage == 1: self._enter(f, 2 if lifted else 1)
            elif cup.supports: self._enter(f, 1)            # 落回接水盘或台面，可重抓，回阶段①
            elif cup.robot_contact: self._enter(f, 2)       # 仍持有但不满足取出条件，回阶段②
        self._charge()
        kept = (cup.tilt_deg <= t['cup_upright_tilt_deg'] if f.liquid.mode == 'visual_cpu_v1'
                else f.liquid.in_cup * 100 >= t['keep_percent'] * self._initial_in_cup)
        return 'success' if self._hold(f, 1 in self._awards and out and kept, cup.position) else None

    # ---- end of episode ----
    def finalize(self, reason):
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('Termination reason required')
        if self._finalized:
            if reason != self._termination: raise ValueError('Contradictory repeated finalization')
            return
        if self._facts is None: raise ValueError('Cannot settle an episode without evidence')
        if reason == 'success' and not self._success: raise ValueError('No measured success behind the requested result')
        self._settle(self._facts)
        self._termination, self._finalized = reason, True

    def _final_counts(self):
        f, t, out = self._facts, self.spec.thresholds, {}
        if f is None: return out
        if 'beans_required' in t:
            out.update(beans_in_hopper=f.beans_in_hopper, beans_total=f.beans_total, beans_band=self._beans_band)
        if 'spill_base' in t:
            whole = f.liquid.emitted if t['spill_base'] == 'emitted' else self._initial_in_cup
            out.update(liquid_mode=f.liquid.mode, physical_liquid=f.liquid.mode == 'pbd')
            suffix = '_volume_units' if f.liquid.mode == 'visual_cpu_v1' else '_particles'
            out.update({k + suffix: getattr(f.liquid, k) for k in ('emitted', 'in_cup', 'airborne', 'spilled')})
            out.update(in_cup_ml=f.liquid.in_cup * ML_PER_PARTICLE, denominator=t['spill_base'], denominator_particles=whole,
                       kept_ratio=f.liquid.in_cup / whole if whole else None, spilled_ratio=f.liquid.spilled / whole if whole else None)
            if f.liquid.mode == 'visual_cpu_v1':
                out.pop('kept_ratio', None); out.pop('spilled_ratio', None)
                out['denominator_volume_units'] = out.pop('denominator_particles')
        if 'free_button_events' in t:
            out.update(valid_button_events=self._valid_events, stopped_by_robot=self._stopped_by_robot, dispensing=f.button.dispensing)
        return out

    def summary(self):
        points, deducted, budgets = self._points(), self._deducted(), self.spec.stage_budget_s
        return dict(
            schema=RULE_VERSION, api=self.api, title=self.spec.title, instruction=self.spec.instruction,
            time_limit_s=self.spec.time_limit_s, stage=self._stage, maximum_score=self.spec.maximum_score,
            score=self.score, raw_score=points - deducted, points_awarded=points, deduction_total=deducted,
            stage_points={str(s): dict(points=p, awarded=s in self._awards, step=self._awards.get(s, {}).get('step'),
                                       time_s=self._awards.get(s, {}).get('time_s')) for s, p in self.spec.stage_points.items()},
            deductions=[dict(d) for d in self._deductions],
            budgets={str(s): dict(budget_s=budgets[s], used_s=round(self._used[s] * CONTROL_DT_S, 6),
                                  exhausted=self._failure == 'stage%d_budget_exhausted' % s) for s in budgets},
            hold=dict(required_s=self.spec.thresholds['hold_s'], span_limit_m=self.spec.thresholds['hold_span_m'],
                      window_s=round(self._window_s(), 6), restarts=self._window_restarts),
            success=self._success, failure=self._failure, ended=self._end is not None,
            end_step=self._end and self._end['step'], end_time_s=self._end and self._end['time_s'],
            settled=self._settled, finalized=self._finalized, termination=self._termination,
            regressions=self._regressions, events=[dict(e) for e in self._events],
            frames=self._frames, last_step=self._last_step, last_time_s=self._last_time_s,
            post_terminal_frames=self._post_terminal_frames, final_counts=self._final_counts())

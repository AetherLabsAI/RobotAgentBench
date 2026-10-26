"""Process rules of the Task 2 family: 2.1 / 2.2 放入蛋糕, 2.3 拉出炸篮, 2.5 取出蛋糕 (DEFINITION_TABLE.md, 第三版 2026-10-02).

Arithmetic on Facts only (facts.py): no Isaac, USD, controller, parameter overrides or model-provided verdicts. One
Facts per 20 Hz control frame, in order. The single source of every rule and constant is the definition table: change
the table first, then this file.

Order inside one frame (失败优先于成功，成功优先于超时；预算边界同一步达成时先转阶段):
  1. the interval since the previous frame is charged to the operation stage that was active during it;
  2. failure conditions: a failing frame ends the episode and neither charges a deduction nor awards a stage;
  3. once-per-episode deductions;
  4. stage conditions: points (historical, once, never revoked), regression, the 1 s stability window, success;
  5. the budget of the operation stage that is still unachieved after 4.
The total time limit is enforced by the public framework; finalize() performs the settlement for it.

Stages (stage = the one currently being worked on):
  2.1 / 2.2  ① opening >= 100 mm +2  ② target whole inside the basket, on the basket floor, released +2
             ③ remaining opening <= 5 mm, ② still true, the other cake not inside the basket +1; ③ held 1 s = success
  2.3        ① opening >= 100 mm +1  ② basket clear of the body: no body contact and opening > 217.5 mm +1
             ③ basket on the counter (contact normal up), released, tilt <= 5 deg, its cake inside on the floor +1; 1 s = success
  2.5        ① both finger pads of one gripper hold the cake and it is off the basket floor +2
             ② whole cake outside the basket outline, still grasped, nothing else supports it +2; ② held 1 s = success

Regression never pays points again and never resets a used budget:
  2.1 / 2.2  ② lost -> ②; ② lost while the basket is closed again -> ① (需开篮才能重放时先回阶段①)
  2.3        basket touching the body again / stroke below the clearance -> ②; ③ lost -> ③
  2.5        cake supported by anything -> ① (needs a new grasp); held but not outside -> ②

The 1 s window: the condition holds in every frame, the origin of each checked object stays within 5 mm of every other
sample of the window (max pairwise distance) and its orientation within 5 deg (max pairwise rotation angle).
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
from .facts import Facts, CakeFacts, TrayFacts, SURFACES, BASKET_FLOOR, BASKET_WALL, PLATE, GROUND

RULE_VERSION = 'arena.air_fryer.process.v2'       # v2 2026-10-03: data subtasks 2.1 / 2.2 / 2.3 / 2.5 of DEFINITION_TABLE.md 第三版
CONTROL_DT_S = 0.05
CLOCK_TOLERANCE_S = 1e-6
EPS = 1e-9


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    instruction: str                    # DEFINITION_TABLE.md row 发给 VLA 的指令, verbatim
    time_limit_s: float                 # row 时限与终止; enforced by the public framework
    stage_budget_s: MappingProxyType    # operation stage -> budget (row 阶段时限)
    stage_points: MappingProxyType      # stage -> points (row 阶段与评测点)
    maximum_score: int
    deductions: tuple                   # once-per-episode deductions of this subtask (row 评分规则)
    thresholds: MappingProxyType

    def as_dict(self):
        return dict(key=self.key, title=self.title, instruction=self.instruction, time_limit_s=self.time_limit_s,
                    maximum_score=self.maximum_score, stage_budget_s={str(k): v for k, v in self.stage_budget_s.items()},
                    stage_points={str(k): v for k, v in self.stage_points.items()}, deductions=list(self.deductions),
                    thresholds=dict(self.thresholds))


def _spec(key, title, instruction, time_limit_s, budgets, points, deductions, **thresholds):
    common = dict(hold_s=1.0, hold_span_m=0.005, hold_rotation_deg=5.0, open_m=0.100, closed_m=0.010)   # closed 10 mm (user 2026-10-04; was 5 mm)
    return Spec(key, title, instruction, float(time_limit_s), MappingProxyType(dict(budgets)), MappingProxyType(dict(points)),
                sum(points.values()), tuple(deductions), MappingProxyType(dict(common, **thresholds)))


INSERT_DEDUCTIONS = ('target_set_down_elsewhere', 'other_touched_ground')
# budgets = producer's longest stage time x 2 (ceil), time limit = producer's longest episode x 2 (user 2026-10-06)
SPECS = {
    '2.1': _spec('2.1', '黄色蛋糕放入炸篮', '拉开空气炸锅的炸篮，把盘子里的黄色蛋糕放进炸篮，然后把炸篮推回关好。', 216,   # shared runtime id with 2.2: the longer of the two doubled limits
                 {1: 18., 2: 108., 3: 31.}, {1: 2, 2: 2, 3: 1}, INSERT_DEDUCTIONS),
    '2.2': _spec('2.2', '白色蛋糕放入炸篮', '拉开空气炸锅的炸篮，把盘子里的白色蛋糕放进炸篮，然后把炸篮推回关好。', 216,
                 {1: 20., 2: 127., 3: 27.}, {1: 2, 2: 2, 3: 1}, INSERT_DEDUCTIONS),
    '2.3': _spec('2.3', '拉出炸篮', '把空气炸锅的炸篮整个取出来，放到台面上并松手放稳，保持蛋糕留在篮内。', 48,
                 {1: 14., 2: 6., 3: 19.}, {1: 1, 2: 1, 3: 1}, ('target_out_of_basket', 'other_knocked_off_plate'),
                 clear_m=0.2175, tray_tilt_deg=5.0),
    '2.5': _spec('2.5', '把面包从炸篮里拿出来', '把炸篮里的蛋糕拿出来，端在手里并保持稳定。', 96,
                 {1: 37., 2: 59.}, {1: 2, 2: 2}, ('target_set_down_outside', 'tray_displaced', 'other_knocked_off_plate'),
                 tray_displacement_m=0.02),
}


def _number(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Non-finite or non-numeric fact: ' + name)
    if low is not None and value < low:
        raise ValueError('Fact out of range: ' + name)


def _flag(name, value):
    if type(value) is not bool:
        raise ValueError('Boolean fact required: ' + name)


def _pose(name, o):
    if not isinstance(o.position, (tuple, list)) or len(o.position) != 3:
        raise ValueError('Position (x, y, z) required: ' + name)
    if not isinstance(o.quaternion, (tuple, list)) or len(o.quaternion) != 4:
        raise ValueError('Quaternion (w, x, y, z) required: ' + name)
    for i, v in enumerate(o.position):
        _number('%s.position[%d]' % (name, i), v)
    for i, v in enumerate(o.quaternion):
        _number('%s.quaternion[%d]' % (name, i), v)
    if abs(math.sqrt(sum(v * v for v in o.quaternion)) - 1) > 1e-4:
        raise ValueError('Unit quaternion required: ' + name)
    _number(name + '.tilt_deg', o.tilt_deg, 0.)
    if o.tilt_deg > 180.:
        raise ValueError('Fact out of range: ' + name + '.tilt_deg')
    _number(name + '.displacement_m', o.displacement_m, 0.)


def _cake(name, o):
    if not isinstance(o, CakeFacts):
        raise ValueError('Cake facts required: ' + name)
    _pose(name, o)
    for key in ('pads_bilateral', 'grasped', 'robot_contact', 'ground_contact', 'basket_contact', 'inside_basket', 'outside_basket'):
        _flag(name + '.' + key, getattr(o, key))
    if not isinstance(o.supports, (frozenset, set)) or not o.supports <= SURFACES:
        raise ValueError('Unknown support surface: ' + name)
    if o.pads_bilateral and not o.robot_contact:
        raise ValueError('Finger pads touch but the robot does not: ' + name)
    if o.grasped and not o.pads_bilateral:
        raise ValueError('Grasped without bilateral pad contact: ' + name)
    if GROUND in o.supports and not o.ground_contact:
        raise ValueError('Ground supports without ground contact: ' + name)
    if (BASKET_FLOOR in o.supports or BASKET_WALL in o.supports) and not o.basket_contact:
        raise ValueError('Basket supports without basket contact: ' + name)
    if o.inside_basket and o.outside_basket:
        raise ValueError('A cake cannot be inside and outside the basket at once: ' + name)


def _tray(name, t):
    if not isinstance(t, TrayFacts):
        raise ValueError('Tray facts required: ' + name)
    _pose(name, t)
    _number(name + '.opening_m', t.opening_m)
    for key in ('body_contact', 'counter_supported', 'robot_contact', 'pads_bilateral', 'ground_contact', 'joint_released'):
        _flag(name + '.' + key, getattr(t, key))
    if t.pads_bilateral and not t.robot_contact:
        raise ValueError('Finger pads touch the tray but the robot does not')


def rotation_angle_deg(q1, q2):
    """Angle of the relative rotation between two unit quaternions (w, x, y, z)."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1., dot)))


class Process:
    """Rule engine of one episode of one data subtask. observe() once per control frame, finalize() once at the end."""

    def __init__(self, key):
        if key not in SPECS:
            raise ValueError('Unknown air_fryer subtask: %r' % (key,))
        self.key, self.spec = key, SPECS[key]
        self._advance = {'2.1': self._insert, '2.2': self._insert, '2.3': self._pull, '2.5': self._take}[key]
        self._stage = 1
        self._awards = {}
        self._deductions = []
        self._once = set()
        self._used = {s: 0 for s in self.spec.stage_budget_s}
        self._charging = 1
        self._window = []               # (step, {object: (position, quaternion)}) of the current stability window
        self._window_restarts = 0
        self._regressions = 0
        self._events = []
        self._success, self._failure = False, None
        self._end = None
        self._finalized, self._termination = False, None
        self._facts = None
        self._last_step = self._last_time_s = None
        self._frames = self._post_terminal_frames = 0
        self._released_events = 0

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
        if not isinstance(f, Facts):
            raise ValueError('Facts of one control frame required')
        expected = 0 if self._last_step is None else self._last_step + 1
        if type(f.step) is not int or f.step != expected:
            raise ValueError('Skipped, repeated or out-of-order control frame: expected step %d, got %r' % (expected, f.step))
        _number('time_s', f.time_s)
        if abs(f.time_s - f.step * CONTROL_DT_S) > CLOCK_TOLERANCE_S:
            raise ValueError('time_s is not on the 20 Hz control clock')
        _cake('target', f.target)
        _cake('other', f.other)
        _tray('tray', f.tray)

    # ---- bookkeeping ----
    def _award(self, f, stage):
        if stage in self._awards:
            return
        points = self.spec.stage_points[stage]
        self._awards[stage] = dict(points=points, step=f.step, time_s=float(f.time_s))
        self._events.append(dict(kind='stage_points', stage=stage, points=points, step=f.step, time_s=float(f.time_s)))

    def _deduct(self, f, kind, amount=1):
        if kind in self._once:
            return
        self._once.add(kind)
        self._deductions.append(dict(kind=kind, amount=amount, step=f.step, time_s=float(f.time_s)))
        self._events.append(dict(kind='deduction', name=kind, points=-amount, step=f.step, time_s=float(f.time_s)))

    def _enter(self, f, stage):
        if stage == self._stage:
            return
        back = stage < self._stage
        self._regressions += back
        self._events.append(dict(kind='regression' if back else 'stage', from_stage=self._stage, to_stage=stage, step=f.step, time_s=float(f.time_s)))
        self._stage = stage

    def _charge(self, holding=False):
        """The next interval is paid by the current stage unless it has no budget or its condition is being held."""
        self._charging = None if holding or self._stage not in self.spec.stage_budget_s else self._stage

    def _hold(self, f, holding, poses):
        """1 s window: every frame meets the condition; every checked object stays within 5 mm / 5 deg of all its other samples."""
        t = self.spec.thresholds
        if not holding:
            self._window = []
            return False
        for _, previous in self._window:
            for name, (p, q) in poses.items():
                pp, pq = previous[name]
                if math.dist(pp, p) > t['hold_span_m'] or rotation_angle_deg(pq, q) > t['hold_rotation_deg']:
                    self._window = []
                    self._window_restarts += 1
                    self._events.append(dict(kind='window_restarted', object=name, step=f.step, time_s=float(f.time_s)))
                    break
            else:
                continue
            break
        self._window.append((f.step, {k: (tuple(p), tuple(q)) for k, (p, q) in poses.items()}))
        return self._window_s() >= t['hold_s'] - EPS

    def _window_s(self):
        return (self._window[-1][0] - self._window[0][0]) * CONTROL_DT_S if self._window else 0.

    def _finish(self, f, *, success=False, failure=None):
        self._success, self._failure = success, failure
        self._end = dict(step=f.step, time_s=float(f.time_s))
        self._events.append(dict(kind='success' if success else 'terminal_failure', reason=failure, stage=self._stage, step=f.step, time_s=float(f.time_s)))

    # ---- one control frame ----
    def observe(self, facts):
        if self._finalized:
            raise RuntimeError('Episode already finalized; a new episode needs a new Process')
        self._validate(facts)
        self._last_step, self._last_time_s = facts.step, float(facts.time_s)
        self._frames += 1
        if self._end is not None:
            self._post_terminal_frames += 1
            return
        self._facts = facts
        if facts.step > 0 and self._charging is not None:
            self._used[self._charging] += 1
        reason = self._failure_now(facts)
        if reason is not None:
            return self._finish(facts, failure=reason)
        self._charge_deductions(facts)
        verdict = self._advance(facts)
        if verdict == 'success':
            return self._finish(facts, success=True)
        if verdict is not None:
            return self._finish(facts, failure=verdict)
        stage = self._charging
        if stage is not None and self._used[stage] * CONTROL_DT_S >= self.spec.stage_budget_s[stage] - EPS:
            return self._finish(facts, failure='stage%d_budget_exhausted' % stage)

    # ---- shared predicates ----
    @staticmethod
    def _placed(cake):
        """DEFINITION_TABLE 2.1 / 2.2 ②: whole inside the basket, on the basket floor, robot released."""
        return cake.inside_basket and BASKET_FLOOR in cake.supports and cake.released

    def _failure_now(self, f):
        if f.target.ground_contact:
            return 'target_on_ground'
        if self.key in ('2.1', '2.2') and self._placed(f.other):
            return 'other_placed_in_basket'
        if self.key == '2.3' and f.tray.ground_contact:
            return 'tray_on_ground'
        return None

    def _charge_deductions(self, f):
        target, other, tray = f.target, f.other, f.tray
        off_plate = other.released and bool(other.supports - {PLATE})      # the other cake rests on something that is not its plate
        happened = dict(
            # 2.1 / 2.2: 黄色目标脱手后由篮底、源纸盘之外的表面承托 −1；干扰蛋糕接触地面 −1
            target_set_down_elsewhere=target.released and bool(target.supports - {BASKET_FLOOR, PLATE}),
            other_touched_ground=other.ground_contact,
            # 2.3: 原篮内蛋糕掉到篮外承托面 −1；盘上蛋糕被碰落盘外承托面 −1
            target_out_of_basket=target.released and bool(target.supports - {BASKET_FLOOR, BASKET_WALL}),   # 篮内滑动/贴壁不扣
            other_knocked_off_plate=off_plate,
            # 2.5: 目标脱手后落在篮外承托面 −1（落回篮内不扣）；炸篮偏离初态 >2 cm −1
            target_set_down_outside=target.released and bool(target.supports - {BASKET_FLOOR, BASKET_WALL}),
            tray_displaced=tray.displacement_m > self.spec.thresholds.get('tray_displacement_m', math.inf))
        for kind in self.spec.deductions:
            if happened[kind]:
                self._deduct(f, kind)

    # ---- 2.1 / 2.2 放入 ----
    def _insert(self, f):
        t, target, other, tray = self.spec.thresholds, f.target, f.other, f.tray
        opened = tray.opening_m >= t['open_m']
        placed = self._placed(target)
        closed = abs(tray.opening_m) <= t['closed_m']
        done = placed and closed and not other.inside_basket
        if opened:
            self._award(f, 1)
        if 1 in self._awards:
            if placed:
                self._award(f, 2)
            if done:
                self._award(f, 3)
            if done:
                self._enter(f, 3)
            elif placed:
                self._enter(f, 3)                                   # placed, closing still outstanding
            elif closed and 2 in self._awards and not opened:
                self._enter(f, 1)                                   # 需开篮才能重放时先回阶段①
            else:
                self._enter(f, 2)
        self._charge(holding=done)
        return 'success' if self._hold(f, done, dict(target=(target.position, target.quaternion), tray=(tray.position, tray.quaternion))) else None

    # ---- 2.3 拉出 ----
    def _pull(self, f):
        t, target, tray = self.spec.thresholds, f.target, f.tray
        opened = tray.opening_m >= t['open_m']
        clear = not tray.body_contact and tray.opening_m > t['clear_m']            # ② first clearance: geometry + no contact
        placed = (tray.counter_supported and not tray.robot_contact and tray.tilt_deg <= t['tray_tilt_deg']
                  and target.inside_basket and BASKET_FLOOR in target.supports)
        # 2026-10-03 (DEFINITION_TABLE 2.3 恢复与回退): back to ② only when the basket re-enters the body — body contact,
        # or the opening falls back inside the clearance while the basket is no longer on the counter. A basket resting
        # on the counter that slides a few mm does not regress.
        reentered = tray.body_contact or (tray.opening_m <= t['clear_m'] and not tray.counter_supported)
        out = not tray.body_contact and (clear or tray.counter_supported)
        if opened:
            self._award(f, 1)
        if 1 in self._awards and clear:
            self._award(f, 2)
        if 2 in self._awards and out and placed:
            self._award(f, 3)
        if 2 in self._awards:
            if reentered:
                self._enter(f, 2)
            else:
                self._enter(f, 3)
        elif 1 in self._awards:
            self._enter(f, 2)
        holding = 3 in self._awards and out and placed
        self._charge(holding=holding)
        return 'success' if self._hold(f, holding, dict(target=(target.position, target.quaternion), tray=(tray.position, tray.quaternion))) else None

    # ---- 2.5 取出 ----
    def _take(self, f):
        target = f.target
        lifted = target.grasped and BASKET_FLOOR not in target.supports
        out = target.grasped and target.outside_basket and not target.supports
        if lifted:
            self._award(f, 1)
        if 1 in self._awards and out:
            self._award(f, 2)
        if 1 in self._awards:
            if out:
                self._enter(f, 3)
            elif target.supports or not target.grasped:
                self._enter(f, 1)                                   # 落回篮内或台面可重抓，回阶段①
            else:
                self._enter(f, 2)                                   # 仍夹持但未完整移出则回阶段②
        holding = 2 in self._awards and out
        self._charge(holding=holding)
        return 'success' if self._hold(f, holding, dict(target=(target.position, target.quaternion))) else None

    # ---- end of episode ----
    def finalize(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Termination reason required')
        if self._finalized:
            if reason != self._termination:
                raise ValueError('Contradictory repeated finalization')
            return
        if self._facts is None:
            raise ValueError('Cannot settle an episode without evidence')
        if reason == 'success' and not self._success:
            raise ValueError('No measured success behind the requested result')
        self._termination, self._finalized = reason, True

    def summary(self):
        points, deducted, budgets = self._points(), self._deducted(), self.spec.stage_budget_s
        f = self._facts
        final = {} if f is None else dict(
            opening_m=f.tray.opening_m, target_inside_basket=f.target.inside_basket, target_outside_basket=f.target.outside_basket,
            target_supports=sorted(f.target.supports), target_grasped=f.target.grasped, target_released=f.target.released,
            other_inside_basket=f.other.inside_basket, other_supports=sorted(f.other.supports), tray_counter_supported=f.tray.counter_supported,
            tray_body_contact=f.tray.body_contact, tray_tilt_deg=f.tray.tilt_deg, tray_displacement_m=f.tray.displacement_m, joint_released=f.tray.joint_released)
        return dict(
            schema=RULE_VERSION, key=self.key, title=self.spec.title, instruction=self.spec.instruction,
            time_limit_s=self.spec.time_limit_s, stage=self._stage, maximum_score=self.spec.maximum_score,
            score=self.score, raw_score=points - deducted, points_awarded=points, deduction_total=deducted,
            stage_points={str(s): dict(points=p, awarded=s in self._awards, step=self._awards.get(s, {}).get('step'),
                                       time_s=self._awards.get(s, {}).get('time_s')) for s, p in self.spec.stage_points.items()},
            deductions=[dict(d) for d in self._deductions],
            budgets={str(s): dict(budget_s=budgets[s], used_s=round(self._used[s] * CONTROL_DT_S, 6),
                                  exhausted=self._failure == 'stage%d_budget_exhausted' % s) for s in budgets},
            hold=dict(required_s=self.spec.thresholds['hold_s'], span_limit_m=self.spec.thresholds['hold_span_m'],
                      rotation_limit_deg=self.spec.thresholds['hold_rotation_deg'], window_s=round(self._window_s(), 6), restarts=self._window_restarts),
            success=self._success, failure=self._failure, ended=self._end is not None,
            end_step=self._end and self._end['step'], end_time_s=self._end and self._end['time_s'],
            finalized=self._finalized, termination=self._termination,
            regressions=self._regressions, events=[dict(e) for e in self._events],
            frames=self._frames, last_step=self._last_step, last_time_s=self._last_time_s,
            post_terminal_frames=self._post_terminal_frames, final_facts=final)

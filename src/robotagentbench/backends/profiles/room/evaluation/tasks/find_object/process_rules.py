"""Process rules of the Task 4 family (DEFINITION_TABLE.md 第二版, approved 2026-10-05 for 4.1 … 4.17 except 4.6 / 4.12).

Arithmetic on Facts only (facts.py). One Facts per 20 Hz control frame, in order. The definition table is the single source of every
rule and constant: change the table first, then this file.

Order inside one frame (失败优先于成功，成功优先于超时；预算边界同一步达成时先转阶段):
  1. the interval since the previous frame is charged to the stage that was active during it (回退不清零已用预算，其他阶段暂停);
  2. failure: the target touches the ground -> the episode ends; no deduction and no stage points in that frame;
  3. once-per-episode deductions (per distractor instance: robot touched it, then it touched the ground);
  4. stage conditions: points (historical, once, never revoked), regression, the continuous hold window, success;
  5. the budget of the stage that is still unachieved after 4; the total time limit (阶段到时未达成或总时限到达 -> 超时未成功).

Pick (4.1 4.2 4.3 4.7 4.8 4.9 4.13 4.14 4.16) and drawer (4.4 4.10):
  ① grasped (+1)  ② grasped and no external support [drawer: and whole geometry above the drawer rim] (+2)
  ③ ② held continuously 3 s -> success (+0). Lost grasp -> ①; grasped but supported again (or below the rim) -> ②.
  Deductions: hold not completed at the end -1 (once, at settlement); distractor touched by the robot then on the ground -1 per
  instance (not for 4.9 / 4.14, which keep the confirmed 11-x rules).
Place (4.17):
  ① target whole inside the TV-table top outline, supported by the TV table, released (+2)
  ② ① held 1 s with span <= 5 mm and rotation <= 5 deg -> success (+2).
  Deductions: target released onto a surface that is neither the TV table nor the ground -1 once; robot touches the television -1 once.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
from .facts import Facts, TargetFacts, DistractorFacts, SURFACES, GROUND, ROBOT, DESTINATION

RULE_VERSION = 'arena.find_object.process.v1'
CONTROL_DT_S = 0.05
CLOCK_TOLERANCE_S = 1e-6
EPS = 1e-9


@dataclass(frozen=True)
class Spec:
    key: str
    kind: str
    title: str
    instruction: str
    time_limit_s: float
    stage_budget_s: MappingProxyType
    stage_points: MappingProxyType
    maximum_score: int
    deductions: tuple
    thresholds: MappingProxyType

    def as_dict(self):
        return dict(key=self.key, kind=self.kind, title=self.title, instruction=self.instruction, time_limit_s=self.time_limit_s, maximum_score=self.maximum_score,
                    stage_budget_s={str(k): v for k, v in self.stage_budget_s.items()}, stage_points={str(k): v for k, v in self.stage_points.items()},
                    deductions=list(self.deductions), thresholds=dict(self.thresholds))


def _pick(key, title, instruction, limit, b1, b2=20, distractor=True, kind='pick'):
    ded = ('hold_incomplete',) + (('distractor_grounded',) if distractor else ())
    return Spec(key, kind, title, instruction, float(limit), MappingProxyType({1: float(b1), 2: float(b2), 3: 10.}), MappingProxyType({1: 1, 2: 2, 3: 0}), 3, ded,
                MappingProxyType(dict(hold_s=3.0)))


SPECS = {
    '4.1': _pick('4.1', '在沙发座面找到并抓取墨镜盒（含靠枕下）', '从沙发座面抓取墨镜盒，抬离承托面并稳定持有 3 秒。', 90, 60),
    '4.2': _pick('4.2', '在茶几表面找到并抓取墨镜盒', '从茶几桌面抓取墨镜盒，抬离承托面并稳定持有 3 秒。', 60, 40, 10),
    '4.3': _pick('4.3', '在边柜找到并抓取墨镜盒', '从小茶几桌面抓取墨镜盒，抬离承托面并稳定持有 3 秒。', 60, 40, 10),
    '4.4': _pick('4.4', '在茶几抽屉里找到并抓取墨镜盒', '从已打开的茶几抽屉抓取墨镜盒，抬出抽屉并稳定持有 3 秒。', 90, 60, kind='drawer'),
    '4.7': _pick('4.7', '在沙发座面找到并抓取钱包（含靠枕下）', '从沙发座面抓取钱包，抬离承托面并稳定持有 3 秒。', 120, 90),
    '4.8': _pick('4.8', '在茶几表面找到并抓取钱包', '从茶几桌面抓取钱包，抬离承托面并稳定持有 3 秒。', 60, 40, 10),
    '4.9': _pick('4.9', '在小茶几找到并抓取钱包', '从小茶几桌面抓取钱包，抬离承托面并稳定持有 3 秒。', 180, 150, distractor=False),
    '4.10': _pick('4.10', '在茶几抽屉找到并抓取钱包', '从已打开的茶几抽屉抓取钱包，抬出抽屉并稳定持有 3 秒。', 120, 90, kind='drawer'),
    '4.13': _pick('4.13', '在沙发座面找到并抓取电视遥控器（含靠枕下）', '从沙发座面抓取电视遥控器，抬离承托面并稳定持有 3 秒。', 120, 90),
    '4.14': _pick('4.14', '在小茶几表面找到并抓取电视遥控器', '从小茶几桌面抓取电视遥控器，抬离承托面并稳定持有 3 秒。', 180, 150, distractor=False),
    '4.16': _pick('4.16', '在零食架上找到并抓取电视遥控器', '从零食架层板抓取电视遥控器，抬离层板并稳定持有 3 秒。', 60, 40, 10),
    '4.17': Spec('4.17', 'place', '携带电视遥控器并放到电视柜上', '把手里的电视遥控器放到电视柜台面上，松手并放稳。', 90.,
                 MappingProxyType({1: 80., 2: 10.}), MappingProxyType({1: 2, 2: 2}), 4, ('target_set_down_elsewhere', 'tv_touched'),
                 MappingProxyType(dict(hold_s=1.0, hold_span_m=0.005, hold_rotation_deg=5.0))),
}


def _number(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Non-finite or non-numeric fact: ' + name)
    if low is not None and value < low:
        raise ValueError('Fact out of range: ' + name)


def _flag(name, value):
    if type(value) is not bool:
        raise ValueError('Boolean fact required: ' + name)


def _target(name, o):
    if not isinstance(o, TargetFacts):
        raise ValueError('Target facts required: ' + name)
    if not isinstance(o.position, (tuple, list)) or len(o.position) != 3 or not isinstance(o.quaternion, (tuple, list)) or len(o.quaternion) != 4:
        raise ValueError('Pose required: ' + name)
    for i, v in enumerate(o.position): _number('%s.position[%d]' % (name, i), v)
    for i, v in enumerate(o.quaternion): _number('%s.quaternion[%d]' % (name, i), v)
    if abs(math.sqrt(sum(v * v for v in o.quaternion)) - 1) > 1e-4:
        raise ValueError('Unit quaternion required: ' + name)
    for key in ('pads_bilateral', 'grasped', 'robot_contact', 'ground_contact', 'above_rim', 'in_destination'):
        _flag(name + '.' + key, getattr(o, key))
    _number(name + '.hull_min_z', o.hull_min_z)
    if not isinstance(o.supports, (frozenset, set)) or not o.supports <= SURFACES:
        raise ValueError('Unknown support surface: ' + name)
    if o.pads_bilateral and not o.robot_contact:
        raise ValueError('Finger pads touch but the robot does not: ' + name)
    if o.grasped and not o.pads_bilateral:
        raise ValueError('Grasped without bilateral pad contact: ' + name)
    if GROUND in o.supports and not o.ground_contact:
        raise ValueError('Ground supports without ground contact: ' + name)
    if ROBOT in o.supports and not o.robot_contact:
        raise ValueError('Robot supports without robot contact: ' + name)


def rotation_angle_deg(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1., dot)))


class Process:
    """Rule engine of one episode of one data subtask. observe() once per control frame, finalize() once at the end."""

    def __init__(self, key, floor_clutter=()):
        if key not in SPECS:
            raise ValueError('Unknown find_object subtask: %r' % (key,))
        self.key, self.spec = key, SPECS[key]
        # distractor names that the source episode already places on the floor at frame 0 (producer scene clutter): the approved
        # deduction is for clutter the robot knocks to the ground, which cannot happen to an object lying there from the start
        self.floor_clutter = frozenset(floor_clutter)
        self._advance = self._place if self.spec.kind == 'place' else self._pick
        self._stage = 1
        self._awards = {}
        self._deductions = []
        self._once = set()
        self._used = {s: 0 for s in self.spec.stage_budget_s}
        self._charging = 1
        self._window = []               # (step, poses) of the current continuous-hold window
        self._window_restarts = 0
        self._regressions = 0
        self._events = []
        self._success, self._failure = False, None
        self._end = None
        self._finalized, self._termination = False, None
        self._facts = None
        self._last_step = self._last_time_s = None
        self._frames = self._post_terminal_frames = 0
        self._hold_completed = False

    @property
    def stage(self): return self._stage

    @property
    def score(self): return max(0., float(self._points() - self._deducted()))

    @property
    def success(self): return self._success

    @property
    def failure(self): return self._failure

    def _points(self): return sum(a['points'] for a in self._awards.values())

    def _deducted(self): return sum(d['amount'] for d in self._deductions)

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
        _target('target', f.target)
        if not isinstance(f.distractors, dict):
            raise ValueError('Distractor facts required')
        for name, d in f.distractors.items():
            if not isinstance(d, DistractorFacts):
                raise ValueError('Distractor facts required: ' + str(name))
            _flag(name + '.robot_touched', d.robot_touched); _flag(name + '.ground_contact', d.ground_contact)
        _flag('tv_contact', f.tv_contact)
        _number('drawer_rim_z', f.drawer_rim_z)

    # ---- bookkeeping ----
    def _award(self, f, stage):
        if stage in self._awards:
            return
        points = self.spec.stage_points[stage]
        self._awards[stage] = dict(points=points, step=f.step, time_s=float(f.time_s))
        self._events.append(dict(kind='stage_points', stage=stage, points=points, step=f.step, time_s=float(f.time_s)))

    def _deduct(self, f, kind, amount=1, instance=None):
        key = kind if instance is None else f'{kind}:{instance}'
        if key in self._once:
            return
        self._once.add(key)
        self._deductions.append(dict(kind=kind, instance=instance, amount=amount, step=f.step, time_s=float(f.time_s)))
        self._events.append(dict(kind='deduction', name=key, points=-amount, step=f.step, time_s=float(f.time_s)))

    def _enter(self, f, stage):
        if stage == self._stage:
            return
        back = stage < self._stage
        self._regressions += back
        self._events.append(dict(kind='regression' if back else 'stage', from_stage=self._stage, to_stage=stage, step=f.step, time_s=float(f.time_s)))
        self._stage = stage

    def _hold(self, f, holding, poses, span_m=None, rotation_deg=None):
        """Continuous window: the condition holds in every frame; with span/rotation limits every checked object also stays within them."""
        if not holding:
            if self._window:
                self._window_restarts += 1
                self._events.append(dict(kind='window_restarted', step=f.step, time_s=float(f.time_s), reason='condition_lost'))
            self._window = []
            return False
        if span_m is not None:
            for _, previous in self._window:
                broken = False
                for name, (p, q) in poses.items():
                    pp, pq = previous[name]
                    if math.dist(pp, p) > span_m or rotation_angle_deg(pq, q) > rotation_deg:
                        broken = True
                        break
                if broken:
                    self._window = []
                    self._window_restarts += 1
                    self._events.append(dict(kind='window_restarted', step=f.step, time_s=float(f.time_s), reason='pose_drift'))
                    break
        self._window.append((f.step, {k: (tuple(p), tuple(q)) for k, (p, q) in poses.items()}))
        return self._window_s() >= self.spec.thresholds['hold_s'] - EPS

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
        if facts.target.ground_contact:
            return self._finish(facts, failure='target_on_ground')
        self._charge_deductions(facts)
        verdict = self._advance(facts)
        if verdict == 'success':
            self._hold_completed = True
            return self._finish(facts, success=True)
        stage = self._charging
        if stage is not None and self._used[stage] * CONTROL_DT_S >= self.spec.stage_budget_s[stage] - EPS:
            return self._finish(facts, failure='stage%d_budget_exhausted' % stage)
        if facts.time_s >= self.spec.time_limit_s - EPS:
            return self._finish(facts, failure='time_limit_exhausted')

    def _charge_deductions(self, f):
        if 'distractor_grounded' in self.spec.deductions:
            for name, d in f.distractors.items():
                if d.robot_touched and d.ground_contact and name not in self.floor_clutter:
                    self._deduct(f, 'distractor_grounded', 1, name)
        if 'target_set_down_elsewhere' in self.spec.deductions and f.target.released and bool(f.target.supports - {DESTINATION, GROUND}):
            self._deduct(f, 'target_set_down_elsewhere')
        if 'tv_touched' in self.spec.deductions and f.tv_contact:
            self._deduct(f, 'tv_touched')

    # ---- pick / drawer ----
    def _pick(self, f):
        t = f.target
        grasped = t.grasped
        free = grasped and not t.supports and (self.spec.kind != 'drawer' or t.above_rim)
        if grasped:
            self._award(f, 1)
        if 1 in self._awards and free:
            self._award(f, 2)
        if 1 in self._awards:
            if not grasped:
                self._enter(f, 1)
            elif free:
                self._enter(f, 3)
            else:
                self._enter(f, 2)
        self._charging = None if free else (self._stage if self._stage in self.spec.stage_budget_s else None)
        if free:
            self._charging = 3        # ③ accumulates while the hold window runs (阶段累计预算 10 s)
        return 'success' if self._hold(f, free, dict(target=(t.position, t.quaternion))) else None

    # ---- place (4.17) ----
    def _place(self, f):
        t = f.target
        placed = t.in_destination and DESTINATION in t.supports and t.released
        if placed:
            self._award(f, 1)
        if 1 in self._awards:
            self._enter(f, 2 if placed else 1)
        self._charging = 2 if placed else 1
        held = self._hold(f, placed, dict(target=(t.position, t.quaternion)), self.spec.thresholds['hold_span_m'], self.spec.thresholds['hold_rotation_deg'])
        if held:
            self._award(f, 2)
            return 'success'
        return None

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
        if 'hold_incomplete' in self.spec.deductions and not self._hold_completed:
            self._deduct(self._facts, 'hold_incomplete')
        self._termination, self._finalized = reason, True

    def summary(self):
        points, deducted, budgets = self._points(), self._deducted(), self.spec.stage_budget_s
        f = self._facts
        final = {} if f is None else dict(target_supports=sorted(f.target.supports), target_grasped=f.target.grasped, target_released=f.target.released,
                                         target_above_rim=f.target.above_rim, target_in_destination=f.target.in_destination, target_ground_contact=f.target.ground_contact,
                                         distractors_grounded=sorted(n for n, d in f.distractors.items() if d.ground_contact))
        return dict(
            schema=RULE_VERSION, key=self.key, kind=self.spec.kind, title=self.spec.title, instruction=self.spec.instruction, time_limit_s=self.spec.time_limit_s,
            floor_clutter=sorted(self.floor_clutter), stage=self._stage, maximum_score=self.spec.maximum_score, score=self.score, raw_score=points - deducted, points_awarded=points, deduction_total=deducted,
            stage_points={str(s): dict(points=p, awarded=s in self._awards, step=self._awards.get(s, {}).get('step'), time_s=self._awards.get(s, {}).get('time_s'))
                          for s, p in self.spec.stage_points.items()},
            deductions=[dict(d) for d in self._deductions],
            budgets={str(s): dict(budget_s=budgets[s], used_s=round(self._used[s] * CONTROL_DT_S, 6), exhausted=self._failure == 'stage%d_budget_exhausted' % s) for s in budgets},
            hold=dict(required_s=self.spec.thresholds['hold_s'], window_s=round(self._window_s(), 6), restarts=self._window_restarts, completed=self._hold_completed),
            success=self._success, failure=self._failure, ended=self._end is not None, end_step=self._end and self._end['step'], end_time_s=self._end and self._end['time_s'],
            finalized=self._finalized, termination=self._termination, regressions=self._regressions, events=[dict(e) for e in self._events],
            frames=self._frames, last_step=self._last_step, last_time_s=self._last_time_s, post_terminal_frames=self._post_terminal_frames, final_facts=final)

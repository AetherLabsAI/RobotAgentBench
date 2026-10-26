"""Scoring arithmetic for the tidy_sofa family on trusted, contiguous physical evidence.

No USD, controller, parameter overrides or model-provided verdicts. One PhysicalFrame per physics
substep; a batch is one control observation. Rule sets per kind (apis/<api>/README.md):
  pick   (v3 2026-09-15, 15-1/13-6 user-approved values): grasp +3, lift 5 cm & hold 1 s +2; wrong-object
         lift = failure; target leaving the gripper after the grasp = failure ("脱离夹爪即掉落").
  drawer (16-1 v3 2026-09-15): open >= 10 cm +2; released & "inside" (centre in the interior x/y, below the rim
         by 1 cm, supported by the drawer) +3; closed <= 3 cm with the target inside +1; success after 1 s closed &
         inside. After release the target is in one of three states: placed (inside), recoverable (on the drawer
         but not inside: rim, half in — no points, no failure, may be re-grasped) or dropped (after the settling
         tolerance, held up by the table/floor/anything not the drawer while touching no part of the drawer).
         The settling window itself never triggers a failure; a target in free flight is not judged.
  place / basket (draft v1): released & supported & inside zone +2, 1 s settled +2; released target not
         supported by the destination = failure.
Deductions (all kinds): robot contact with listed furniture −1 at contact start and −1 per further full
second of continuous contact; each scenery object pushed onto the floor −1 once. Score = max(0, raw).
"""
from dataclasses import dataclass, field
import math

RULE_VERSION = 'arena.livingroom_cleanup.process.v1'
COLLISION_INTERVAL_S = 1.0
DROP_DEBOUNCE_S = 0.25      # gripper contact lost for this long after a grasp = dropped
RELEASE_SETTLE_S = 0.5      # place/drawer settling tolerance after release: free fall / contact chatter inside it is not judged


@dataclass(frozen=True)
class PhysicalFrame:
    episode_id: str
    sample_index: int
    time_s: float
    target_grasped: bool            # confirmed bilateral grasp by any hand (relative motion window)
    target_pad_contact: bool        # any finger pad touching the target
    target_supported: bool          # resting on the task support/destination surface (pick: sofa seat; place: destination; drawer: drawer floor)
    target_in_zone: bool            # place: projection inside the destination zone; drawer/basket: inside the container interior
    lift_m: float                   # target height above its accepted initial height
    target_on_floor: bool
    wrong_grasped: frozenset = frozenset()     # scenery objects grasped and lifted (pick failure)
    furniture_contacts: frozenset = frozenset()  # furniture roots touched by the robot (deductions)
    scenery_on_floor: frozenset = frozenset()
    drawer_travel_m: float = 0.0    # drawer tasks: opening from the flush pose
    complete: bool = True
    target_support_contact: bool = False   # touching any part of the support/destination body (drawer rim/wall counts)
    target_foreign_support: bool = False   # held up by a body that is neither the support/destination nor the robot (table, floor, other props)


class BaseScorer:
    def __init__(self, episode_id, physics_dt_s, spec):
        if not isinstance(episode_id, str) or not episode_id.strip(): raise ValueError('Missing episode identity')
        if isinstance(physics_dt_s, bool) or not math.isfinite(physics_dt_s) or physics_dt_s <= 0: raise ValueError('Invalid physical sampling interval')
        self.episode_id = episode_id; self.dt = physics_dt_s; self.spec = spec
        self.last = None; self.awarded = {}; self.failure = None; self.terminal = False; self.finalized = False; self.reason = None
        self.events = []; self.contact_started = {}; self.contact_ticks = {}; self.collision_deduction = 0
        self.floor_deduction = 0; self.floored = set(); self.hold_start = None; self.hold_elapsed = 0.0

    # ---- validation ----
    def _validate_batch(self, frames):
        if not isinstance(frames, tuple) or not frames: raise ValueError('Nonempty physical evidence batch required')
        previous = self.last
        for f in frames:
            if not isinstance(f, PhysicalFrame) or f.episode_id != self.episode_id: raise ValueError('Wrong episode or physical frame')
            expected = 0 if previous is None else previous.sample_index + 1
            if type(f.sample_index) is not int or f.sample_index != expected: raise ValueError('Missing, duplicate or out-of-order physical sample')
            for v in (f.time_s, f.lift_m, f.drawer_travel_m):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): raise ValueError('Nonfinite physical measurement')
            if f.time_s < 0 or (previous is not None and abs(f.time_s - previous.time_s - self.dt) > 1e-6): raise ValueError('Physical evidence gap or changed sampling interval')
            for name in ('target_grasped', 'target_pad_contact', 'target_supported', 'target_in_zone', 'target_on_floor', 'complete'):
                if type(getattr(f, name)) is not bool: raise ValueError('Missing physical boolean: ' + name)
            if not f.complete: raise ValueError('Incomplete evidence is not a zero-score episode')
            for values in (f.wrong_grasped, f.furniture_contacts, f.scenery_on_floor):
                if not isinstance(values, frozenset) or any(not isinstance(v, str) or not v.startswith('/') for v in values): raise ValueError('Expected resolved prim identities')
            previous = f

    def _event(self, f, kind, points, **details):
        self.events.append(dict(kind=kind, sample_index=f.sample_index, time_s=f.time_s, points=points, **details))

    def _award(self, f, stage):
        if stage not in self.awarded:
            self.awarded[stage] = f.time_s; self._event(f, stage, self.spec.points[stage])

    def _deductions(self, f):
        for obj in set(self.contact_started) - f.furniture_contacts:
            del self.contact_started[obj]; del self.contact_ticks[obj]
        for obj in sorted(f.furniture_contacts):
            if obj not in self.contact_started: self.contact_started[obj] = f.time_s; self.contact_ticks[obj] = 0
            ticks = 1 + math.floor((f.time_s - self.contact_started[obj] + 1e-9) / COLLISION_INTERVAL_S)
            while self.contact_ticks[obj] < ticks:
                tick = self.contact_ticks[obj]; self.contact_ticks[obj] += 1; self.collision_deduction += 1
                self._event(f, 'furniture_contact', -1, object_path=obj, charged_at_s=self.contact_started[obj] + tick * COLLISION_INTERVAL_S)
        for obj in sorted(f.scenery_on_floor - self.floored):
            self.floored.add(obj); self.floor_deduction += 1; self._event(f, 'scenery_on_floor', -1, object_path=obj)

    def _fail(self, f, reason):
        self.failure = reason; self.hold_start = None; self.hold_elapsed = 0.0; self._event(f, 'terminal_failure', 0, reason=reason)

    def _hold(self, f, condition):
        """Continuous-hold timer; returns True once the hold requirement is met."""
        if condition:
            if self.hold_start is None: self.hold_start = f.time_s
            self.hold_elapsed = f.time_s - self.hold_start
        else:
            self.hold_start = None; self.hold_elapsed = 0.0
        return self.hold_start is not None and self.hold_elapsed + 1e-9 >= self.spec.hold_s

    def observe_batch(self, frames):
        if self.finalized or self.terminal: raise RuntimeError('Terminal episode; reset requires a new scorer')
        self._validate_batch(frames)
        success = False
        for f in frames:
            self.last = f
            if self.failure is not None: continue
            self._deductions(f)
            success = self._observe(f) or success
            if self.failure is not None: continue
        self.terminal = self.failure is not None or success
        if success: self.success = True
        self.reason = self.failure or ('success' if getattr(self, 'success', False) else None)
        return self.summary()

    def _observe(self, f):
        raise NotImplementedError

    def summary(self):
        earned = sum(self.spec.points[s] for s in self.awarded)
        raw = earned - self.collision_deduction - self.floor_deduction
        return dict(schema=RULE_VERSION, episode_id=self.episode_id, task_id=self.spec.api_id, kind=self.spec.kind,
                    score=max(0, raw), raw_score=raw, maximum_score=self.spec.maximum_score, earned_points=earned,
                    stages={s: dict(points=self.spec.points[s], awarded_at_s=self.awarded.get(s)) for s in self.spec.points},
                    collision_deduction=self.collision_deduction, scenery_floor_deduction=self.floor_deduction,
                    hold_elapsed_s=self.hold_elapsed, failure=self.failure,
                    success=bool(getattr(self, 'success', False)) and self.failure is None, reason=self.reason, finalized=self.finalized)

    def finalize(self, reason):
        if self.last is None: raise ValueError('Cannot settle absent evidence')
        if self.finalized:
            if reason != self.reason: raise ValueError('Contradictory repeated finalization')
            return self.summary()
        if self.reason is not None and reason != self.reason: raise ValueError('Finalization disagrees with measured terminal event')
        if self.reason is None and reason not in ('time_limit', 'client_closed'): raise ValueError('No evidence for requested terminal result')
        self.reason = reason; self.finalized = True
        return self.summary()


class PickScorer(BaseScorer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.contact_lost_since = None

    def _observe(self, f):
        if f.target_grasped: self._award(f, 'grasp')
        if f.wrong_grasped:
            self._fail(f, 'wrong_object_grasped'); return False
        if f.target_on_floor:
            self._fail(f, 'target_dropped'); return False
        if 'grasp' in self.awarded:
            if f.target_pad_contact: self.contact_lost_since = None
            else:
                if self.contact_lost_since is None: self.contact_lost_since = f.time_s
                if f.time_s - self.contact_lost_since + 1e-9 >= DROP_DEBOUNCE_S:
                    self._fail(f, 'target_dropped'); return False
        carrying = 'grasp' in self.awarded and f.target_grasped and not f.target_supported and f.lift_m >= self.spec.lift_m
        if self._hold(f, carrying):
            self._award(f, 'lift'); return True
        return False


class DrawerScorer(BaseScorer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.released_since = None; self.max_travel = 0.0

    def _observe(self, f):
        from .specs import DRAWER_OPEN_M, DRAWER_CLOSED_M
        self.max_travel = max(self.max_travel, f.drawer_travel_m)
        if f.target_on_floor:
            self._fail(f, 'target_dropped'); return False
        if f.drawer_travel_m >= DRAWER_OPEN_M: self._award(f, 'open')
        inside = f.target_in_zone and f.target_supported
        if f.target_pad_contact: self.released_since = None
        else:
            if self.released_since is None: self.released_since = f.time_s
            settled = f.time_s - self.released_since + 1e-9 >= RELEASE_SETTLE_S
            # dropped = after settling, held up by a non-drawer body while touching no part of the drawer; on the drawer but
            # not inside is the recoverable state (no failure); free flight is not judged
            if settled and f.target_foreign_support and not f.target_support_contact:
                self._fail(f, 'target_dropped'); return False
        if 'open' in self.awarded and not f.target_pad_contact and inside: self._award(f, 'place')
        if 'place' in self.awarded and f.target_in_zone and f.drawer_travel_m <= DRAWER_CLOSED_M: self._award(f, 'close')
        done = 'close' in self.awarded and f.target_in_zone and f.drawer_travel_m <= DRAWER_CLOSED_M
        return self._hold(f, done)

    def summary(self):
        return dict(super().summary(), max_drawer_travel_m=self.max_travel)


class PlaceScorer(BaseScorer):
    """Surface place and basket place: released target resting on/in the destination inside the zone."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.released_since = None

    def _observe(self, f):
        if f.target_on_floor:
            self._fail(f, 'target_dropped'); return False
        placed = not f.target_pad_contact and f.target_supported and f.target_in_zone
        if f.target_pad_contact: self.released_since = None
        else:
            if self.released_since is None: self.released_since = f.time_s
            if not f.target_supported and f.time_s - self.released_since + 1e-9 >= RELEASE_SETTLE_S:
                self._fail(f, 'target_dropped'); return False
        if placed: self._award(f, 'place')
        if self._hold(f, placed):
            self._award(f, 'settle'); return True
        return False


def scorer_for(spec, episode_id, physics_dt_s):
    cls = {'pick': PickScorer, 'drawer': DrawerScorer, 'place': PlaceScorer, 'basket': PlaceScorer}.get(spec.kind)
    if cls is None: raise NotImplementedError('No scorer implemented yet for kind ' + spec.kind)
    return cls(episode_id, physics_dt_s, spec)

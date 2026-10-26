"""Approved scene-bound 7-1 + basket placement adapters; source APIs unchanged."""
from dataclasses import replace
from evaluation.tasks.tabletop_pick import process_rules as pick
from evaluation.tasks.basket_place import process_rules as place


class RecoverablePick(pick.ProcessScorer):
    def __init__(self, episode_id, dt, target):
        super().__init__(episode_id, dt, '7-1')
        self.target = target
        self.ever_complete = False

    def pause(self):
        self.hold_start = None
        self.hold_elapsed = 0.
        self.carrying = False

    def observe_one(self, f, active):
        # Composite validates dense native indices/dt, including inactive 60 Hz.
        self.last = f
        if self.origin is None: self.origin = f.time_s
        if not active or f.target_on_floor:
            self.pause()
            return False
        if f.target_held and not self.grasp_awarded:
            self.grasp_awarded = True
            self._event(f, 'correct_grasp', pick.GRASP_POINTS)
        self.carrying = f.target_held and f.target_separated_from_table
        if not self.carrying:
            self.pause()
            return False
        if not self.lift_awarded:
            self.lift_awarded = True
            self._event(f, 'lifted', pick.LIFT_POINTS)
        if self.hold_start is None: self.hold_start = f.time_s
        self.hold_elapsed = f.time_s - self.hold_start
        ready = self.hold_elapsed + 1e-9 >= pick.HOLD_S
        if ready and not self.ever_complete:
            self.ever_complete = True
            self._event(f, 'stable_hold', 0)
        return ready

    def summary(self):
        return dict(super().summary(), success=self.ever_complete, failure=None,
                    time_limit_s=None, timeout_owner='room',
                    rule_adapter='openend_glasses_recoverable_7_1')


class BoundBasket(place.BasketPlacementScorer):
    def __init__(self, target, destination):
        super().__init__(public_clock=True, target=target)
        self.destination = destination
        self.continuity = False

    def _validate(self, f):
        if not isinstance(f, place.PlacementFacts) or f.destination_path != self.destination:
            raise ValueError('Wrong scene bag identity')
        # Source validation only; no global path mutation or scene writes.
        super()._validate(replace(f, destination_path=place.DESTINATION))

    def pause(self):
        self._since = None
        self.continuity = False

    def observe_one(self, f, *, final_sample):
        previous_stable = self._stable
        result = super().observe(f)
        if result.success and not final_sample:
            # Only the final measured substep of a control batch can settle.
            self._stable = previous_stable
            result = replace(result, stable_awarded=previous_stable,
                score=place.PLACE_POINTS * self._place + place.STABLE_POINTS * previous_stable,
                success=False, reason=None,
                new_events=tuple(e for e in result.new_events if e != 'stable_in_basket'))
            self._result = result
        self.continuity = f.placed
        return result

    def summary(self):
        return dict(self._result.to_dict() if self._result else dict(score=0, success=False),
            raw_score=self._result.score if self._result else 0,
            maximum_score=place.MAXIMUM_SCORE, timeout_owner='room',
            required_hold_s=place.HOLD_SECONDS,
            rule_adapter='scene_bound_fixed_cpu_basket')

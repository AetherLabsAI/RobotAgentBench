"""Task-local arithmetic adapter; approved source-furniture pickup deductions."""
from dataclasses import replace
from evaluation.tasks.tidy_sofa import process_rules as pick
from evaluation.tasks.personal_item_pick.specs import task_spec
from evaluation.tasks.basket_place import process_rules as place


class RecoverablePick(pick.BaseScorer):
    def __init__(self, episode_id, dt, target):
        super().__init__(episode_id, dt, task_spec('13-3'))
        self.target = target
        self.ever_complete = False

    def pause(self):
        self.hold_start = None
        self.hold_elapsed = 0.
        self.contact_started.clear()
        self.contact_ticks.clear()

    def observe_one(self, f, active, *, pickup_phase=True):
        # Dense evidence/identity validation belongs to the composite.
        # Composite filters contacts to the episode's accepted source furniture.
        self.last = f
        if not active:
            self.pause()
            return False
        if pickup_phase:
            self._deductions(f)
        else:
            self.contact_started.clear()
            self.contact_ticks.clear()
        if f.target_on_floor:
            self._hold(f, False)
            return False
        if f.target_grasped:
            self._award(f, 'grasp')
        carrying = (f.target_grasped and not f.target_supported
                    and f.lift_m >= self.spec.lift_m)
        ready = self._hold(f, carrying)
        if ready:
            self._award(f, 'lift')
            self.ever_complete = True
        return ready

    def summary(self):
        return dict(super().summary(), success=self.ever_complete, failure=None,
                    time_limit_s=None, timeout_owner='room',
                    rule_adapter='openend_powerbank_recoverable_13_3')


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

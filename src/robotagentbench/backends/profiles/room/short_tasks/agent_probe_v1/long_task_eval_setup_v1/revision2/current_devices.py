"""Read-only reuse of the existing fryer 2.5 current extraction predicate."""
import math
import json


def fryer_current_result(*, history_success, facts, sample_time_s, current_time_s):
    if type(history_success) is not bool:
        raise ValueError('Original typed process completion required')
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in (sample_time_s,current_time_s)) or sample_time_s != current_time_s:
        raise ValueError('Current native fryer sample required')
    target = facts.target
    for name in ('grasped','outside_basket','ground_contact'):
        if type(getattr(target,name)) is not bool:
            raise ValueError('Typed current fryer target fact required: '+name)
    valid = history_success and target.grasped and target.outside_basket and not target.supports and not target.ground_contact
    return dict(available=True, met=bool(valid), history_success=history_success,
        simulation_time_s=current_time_s, source='original_fryer_2_5_current_take_predicate',
        added_score=0, new_hold_required=False)


def bread_port():
    """The original scorer ends; the same native reader keeps monitoring it."""
    from short_tasks.tasks.bake_bread.native_port import BakeBreadPort
    from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Frame

    class LongBakeBreadPort(BakeBreadPort):
        @property
        def ended(self):
            if getattr(self, '_long_finalized', False):
                return True
            return self.composite is not None and self.composite.termination is not None and not self.composite.success

        def observe(self):
            if self.clock.hz not in self.supported_physics_hz:
                raise RuntimeError('Unsupported original fryer reader clock')
            if self.source.latest['time_s'] != self.clock.current_time:
                raise RuntimeError('Stale native fryer evidence')
            self.index += 1
            # One reader snapshot serves both the original scorer and the
            # current predicate. selected_facts may consume its contact window.
            facts = self.source.selected_facts(self.index)
            if self.composite.termination is None:
                self.score = self.composite.observe(Frame(self.binding, facts))
            else:
                if not self.composite.success:
                    raise RuntimeError('Cannot continue a failed fryer process')
            # Uses the unchanged reader; no scene, phase, baseline or score writes.
            self._long_current_result = fryer_current_result(
                history_success=bool(self.composite.success), facts=facts,
                sample_time_s=self.source.latest['time_s'], current_time_s=self.clock.current_time)
            self.trace.write(json.dumps(dict(facts=facts.as_dict(),score=self.score,
                simulation_time_s=self.clock.current_time,long_current_result=self._long_current_result),default=list)+'\n')
            self.trace.flush()
            return self.score

        def finalize(self, reason):
            result = super().finalize(reason)
            self._long_finalized = True
            return result

    return LongBakeBreadPort


def coffee_port():
    from short_tasks.tasks.prepare_coffee.port import CoffeeShortPort

    class LongCoffeePort(CoffeeShortPort):
        def observe(self):
            result = super().observe()
            self._long_current_observed_time = self.clock.current_time
            return result

    return LongCoffeePort

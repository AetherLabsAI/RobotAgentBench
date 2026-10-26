"""Room-selected task score adapters. Original Process scoring rules unchanged.

The user-approved room watchdog replaces composite per-phase terminal timeouts.
All time-based termination belongs to the shared room watchdog.
Interruption clears unfinished holds, not awarded points, baselines or stages.
"""
from copy import deepcopy
from dataclasses import replace
import math
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import CoffeeComposite, PHASES
from evaluation.tasks.coffee_machine.process_rules import CONTROL_DT_S
from .components import CoffeeProcess as Process, FryerProcess as _HandoffProcess
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import AirFryerComposite
from evaluation.tasks.air_fryer.facts import PLATE, COUNTER

def interrupt_hold(composite):
    p=composite.process
    if p is not None:
        p._window=[]
        if hasattr(p,'_released_step'):p._released_step=None

class CoffeeTask(CoffeeComposite):
    def summary(self):
        result=super().summary()
        result.update(schema='openend.coffee.v1', task_type='coffee',
                      total_timeout_policy='room_no_score_increase',
                      leave_resume_policy='preserve_scores_clear_unfinished_hold',
                      phase_budgets='usage_telemetry_only_not_enforced')
        result['termination']=getattr(self,'termination',None)
        return result

    def finalize(self, reason):
        if getattr(self,'termination',None) is not None:
            if self.termination != reason: raise ValueError('Contradictory finalization')
            return self.summary()
        if self.process is not None and not self.process._finalized:
            self.process.finalize('success' if self.success else reason)
            if self.process.failure:
                self.failure=dict(phase=self.phase,reason=self.process.failure)
        self.termination=reason
        self.ended=True
        return self.summary()

    def observe(self, frame):
        self._validate_global(frame)
        f = frame.facts
        entering = self.process is None
        baseline = (self.last or f) if entering else self.baseline
        start = f.step if entering else self.phase_start
        # Frozen Spec includes MappingProxyType. Share this immutable definition,
        # copy only per-instance state (bound advance method is rebound by deepcopy).
        p = Process(self.phase) if entering else deepcopy(
            self.process, {id(self.process.spec): self.process.spec})
        # Pure measurement baselines, not changes to scene poses or original rules.
        cup = replace(f.cup, displacement_m=math.dist(f.cup.position, baseline.cup.position))
        bag = replace(f.bag, displacement_m=math.dist(f.bag.position, baseline.bag.position))
        liquid = f.liquid
        if self.phase == '1-4':
            # The actual amount at the brew->retrieve boundary; never old prefill.
            liquid = replace(liquid, initial_in_cup=baseline.liquid.in_cup,
                             spilled=f.liquid.spilled - baseline.liquid.spilled)
        local = replace(f, step=f.step-start, time_s=(f.step-start)*CONTROL_DT_S,
                        cup=cup, bag=bag, liquid=liquid)
        p.observe(local)  # Existing validation/thresholds/budgets/deductions intact.
        timed_out = False  # room's shared no-progress watchdog owns episode termination
        if timed_out:
            p.finalize('timeout')
        # Commit only after valid evidence: rejected frames do not consume steps.
        self.process, self.baseline, self.phase_start, self.last = p, baseline, start, f
        if p.failure or timed_out:
            self.failure = dict(phase=self.phase, reason=p.failure or 'timeout')
            self.ended = True
        elif p.success:
            p.finalize('success')
            self.completed.append(dict(api=self.phase, start_step=start,
                                       end_step=f.step, result=p.summary()))
            self.transitions.append(dict(from_phase=self.phase, step=f.step,
                                         scene_reset=False, liquid_in_cup=f.liquid.in_cup,
                                         beans_in_hopper=f.beans_in_hopper))
            self.index += 1
            self.process = None
            self.success = self.index == len(PHASES)
            self.ended = self.success
        return self.summary()


class FryerTask(AirFryerComposite):
    def __init__(self, binding, bread_variant):
        super().__init__(binding, bread_variant)
        self.process = _HandoffProcess(self.phase)

    def summary(self):
        result=super().summary()
        result.update(schema='openend.fryer.v1', task_type='fryer', timeout_owner='room',
                      phase_time_limits_enforced=False)
        return result

    def _observe(self, f):
        local = self._local(f)
        self.process._validate(local)
        if self.last is None:
            # Logical start only; separate live USD/settle acceptance remains necessary.
            if abs(f.tray.opening_m) > 0.005 or f.tray.joint_released:
                raise ValueError('Unified scene must start with closed, guided basket')
            if f.tray.robot_contact or f.tray.ground_contact:
                raise ValueError('Invalid basket start')
            for c in (f.target, f.other):
                if c.supports != frozenset({PLATE}) or not c.released or c.inside_basket or c.ground_contact or c.tilt_deg > 5:
                    raise ValueError('Both released cakes must start on source plate')
        if self.last and self.last.tray.joint_released and not f.tray.joint_released:
            raise ValueError('all_assets rail cannot relatch during an episode')
        if self.index > 0 and self.inherited_counter_support:
            c = f.other
            # Exempt only uninterrupted inherited countertop support. Rehandling,
            # leaving that surface, floor or basket contact revokes the exemption.
            self.inherited_counter_support = (c.released and c.supports == frozenset({COUNTER})
                                              and not c.ground_contact and not c.basket_contact)
        self.process.inherited_counter_support = self.inherited_counter_support
        self.process.observe(local)
        self.last = f
        if self.process.failure:
            self.failure = self.process.failure
            self.termination = 'failure'
            return
        if not self.process.success:
            # Shared room watchdog replaces per-phase episode termination.
            return
        self.process.finalize('success')
        if self.index == 2:
            self.success = True
            self.termination = 'success'
            return
        previous = self.phase
        self.completed.append(self.process.summary())
        if self.index == 0:
            c = f.other
            self.inherited_counter_support = (c.released and c.supports == frozenset({COUNTER})
                                              and not c.ground_contact and not c.basket_contact)
        else:
            self.basket_baseline = tuple(f.tray.position)
        self.index += 1
        self.phase_start = f.step
        self.transitions.append(dict(from_phase=previous, to_phase=self.phase, step=f.step, time_s=f.time_s,
                                     basket_baseline=self.basket_baseline,
                                     inherited_counter_support=self.inherited_counter_support))
        self.process = _HandoffProcess(self.phase)
        self.process.inherited_counter_support = self.inherited_counter_support
        # Same measured boundary is local frame zero. No world/reset/pose changes.
        self.process.observe(self._local(f))

"""Independent operation and final-state scoring on trusted measurements.

This module is not a native reader. No USD, reset, device or robot actions.
Pickup predicates come from Openend's original timeout-adapted components.
Final state follows the latest user rule after required process milestones.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

from openend.tasks.components import CoffeeProcess, FryerProcess
from evaluation.tasks.coffee_machine.facts import Facts as CoffeeFacts, TRAY
from evaluation.tasks.air_fryer.facts import Facts as FryerFacts, BASKET_FLOOR, COUNTER
from .final_state import Binding, PlacementFrame, FinalState, path

KINDS = ('coffee_to_meal_tray', 'bread_to_meal_tray')


@dataclass(frozen=True)
class PickupFrame:
    binding: Binding
    facts: object


class MealTrayTask:
    def __init__(self, binding):
        if not isinstance(binding, Binding) or binding.kind not in KINDS:
            raise ValueError('Scene-supplied binding required')
        self.binding = binding
        self.pick = (CoffeeProcess('1-4') if binding.kind == KINDS[0]
                     else FryerProcess('2.5'))
        self.phase = 'pickup'
        self.last_step = None
        self.baseline = None
        self.pick_result = None
        self.final_state = FinalState(binding)
        self.success = False
        self.termination = None
        self.events = []
        self.inherited_counter_support = False
        self.upstream_instance = None
        self.operation_step = -1
        self.operation_failure = None
        self.operation_closed_by_final = False

    def accept_preparation(self, preparation):
        """Bind a measured upstream completion, never a request-supplied JSON flag.

        The physical reader stays owned by the room. This transfers measurement
        baselines and exemption provenance only, not poses, devices or rewards.
        """
        from .preparation import Preparation
        if self.operation_step >= 0 or self.baseline is not None:
            raise ValueError('Preparation can be handed off exactly once, before pickup')
        if not isinstance(preparation, Preparation) or not preparation.success:
            raise ValueError('Measured preparation completion required')
        h = preparation.handoff
        expected = 'coffee' if self.binding.kind == KINDS[0] else 'fryer'
        if (h['kind'] != expected or h['episode_id'] != self.binding.episode_id
                or h['target'] != self.binding.target or h['source'] != self.binding.source
                or h['source_task_instance_id'] == self.binding.instance_id):
            raise ValueError('Upstream episode, target, source or identity mismatch')
        self.baseline = preparation.component.last
        self.inherited_counter_support = (h['inherited_counter_support'] and
                                           preparation.inherited_counter_support)
        self.upstream_instance = h['source_task_instance_id']

    def observe_inactive_other(self, other):
        """Room reader must call this for contact/support changes while inactive."""
        if self.inherited_counter_support:
            self.inherited_counter_support = self._counter_continuous(other)

    @staticmethod
    def _counter_continuous(other):
        return (other.released and other.supports == frozenset({COUNTER})
                and not other.ground_contact and not other.basket_contact)

    @property
    def ended(self):
        return self.termination is not None

    def _check(self, binding, step, time_s):
        if self.ended:
            raise ValueError('Task already finalized')
        if binding != self.binding:
            raise ValueError('Wrong episode, task, target or meal tray')
        expected = 0 if self.last_step is None else self.last_step + 1
        if type(step) is not int or step != expected:
            raise ValueError('Repeated, skipped or unordered selected frame')
        if type(time_s) not in (int, float) or not math.isfinite(time_s) or abs(time_s-step*.05) > 1e-6:
            raise ValueError('Existing selected 20 Hz clock required')

    def observe_pickup(self, frame):
        if not isinstance(frame, PickupFrame) or self.phase != 'pickup':
            raise ValueError('Pickup evidence cannot score during placement')
        f = frame.facts
        expected = CoffeeFacts if self.binding.kind == KINDS[0] else FryerFacts
        if not isinstance(f, expected):
            raise ValueError('Original typed pickup facts required')
        self._check(frame.binding, f.step, f.time_s)
        base = self.baseline or f
        if isinstance(f, CoffeeFacts):
            if f.liquid is None or f.liquid.mode != 'visual_cpu_v1':
                raise ValueError('Openend coffee must use existing CPU animation mode')
            if self.baseline is None and (not f.cup.released or TRAY not in f.cup.supports or f.placement.outside_bay):
                raise ValueError('Unified scene must supply cup on coffee machine tray')
            local = replace(f, cup=replace(f.cup, displacement_m=math.dist(f.cup.position, base.cup.position)),
                            bag=replace(f.bag, displacement_m=math.dist(f.bag.position, base.bag.position)),
                            liquid=replace(f.liquid, initial_in_cup=base.liquid.in_cup,
                                           spilled=f.liquid.spilled-base.liquid.spilled))
        else:
            if self.baseline is not None and not f.tray.joint_released:
                raise ValueError('Released all_assets rail cannot relatch')
            if self.baseline is None and (not f.target.released or BASKET_FLOOR not in f.target.supports
                    or not f.target.inside_basket or not f.tray.counter_supported
                    or f.tray.robot_contact or f.tray.body_contact or not f.tray.joint_released):
                raise ValueError('Scene or upstream task must supply released basket on counter with target inside')
            local = replace(f, tray=replace(f.tray, displacement_m=math.dist(f.tray.position, base.tray.position)))
        # Rejecting a frame must not partially consume it or change score.
        candidate = deepcopy(self.pick, {id(self.pick.spec): self.pick.spec})
        inherited = self.inherited_counter_support
        if isinstance(f, FryerFacts):
            inherited = inherited and self._counter_continuous(f.other)
            candidate.inherited_counter_support = inherited
        # The operation component owns only its selected samples. Final-state
        # observations neither advance it nor fabricate missing pickup frames.
        local = replace(local,step=self.operation_step+1,time_s=(self.operation_step+1)*.05)
        candidate.observe(local)
        self.pick, self.baseline, self.last_step = candidate, base, f.step
        self.operation_step += 1
        self.final_state.invalidate(); self.success = False
        self.inherited_counter_support = inherited
        if candidate.failure:
            candidate.finalize('failure')
            self.operation_failure = candidate.failure
            self.pick_result = candidate.summary()
            self.phase = 'operation_failed'
        elif candidate.success:
            candidate.finalize('success')
            self.pick_result = candidate.summary()
            self.phase = 'place'
            self.events.append(dict(event='pickup_complete', step=f.step, scene_reset=False))
        return self.summary()

    def observe_placement(self, frame):
        if not isinstance(frame, PlacementFrame):
            raise ValueError('Typed final-state evidence required')
        self._check(frame.binding, frame.step, frame.time_s)
        self.final_state.observe(frame,process_complete=bool((self.pick_result or self.pick.summary())['success']))
        self.last_step = frame.step
        self.success = self.final_state.met
        if self.success and not any(e['event']=='meal_tray_placed' for e in self.events):
            self.events.append(dict(event='meal_tray_placed', points=3, step=frame.step))
        return self.summary()

    def observe(self, placement, pickup=None):
        """Atomic same-control-frame entry. Native callers must use this method.

        Optional pickup evidence is evaluated only while its operation is open;
        final state is ALWAYS evaluated. A bad second channel commits neither.
        """
        self._check(placement.binding, placement.step, placement.time_s)
        candidate=deepcopy(self,{id(self.pick.spec):self.pick.spec})
        final_probe=deepcopy(self.final_state)
        final_probe.observe(placement)
        previous=self.last_step
        if pickup is not None:
            if (not isinstance(pickup,PickupFrame) or pickup.binding!=placement.binding
                    or pickup.facts.step!=placement.step or pickup.facts.time_s!=placement.time_s):
                raise ValueError('Both score channels require the same physical frame')
            if candidate.phase=='pickup': candidate.observe_pickup(pickup)
        candidate.last_step=previous
        candidate.observe_placement(placement)
        self.__dict__.update(candidate.__dict__)
        return self.summary()

    def pause(self):
        # Score and initial measurement baselines persist. Holds never bridge absence.
        self.final_state.invalidate(); self.success = False
        self.pick._window = []
        if hasattr(self.pick, '_released_step'):
            self.pick._released_step = None

    def finalize(self, reason):
        if not isinstance(reason,str) or not reason:
            raise ValueError('Termination reason required')
        if self.ended:
            if reason != self.termination:
                raise ValueError('Contradictory finalization')
            return self.summary()
        if reason == 'success' and not self.success:
            raise ValueError('No measured placement success')
        if self.operation_step >= 0 and self.phase == 'pickup':
            self.pick.finalize('source_exhausted' if reason=='success' else reason)
        self.termination = reason
        return self.summary()

    def summary(self):
        p = self.pick_result or self.pick.summary()
        return dict(schema='openend.meal_tray.score.v2',task_type=self.binding.kind,
                    phase=self.phase,score=p['score']+(3 if self.success else 0),
                    maximum_score=p['maximum_score']+3,success=self.success,ended=self.ended,
                    termination=self.termination,pickup=p,placement_points=3 if self.success else 0,
                    operation_score=p['score'],operation_complete=bool(p['success']),
                    operation_failure=self.operation_failure,final_state=self.final_state.summary(),
                    operation_closed_by_final=self.operation_closed_by_final,
                    completion='process_milestones_then_current_destination',
                    events=deepcopy(self.events),
                    timeout_owner='room_120s_no_score_increase',scene_reset=False,
                    upstream_instance=self.upstream_instance,
                    inherited_counter_support=self.inherited_counter_support,
                    runtime_registered=False,native_physics_verified=False)

"""CPU-only composition prototype. No scene/reset/device/controller imports.

Consumes trusted, dense 20 Hz facts from ONE unified scene. Not a registered
WorkerTask: live binding, episode timeout and leave/resume policy remain pending.
The four existing Process implementations and all their constants are unchanged.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

from evaluation.tasks.coffee_machine.facts import Facts, COUNTER
from evaluation.tasks.coffee_machine.process_rules import (
    Process, SPECS, CONTROL_DT_S, CLOCK_TOLERANCE_S, RULE_VERSION,
)

PHASES = ('1-1', '1-2', '1-3', '1-4')
ROLES = frozenset({'machine', 'cup', 'bag', 'beans', 'button', 'cup_support', 'counter'})


@dataclass(frozen=True)
class Binding:
    episode_id: str
    task_instance_id: str
    objects: tuple  # sorted (role, instance identity) pairs, supplied by scene owner

    def __post_init__(self):
        if not self.episode_id or not self.task_instance_id:
            raise ValueError('Episode and task instance identity required')
        if not isinstance(self.objects, tuple) or any(
                not isinstance(x, tuple) or len(x) != 2 for x in self.objects):
            raise ValueError('Immutable object bindings required')
        if len(self.objects) != len(ROLES) or set(dict(self.objects)) != ROLES:
            raise ValueError('Missing or duplicate object roles')
        values = [v for _, v in self.objects]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError('Each role must resolve to a distinct concrete instance')


@dataclass(frozen=True)
class Frame:
    binding: Binding
    facts: Facts


class UnresolvedSequence(ValueError):
    """Unsupported ordering requires a defined policy, not invented failure points."""


class CoffeeComposite:
    def __init__(self, binding):
        if not isinstance(binding, Binding):
            raise ValueError('Binding required; task never creates missing objects')
        self.binding = binding
        self.index = 0
        self.process = None
        self.completed = []
        self.last = None
        self.baseline = None
        self.phase_start = None
        self.failure = None
        self.success = False
        self.ended = False
        self.transitions = []
        self.source = 'synthetic_or_external_facts_not_physics_verified'

    @property
    def phase(self):
        return PHASES[self.index] if self.index < len(PHASES) else 'complete'

    def _validate_global(self, frame):
        if self.ended:
            raise ValueError('Task is terminal; cannot score again')
        if not isinstance(frame, Frame) or frame.binding != self.binding:
            raise ValueError('Wrong episode, task or object instance')
        f = frame.facts
        if not isinstance(f, Facts):
            raise ValueError('Typed facts required')
        expected = 0 if self.last is None else self.last.step + 1
        if type(f.step) is not int or f.step != expected:
            raise ValueError('Repeated, missing or out-of-order global control frame')
        if not math.isfinite(f.time_s) or abs(f.time_s - f.step * CONTROL_DT_S) > CLOCK_TOLERANCE_S:
            raise ValueError('Global time must remain on the existing 20 Hz clock')
        if f.liquid is None or f.button is None:
            raise ValueError('Persistent device and liquid facts required in every phase')
        if self.last:
            if f.beans_total != self.last.beans_total:
                raise ValueError('Bean inventory changed across the continuous task')
            if f.liquid.mode != self.last.liquid.mode:
                raise ValueError('Liquid evidence mode changed')
            if f.liquid.emitted < self.last.liquid.emitted or f.liquid.spilled < self.last.liquid.spilled:
                raise ValueError('Device/liquid counters reset during task')
        if self.index < 2 and (f.button.events or f.button.dispensing or f.liquid.emitted):
            raise UnresolvedSequence('Device operated before brew phase; no credit or silent reset')
        if self.index == 3 and (f.button.events or f.button.dispensing):
            raise UnresolvedSequence('Device restarted during retrieval; policy not defined')
        if self.last is None:
            # Only observable logical readiness. Not a USD/settle acceptance receipt.
            if f.beans_in_hopper or f.beans_total <= 0 or f.liquid.in_cup or f.liquid.airborne or f.liquid.spilled:
                raise ValueError('Expected empty hopper/cup with existing bean inventory')
            for obj in (f.cup, f.bag):
                if COUNTER not in obj.supports or not obj.released:
                    raise ValueError('Scene must supply released cup and bag on counter')

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
        timed_out = not p.success and not p.failure and local.time_s >= p.spec.time_limit_s
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

    def summary(self):
        active = self.process.summary() if self.process else None
        score = sum(x['result']['score'] for x in self.completed) + (active['score'] if active else 0)
        return deepcopy(dict(
            schema='coffee.openended.composition.prototype.v1',
            task_instance_id=self.binding.task_instance_id,
            runtime_registered=False, physics_verified=False, evidence_source=self.source,
            phase=self.phase, completed_phases=len(self.completed),
            score=score, maximum_score=sum(SPECS[x].maximum_score for x in PHASES),
            success=self.success, failure=self.failure, ended=self.ended,
            completed=self.completed, active=active, transitions=self.transitions,
            rule_version=RULE_VERSION, initialization_owner='unified_scene',
            total_timeout_policy='pending', leave_resume_policy='pending',
            phase_budgets='unchanged_source_rules_dense_20hz_only',
            cumulative_score_policy='prototype_sum_of_original_phase_scores',
        ))

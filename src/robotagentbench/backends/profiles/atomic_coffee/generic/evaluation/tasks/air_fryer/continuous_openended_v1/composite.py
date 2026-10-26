"""Pure scoring: one unified scene, one target, insert -> pull -> take.

No USD/reset/controller/device writes. The scene owner supplies dense trusted
20 Hz Facts. This component is not proof of physics or a registered WorkerTask.
Only the two user-approved handoff rules differ from the original Process.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

from evaluation.tasks.air_fryer.facts import Facts, COUNTER, PLATE
from evaluation.tasks.air_fryer.process_rules import Process, CONTROL_DT_S, CLOCK_TOLERANCE_S

ROLES = frozenset({'fryer', 'basket', 'target', 'other', 'plate', 'counter'})


@dataclass(frozen=True)
class Binding:
    episode_id: str
    task_instance_id: str
    objects: tuple

    def __post_init__(self):
        if not all(isinstance(s, str) and s for s in (self.episode_id, self.task_instance_id)):
            raise ValueError('Episode and task identity required')
        if not isinstance(self.objects, tuple) or any(not isinstance(p, tuple) or len(p) != 2 for p in self.objects):
            raise ValueError('Immutable role bindings required')
        if len(self.objects) != len(ROLES) or set(dict(self.objects)) != ROLES:
            raise ValueError('Missing or duplicate role')
        values = [v for _, v in self.objects]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError('Distinct concrete object identities required')


@dataclass(frozen=True)
class Frame:
    binding: Binding
    facts: Facts


class _HandoffProcess(Process):
    """Keep source predicates/thresholds. Narrowly waive inherited counter support."""
    def __init__(self, key):
        super().__init__(key)
        self.inherited_counter_support = False

    def _deduct(self, f, kind, amount=1):
        if kind == 'other_knocked_off_plate' and self.inherited_counter_support:
            return
        super()._deduct(f, kind, amount)


class AirFryerComposite:
    def __init__(self, binding, bread_variant):
        if not isinstance(binding, Binding):
            raise ValueError('Unified-scene Binding required')
        if bread_variant not in ('sugar', 'no_sugar'):
            raise ValueError('bread_variant must be sugar or no_sugar')
        self.binding = binding
        self.phases = ('2.2' if bread_variant == 'sugar' else '2.1', '2.3', '2.5')
        self.index = 0
        self.process = _HandoffProcess(self.phases[0])
        self.phase_start = 0
        self.last = None
        self.completed = []
        self.transitions = []
        self.basket_baseline = None
        self.inherited_counter_support = False
        self.success = False
        self.failure = None
        self.termination = None

    @property
    def phase(self):
        return 'complete' if self.success else self.phases[self.index]

    def observe(self, frame):
        # Transactional: rejected evidence cannot consume a frame or revoke an exemption.
        if self.termination is not None:
            raise ValueError('Task is terminal')
        if not isinstance(frame, Frame) or frame.binding != self.binding:
            raise ValueError('Wrong episode, task or object identity')
        f = frame.facts
        if not isinstance(f, Facts):
            raise ValueError('Typed physical facts required')
        expected = 0 if self.last is None else self.last.step + 1
        if type(f.step) is not int or f.step != expected:
            raise ValueError('Missing, repeated or unordered control frame')
        if isinstance(f.time_s, bool) or not isinstance(f.time_s, (int, float)) or not math.isfinite(f.time_s) or abs(f.time_s - f.step * CONTROL_DT_S) > CLOCK_TOLERANCE_S:
            raise ValueError('Expected continuous 20 Hz simulation time')
        candidate = deepcopy(self, {id(self.process.spec): self.process.spec})
        candidate._observe(f)
        self.__dict__.update(candidate.__dict__)
        return self.summary()

    def _local(self, f):
        tray = f.tray
        if self.phase == '2.5':
            tray = replace(tray, displacement_m=math.dist(tray.position, self.basket_baseline))
        step = f.step - self.phase_start
        return replace(f, step=step, time_s=step * CONTROL_DT_S, tray=tray)

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
            # Original phase total budgets were enforced outside Process; retain them here.
            if local.time_s >= self.process.spec.time_limit_s:
                self.process.finalize('time_limit')
                self.termination = 'time_limit'
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

    def finalize(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError('Termination reason required')
        if self.termination is not None:
            if reason != self.termination:
                raise ValueError('Contradictory finalization')
            return self.summary()
        if reason == 'success':
            raise ValueError('No measured continuous success')
        self.process.finalize(reason)
        self.termination = reason
        return self.summary()

    def summary(self):
        phases = self.completed + [self.process.summary()]
        return dict(schema='arena.air_fryer.continuous.prototype.v1', phase=self.phase,
                    score=max(0., sum(p['raw_score'] for p in phases)), maximum_score=12,
                    success=self.success, failure=self.failure, termination=self.termination,
                    phase_results=phases, transitions=deepcopy(self.transitions),
                    last_global_step=None if self.last is None else self.last.step,
                    basket_baseline=self.basket_baseline,
                    inherited_counter_support=self.inherited_counter_support,
                    evidence_scope='synthetic_or_external_facts_not_physics_verified',
                    registered_runtime=False)

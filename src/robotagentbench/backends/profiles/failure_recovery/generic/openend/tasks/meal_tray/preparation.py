"""Stop existing Openend composites before their extraction phase.

The legacy public APIs and old complete Openend classes remain untouched.
Prepared products remain where the measured preceding phase left them.
"""
from copy import deepcopy
from openend.tasks.scores import CoffeeTask, FryerTask, interrupt_hold


class Preparation:
    def __init__(self, component, *, kind):
        if kind == 'coffee' and not isinstance(component, CoffeeTask):
            raise ValueError('Original Openend coffee component required')
        if kind == 'fryer' and not isinstance(component, FryerTask):
            raise ValueError('Original Openend fryer component required')
        if kind not in ('coffee', 'fryer'):
            raise ValueError('Unknown preparation kind')
        if component.last is not None:
            raise ValueError('Preparation must own a fresh component')
        self.component = component
        self.kind = kind
        self.boundary = '1-4' if kind == 'coffee' else '2.5'
        self.success = False
        self.termination = None
        self.handoff = None

    @property
    def phase(self):
        return 'complete' if self.success else self.component.phase

    @property
    def ended(self):
        return self.termination is not None

    @property
    def process(self):
        return self.component.process

    @property
    def inherited_counter_support(self):
        return getattr(self.component,'inherited_counter_support',False)

    @inherited_counter_support.setter
    def inherited_counter_support(self,value):
        self.component.inherited_counter_support=value

    def observe(self, frame):
        if self.ended:
            raise ValueError('Preparation complete; extraction belongs to meal task')
        self.component.observe(frame)
        if self.component.phase == self.boundary:
            # The full source composite constructed its next process, but no
            # later extraction frames are ever sent to it by this wrapper.
            self.success = True
            self.termination = 'success'
            roles = dict(self.component.binding.objects)
            self.handoff = dict(episode_id=self.component.binding.episode_id,
                source_task_instance_id=self.component.binding.task_instance_id,
                kind=self.kind, target=roles['cup' if self.kind=='coffee' else 'target'],
                source=roles['machine' if self.kind=='coffee' else 'basket'],
                source_phase=self.boundary,scene_reset=False,
                basket_baseline=getattr(self.component,'basket_baseline',None),
                inherited_counter_support=getattr(self.component,'inherited_counter_support',False))
        elif self.component.failure:
            self.termination = 'failure'
        return self.summary()

    def pause(self):
        interrupt_hold(self.component)

    def finalize(self, reason):
        if self.ended:
            if reason != self.termination:
                raise ValueError('Contradictory finalization')
            return self.summary()
        if reason == 'success':
            raise ValueError('No measured preparation success')
        self.component.finalize(reason)
        self.termination = reason
        return self.summary()

    def summary(self):
        raw = self.component.summary()
        if self.kind == 'coffee':
            phases = [p['result'] for p in raw['completed']]
            if not self.success and raw['active'] is not None:
                phases.append(raw['active'])
        else:
            phases = raw['phase_results'][:2]
        return dict(schema='openend.meal_preparation.v1',kind=self.kind,
            phase=self.phase,score=max(0,sum(p['raw_score'] for p in phases)),
            maximum_score=16 if self.kind=='coffee' else 8,
            success=self.success,ended=self.ended,termination=self.termination,
            phase_results=deepcopy(phases),handoff=deepcopy(self.handoff),
            extraction_scored=False,scene_reset=False,runtime_registered=False,
            native_physics_verified=False)

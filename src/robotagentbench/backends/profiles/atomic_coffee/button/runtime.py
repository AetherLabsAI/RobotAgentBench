"""Task-local WorkerTask adapter: original 1-3 evidence/rules, real light hook.

No global registration, source-output guard bypass, or new action protocol.
The existing CoffeeMachineRuntime constructor owns another task's output, so
this subclass initializes its own fields explicitly and reuses its unchanged
attach/acceptance/evaluate/finalize methods. State cannot leak between episodes.
"""
from pathlib import Path
from uuid import UUID
from evaluation.tasks.coffee_machine.api_runtime import CoffeeMachineRuntime
from evaluation.tasks.coffee_machine.registration import normalizer
from evaluation.tasks.coffee_machine.process_rules import Process
from evaluation.tasks.coffee_machine.specs import spec
from arena_api.physics_defaults import steps_for_seconds
from evaluation.tasks.coffee_machine.initial_state import SETTLE_SECONDS
from .initial import build, task_output


class LightingButtonRuntime(CoffeeMachineRuntime):
    def __init__(self, context):
        from .source_audit import verify
        from .initial import HERE
        verify(HERE / 'component_sources.json')
        UUID(context.episode_id)
        if any(context.randomization.values()):
            raise ValueError('Use original task_parameters.random_layout, not new house flags')
        self.context = context
        self.spec = spec('1-3')
        self.parameters = normalizer('1-3')(context.task_parameters)
        task_output(context.output)
        self.setup = Path(context.output) / 'setup'
        self.settle_steps = steps_for_seconds(SETTLE_SECONDS['1-3'])
        self.requires_gpu_dynamics = False
        self.process = Process('1-3')
        self.reader = self.plan = self.pending = self.lighting = None
        self.control_index = -1
        self.admitted = False

    def initialize(self, source):
        from arena_api import l1_scene
        if Path(source).resolve(strict=True) != l1_scene.BASE.resolve():
            raise ValueError('Original atomic baseline required')
        self.plan = build(self.setup, seed=self.context.seed,
                          role='random' if self.parameters['random_layout'] else 'source',
                          episode=self.parameters['source_episode'])
        self.appearance = self.plan['appearance']
        return Path(self.plan['scene'])

    def attach(self, stage):
        super().attach(stage)
        from .lighting import SceneLighting
        self.lighting = SceneLighting(stage, self.context.seed)

    def validate_initial(self, metadata):
        super().validate_initial(metadata)
        self.lighting.arm(float(self.world.current_time))
        self.admitted = True

    def after_physics(self, helper_report):
        super().after_physics(helper_report)
        if self.admitted:
            self.lighting.tick(float(self.world.current_time))

    def close(self):
        try:
            if self.lighting:
                import json
                (task_output(self.context.output) / 'lighting.json').write_text(
                    json.dumps(self.lighting.snapshot(), indent=2) + '\n')
                self.lighting.close()
        finally:
            super().close()

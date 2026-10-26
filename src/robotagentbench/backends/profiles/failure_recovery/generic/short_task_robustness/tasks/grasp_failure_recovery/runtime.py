"""Task-local prepared adapters. The source scene composers are not redirected."""
import json
from copy import deepcopy
from pathlib import Path
from .common import CURRENT, COFFEE_SOURCE, owned, require_source, verify_sources


def make_runtime(context, initial_file):
    initial_file = owned(initial_file)
    initial = json.loads(initial_file.read_text())
    verify_sources(initial['source_layers'])
    verify_sources(initial['code_sources'])
    if initial['variant'] == 'shelf_drink':
        from evaluation.tasks.shelf_pick import api_runtime as source, initial_state
        from short_tasks.tasks.stock_wooden_tray.adapter import RecoverableShelfProcess
        require_source(source, CURRENT)

        class PreparedShelf(source.ShelfPickRuntime):
            def __init__(self):
                super().__init__(context)
                self.scorer = RecoverableShelfProcess(context.episode_id, source.PHYSICS_DT, '4-3')
                self.snapshot = initial_file.parent/'source_snapshot.json'
                snapshot = json.loads(self.snapshot.read_text())
                self.plan = dict(initial, api_id='4-3', targets=[initial['target']],
                    robot_initial_positions=initial['robot_initialization']['positions'],
                    stock_roots=list(initial_state.STOCK_ROOTS), dependency_sha256=initial['source_layers'],
                    source_episode=dict(snapshot['source'], item_supports={item['prim_path']: item['support']
                        for item in snapshot['items']}))

            def initialize(self, source):
                return Path(self.plan['scene'])

            def randomize(self, scene):
                return scene

        return PreparedShelf()
    if initial['variant'] not in ('coffee_bag', 'coffee_cup', 'coffee_button'):
        raise ValueError('Unknown failure-recovery variant')
    api = {'coffee_cup': '1-1', 'coffee_bag': '1-2', 'coffee_button': '1-3'}[initial['variant']]
    if initial['source_api'] != api or context.task_id != api:
        raise ValueError('Prepared coffee variant and runtime API disagree')
    if initial['variant'] == 'coffee_cup':
        if initial.get('recovery_initial_contract') != 'cup_upright_released_on_counter_outside_tray.v1' \
                or initial.get('preparation_accepted') is not True:
            raise ValueError('Cup recovery requires the task-owned outside-tray preparation gate')
    from evaluation.tasks.coffee_machine import api_runtime as source
    from evaluation.tasks.coffee_machine.specs import spec
    from evaluation.tasks.coffee_machine.registration import normalizer
    from evaluation.tasks.coffee_machine.initial_state import SETTLE_SECONDS
    from evaluation.tasks.coffee_machine.process_rules import Process
    from arena_api.physics_defaults import steps_for_seconds
    require_source(source, COFFEE_SOURCE)

    class PreparedCoffee(source.CoffeeMachineRuntime):
        def __init__(self):
            # This adapter owns its prepared scene/output; the old source
            # initializer's output guard is never patched or bypassed.
            self.context = context
            self.spec = spec(api)
            self.parameters = normalizer(api)({'source_episode': initial['source_episode'], 'random_layout': False})
            self.settle_steps = steps_for_seconds(SETTLE_SECONDS[api])
            self.requires_gpu_dynamics = False
            self.plan = deepcopy(initial)
            self.appearance = self.plan['appearance']
            self.process = Process(api)
            self.reader = self.pending = None
            self.control_index = -1

        def initialize(self, source):
            return Path(self.plan['scene'])

        def validate_initial(self, metadata):
            super().validate_initial(metadata)
            if initial['variant'] == 'coffee_cup':
                from .cup_initial import native_checks
                from .common import write_json
                receipt = native_checks(self.reader, metadata)
                metadata['recovery_initial_acceptance'] = receipt
                write_json(Path(self.context.output)/'cup_recovery_initial_acceptance.json', receipt)

    return PreparedCoffee()

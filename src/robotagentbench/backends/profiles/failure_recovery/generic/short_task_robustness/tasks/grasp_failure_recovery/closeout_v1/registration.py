"""Task-owned lifecycle declarations, NOT a change to the public registry.

The full source registration owns instruction, budgets, normalization and
execution mode. Only its trusted runtime factory selects our frozen failed
initial state. No request-specified Python factory or source path is accepted.
"""
from dataclasses import replace
import json
from ..common import CURRENT, COFFEE_SOURCE, owned, digest, verify_sources, require_source


def registration_for(initial_file):
    initial_file = owned(initial_file)
    initial = json.loads(initial_file.read_text())
    expected = {'shelf_drink': '4-3', 'coffee_bag': '1-2'}
    if initial.get('variant') not in expected or initial.get('source_api') != expected[initial['variant']]:
        raise ValueError('Only the two procedural recovery starts are implemented here')
    if initial.get('initialization_policy') != 'exact_training_bank_original_object_sampler.v3' or \
            initial.get('preparation_accepted') is not True or \
            initial.get('generation_reads_recorded_trajectory') is not False:
        raise ValueError('Accepted trajectory-free failed initial required')
    scene = owned(initial['scene'])
    if scene.parent != initial_file.parent or digest(scene) != initial['scene_sha256']:
        raise ValueError('Frozen scene identity mismatch')
    verify_sources(initial['source_layers'])
    verify_sources(initial['code_sources'])
    initial_hash = digest(initial_file)
    if initial['variant'] == 'shelf_drink':
        from evaluation.tasks.shelf_pick import registration as source
        source_root = CURRENT
        parameters = {}
    else:
        from evaluation.tasks.coffee_machine import registration as source
        source_root = COFFEE_SOURCE
        parameters = dict(source_episode=initial['source_episode'], random_layout=False)
    source_identity = require_source(source, source_root)
    original = source.registration(initial['source_api'])
    parameters = original.resolve_parameters(parameters)

    def factory(context):
        # The initial is frozen independently of action input. Reject attempts
        # to silently rerandomize or run a different task/seed/parameter set.
        if digest(initial_file) != initial_hash or digest(scene) != initial['scene_sha256']:
            raise ValueError('Prepared recovery input changed after registration')
        verify_sources(source_identity)
        if context.task_id != original.task_id or context.seed != initial['seed'] or \
                any(context.randomization.values()) or \
                original.resolve_parameters(context.task_parameters) != parameters:
            raise ValueError('Runtime context differs from frozen recovery input')
        original.validate(context.max_steps, context.randomization)
        owned(context.output)
        from ..runtime import make_runtime
        return make_runtime(context, initial_file)

    declaration = replace(original, runtime_factory=factory)
    receipt = dict(schema='arena.recovery.task_local_registration.v1',
        public_registry_modified=False, public_runtime_registered=False,
        variant=initial['variant'], source_api=original.task_id,
        initial=str(initial_file), initial_sha256=initial_hash,
        scene_sha256=initial['scene_sha256'], seed=initial['seed'],
        task_parameters=parameters, instruction=declaration.instruction,
        max_steps=declaration.max_steps, default_steps=declaration.default_steps,
        max_inferences=declaration.max_inferences, evaluator_id=declaration.evaluator_id,
        gripper_execution_mode=declaration.gripper_execution_mode,
        source_registration=source_identity,
        inherited_fields_unchanged=all(getattr(original, field) == getattr(declaration, field)
            for field in original.__dataclass_fields__ if field != 'runtime_factory'))
    return declaration, receipt

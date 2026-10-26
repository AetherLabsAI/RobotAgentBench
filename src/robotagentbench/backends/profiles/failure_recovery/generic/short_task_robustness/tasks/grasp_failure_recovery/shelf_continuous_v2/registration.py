"""Trusted task-local original beverage registration; no shared writes."""
from dataclasses import replace
import json
from ..common import CURRENT,owned,digest,verify_sources,require_source
from .catalog import APIS,POLICY

def registration_for(path):
    from evaluation.tasks.shelf_pick import registration as source
    from evaluation.tasks.shelf_pick.targets import spec
    path=owned(path);initial=json.loads(path.read_text());api=initial['source_api']
    if api not in APIS or initial['variant']!='shelf_drink' or initial['initialization_policy']!=POLICY:raise ValueError('Existing beverage identity required')
    if initial.get('preparation_accepted') is not True or initial.get('online_ik_calls')!=0 or initial.get('runtime_training_dataset_required') is not False:raise ValueError('Frozen trajectory-free start required')
    if initial['instruction']!=spec(api).instruction or not spec(api).is_target(initial['target']):raise ValueError('Wrong product instruction or target')
    scene=owned(initial['scene']);before=digest(path);verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    if digest(scene)!=initial['scene_sha256']:raise ValueError('Frozen scene changed')
    source_identity=require_source(source,CURRENT);original=source.registration(api);parameters=original.resolve_parameters({})
    def factory(context):
        if digest(path)!=before or digest(scene)!=initial['scene_sha256']:raise ValueError('Input changed')
        if context.task_id!=api or context.seed!=initial['seed'] or any(context.randomization.values()) or original.resolve_parameters(context.task_parameters)!=parameters:raise ValueError('Wrong context')
        original.validate(context.max_steps,context.randomization);owned(context.output)
        from .runtime import make_runtime
        return make_runtime(context,path)
    registration=replace(original,runtime_factory=factory)
    return registration,dict(source_api=api,task_parameters=parameters,instruction=original.instruction,
        initial=str(path),initial_sha256=before,source_registration=source_identity,public_registry_modified=False,
        inherited_fields_unchanged=all(getattr(original,k)==getattr(registration,k) for k in original.__dataclass_fields__ if k!='runtime_factory'))


"""Trusted task-local five-way selection; not shared registry publication."""
import json
from ..common import owned,digest,verify_sources
from ..expansion_v2.catalog import definition

VARIANTS=('shelf_drink','coffee_bag','sofa','table_waste_empty_grasp','table_waste_tipped_cup')

def admit_initial(path):
    path=owned(path);initial=json.loads(path.read_text());variant=initial.get('variant')
    if variant not in VARIANTS or not initial.get('preparation_accepted') or initial.get('source_progress_inherited') is not False:
        raise ValueError('Accepted fresh five-case failed initial required')
    if initial.get('runtime_training_dataset_required') is not False or initial.get('online_ik_calls')!=0:
        raise ValueError('Frozen training-free solver-free runtime required')
    if digest(owned(initial['scene']))!=initial['scene_sha256']:raise ValueError('Prepared scene changed')
    verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    if variant in ('shelf_drink','coffee_bag') and initial['initialization_policy']!='exact_training_bank_original_object_sampler.v3':
        raise ValueError('New exact training bank required')
    if variant=='table_waste_tipped_cup' and initial.get('initialization_contract')!='authored_tipped_open.v1':
        raise ValueError('Current authored tipped/open contract required')
    spec=definition(variant)
    instruction=spec.get('instruction')
    if not instruction:instruction=initial['original_bindings']['instruction']
    return path,dict(task_name='failure_recovery_'+variant,instruction=instruction,
        variant=variant,global_registered=False,source_scoring_unchanged=True)

def owner_for(variant):
    if variant not in VARIANTS:raise ValueError('Unknown recovery variant')
    return 'atomic_owner' if variant in ('shelf_drink','coffee_bag') else 'room_owner'

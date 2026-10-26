"""Only existing beverage APIs, plan instruction always matches actual target."""
import json
from ..common import owned,digest,verify_sources
from .catalog import APIS,POLICY

def admit_initial(path):
    from evaluation.tasks.shelf_pick.targets import spec
    path=owned(path);initial=json.loads(path.read_text());api=initial.get('source_api')
    if api not in APIS or initial.get('variant')!='shelf_drink' or initial.get('initialization_policy')!=POLICY:raise ValueError('Wrong product')
    if not initial.get('preparation_accepted') or initial.get('online_ik_calls')!=0 or initial.get('runtime_training_dataset_required') is not False:raise ValueError('Frozen start required')
    if initial['instruction']!=spec(api).instruction or not spec(api).is_target(initial['target']):raise ValueError('Wrong target/prompt')
    if digest(owned(initial['scene']))!=initial['scene_sha256']:raise ValueError('Changed scene')
    verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    return path,dict(task_name='failure_recovery_shelf_drink',source_api=api,instruction=initial['instruction'],
        variant='shelf_drink',global_registered=False,source_scoring_unchanged=True)


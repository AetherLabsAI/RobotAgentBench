"""Five approved recovery tasks, full-tree trusted registration adapter v2."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from ..common import HERE,CURRENT,COFFEE_SOURCE,owned,digest,verify_sources,write_json
from ..public_adapter import verify_pins
from ..expansion_v2.catalog import base_request,definition
from .adapter import admit_initial,owner_for,VARIANTS

NAMES={('failure_recovery_'+v):v for v in VARIANTS}
RULE_VERSION='short_task_robustness.grasp_failure_recovery.frozen5.v2'
RECORD=HERE/'closeout_v1/component_sources.json'

def rule_version_for(name):
    if name=='failure_recovery_sofa':
        from .sofa_whole_table import RULE_VERSION as sofa_version
        return sofa_version
    return RULE_VERSION

def request_for(name):
    if name not in NAMES:raise ValueError('Only the five approved recoveries')
    return dict(schema='openend.short_task.request.v1',task_name=name,seed=1,rule_version=rule_version_for(name))

def resolve(request):
    name=request['task_name']
    if name not in NAMES or request['rule_version']!=rule_version_for(name):raise ValueError('Unknown recovery task/version')
    if any(request.get(k) for k in ('options','targets','source','destination','background','position_constraints')) or request.get('layout_id') is not None:
        raise ValueError('Only existing approved seeded initial samplers, no scene/goal overrides')
    variant=NAMES[name]
    if variant in ('shelf_drink','coffee_bag'):
        spec=definition(variant);api={'shelf_drink':'4-3','coffee_bag':'1-2'}[variant]
        plan=dict(available=True,instruction=spec['instruction'],score_policy=dict(maximum=spec['maximum_score'],inherited_original=True),
            source_api=api,scene_request=dict(tasks=[dict(id='recovery',type='original_atomic_recovery')]),
            runtime_components=[dict(instance_id='recovery',type='original_atomic_recovery',parameters=dict(variant=variant,source_api=api))])
    else:
        from short_tasks import registry
        original=base_request(variant,request['seed'],sofa_asset='remote_tv' if variant=='sofa' else None,
            waste_categories=None if variant=='sofa' else ['cup'])
        plan=deepcopy(registry.resolve(original))
        plan['original_task_name']=original['task_name']
        plan['recovery_original_execution']={k:deepcopy(plan[k]) for k in ('no_progress_timeout_s','physics_policy')}
    plan.update(task_name=name,rule_version=rule_version_for(name),failure_recovery_variant=variant)
    if variant=='sofa':
        from .sofa_whole_table import plan_metadata
        plan_metadata(plan)
    return plan

def verify_sources_for_plan(plan):
    if plan['task_name'] not in NAMES or plan['rule_version']!=rule_version_for(plan['task_name']):raise ValueError('Wrong recovery version')
    result=verify_pins(json.loads(RECORD.read_text()),CURRENT,HERE,COFFEE_SOURCE)
    return dict(result,source_record_sha256=digest(RECORD))

def bind(plan,manifest):
    variant=NAMES[plan['task_name']]
    if manifest.get('schema')!='arena.failure_recovery.frozen5_manifest.v2' or manifest.get('variant')!=variant or manifest.get('seed')!=plan['request']['seed']:
        raise ValueError('Prepared manifest identity mismatch')
    initial_path=owned(manifest['initial'])
    if digest(initial_path)!=manifest['initial_sha256']:raise ValueError('Frozen initial changed')
    path,_=admit_initial(initial_path);v=json.loads(path.read_text())
    if v['variant']!=variant or v['seed']!=manifest['seed'] or v['scene']!=manifest['scene'] or v['scene_sha256']!=manifest['scene_sha256']:
        raise ValueError('Frozen scene or variant changed')
    return deepcopy(plan['runtime_components'] if variant in ('shelf_drink','coffee_bag') else v['original_bindings']['runtime_components'])

def prepare_registered(request,ready_pose,sensor_file,output):
    from short_tasks import registry
    from .sampler import build
    if ready_pose is not None or sensor_file is not None:raise ValueError('Original per-task posture/cameras must be preserved')
    plan=registry.resolve(request);verify_sources_for_plan(plan);out=owned(output,fresh=True);out.mkdir(parents=True)
    write_json(out/'request.json',request);v=build(NAMES[request['task_name']],request['seed'],out/'initial')
    manifest=dict(schema='arena.failure_recovery.frozen5_manifest.v2',variant=v['variant'],seed=v['seed'],
        initial=str(out/'initial/initial.json'),initial_sha256=digest(out/'initial/initial.json'),scene=v['scene'],scene_sha256=v['scene_sha256'],
        generated=True,native_physics_validated=False,source_progress_inherited=False)
    bind(plan,manifest);write_json(out/'manifest.json',manifest)
    launch=dict(schema='openend.short_task.launch.v1',task_name=plan['task_name'],request=str(out/'request.json'),request_sha256=digest(out/'request.json'),
        manifest=str(out/'manifest.json'),manifest_sha256=digest(out/'manifest.json'),scene=v['scene'],scene_sha256=v['scene_sha256'],
        source_record_sha256=digest(RECORD),native_physics_validated=False,action_acceptance_verified=False)
    write_json(out/'launch.json',launch);return launch

def admit_launch(launch):
    from short_tasks import registry
    if launch.get('schema')!='openend.short_task.launch.v1' or launch.get('task_name') not in NAMES or launch.get('source_record_sha256')!=digest(RECORD):
        raise ValueError('Frozen5 launch version mismatch')
    qpath,mpath=owned(launch['request']),owned(launch['manifest'])
    if digest(qpath)!=launch['request_sha256'] or digest(mpath)!=launch['manifest_sha256']:raise ValueError('Launch input changed')
    request,manifest=json.loads(qpath.read_text()),json.loads(mpath.read_text());plan=registry.resolve(request)
    if request['task_name']!=launch['task_name'] or manifest['scene']!=launch['scene'] or manifest['scene_sha256']!=launch['scene_sha256']:
        raise ValueError('Launch identity differs')
    verify_sources_for_plan(plan);bind(plan,manifest);return manifest,plan

def exec_native():
    # No request-selected code. Keep original source transport arguments, select a trusted family.
    args=list(sys.argv[1:])
    if '--launch' not in args:raise ValueError('Prepared launch required')
    index=args.index('--launch');launch=json.loads(owned(args[index+1]).read_text());manifest,plan=admit_launch(launch)
    args[index:index+2]=['--initial',manifest['initial']]
    if '--coding-policy' in args:raise ValueError('This recovery version exposes the unchanged 22D action protocol only')
    if '--allow-shared-gpu' in args or '--allow-idle-utilization' in args:raise ValueError('No GPU sharing override accepted')
    if '--gpu-uuid' not in args:
        import subprocess
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
        if len(visible)!=1:raise ValueError('Exactly one isolated GPU required')
        args+=['--gpu-uuid',visible[0]]
    module=(__package__+'.room_owner_v3' if NAMES[plan['task_name']]=='sofa'
        else __package__+'.'+owner_for(NAMES[plan['task_name']])+'_v2')
    os.execv(sys.executable,[sys.executable,'-B','-u','-m',module,*args])

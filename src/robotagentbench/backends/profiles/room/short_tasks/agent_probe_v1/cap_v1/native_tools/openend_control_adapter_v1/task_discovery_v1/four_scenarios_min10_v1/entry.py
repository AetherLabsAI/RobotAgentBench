"""Closed, source-pinned selection of the four new task-owned Discovery rooms."""
from pathlib import Path
from .profiles import FAMILIES, PROMPTS
from .object_bindings import catalogue
from ..inputs import model_input, wire_instruction
from ..policy import RULE_VERSION


def admit_launch(launch, output):
    from ... import native_entry as parent
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from openend.room import validate_request, prepare_registration
    root=parent.HERE/'task_discovery_v1/four_scenarios_min10_v1'
    fields={'schema','task_name','manifest','request','ready_pose','initial_base_xyyaw',
            'sensor','prompt_id','source_hashes','bindings'}
    if set(launch)!=fields or launch['schema']!='openend.coding_room.launch.v1' or launch['task_name']!='task_discovery':
        raise ValueError('Explicit versioned Discovery launch required')
    out=parent.owned(output,exists=False)
    if not out.is_relative_to(root) or out.exists():
        raise ValueError('Fresh four-scenario output required')
    paths={k:parent.owned(launch[k]) for k in ('manifest','request','ready_pose')}
    case=paths['manifest'].parent
    if case.parent!=root/'generated_rooms' or case.name not in FAMILIES or any(p.parent!=case for p in paths.values()):
        raise ValueError('One explicitly frozen four-scenario case required')
    if launch['prompt_id']!=PROMPTS[case.name]:
        raise ValueError('Scene and original public request differ')
    if launch['bindings']!={k:parent.sha(p) for k,p in paths.items()}:
        raise ValueError('Frozen input binding changed')
    verify_scene_sources()
    verify_manifest({'source_hashes':launch['source_hashes']})
    manifest,request=parent.read(paths['manifest']),parent.read(paths['request'])
    if parent.owned(manifest['scene']).parent!=case or parent.sha(manifest['scene'])!=manifest['scene_sha256']:
        raise ValueError('Frozen physical room changed')
    verify_manifest(manifest);validate_request(request)
    binding=catalogue(request,manifest,prepare_registration(request,manifest))
    frozen=parent.read(case/'object_bindings.json')
    if frozen!=__import__('json').loads(__import__('json').dumps(binding)):
        raise ValueError('Real object binding differs from reviewed catalogue')
    public=model_input(launch['prompt_id'])
    return request,manifest,dict(task_name='task_discovery',instruction=wire_instruction(public),
        rule_version=RULE_VERSION,score_policy=dict(maximum=binding['unique_target_count'],
        metric='unique_target_object_coverage',non_target_penalty=0,operation_points_enabled=False))


def build_room(request,stage,manifest,output,*,episode_id,sensor):
    from .runtime import DiscoveryRoom
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    return DiscoveryRoom(stage,manifest,output,episode_id,request=request,sensor=sensor,
                         workspace_selector=from_scene(stage,manifest,sensor))

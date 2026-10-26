"""Trusted task-owned admission for the first person-aware discovery fixture."""
from .inputs import model_input, wire_instruction
from .policy import RULE_VERSION


def admit_launch(launch, output):
    from pathlib import Path
    own = Path(__file__).resolve().parent / 'four_scenarios_min10_v1/generated_rooms'
    if Path(launch.get('manifest', '')).parent.parent == own:
        from .four_scenarios_min10_v1.entry import admit_launch as admit_four
        return admit_four(launch, output)
    from .. import native_entry as parent
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from openend.room import validate_request, prepare_registration
    fields = {'schema','task_name','manifest','request','ready_pose','initial_base_xyyaw',
              'sensor','prompt_id','source_hashes','bindings'}
    if (set(launch) != fields or launch['schema'] != 'openend.coding_room.launch.v1'
            or launch['task_name'] != 'task_discovery' or launch['prompt_id'] != 'outing_emma_work'):
        raise ValueError('Explicit frozen person-aware discovery launch required')
    out = parent.owned(output, exists=False)
    if not out.is_relative_to(parent.HERE/'task_discovery_v1') or out.exists():
        raise ValueError('Fresh discovery-owned output required')
    paths = {k:parent.owned(launch[k]) for k in ('manifest','request','ready_pose')}
    if any(not p.is_relative_to(parent.HERE/'task_discovery_v1/fixture') for p in paths.values()):
        raise ValueError('Discovery fixture inputs required')
    if launch['bindings'] != {k:parent.sha(p) for k,p in paths.items()}:
        raise ValueError('Discovery input binding changed')
    verify_scene_sources(); verify_manifest({'source_hashes':launch['source_hashes']})
    manifest, request = parent.read(paths['manifest']), parent.read(paths['request'])
    parent.owned(manifest['scene'])
    if parent.sha(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Frozen scene changed')
    verify_manifest(manifest); validate_request(request)
    if prepare_registration(request,manifest).get('pending'):
        raise ValueError('Unbound discovery registration')
    public = model_input(launch['prompt_id'])
    return request,manifest,dict(task_name='task_discovery',
        instruction=wire_instruction(public),
        rule_version=RULE_VERSION,score_policy=dict(maximum=len(request['tasks']),metric='unique_target_object_coverage',
        non_target_penalty=0,operation_points_enabled=False))


def build_room(request,stage,manifest,output,*,episode_id,sensor):
    from pathlib import Path
    own = Path(__file__).resolve().parent / 'four_scenarios_min10_v1'
    if Path(output).is_relative_to(own):
        from .four_scenarios_min10_v1.entry import build_room as build_four
        return build_four(request,stage,manifest,output,episode_id=episode_id,sensor=sensor)
    from .runtime import DiscoveryRoom
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    return DiscoveryRoom(stage,manifest,output,episode_id,request=request,sensor=sensor,
                         workspace_selector=from_scene(stage,manifest,sensor))

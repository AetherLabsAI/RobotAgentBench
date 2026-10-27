"""Derive Discovery inputs from the same already generated physical room."""
import hashlib
import json
from pathlib import Path
import shutil

from . import INSTANCE, identity
from .prepare import write


def prepare(config, scenario, seed):
    key = identity('task_discovery', seed, scenario)
    bank = Path(config['bank'])
    case = bank/config['case_subdirectory']
    case.mkdir(parents=True, exist_ok=False)
    source = config['shared_instance']
    root = Path(config['discovery_case'])
    if any(root.iterdir()):
        raise ValueError('Fresh per-instance Discovery namespace required')
    original_launch = json.loads(Path(source['launch']).read_text())
    if hashlib.sha256(Path(source['launch']).read_bytes()).hexdigest() != source['hash']:
        raise ValueError('Shared prepared launch changed')
    manifest = json.loads(Path(original_launch['manifest']).read_text())
    request = json.loads(Path(original_launch['request']).read_text())
    if hashlib.sha256(Path(source['scene']).read_bytes()).hexdigest() != source['scene_sha256']:
        raise ValueError('Shared prepared physical room changed')
    # Exact existing Discovery clock, not the OpenEnd execution clock.
    request['no_progress_timeout_s'] = config['discovery_clock']['no_progress_timeout_s']
    request['selection_order'] = config['discovery_clock']['selection_order']
    shutil.copyfile(source['scene'], root/'room.usda')
    shutil.copyfile(original_launch['ready_pose'], root/'ready_pose.json')
    manifest['scene'] = str(root/'room.usda')
    manifest['scene_sha256'] = source['scene_sha256']
    write(root/'request.json', request)
    write(root/'composition.json', manifest)
    from openend.room import prepare_registration
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.task_discovery_v1.four_scenarios_min10_v1.object_bindings import catalogue
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.task_discovery_v1.four_scenarios_min10_v1 import entry
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.task_discovery_v1.inputs import model_input, wire_instruction
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.task_discovery_v1.four_scenarios_min10_v1.profiles import PROMPTS
    binding = catalogue(request, manifest, prepare_registration(request, manifest))
    write(root/'object_bindings.json', binding)
    launch = dict(original_launch, task_name='task_discovery',
                  manifest=str(root/'composition.json'), request=str(root/'request.json'),
                  ready_pose=str(root/'ready_pose.json'), prompt_id=PROMPTS[scenario])
    launch['bindings'] = {name: hashlib.sha256(Path(launch[name]).read_bytes()).hexdigest()
                          for name in ('manifest', 'request', 'ready_pose')}
    write(root/'launch.json', launch)
    admitted = entry.admit_launch(launch, root.parent.parent/('diagnostic_admit_'+str(seed)))
    public = model_input(launch['prompt_id'])
    write(case/'model_input.json', dict(instruction=wire_instruction(public), context=public['context']))
    write(case/'original_cpu_admission.private.json', dict(passed=True, returned_items=len(admitted),
          native_started=False, unique_target_count=binding['unique_target_count']))
    from .variation import physical_signature
    semantic = physical_signature(root/'launch.json', dict(task_name='task_discovery',
                                  rule_version=entry.RULE_VERSION))
    semantic['task_object_bindings'] = binding
    write(case/'variation.private.json', semantic)
    record = dict(schema=INSTANCE, instance_id=key, task_name='task_discovery', seed=seed,
        scenario=scenario, status='prepared_cpu_not_native', launch=str(root/'launch.json'),
        model_input=str((case/'model_input.json').relative_to(bank)), output_logical=str(root),
        original_launch_sha256=hashlib.sha256((root/'launch.json').read_bytes()).hexdigest(),
        variation_sha256=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest(),
        native_accepted=False, model_called=False, scene_bytes_changed=False,
        shared_physical_room=source, clock_source=config['discovery_clock_source'],
        physics_scoring_camera_clock_timeout_modified=False, files_sha256={})
    for path in case.rglob('*'):
        if path.is_file():record['files_sha256'][str(path.relative_to(bank))]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(case/'record.pending.json', record)
    return record

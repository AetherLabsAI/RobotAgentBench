"""Public preparation delegates scene creation to clear_table_trash, not scoring."""
from copy import deepcopy
import json
from pathlib import Path
from .prepare import HERE, read, write, sha, prepare

def admit_bundle(bundle_path):
    from short_tasks import registry
    from short_tasks.agent_probe_v1.native_owner import admit_launch
    bundle = read(bundle_path)
    directory = Path(bundle_path).resolve().parent
    selection = read(bundle['selection'])
    request = dict(schema='openend.short_task.request.v1', task_name='select_trash',
        rule_version=selection['rule_version'], seed=selection['seed'],
        layout_id=bundle['actual_layout_id'], options=dict(category=selection['category'], count=len(selection['candidates'])))
    manifest = read(bundle['manifest'])
    registry.bind(registry.resolve(request), manifest)
    # Pin every byte of the selected self-contained assets, including textures.
    for asset in selection['candidates']:
        for path in Path(asset['entry']).parent.rglob('*'):
            if path.is_file():
                manifest['source_hashes'][str(path)] = sha(path)
    manifest['runtime_registration_pending'] = False
    write(directory/'manifest_registered.json', manifest)
    write(directory/'request_registered.json', request)
    launch = dict(schema='openend.short_task.launch.v1', task_name='select_trash',
        request=str(directory/'request_registered.json'), manifest=str(directory/'manifest_registered.json'),
        ready_pose=bundle['ready_pose'], initial_base_xyyaw=bundle['initial_base_xyyaw'], sensor=bundle['sensor'],
        startup_spawn_scope='clear_table_trash_source_range_requires_native_acceptance',
        startup_physics_hz=60, control_hz=20, physics_backend='cpu',
        physics_frequency_owner='selected_workspace', preparation_receipt=str(bundle_path))
    admit_launch(launch, directory/'admission_check_not_created')
    write(directory/'launch_registered.json', launch)
    return dict(status='prepared_not_native_accepted', task_name='select_trash',
        launch=str(directory/'launch_registered.json'), native_admission_passed=True,
        native_physics_validated=False)

def prepare_registered(request, ready_pose, sensor_file, output):
    from short_tasks.agent_probe_v1.prepare_task import prepare as prepare_base
    from short_tasks import registry
    from .adapter import resolve
    plan = registry.resolve(request)
    registry.verify_sources(plan)
    out = Path(output).resolve()
    if out == HERE or not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Context preparation output must be a fresh task-local directory')
    out.mkdir(parents=True)
    base_request = dict(schema='openend.short_task.request.v1', task_name='clear_table_trash',
        rule_version='openend.short_task.clear_table_trash.v1', seed=request['seed'], options=dict(count=2))
    if request.get('layout_id') is not None:
        base_request['layout_id'] = request['layout_id']
    base = prepare_base(base_request, ready_pose, sensor_file, out/'base_trash')
    selected = plan['selection']
    prepare(base['launch'], seed=request['seed'], category=selected['category'],
            count=len(selected['candidates']), output=out/'selection')
    result = admit_bundle(out/'selection/preparation.json')
    write(out/'preparation.json', result)
    return result

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    print(json.dumps(admit_bundle(ap.parse_args().bundle)))

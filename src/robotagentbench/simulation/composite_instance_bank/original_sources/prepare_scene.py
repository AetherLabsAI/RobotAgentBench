"""Compose a new current-version OpenEnd room through the existing planner."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[4]
SCENE = SOURCE / 'openend/tasks/scene_requirements'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'api-protocol'), str(SCENE)]
from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.native_entry import POLICY, sha


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def plan(case, node):
    from room_request import resolve
    from planner_v2 import compile_request
    request = json.loads((case / 'request.json').read_text())
    resolved = resolve(request)
    layouts = json.loads((SCENE / 'placement_sources/random-layouts-v1.json').read_text())['layouts']
    attempts = []
    for index, lid in enumerate(resolved['layout_candidates']):
        resolved['layout'] = next(row for row in layouts if row['id'] == lid)
        resolved['request']['layout_id'] = lid
        resolved['requirements'] = compile_request(resolved['request'])
        path = case / ('resolved_%02d.json' % index)
        save(path, resolved)
        p = subprocess.run([node, str(SCENE / 'room_plan.cjs'), str(path)], capture_output=True, text=True)
        (case / ('planner_%02d.log' % index)).write_text(p.stderr)
        if p.returncode not in (0, 2):
            raise RuntimeError(p.stderr)
        candidate = json.loads(p.stdout)
        attempts.append({'layout_id': lid, 'status': candidate['plan']['status']})
        if p.returncode == 0:
            candidate['layout_selection'].update(selected=lid, rerolls=index, attempts=attempts)
            candidate['openend_episode_policy'] = dict(POLICY)
            # Existing ordinary Short Task sampler, not a new packing policy.
            shelf = candidate['requirements'].get('shelf_task_requirements', [])
            if shelf:
                from short_tasks.tasks.stock_wooden_tray.generation import POLICY as shelf_policy, validate_policy
                validate_policy(shelf_policy, shelf)
                candidate['task_scene_policy'] = dict(shelf_policy)
            save(case / 'input.json', candidate)
            save(case / 'layout_attempts.json', attempts)
            return
    save(case / 'layout_attempts.json', attempts)
    raise ValueError('Existing planner exhausted compatible layouts')


def finalize(case, prompt_id='tidy_room'):
    import math
    from pxr import Usd, UsdGeom
    from openend.tasks.layout_repair_v1.spawn_v2 import choose
    from openend.room import prepare_registration
    manifest = json.loads((case / 'composition.json').read_text())
    request = json.loads((case / 'runtime_request.json').read_text())
    prepare_registration(request, manifest)
    stage = Usd.Stage.Open(manifest['scene'])
    spawn = choose(stage, manifest)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
    p = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationMatrix()
    yaw = math.atan2(float(rotation[0][1]), float(rotation[0][0]))
    base = [float(p[0]) + spawn['delta'][0], float(p[1]) + spawn['delta'][1], yaw]
    reg = SOURCE / 'layout_task_activation_v1/task_scene_registration_v1'
    ready = reg / 'recorded_ready_pose_042_v1.json'
    sensor_source = reg.parent / 'hotwake_probe_inputs.json'
    save(case / 'ready_pose.json', json.loads(ready.read_text()))
    save(case / 'spawn.json', spawn)
    sources = {str(p): sha(p) for p in (ready, sensor_source,
        SOURCE / 'short_tasks/agent_probe_v1/native_owner.py',
        SOURCE / 'short_tasks/scene_sources.json')}
    # Freeze current executable task/control sources, not historical artifact trees.
    for folder in (SOURCE / 'openend', SOURCE / 'openend/tasks'):
        for p in folder.glob('*.py'):
            sources[str(p)] = sha(p)
    for folder in (SOURCE / 'openend/tasks').iterdir():
        if folder.is_dir():
            for p in folder.glob('*.py'):
                sources[str(p)] = sha(p)
    for p in (SOURCE / 'short_tasks/agent_probe_v1/cap_v1/native_tools').glob('*.py'):
        sources[str(p)] = sha(p)
    for p in HERE.glob('*.py'):
        sources[str(p)] = sha(p)
    for folder in ('scenario_alignment_v1', 'contact_credit_v1', 'positive_only_v1', 'frozen_context'):
        for p in (HERE / folder).glob('*'):
            if p.is_file() and p.suffix in ('.py', '.json', '.txt'):
                sources[str(p)] = sha(p)
    launch = dict(schema='openend.coding_room.launch.v1', task_name='openend',
        manifest=str(case / 'composition.json'), request=str(case / 'runtime_request.json'),
        ready_pose=str(case / 'ready_pose.json'), initial_base_xyyaw=base,
        sensor=json.loads(sensor_source.read_text())['sensor'], prompt_id=prompt_id,
        source_hashes=sources,
        bindings={key: sha(case / name) for key, name in (
            ('manifest', 'composition.json'), ('request', 'runtime_request.json'), ('ready_pose', 'ready_pose.json'))})
    save(case / 'launch.json', launch)
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.native_entry import admit_launch
    admit_launch(launch, HERE / ('native_admission_' + case.name))
    print(json.dumps({'status': 'new_scene_composed_and_cpu_admitted', 'scene': manifest['scene'],
        'tasks': request['tasks'], 'launch': str(case / 'launch.json'), 'source_files': len(sources)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('plan', 'compose', 'finalize'))
    ap.add_argument('case')
    ap.add_argument('--node')
    ap.add_argument('--prompt-id', default='tidy_room')
    args = ap.parse_args()
    case = HERE / 'generated_rooms' / args.case
    if case.resolve() != case or case.parent != HERE / 'generated_rooms':
        raise ValueError('Invalid case')
    if args.phase == 'plan':
        plan(case, args.node)
    elif args.phase == 'compose':
        subprocess.run([sys.executable, '-B', str(SCENE / 'build_selected_room.py'), str(case / 'input.json')], check=True)
    else:
        finalize(case, args.prompt_id)

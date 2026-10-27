"""Task-owned CPU preparation using original unchanged composite preparers."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from robotagentbench.simulation.composite_instance_bank.templates import select

HERE = Path(__file__).resolve().parent
OWNER = {'owner': 'RobotAgentBench-composite-instance-bank-v31'}
ADAPTER = 'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1'
LONG = 'short_tasks/agent_probe_v1/long_task_eval_setup_v1'
TASK_NAMES=frozenset(('long_task_15','long_task_16','long_task_17','long_task_19','long_task_20','long_task_21','long_task_22',
                      'prepare_dinner','host_guests','prepare_to_leave','tidy_room'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def guard(event, args):
    if event == 'import' and args[0] in ('isaacsim.simulation_app', 'omni.kit.app', 'omni.physx'):
        raise RuntimeError('Instance-bank preparation cannot start Isaac or PhysX')


def prepare_one(config, task, seed, attempt=1):
    template = select(config, task, seed)
    root = Path(config['root'])
    if json.loads((root/'OWNER.json').read_text()) != OWNER:
        raise ValueError('Wrong task directory owner')
    source = Path(config['source_root'])
    if root.resolve()!=root or source.resolve()!=source or not source.is_relative_to(root) or source==root:
        raise ValueError('All preparation writes must remain in this owned source view')
    site = Path(config['sdk_site'])
    usd = next((site/'isaacsim/extscache').glob('omni.usd.libs-*'))
    schema = next((site/'isaacsim/extscache').glob('omni.usd.schema.physx-*'))
    sys.dont_write_bytecode = True
    sys.addaudithook(guard)
    sys.path[:0] = [str(HERE), str(source), str(source/'api-protocol'), str(usd), str(schema), str(site)]
    os.environ['PYTHONPATH'] = os.pathsep.join(sys.path)
    identity = task + '_seed' + str(seed).zfill(3)
    case_name = identity + (('_attempt'+str(attempt).zfill(3)) if attempt>1 else '')
    report_dir = root/'records'/identity
    if attempt>1:
        report_dir=report_dir/('attempt_'+str(attempt).zfill(3))
    report_dir.mkdir(parents=True, exist_ok=False)
    row = dict(task_name=task, seed=seed, attempt=attempt, launch=None, hash=None,
               status='preparing', native_physics_validated=False, model_calls=0)
    try:
        if task == 'long_task_15':
            request = template['request']; request['seed'] = seed
            out = source/LONG/'generated_rooms'/case_name
            from short_tasks.agent_probe_v1.prepare_task import prepare
            reg = source/'layout_task_activation_v1/task_scene_registration_v1'
            result = prepare(request, reg/'recorded_ready_pose_042_v1.json',
                             reg.parent/'hotwake_probe_inputs.json', out)
            launch = Path(result['launch'])
        elif task.startswith('long_task_'):
            request = template['request']; request['parameters']['seed'] = seed
            module = importlib.import_module(LONG.replace('/', '.')+'.revision2.prepare_long')
            result = module.prepare(request, case_name)
            out = source/LONG/'generated_rooms'/case_name
            launch = Path(result['launch'])
        else:
            request = template['request']; request['seed'] = seed
            out = source/ADAPTER/'generated_rooms'/case_name
            out.mkdir(parents=True, exist_ok=False)
            save(out/'request.json', request)
            save(out/'public_input.json', template['public_input'])
            module = importlib.import_module(ADAPTER.replace('/', '.')+'.prepare_scene')
            module.plan(out, config['node'])
            composer = source/'openend/tasks/scene_requirements/build_selected_room.py'
            with (out/'compose.log').open('x') as log:
                stage = subprocess.run([sys.executable, '-B', str(composer), str(out/'input.json')],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if stage.returncode:
                raise RuntimeError('Original OpenEnd composer rejected; see compose.log')
            module.finalize(out, template['prompt_id'])
            launch = out/'launch.json'
            # The legacy prompt_id remains valid for native admission. The
            # public binding is authoritative for this newly authored cohort.
            save(out/'public_input_binding.json', dict(
                schema='robotagentbench.public_input_binding.v1',
                source=str(out/'public_input.json'), sha256=sha(out/'public_input.json'),
                launch_sha256=sha(launch)))
            if 'context_variant' in template:
                save(out/'context_variant.json', template['context_variant'])
            result = {'status': 'prepared_cpu_admitted'}
        payload = json.loads(launch.read_text())
        manifest = Path(payload['manifest'])
        manifest_value=json.loads(manifest.read_text())
        scene = Path(manifest_value['scene'])
        geometry_keys=('id','path','visible_bounds_world_m','polygon','placement','support','position_m','matrix','world_transform')
        geometry=dict(layout_id=manifest_value['layout_id'],initial_base_xyyaw=payload['initial_base_xyyaw'],
            objects=[{key:obj[key]for key in geometry_keys if key in obj}for obj in manifest_value.get('objects',[])])
        geometry_sha=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        # Complete saved input artifacts, not seed-only copies. Each generated
        # recipe keeps the original placement choices and initial spawn.
        inputs = {'launch': launch, 'manifest': manifest, 'scene': scene}
        for name in ('request', 'ready_pose', 'spawn'):
            if name in payload:
                inputs[name] = Path(payload[name])
        for name in ('input.json', 'public_input.json', 'public_input_binding.json',
                     'context_variant.json', 'spawn.json', 'long_request.json'):
            if (out/name).is_file():
                inputs[name] = out/name
        files = {key: {'path': str(path), 'sha256': sha(path)} for key, path in inputs.items()}
        data = dict(schema='robotagentbench.composite_instance.v31', task_name=task,
                    seed=seed, launch=str(launch), actual_runtime_task=payload['task_name'],
                    status='actual_scene_cpu_admitted', files=files,
                    geometry_sha256=geometry_sha,
                    randomization_source='original_seeded_planner_and_composer',
                    scoring_or_physics_modified=False, native_physics_validated=False,
                    model_calls=0, preparation_result=result)
        instance = report_dir/'instance.json'
        save(instance, data)
        from bank import load
        load(instance)
        row.update(launch=str(launch), hash=sha(launch), instance=str(instance),
                   instance_hash=sha(instance), status=data['status'],
                   scene=str(scene), scene_sha256=files['scene']['sha256'],
                   geometry_sha256=geometry_sha,
                   recipe_sha256=files.get('input.json', {}).get('sha256'),
                   read_only_reload_passed=True)
    except BaseException as error:
        row.update(status='blocked', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    save(report_dir/'receipt.json', row)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.task not in TASK_NAMES or args.task not in config['templates'] or not 1 <= args.seed <= 30:
        raise ValueError('This bank contains exactly seeds 1..30 for the selected tasks')
    if not 1<=args.attempt<=8:raise ValueError('Only explicit diagnosed resource-gap corrections are admitted')
    row = prepare_one(config, args.task, args.seed,args.attempt)
    return 0 if row['status'] == 'actual_scene_cpu_admitted' else 2


if __name__ == '__main__':
    raise SystemExit(main())

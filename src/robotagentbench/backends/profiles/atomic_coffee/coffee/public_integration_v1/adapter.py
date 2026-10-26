"""CPU-only public admission. Native imports are isolated in the worker.

The public action contract is the existing robot22 contract. These tasks retain
their atomic source scene and scorer; they are NOT complete prepare_coffee.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .contact_award import VERSION

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source')
SOURCE = Path('/home/lingjun/robot_agent_arena/api-protocol-st18-20261013-frozen4')
BUTTON = Path('/mnt/agent-bench-data/lingjun/short_task_robustness/tasks/coffee_button_lighting_interference')
MACHINE = HERE.parent
BAG = ROOT / 'short_task_robustness/tasks/coffee_bag_interference'
NAMES = {
    'coffee_bag_interference': ('1-2', 'bag_only', BAG),
    'coffee_machine_push_interference': ('1-2', 'machine_only', MACHINE),
    'coffee_bag_then_machine_interference': ('1-2', 'bag_then_machine', MACHINE),
    'coffee_button_lighting_interference': ('1-3', 'button_lighting', BUTTON),
}
PACKAGE = 'short_task_robustness.tasks.coffee_machine_push_interference.public_integration_v1'


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rule(name):
    return 'openend.short_task.' + name + '.v1'


def request_for(name):
    if name not in NAMES:
        raise ValueError('Unknown atomic robustness task')
    return dict(schema='openend.short_task.request.v1', task_name=name,
                rule_version=rule(name), seed=2, options={})


def resolve(request):
    name = request['task_name']
    api, variant, _ = NAMES[name]
    if request['rule_version'] != rule(name):
        raise ValueError('Explicit atomic robustness rule version required')
    if request.get('layout_id') is not None:
        raise ValueError('This variant uses the original atomic scene, not a numbered room layout')
    if any(request.get(k) for k in ('targets', 'source', 'destination', 'background', 'position_constraints', 'options')):
        raise ValueError('No unimplemented semantic or parameter overrides are accepted')
    maximum, seconds = (5, 67) if api == '1-3' else (6, 93)
    instruction = ('Press the coffee machine button to start dispensing coffee, then press it again to stop after collecting 25–60 ml.'
                   if api == '1-3' else
                   'Pick up the coffee bean bag, pour the beans into the machine hopper, then return the bag upright to the counter.')
    return dict(task_name=name, rule_version=rule(name), available=True,
        execution_mode='atomic_coffee_robustness_v1', layout_scope='atomic_source',
        source_api=api, variant=variant, instruction=instruction,
        score_policy=dict(maximum=maximum+1, source='original_atomic_process_plus_target_contact_v2', time_limit_s=seconds),
        scoring_revision=VERSION,
        runtime_components=[dict(instance_id=name, type='atomic_coffee', parameters=dict(api=api, variant=variant))],
        native_validation='public_owner_not_yet_native_accepted', full_actions_validated=False)


def owned(name, path, *, fresh=False):
    root = NAMES[name][2].resolve(strict=True)
    # Data-disk integration artifacts are segregated per task; original source
    # initializer guards are unchanged. Avoid filling the system disk with Kit cache.
    native_root = (HERE/'native_runs'/name).resolve()
    out = Path(path).resolve()
    inside = ((out != root and out.is_relative_to(root)) or
              (name != 'coffee_button_lighting_interference' and out != native_root and out.is_relative_to(native_root)))
    if not inside or (fresh and out.exists()):
        raise ValueError('Fresh output inside the selected robustness task is required')
    return out


def verify_sources(plan):
    record = json.loads((HERE / 'component_sources.json').read_text())
    for path, expected in record['absolute_sources'].items():
        if digest(path) != expected:
            raise ValueError('Pinned integration dependency changed: ' + path)
    return dict(verified=True, file_count=len(record['absolute_sources']),
                source_record_sha256=digest(HERE / 'component_sources.json'))


def bind(plan, manifest):
    if (manifest.get('schema') != 'short_tasks.atomic_coffee.scene.v1'
            or manifest.get('task_name') != plan['task_name']
            or manifest.get('seed') != plan['request']['seed']
            or manifest.get('source_api') != plan['source_api']
            or manifest.get('variant') != plan['variant']
            or manifest.get('layout_scope') != 'atomic_source'
            or manifest.get('layout_id') is not None):
        raise ValueError('Atomic robustness scene identity mismatch')
    path = owned(plan['task_name'], manifest['initial_file'])
    if digest(path) != manifest['initial_sha256']:
        raise ValueError('Prepared initial manifest changed')
    initial = json.loads(path.read_text())
    scene = owned(plan['task_name'], initial['scene'])
    if str(scene) != manifest['scene'] or digest(scene) != manifest['scene_sha256']:
        raise ValueError('Prepared scene changed')
    if initial['seed'] != manifest['seed']:
        raise ValueError('Initial seed mismatch')
    for source, sha in initial['source_layers'].items():
        if digest(source) != sha:
            raise ValueError('Initial source layer changed: ' + source)
    result = deepcopy(plan)
    result['actual_layout_id'] = None
    result['initial_file'] = str(path)
    return result


def admit_launch(launch, output):
    from short_tasks import registry
    request = json.loads(Path(launch['request']).read_text())
    manifest = json.loads(Path(launch['manifest']).read_text())
    plan = registry.resolve(request)
    if (launch.get('schema') != 'openend.short_task.launch.v1'
            or launch.get('task_name') != plan['task_name']
            or launch.get('scene') != manifest['scene']
            or launch.get('physics_backend') != 'cpu'
            or launch.get('physics_hz') != 240 or launch.get('control_hz') != 20):
        raise ValueError('Atomic launch contract mismatch')
    owned(plan['task_name'], output, fresh=True)
    registry.verify_sources(plan)
    return request, manifest, registry.bind(plan, manifest)


def environment(*, cpu=False):
    env = dict(os.environ)
    paths = [str(BUTTON.parent), str(SOURCE), str(ROOT)]
    if cpu:
        libs = Path('/home/lingjun/robot_agent_arena/.venv/lib/python3.11/site-packages/isaacsim/extscache')
        usd = libs / 'omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311'
        physx = libs / 'omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353'
        paths += [str(ROOT / 'openend/tasks/scene_requirements/cpu_dependencies'), str(usd), str(physx)]
        env['LD_LIBRARY_PATH'] = ':'.join([str(usd/'bin'), str(physx/'bin'), env.get('LD_LIBRARY_PATH', '')])
        env['PXR_PLUGINPATH_NAME'] = ':'.join(map(str, [usd/'resources', physx/'plugins/PhysxSchema/resources',
            physx/'plugins/PhysxSchemaAddition/resources', physx/'plugins/OmniUsdPhysicsDeformableSchema/resources']))
    env.update(PYTHONPATH=':'.join(paths), PYTHONDONTWRITEBYTECODE='1')
    return env


def prepare_registered(request, ready_pose, sensor_file, output):
    # These atomic variants use their source reset/calibration, not generic room overrides.
    if ready_pose is not None or sensor_file is not None:
        raise ValueError('Atomic source reset and cameras are owned by this task; omit generic pose/sensor inputs')
    owned(request['task_name'], output, fresh=True)
    result = subprocess.run([sys.executable, '-B', '-m', PACKAGE+'.prepare_worker',
        '--request-json', json.dumps(request), '--output', str(output)], env=environment(cpu=True),
        cwd=HERE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('Atomic preparation failed (no scene substitution): '+result.stderr[-4000:])
    return json.loads(result.stdout)


def exec_native():
    # -m places cwd before PYTHONPATH. Never let the generic deployment cwd
    # shadow the frozen atomic source, even when launched from that deployment.
    os.chdir(HERE)
    os.execve(sys.executable, [sys.executable, '-B', '-m', PACKAGE+'.native_worker', *sys.argv[1:]], environment())

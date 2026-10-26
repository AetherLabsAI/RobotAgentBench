"""Trusted Long Task admission through the existing native robot22 owner."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
SOURCE = TASK.parents[2]
POLICY = {'schema': 'openend.task_owned_generation.v1', 'owner': 'long_task_eval_setup_v1'}
RUNTIME_IDS = frozenset('long_task_'+str(i) for i in range(15, 23))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned(path, *, exists=True):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or not path.is_relative_to(TASK) or path == TASK):
        raise ValueError('Canonical Long Task owned path required')
    if exists and not path.is_file():
        raise ValueError('Missing Long Task input: '+str(path))
    return path


def compile_input(request):
    from .long_registry import compile_task
    if (not isinstance(request, dict) or set(request) != {'document_id', 'parameters'}
            or request['document_id'] not in {str(i) for i in range(15, 23)}
            or not isinstance(request['parameters'], dict)):
        raise ValueError('Explicit Long document identity and parameters required')
    result = compile_task(request['document_id'], request['parameters'])
    if not result['available']:
        raise ValueError('Long Task unavailable: '+str(result.get('unavailable', result.get('pending'))))
    return result


def prepare_recipe(candidate, spec):
    """Reuse the existing Short preparer's support and fryer initialization."""
    from short_tasks.agent_probe_v1.prepare_workspace import bind_supports
    candidate, receipts = bind_supports(candidate)
    if spec['document_id'] == '21':
        candidate['short_task_fryer_initial'] = dict(
            schema='short_tasks.fryer_initial.v1', variant=spec['parameters']['bread_variant'])
    candidate['openend_episode_policy'] = dict(POLICY)
    shelf = candidate['requirements'].get('shelf_task_requirements', [])
    if shelf:
        from short_tasks.tasks.stock_wooden_tray.generation import POLICY as shelf_policy, validate_policy
        validate_policy(shelf_policy, shelf)
        candidate['task_scene_policy'] = dict(shelf_policy)
    return candidate, receipts


def validate_generation_input(path, policy):
    path = owned(path)
    if (policy != POLICY or path.name != 'input.json'
            or path.parent.parent != TASK/'generated_rooms'):
        raise ValueError('Only explicit Long Task generated-room input is admitted')
    data = read(path)
    spec = compile_input(read(owned(path.parent/'long_request.json')))
    expected = dict(spec['scene_request'])
    # The existing seeded planner owns the selected layout, not the model.
    from short_tasks.scene import _scene_imports, _candidate
    with _scene_imports():
        from room_request import resolve
        resolved = resolve(expected)
    actual = data['request']
    expected['layout_id'] = actual.get('layout_id')
    if actual != expected or actual['layout_id'] not in resolved['layout_candidates']:
        raise ValueError('Generated recipe differs from the compiled Long Task')
    index = resolved['layout_candidates'].index(actual['layout_id'])
    candidate = _candidate(spec['scene_request'], resolved, actual['layout_id'], index)
    candidate, _ = prepare_recipe(candidate, spec)
    if data != candidate:
        raise ValueError('Layout, requirements or placements differ from the original seeded planner')
    return path


def admit_launch(launch, output):
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from openend.room import prepare_registration
    fields = {'schema', 'task_name', 'document_id', 'manifest', 'request', 'ready_pose', 'spawn',
              'initial_base_xyyaw', 'sensor', 'source_hashes', 'bindings'}
    if (not isinstance(launch, dict) or set(launch) != fields
            or launch['schema'] != 'openend.coding_room.launch.v1'
            or launch['task_name'] not in RUNTIME_IDS):
        raise ValueError('Explicit trusted Long Task launch required')
    out = owned(output, exists=False)
    if out.exists():
        raise ValueError('Fresh native output required')
    paths = {key: owned(launch[key]) for key in ('manifest', 'request', 'ready_pose', 'spawn')}
    if launch['bindings'] != {key: sha(path) for key, path in paths.items()}:
        raise ValueError('Long launch inputs changed')
    request = read(paths['request'])
    spec = compile_input(request)
    if (launch['document_id'] != request['document_id']
            or launch['task_name'] != spec['runtime_id']):
        raise ValueError('Long Task identity mismatch')
    manifest = read(paths['manifest'])
    from .long_registry import bind_compiled
    spec = bind_compiled(spec, manifest)
    owned(manifest['scene'])
    if sha(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Prepared scene changed')
    if manifest.get('long_task_request') != request:
        raise ValueError('Scene is bound to another Long Task instance')
    reg = SOURCE/'layout_task_activation_v1/task_scene_registration_v1'
    if read(paths['ready_pose']) != read(reg/'recorded_ready_pose_042_v1.json'):
        raise ValueError('Source ready pose changed')
    if launch['sensor'] != read(reg.parent/'hotwake_probe_inputs.json')['sensor']:
        raise ValueError('Existing calibrated sensor required')
    spawn = read(paths['spawn'])
    if (spawn.get('scene_sha256') != manifest['scene_sha256']
            or spawn.get('initial_base_xyyaw') != launch['initial_base_xyyaw']):
        raise ValueError('Launch base differs from bound original spawn receipt')
    verify_scene_sources()
    if any(launch['source_hashes'].get(path) != digest
           for path, digest in spec['source_hashes'].items()):
        raise ValueError('Launch omits or changes a required Long source pin')
    verify_manifest({'source_hashes': launch['source_hashes']})
    verify_manifest(manifest)
    registration = prepare_registration(spec['runtime_request'], manifest)
    if registration.get('pending'):
        raise ValueError('Unbound Long Task registration: '+str(registration['pending']))
    return spec['runtime_request'], manifest, dict(task_name=spec['runtime_id'],
        instruction=spec['instruction'], rule_version=spec['rule_version'],
        score_policy=spec['score_policy'])


def build_room(request, stage, manifest, output, *, episode_id, sensor, evaluation_mode):
    from .runtime import build_room as build
    from .long_registry import bind_compiled
    spec = compile_input(manifest['long_task_request'])
    spec = bind_compiled(spec, manifest)
    if evaluation_mode != spec['runtime_id'] or request != spec['runtime_request']:
        raise ValueError('Native request differs from admitted Long Task')
    room = build(request, stage, manifest, output, episode_id=episode_id,
                 sensor=sensor, evaluation_mode=evaluation_mode, compiled=spec)
    room._long_compiled = spec
    return room


def accepted_session(room, *, episode_id, instruction):
    from .runtime import accepted_session as accept
    spec = room._long_compiled
    if instruction != spec['instruction']:
        raise ValueError('Public Long instruction changed')
    return accept(room, episode_id=episode_id, instruction=instruction)

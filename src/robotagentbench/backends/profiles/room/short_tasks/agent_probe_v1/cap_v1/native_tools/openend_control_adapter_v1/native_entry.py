"""Trusted OpenEnd factory; positive-only scores, unchanged physical controls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[4]
POLICY = {'schema': 'openend.task_owned_generation.v1', 'owner': 'openend_control_adapter_v1'}
LONG_RUNTIME_IDS = frozenset('long_task_'+str(i) for i in range(15, 23))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned(path, *, exists=True):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(HERE) or path == HERE:
        raise ValueError('OpenEnd inputs/outputs must remain inside this real task directory')
    if exists and not path.is_file():
        raise ValueError('Missing task-owned input: ' + str(path))
    return path


def validate_generation_input(path, policy):
    if policy == {'schema': 'openend.task_owned_generation.v1', 'owner': 'long_task_eval_setup_v1'}:
        from short_tasks.agent_probe_v1.long_task_eval_setup_v1.revision2.entry import validate_generation_input as validate_long
        return validate_long(path, policy)
    path = owned(path)
    if (policy != POLICY or path.name != 'input.json' or path.parent.parent not in (HERE / 'generated_rooms', HERE / 'task_discovery_v1/four_scenarios_min10_v1/generated_rooms')):
        raise ValueError('Only fresh OpenEnd task-owned generation is admitted')
    return path


def admit_launch(launch, output):
    if launch.get('task_name') in LONG_RUNTIME_IDS:
        from short_tasks.agent_probe_v1.long_task_eval_setup_v1.revision2.entry import admit_launch as admit_long
        return admit_long(launch, output)
    if launch.get('task_name') == 'task_discovery':
        from .task_discovery_v1.entry import admit_launch as admit_discovery
        return admit_discovery(launch, output)
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from openend.room import validate_request, prepare_registration
    from .frozen_context.bundle import model_input
    fields = {'schema', 'task_name', 'manifest', 'request', 'ready_pose',
              'initial_base_xyyaw', 'sensor', 'prompt_id', 'source_hashes', 'bindings'}
    if set(launch) != fields or launch['schema'] != 'openend.coding_room.launch.v1' or launch['task_name'] != 'openend':
        raise ValueError('Explicit trusted OpenEnd launch required')
    # New scoring rules require a fresh, explicitly pinned OpenEnd launch.
    # Never silently relabel an existing frozen cohort with the new protocol.
    if not all((HERE / 'positive_only_v1' / name).is_file()
               for name in ('__init__.py', 'policy.py', 'runtime.py')):
        raise ValueError('Missing OpenEnd positive-only scoring implementation')
    for source in (HERE / 'positive_only_v1').glob('*.py'):
        if launch['source_hashes'].get(str(source)) != sha(source):
            raise ValueError('Fresh OpenEnd positive-only scoring source pins required')
    out = owned(output, exists=False)
    if out.exists():
        raise ValueError('Native output must be fresh')
    paths = {key: owned(launch[key]) for key in ('manifest', 'request', 'ready_pose')}
    if launch['bindings'] != {key: sha(value) for key, value in paths.items()}:
        raise ValueError('Launch input binding changed')
    verify_scene_sources()
    verify_manifest({'source_hashes': launch['source_hashes']})
    manifest, request = read(paths['manifest']), read(paths['request'])
    owned(manifest['scene'])
    if sha(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Prepared scene changed')
    verify_manifest(manifest)
    validate_request(request)
    registration = prepare_registration(request, manifest)
    if registration.get('pending'):
        raise ValueError('Unbound task registration')
    public = model_input(launch['prompt_id'])
    from .positive_only_v1.policy import RULE_VERSION
    return request, manifest, {'task_name': 'openend', 'instruction': public['prompt'],
        'rule_version': RULE_VERSION, 'score_policy': {'maximum': None,
        'additional_contact_points_per_registered_instance': 1,
        'proximity_points_enabled': False,
        'original_stage_points_removed': 0,
        'numerical_deductions_enabled': False}}


def build_room(request, stage, manifest, output, *, episode_id, sensor, evaluation_mode='openend'):
    if evaluation_mode in LONG_RUNTIME_IDS:
        from short_tasks.agent_probe_v1.long_task_eval_setup_v1.revision2.entry import build_room as build_long
        return build_long(request, stage, manifest, output, episode_id=episode_id,
                          sensor=sensor, evaluation_mode=evaluation_mode)
    if evaluation_mode == 'task_discovery':
        from .task_discovery_v1.entry import build_room as build_discovery
        return build_discovery(request, stage, manifest, output, episode_id=episode_id, sensor=sensor)
    if evaluation_mode != 'openend':
        raise ValueError('Unknown trusted room evaluation mode')
    from .scenario_alignment_v1.runtime import prepare_fridge
    from .positive_only_v1.runtime import PositiveOnlyRoom
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    manifest = prepare_fridge(stage, manifest, request, output)
    return PositiveOnlyRoom(stage, manifest, output, episode_id, request=request,
        sensor=sensor, workspace_selector=from_scene(stage, manifest, sensor))


def accepted_session(room, *, episode_id, instruction):
    if getattr(room, '_long_compiled', None) is not None:
        from short_tasks.agent_probe_v1.long_task_eval_setup_v1.revision2.entry import accepted_session as accept_long
        return accept_long(room, episode_id=episode_id, instruction=instruction)
    from .task_discovery_v1.runtime import DiscoveryRoom, DiscoverySession
    if isinstance(room, DiscoveryRoom):
        return DiscoverySession(room, episode_id=episode_id, instruction=instruction)
    from .session_adapter import OpenEndControlSession
    return OpenEndControlSession(room, episode_id=episode_id, instruction=instruction)

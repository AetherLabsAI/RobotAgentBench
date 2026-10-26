"""Existing ShortTask task-local extension, fresh continuous shelf initials.

Only robot joints are banked. Resolve selects the same beverage as generation;
the source's Chinese instruction remains provenance, the public model receives
the framework's English product label. No model-specific prompt translation.
"""
from copy import deepcopy
import json
import os
import sys
from ..common import HERE, CURRENT, owned, digest, verify_sources, write_json
from .catalog import APIS, POLICY, generator

NAME = 'failure_recovery_shelf_drink'
VERSION = 'short_task_robustness.grasp_failure_recovery.shelf_continuous.v2'


def request_for(name):
    if name != NAME:
        raise ValueError('Shelf recovery only')
    return dict(schema='openend.short_task.request.v1', task_name=NAME,
                seed=1, rule_version=VERSION)


def resolve(request):
    from evaluation.tasks.shelf_pick.targets import spec
    from short_tasks.tasks.stock_wooden_tray.adapter import product_label_en
    if request.get('task_name') != NAME or request.get('rule_version') != VERSION:
        raise ValueError('Continuous shelf recovery requires its new initialization version')
    if any(request.get(k) for k in ('options', 'targets', 'source', 'destination',
                                   'background', 'position_constraints')) or request.get('layout_id') is not None:
        raise ValueError('Only the approved shelf sampler is supported; no scene/goal overrides')
    api = generator(request['seed'], 'product').choice(APIS)
    product = spec(api).target.rsplit('/', 1)[-1]
    return dict(task_name=NAME, available=True, rule_version=VERSION,
        instruction='Pick up the '+product_label_en(product)+' from the shelf and hold it steadily.',
        source_api=api, failure_recovery_variant='shelf_drink', initialization_policy=POLICY,
        score_policy=dict(maximum=4, inherited_original=True),
        scene_request=dict(tasks=[dict(id='recovery', type='original_atomic_recovery')]),
        runtime_components=[dict(instance_id='recovery', type='original_atomic_recovery',
            parameters=dict(variant='shelf_drink', source_api=api))])


def verify_sources_for_plan(plan):
    from ..public_adapter import verify_pins
    if plan.get('task_name') != NAME or plan.get('rule_version') != VERSION:
        raise ValueError('Wrong continuous shelf identity')
    path = HERE/'public_component_sources.json'
    result = verify_pins(json.loads(path.read_text()), CURRENT, HERE)
    return dict(result, source_record_sha256=digest(path))


def bind(plan, manifest):
    from .adapter import admit_initial
    expected = dict(schema='arena.failure_recovery.shelf_continuous_manifest.v2',
        task_name=NAME, variant='shelf_drink', source_api=plan['source_api'],
        seed=plan['request']['seed'], initialization_policy=POLICY)
    if any(manifest.get(k) != value for k, value in expected.items()):
        raise ValueError('Continuous shelf manifest differs from request')
    initial_file = owned(manifest['initial'])
    if digest(initial_file) != manifest['initial_sha256']:
        raise ValueError('Initial manifest changed')
    admit_initial(initial_file)
    initial = json.loads(initial_file.read_text())
    if (initial['source_api'] != plan['source_api'] or initial['seed'] != expected['seed']
            or initial['scene'] != manifest['scene']
            or initial['scene_sha256'] != manifest['scene_sha256']
            or digest(owned(manifest['scene'])) != manifest['scene_sha256']
            or not initial['joint_coordinates_frozen'] or initial['whole_scene_frozen']
            or not initial['target_xy_continuously_sampled'] or not initial['backgrounds_freshly_sampled']):
        raise ValueError('Scene/product/continuous generation contract mismatch')
    return deepcopy(plan['runtime_components'])


def prepare_registered(request, ready_pose, sensor_file, output):
    from short_tasks import registry
    from .generate_v3 import build, dependency_guard, no_optimizer
    if ready_pose is not None or sensor_file is not None:
        raise ValueError('Original reset/cameras required, not generic ready-pose overrides')
    plan = registry.resolve(request)
    verify_sources_for_plan(plan)
    out = owned(output, fresh=True)
    out.mkdir(parents=True)
    write_json(out/'request.json', request)
    # Preserve rejected requests; never change product/seed/profile on failure.
    sys.addaudithook(dependency_guard)
    if sys.getprofile() is not None:
        raise ValueError('Unexpected active profiler before initialization audit')
    sys.setprofile(no_optimizer)
    try:
        initial = build(request['seed'], out/'initial')
    finally:
        sys.setprofile(None)
    manifest = dict(schema='arena.failure_recovery.shelf_continuous_manifest.v2',
        task_name=NAME, variant='shelf_drink', source_api=initial['source_api'],
        seed=request['seed'], initialization_policy=POLICY,
        initial=str(out/'initial/initial.json'), initial_sha256=digest(out/'initial/initial.json'),
        scene=initial['scene'], scene_sha256=initial['scene_sha256'],
        generated=True, native_physics_validated=False, source_progress_inherited=False,
        source_is_recorded_failure=False, optimizer_calls=0, trajectory_reads=0)
    registry.bind(plan, manifest)
    write_json(out/'manifest.json', manifest)
    launch = dict(schema='openend.short_task.launch.v1', task_name=NAME,
        request=str(out/'request.json'), request_sha256=digest(out/'request.json'),
        manifest=str(out/'manifest.json'), manifest_sha256=digest(out/'manifest.json'),
        scene=initial['scene'], scene_sha256=initial['scene_sha256'],
        source_record_sha256=digest(HERE/'public_component_sources.json'),
        native_physics_validated=False, action_acceptance_verified=False)
    write_json(out/'launch.json', launch)
    return launch


def admit_launch(launch, output):
    from short_tasks import registry
    owned(output, fresh=True)
    if launch.get('schema') != 'openend.short_task.launch.v1' or launch.get('task_name') != NAME:
        raise ValueError('Trusted shelf launch required')
    if launch.get('source_record_sha256') != digest(HERE/'public_component_sources.json'):
        raise ValueError('Implementation differs from prepared launch; regenerate explicitly')
    request_path, manifest_path = owned(launch['request']), owned(launch['manifest'])
    if digest(request_path) != launch['request_sha256'] or digest(manifest_path) != launch['manifest_sha256']:
        raise ValueError('Launch inputs changed')
    request, manifest = json.loads(request_path.read_text()), json.loads(manifest_path.read_text())
    if request['task_name'] != NAME or manifest['scene'] != launch['scene'] or manifest['scene_sha256'] != launch['scene_sha256']:
        raise ValueError('Task/scene identity mismatch')
    plan = registry.resolve(request)
    verify_sources_for_plan(plan)
    return request, manifest, registry.bind(plan, manifest)


def exec_native():
    os.execv(sys.executable, [sys.executable, '-B', '-u', '-m',
        __package__+'.public_owner', *sys.argv[1:]])

"""Prepare a compiled Long instance with the unchanged room composer/spawn."""
from pathlib import Path
import json
import math
import subprocess
import sys

from .entry import HERE, TASK, SOURCE, owned, read, sha, compile_input, prepare_recipe


def save(path, value):
    path = owned(path, exists=False)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare(request, case_name):
    from short_tasks.scene import _candidate, _scene_imports, verify_scene_sources, verify_manifest
    from .long_registry import bind_compiled
    spec = compile_input(request)
    case = TASK/'generated_rooms'/case_name
    if (not isinstance(case_name, str) or not case_name
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_name)):
        raise ValueError('Simple unique case name required')
    owned(case, exists=False)
    case.mkdir(parents=True, exist_ok=False)
    save(case/'long_request.json', request)
    save(case/'compiled.json', spec)
    verify_scene_sources()
    verify_manifest({'source_hashes': spec['source_hashes']})
    with _scene_imports():
        from room_request import resolve
        resolved = resolve(spec['scene_request'])
    attempts = []
    for index, layout in enumerate(resolved['layout_candidates']):
        candidate = _candidate(spec['scene_request'], resolved, layout, index)
        attempts.append(dict(layout_id=layout, status=candidate['plan']['status']))
        if candidate['plan']['status'] == 'planar_candidate':
            candidate, receipts = prepare_recipe(candidate, spec)
            save(case/'support_bindings.json', receipts)
            save(case/'input.json', candidate)
            break
    else:
        save(case/'layout_attempts.json', attempts)
        raise ValueError('Original scene planner exhausted selected layouts')
    save(case/'layout_attempts.json', attempts)
    composer = SOURCE/'openend/tasks/scene_requirements/build_selected_room.py'
    command = [sys.executable, '-B', str(composer), str(case/'input.json')]
    save(case/'compose_command.json', command)
    with (case/'compose.log').open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if process.returncode:
        raise RuntimeError('Original room composer failed; see '+str(case/'compose.log'))
    manifest = read(case/'composition.json')
    # Keep the generator's own original output and add an independently bound
    # Long manifest, without overwriting historical/source artifacts.
    manifest['long_task_request'] = request
    requested_runtime = spec['runtime_request']
    spec = bind_compiled(spec, manifest)
    generated = read(case/'runtime_request.json')
    if generated != spec['runtime_request']:
        # A compiler may select an existing explicit variant that the room
        # generator's default runtime list does not carry. Require exact scene
        # identities; only trusted compile/bind selects its approved parameters.
        actual = {(t['instance_id'], t['type']) for t in generated['tasks']}
        expected = {(t['instance_id'], t['type']) for t in requested_runtime['tasks']}
        if actual != expected:
            raise ValueError('Generated runtime identities differ from Long instance')
        if any(generated[key] != spec['runtime_request'][key]
               for key in ('no_progress_timeout_s', 'selection_order')):
            raise ValueError('Original room generator and current profile disagree')
    save(case/'bound_runtime_request.json', spec['runtime_request'])
    from short_tasks.agent_probe_v1.prepare_task import localize
    localized = case/'localized'
    localized.mkdir(exist_ok=False)
    manifest = localize(manifest, localized)
    from pxr import Usd, UsdGeom
    from openend.tasks.layout_repair_v1.spawn_v2 import choose
    from openend.room import prepare_registration
    from short_tasks.tools.prepare_scene import checked_dependencies
    stage = Usd.Stage.Open(manifest['scene'])
    if stage is None:
        raise ValueError('Generated USD cannot open')
    from .runtime import prepare_scene
    effective = prepare_scene(spec, stage, manifest, case/'cpu_validation')
    prepare_registration(spec['runtime_request'], effective)
    layers, assets, special = checked_dependencies(Path(manifest['scene']))
    source_hashes = dict(spec['source_hashes'])
    source_hashes.update(effective.get('source_hashes', {}))
    source_hashes.update({str(p): sha(p) for p in [Path(l.realPath) for l in layers if l.realPath]+list(map(Path, assets))})
    spawn = choose(stage, effective)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
    position = transform.ExtractTranslation()
    rotation = transform.ExtractRotationMatrix()
    base = [float(position[0])+spawn['delta'][0], float(position[1])+spawn['delta'][1],
            math.atan2(float(rotation[0][1]), float(rotation[0][0]))]
    reg = SOURCE/'layout_task_activation_v1/task_scene_registration_v1'
    ready = reg/'recorded_ready_pose_042_v1.json'
    sensor = reg.parent/'hotwake_probe_inputs.json'
    save(case/'ready_pose.json', read(ready))
    save(case/'spawn.json', dict(original_spawn=spawn,
        scene_sha256=manifest['scene_sha256'], initial_base_xyyaw=base))
    save(case/'long_manifest.json', manifest)
    for path in (ready, sensor, composer, SOURCE/'short_tasks/scene_sources.json',
                 SOURCE/'short_tasks/agent_probe_v1/native_owner.py',
                 SOURCE/'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/native_entry.py'):
        source_hashes[str(path)] = sha(path)
    for path in HERE.glob('*.py'):
        source_hashes[str(path)] = sha(path)
    launch = dict(schema='openend.coding_room.launch.v1', task_name=spec['runtime_id'],
        document_id=request['document_id'], manifest=str(case/'long_manifest.json'),
        request=str(case/'long_request.json'), ready_pose=str(case/'ready_pose.json'),
        spawn=str(case/'spawn.json'),
        initial_base_xyyaw=base, sensor=read(sensor)['sensor'], source_hashes=source_hashes,
        bindings={key: sha(case/name) for key, name in (
            ('manifest','long_manifest.json'), ('request','long_request.json'),
            ('ready_pose','ready_pose.json'), ('spawn','spawn.json'))})
    save(case/'launch.json', launch)
    from .entry import admit_launch
    admit_launch(launch, TASK/'native_validation'/case_name)
    result = dict(status='prepared_cpu_admitted', document_id=request['document_id'],
        scene=manifest['scene'], launch=str(case/'launch.json'),
        runtime_request=spec['runtime_request'], native_physics_validated=False,
        full_trajectory_validated=False, model_calls=0)
    save(case/'preparation_result.json', result)
    return result

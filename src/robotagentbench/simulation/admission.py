"""Prepared-scene admission delegated to the original trusted task owners."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from robotagentbench.interfaces.errors import TaskError


def read(path):
    return json.loads(Path(path).read_text())


def verify_manifest(manifest):
    """No source refresh or fallback asset substitution at episode admission."""
    hashes = manifest.get("source_hashes", {})
    if not hashes:
        raise TaskError("Scene owner must supply source hashes", kind="binding", reason="unverified_scene_sources")
    for path, expected in hashes.items():
        source = Path(path)
        if not source.is_file():
            raise TaskError("Scene source missing: " + path, kind="load", reason="scene_asset_missing")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise TaskError("Scene source changed: " + path, kind="binding", reason="scene_source_changed")


@dataclass(frozen=True)
class AdmittedLaunch:
    launch_path: Path
    output: Path
    launch: dict
    request: dict
    manifest: dict
    plan: dict
    owner_kind: str
    input_hashes: tuple[tuple[Path, str], ...]
    context: str = ''

    def verify_inputs(self):
        for path, digest in self.input_hashes:
            if (path.resolve() != path or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError('Admitted native input changed before startup')


def admit_prepared_launch(launch_path, output, *, backend, workspace, public_input_binding=None):
    """Validate before any GPU query, output creation, or simulator imports.

    The workspace replaces deployment-directory allowlists only. Registry
    availability, task-owned admission, and source hashes remain authoritative.
    No manifest is rewritten or repinned during admission.
    """
    path = workspace.input(launch_path)
    out = workspace.output(output)
    launch = read(path)
    if not isinstance(launch, dict):
        raise ValueError('Explicit prepared native launch required')
    schema = launch.get('schema')
    if schema not in ('openend.short_task.launch.v1', 'openend.coding_room.launch.v1'):
        raise ValueError('Unsupported prepared native launch schema')
    public = None
    public_pins = ()
    if public_input_binding is not None:
        from .public_input import load
        public, public_pins = load(public_input_binding, path, workspace=workspace)
    input_paths = [path]
    for key in ('request', 'manifest'):
        input_paths.append(workspace.input(launch[key]))
    if 'ready_pose' in launch:
        input_paths.append(workspace.input(launch['ready_pose']))
    if 'spawn' in launch:
        input_paths.append(workspace.input(launch['spawn']))
    original_manifest = read(launch['manifest'])
    input_paths.append(workspace.input(original_manifest['scene']))
    pins = tuple((item, hashlib.sha256(item.read_bytes()).hexdigest()) for item in dict.fromkeys(input_paths)) + public_pins
    if schema == 'openend.coding_room.launch.v1':
        if 'ready_pose' not in launch:
            raise ValueError('The original room owner requires its prepared ready pose')
        if any(callback is None for callback in (backend.admit_room, backend.build_room, backend.accept_room)):
            raise TaskError('Original room owner is not bound', kind='load', reason='room_owner_required')
        request, manifest, plan = backend.admit_room(deepcopy(launch), out)
        owner_kind = 'room'
    else:
        request, manifest = read(launch['request']), read(launch['manifest'])
        if launch.get('task_name') != request.get('task_name'):
            raise ValueError('Launch task differs from request')
        registry = backend.registry
        plan = registry.resolve(request)
        specialized = plan['task_name'] in registry.FAILURE_RECOVERY_TASKS | registry.ATOMIC_COFFEE_TASKS
        if specialized and backend.exec_special_owner is None:
            raise TaskError('Original task-specific native interpreter is not bound',
                            kind='load', reason='special_native_owner_required')
        if plan['task_name'] in registry.FAILURE_RECOVERY_TASKS:
            request, manifest, plan = registry.failure_recovery_adapter().admit_launch(deepcopy(launch), out)
            owner_kind = 'failure_recovery'
        elif plan['task_name'] in registry.ATOMIC_COFFEE_TASKS:
            request, manifest, plan = registry.atomic_coffee_adapter().admit_launch(deepcopy(launch), out)
            owner_kind = 'atomic_coffee'
        else:
            if 'ready_pose' not in launch:
                raise ValueError('The original standalone owner requires its prepared ready pose')
            if not plan['available']:
                error = plan['unavailable']
                raise TaskError(error['message'], kind=error['kind'], reason=error['reason'])
            if registry.is_gpu_cloth_preview(plan):
                raise TaskError('GPU cloth preview needs its existing GPU scene owner; this owner uses CPU physics',
                                kind='load', reason='gpu_cloth_scene_owner_required')
            plan = registry.bind(plan, manifest)
            registry.verify_sources(plan)
            backend.verify_scene_sources()
            verify_manifest(manifest)
            owner_kind = 'standalone'
    workspace.input(manifest['scene'])
    # Only the trusted original owner may select public text. Never forward the
    # request, scene manifest, target bindings, or arbitrary plan dictionaries.
    from robotagentbench.interfaces.observation_extras import public_context
    if public is not None:
        if owner_kind != 'room':
            raise ValueError('Saved public text binding requires the room owner public-instruction boundary')
        plan = deepcopy(plan)
        plan['instruction'] = public['instruction']
        context = public_context(public['context'])
    else:
        context = public_context(backend.public_context(deepcopy(launch), deepcopy(plan))
                                 if backend.public_context is not None else '')
    admitted = AdmittedLaunch(path, out, deepcopy(launch), deepcopy(request), deepcopy(manifest),
                              deepcopy(plan), owner_kind, pins, context)
    admitted.verify_inputs()
    return admitted

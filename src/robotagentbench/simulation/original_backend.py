"""Concrete routing to unchanged, hash-pinned original task-owner factories."""
from contextlib import contextmanager
import sys

from robotagentbench.runtime.coding_action_host import InputRejected as ControlInputRejected
from .source_tree import NAMESPACES, SourceDependencyError
from .task_backend import NativeBindings, TaskBackend

REGISTRY = 'short_tasks.registry'
SCENE = 'short_tasks.scene'
SHORT_SESSION = 'short_tasks.runtime'
TRASH_SELECTION = 'short_task_ContextReasoning.tasks.select_trash.native'
OTHER_SELECTION = 'short_task_ContextReasoning.tasks.select_shelf.native'
ROOM_OWNER = 'short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.native_entry'
MOBILE = 'arena_api.mobile_base_native'
ALIGNMENT = 'short_tasks.agent_probe_v1.robot_parameter_alignment'
GUARD = 'short_tasks.agent_probe_v1.collision_guard.runtime'
CODING = 'short_tasks.agent_probe_v1.cap_v1.native_tools.broker'
RECOVERY_TASKS = frozenset('failure_recovery_' + name for name in (
    'shelf_drink', 'coffee_bag', 'sofa', 'table_waste_empty_grasp', 'table_waste_tipped_cup'))
ATOMIC_COFFEE_TASKS = frozenset((
    'coffee_bag_interference', 'coffee_machine_push_interference',
    'coffee_bag_then_machine_interference', 'coffee_button_lighting_interference'))


def run_original_worker(launch_path, output, *, options, sources, workspace,
                        enable_coding=False, specialized_owner=None, paths=None,
                        native_cloth_owner=None):
    """Route frozen owners before importing the generic original environment.

    All bindings are supplied by trusted operator code, never by the launch or
    action protocol. A specialized callback must verify its own complete source
    profile and perform a fresh-interpreter handoff; it must not return. The
    child retains original task admission, physics, scoring and shutdown.
    """
    from .admission import read
    from .worker import NativeRunOptions, run_native_worker
    from .cloth_binding import NativeClothBinding, TASK_NAMES as CLOTH_TASKS
    from robotagentbench.interfaces.errors import TaskError

    if not isinstance(options, NativeRunOptions) or type(enable_coding) is not bool:
        raise ValueError('Explicit native options and code-tool selection are required')
    if specialized_owner is not None and not callable(specialized_owner):
        raise ValueError('A trusted specialized-owner callback is required')
    if native_cloth_owner is not None and not isinstance(native_cloth_owner, NativeClothBinding):
        raise ValueError('A trusted NativeClothBinding is required')
    if paths is not None:
        paths.validate_source_tree(sources)
    path, out = workspace.input(launch_path), workspace.output(output)
    launch = read(path)
    if not isinstance(launch, dict) or launch.get('schema') not in (
            'openend.short_task.launch.v1', 'openend.coding_room.launch.v1',
            'robotagentbench.native_cloth.launch.v1'):
        raise ValueError('Explicit prepared native launch required')
    if (launch['schema'] == 'robotagentbench.native_cloth.launch.v1'
            and launch.get('task_name') not in CLOTH_TASKS):
        raise ValueError('Native cloth launch cannot select another task family')
    if launch.get('task_name') in CLOTH_TASKS:
        name = launch['task_name']
        if launch['schema'] != 'robotagentbench.native_cloth.launch.v1':
            request = read(workspace.input(launch['request']))
            if not isinstance(request, dict) or request.get('task_name') != name:
                raise ValueError('Cloth launch task differs from its request')
            expected = ('openend.coding_room.launch.v1' if name == 'long_task_18'
                        else 'openend.short_task.launch.v1')
            if launch['schema'] != expected:
                raise ValueError('Wrong native cloth launch scope')
        if sources.active or any(module.split('.')[0] in NAMESPACES for module in sys.modules):
            raise SourceDependencyError('Native cloth needs its own fresh source environment')
        if native_cloth_owner is None:
            raise TaskError('The scored GPU cloth owner is not bound', kind='load',
                            reason='native_cloth_owner_required')
        return native_cloth_owner.run(path, out, task_name=name, options=options, workspace=workspace)
    family = None
    if launch['schema'] == 'openend.short_task.launch.v1':
        request = read(workspace.input(launch['request']))
        if not isinstance(request, dict) or launch.get('task_name') != request.get('task_name'):
            raise ValueError('Launch task differs from request')
        name = request.get('task_name')
        family = ('failure_recovery' if name in RECOVERY_TASKS else
                  'atomic_coffee' if name in ATOMIC_COFFEE_TASKS else None)
    if family is not None:
        if sources.active or any(name.split('.')[0] in NAMESPACES for name in sys.modules):
            raise SourceDependencyError('Specialized handoff requires a fresh original-source environment')
        if options.coding_policy:
            raise SourceDependencyError('Specialized owners retain their original raw-action protocol')
        if specialized_owner is None:
            raise TaskError('Original task-specific native interpreter is not bound',
                            kind='load', reason='special_native_owner_required')
        if paths is not None:
            paths.verify(name, 'native')
        kwargs = dict(options=options, workspace=workspace)
        if paths is not None:
            kwargs['paths'] = paths
        specialized_owner(family, path, out, **kwargs)
        raise RuntimeError('Task-specific interpreter handoff returned unexpectedly')
    with bind_original_backend(sources, enable_coding=enable_coding, paths=paths) as backend:
        return run_native_worker(path, out, options=options, backend=backend, workspace=workspace)


class _ConfiguredRegistry:
    def __init__(self, original, owner):
        self._original, self._owner = original, owner

    def __getattr__(self, name):
        return getattr(self._original, name)

    def resolve(self, request):
        result = self._original.resolve(request)
        self._owner.verify_paths(result['task_name'], 'source')
        return result

    def bind(self, plan, manifest):
        self._owner.verify_paths(plan['task_name'], 'assets')
        result = self._original.bind(plan, manifest)
        self._owner.task_name = plan['task_name']
        return result

    def verify_sources(self, plan):
        self._owner.verify_paths(plan['task_name'], 'source')
        return self._original.verify_sources(plan)


class _OriginalControlBoundary:
    """Preserve the original broker's recoverable-input exception identity."""
    def __init__(self, host, rejected_type):
        self._host, self._rejected_type = host, rejected_type

    def __getattr__(self, name):
        return getattr(self._host, name)

    def execute(self, *args, **kwargs):
        try:
            return self._host.execute(*args, **kwargs)
        except ControlInputRejected as error:
            raise self._rejected_type(str(error)) from None

    def exchange(self, *args, **kwargs):
        try:
            return self._host.exchange(*args, **kwargs)
        except ControlInputRejected as error:
            raise self._rejected_type(str(error)) from None


class _ControlBrokerBoundary:
    """Return original recoverable tool rejections to the current RPC server."""
    def __init__(self, broker, rejected_type):
        self._broker, self._rejected_type = broker, rejected_type

    def __getattr__(self, name):
        return getattr(self._broker, name)

    def exchange(self, *args, **kwargs):
        try:
            return self._broker.exchange(*args, **kwargs)
        except self._rejected_type as error:
            raise ControlInputRejected(str(error)) from None


class _OriginalTasks:
    def __init__(self, sources, *, coding, paths=None):
        self.sources, self.coding, self.paths = sources, coding, paths
        self.task_name = None
        original = sources.import_module(REGISTRY)
        self.registry = _ConfiguredRegistry(original, self) if paths is not None else original
        self.scene = sources.import_module(SCENE)

    def verify_paths(self, task_name, phase):
        if self.paths is not None:
            return self.paths.verify(task_name, phase)

    def verify_scene_sources(self):
        self.sources.verify()
        if self.paths is not None:
            self.paths.verify_common()
        return self.scene.verify_scene_sources()

    def build_short_session(self, task_name, request, *args, **kwargs):
        if task_name not in self.registry.TASK_NAMES or request.get('task_name') != task_name:
            raise ValueError('Task factory identity differs from the admitted request')
        if task_name in self.registry.FAILURE_RECOVERY_TASKS | self.registry.ATOMIC_COFFEE_TASKS:
            raise SourceDependencyError('This task requires its original separate native interpreter')
        if self.paths is not None and task_name != self.task_name:
            raise SourceDependencyError('Task factory differs from the admitted task path profile')
        self.verify_paths(task_name, 'native')
        module = (TRASH_SELECTION if task_name == 'select_trash' else OTHER_SELECTION
                  if task_name in ('select_sofa', 'select_drink', 'select_snack') else SHORT_SESSION)
        self.sources.verify()
        return self.sources.import_module(module).build_session(request, *args, **kwargs)

    def admit_room(self, launch, output):
        self.sources.verify()
        self.verify_paths(launch.get('task_name'), 'assets')
        result = self.sources.import_module(ROOM_OWNER).admit_launch(launch, output)
        self.task_name = launch.get('task_name')
        if self.task_name == 'task_discovery':
            from robotagentbench.tasks.discovery.binding import describe_plan
            request, manifest, plan = result
            result = request, manifest, describe_plan(plan)
        return result

    def build_room(self, *args, **kwargs):
        self.sources.verify()
        task = kwargs.get('evaluation_mode', 'openend')
        self.verify_paths(task, 'native')
        if self.paths is not None and task != self.task_name:
            raise SourceDependencyError('Room factory differs from the admitted task path profile')
        room = self.sources.import_module(ROOM_OWNER).build_room(*args, **kwargs)
        if task == 'task_discovery':
            from robotagentbench.tasks.discovery.binding import install
            try:
                return install(room)
            except BaseException:
                room.close()
                raise
        return room

    def accept_room(self, *args, **kwargs):
        self.verify_paths(self.task_name, 'native')
        session = self.sources.import_module(ROOM_OWNER).accepted_session(*args, **kwargs)
        if self.task_name == 'task_discovery':
            from robotagentbench.tasks.discovery.binding import DiscoveryControlSession
            return DiscoveryControlSession(session)
        return session

    def load_native_bindings(self):
        self.sources.verify()
        self.verify_paths(self.task_name, 'native')
        self.sources.require_group('native')
        if 'fcl' not in self.sources.namespace_paths or self.sources.module_file('fcl')[0] is None:
            raise SourceDependencyError('Original collision-library sources and native artifacts are not bound')
        self.sources.native_extensions_enabled = True
        # Load the pinned vendor namespace first; an unrelated installed fcl
        # must not be substituted by the original guard's vendor path insertion.
        self.sources.import_module('fcl')
        return NativeBindings(
            self.sources.import_module(MOBILE).Controller,
            self.sources.import_module(GUARD).NativeCollisionGuard,
            self.sources.import_module(ALIGNMENT))

    def wrap_coding_host(self, host, held_targets, output, **kwargs):
        if not self.coding:
            raise SourceDependencyError('Original code-tool dependencies were not enabled')
        self.verify_paths(self.task_name, 'coding')
        self.sources.require_group('coding')
        broker = self.sources.import_module(CODING)
        boundary = _OriginalControlBoundary(host, broker.InputRejected)
        tools = broker.CodingHost(boundary, held_targets, output, **kwargs)
        return _ControlBrokerBoundary(tools, broker.InputRejected)

@contextmanager
def bind_original_backend(sources, *, enable_coding=False, paths=None):
    """Bind real standalone/context/room owners for one fresh worker lifetime.

    Source receipts, native artifacts, code-tool data and original task-specific
    admission remain required. This does not install dependencies or bless
    missing source files as available tasks.
    """
    if type(enable_coding) is not bool:
        raise ValueError('Code-tool dependencies require an explicit boolean selection')
    if paths is not None:
        paths.validate_source_tree(sources)
    if enable_coding:
        sources.require_group('coding')
    with sources.activated():
        owner = _OriginalTasks(sources, coding=enable_coding, paths=paths)
        yield TaskBackend(registry=owner.registry,
            verify_scene_sources=owner.verify_scene_sources,
            build_short_session=owner.build_short_session,
            load_native_bindings=owner.load_native_bindings,
            admit_room=owner.admit_room, build_room=owner.build_room,
            accept_room=owner.accept_room,
            wrap_coding_host=owner.wrap_coding_host if enable_coding else None,
            # Specialized frozen interpreters need their own separately verified
            # source environment; never inherit this generic source binding.
            exec_special_owner=None)

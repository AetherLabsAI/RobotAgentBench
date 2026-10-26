"""Factory-routing tests with named fake modules, not native task validation."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import sys
from unittest.mock import patch
import unittest

from robotagentbench.simulation.original_backend import bind_original_backend, run_original_worker
from robotagentbench.simulation.source_tree import NAMESPACES, SourceDependencyError
from robotagentbench.simulation.worker import NativeRunOptions
from robotagentbench.interfaces.errors import TaskError
from admission_fixtures import AdmissionFixture
from source_fixtures import SourceFixture
from robotagentbench.runtime.coding_action_host import InputRejected as ControlInputRejected


class OriginalBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SourceFixture(self.temp.name)

    def test_registered_generic_tasks_use_original_session_factory(self):
        tree = self.fixture.tree()
        with bind_original_backend(tree) as backend:
            skipped = backend.registry.FAILURE_RECOVERY_TASKS | backend.registry.ATOMIC_COFFEE_TASKS
            for name in backend.registry.TASK_NAMES:
                if name in skipped or name.startswith('select_'):
                    continue
                with self.subTest(task=name):
                    request = {'task_name': name}
                    marker, forwarded, args, kwargs = backend.build_short_session(name, request, 'stage', sensor='sensor')
                    self.assertEqual(marker, 'standalone')
                    self.assertIs(forwarded, request)
                    self.assertEqual(args, ('stage',))
                    self.assertEqual(kwargs, {'sensor': 'sensor'})

    def test_four_context_tasks_retain_their_original_specific_factories(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            for name in ('select_trash', 'select_drink', 'select_snack', 'select_sofa'):
                with self.subTest(task=name):
                    row = backend.build_short_session(name, {'task_name': name}, 'stage', sensor='original')
                    self.assertEqual(row[0], 'trash' if name == 'select_trash' else 'selection')
                    self.assertEqual(row[3]['sensor'], 'original')

    def test_special_tasks_cannot_use_generic_session_or_unbound_interpreter(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            self.assertIsNone(backend.exec_special_owner)
            for name in backend.registry.FAILURE_RECOVERY_TASKS | backend.registry.ATOMIC_COFFEE_TASKS:
                with self.subTest(task=name), self.assertRaises(SourceDependencyError):
                    backend.build_short_session(name, {'task_name': name})

    def test_room_admission_build_and_acceptance_are_unmodified_delegation(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            launch = {'original_request': True}
            self.assertEqual(backend.admit_room(launch, 'output'), ('room_admit', (launch, 'output'), {}))
            self.assertEqual(backend.build_room('request', episode_id='original'),
                             ('room_build', ('request',), {'episode_id': 'original'}))
            self.assertEqual(backend.accept_room('room', instruction='original'),
                             ('room_accept', ('room',), {'instruction': 'original'}))

    def test_native_bindings_require_pinned_native_dependencies(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            with self.assertRaises(SourceDependencyError):
                backend.load_native_bindings()
        with bind_original_backend(self.fixture.tree(native=True)) as backend:
            native = backend.load_native_bindings()
            self.assertEqual(native.mobile_controller.__name__, 'Controller')
            self.assertEqual(native.collision_guard.__name__, 'NativeCollisionGuard')
            self.assertTrue(callable(native.robot_alignment.verify_native_forces))

    def test_coding_is_optional_and_requires_its_original_resources(self):
        with self.assertRaises(SourceDependencyError):
            with bind_original_backend(self.fixture.tree(), enable_coding=True):
                self.fail('Missing original code-tool data must not be accepted')
        with bind_original_backend(self.fixture.tree()) as backend:
            self.assertIsNone(backend.wrap_coding_host)
        with bind_original_backend(self.fixture.tree(coding=True), enable_coding=True) as backend:
            marker, args, kwargs = backend.wrap_coding_host('host', 'held', 'output')._broker
            self.assertEqual(marker, 'code_host')
            self.assertEqual(args[1:], ('held', 'output'))
            self.assertEqual(args[0]._host, 'host')
            self.assertEqual(kwargs, {})

    def test_coding_boundary_preserves_recoverable_errors_and_native_faults(self):
        def rejected(*args, **kwargs):
            raise ControlInputRejected('invalid test action')
        def uncertain(*args, **kwargs):
            raise RuntimeError('uncertain native write')
        tree = self.fixture.tree(coding=True)
        with bind_original_backend(tree, enable_coding=True) as backend:
            host = SimpleNamespace(execute=rejected, exchange=rejected, session='same session')
            tools = backend.wrap_coding_host(host, 'held', 'output')
            boundary = tools._broker[1][0]
            original = tree.import_module('short_tasks.agent_probe_v1.cap_v1.native_tools.broker').InputRejected
            for function in (boundary.execute, boundary.exchange):
                with self.assertRaises(original) as error:
                    function({'invalid': True})
                self.assertNotIsInstance(error.exception, ControlInputRejected)
            self.assertEqual(boundary.session, 'same session')
            host.execute = uncertain
            with self.assertRaisesRegex(RuntimeError, 'uncertain native write'):
                boundary.execute({})
            def broker_rejection(*args, **kwargs):
                raise original('invalid test tool')
            tools._broker = SimpleNamespace(exchange=broker_rejection, faulted=False)
            with self.assertRaises(ControlInputRejected):
                tools.exchange({})
            self.assertFalse(tools.faulted)
            tools._broker.exchange = uncertain
            with self.assertRaisesRegex(RuntimeError, 'uncertain native write'):
                tools.exchange({})

    def test_task_factory_identity_and_missing_source_fail_closed(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            for name, request in (('unknown', {'task_name': 'unknown'}),
                                  ('prepare_coffee', {'task_name': 'bake_bread'})):
                with self.assertRaises(ValueError):
                    backend.build_short_session(name, request)

    def test_scene_source_verification_is_not_replaced_by_local_hash_receipt(self):
        with bind_original_backend(self.fixture.tree()) as backend:
            self.assertEqual(backend.verify_scene_sources(), {'original_scene_check': True})

    def test_frozen_handoff_precedes_generic_imports_and_accepts_original_reset_scope(self):
        f = AdmissionFixture(self.temp.name)
        tree = self.fixture.tree()
        for name, family in (('coffee_bag_interference', 'atomic_coffee'),
                             ('failure_recovery_sofa', 'failure_recovery')):
            with self.subTest(task=name):
                f.configure(task=name)
                launch = json.loads(f.launch.read_text())
                del launch['ready_pose']
                f.launch.write_text(json.dumps(launch))
                called = []
                def handoff(kind, path, output, **kwargs):
                    called.append((kind, path, output))
                    self.assertFalse(tree.active)
                    self.assertFalse(any(module.split('.')[0] in NAMESPACES for module in sys.modules))
                    self.assertEqual(kwargs['options'], NativeRunOptions(0))
                    raise SystemExit(0)
                with patch('robotagentbench.simulation.original_backend.bind_original_backend',
                           side_effect=AssertionError('Generic source imports are forbidden')):
                    with self.assertRaises(SystemExit):
                        run_original_worker(f.launch, f.output, options=NativeRunOptions(0),
                            sources=tree, workspace=f.workspace, specialized_owner=handoff)
                self.assertEqual(called, [(family, f.launch, f.output)])
                self.assertFalse(f.output.exists())

    def test_unbound_or_returning_frozen_handoff_never_uses_generic_owner(self):
        f = AdmissionFixture(self.temp.name)
        f.configure(task='coffee_bag_interference')
        tree = self.fixture.tree()
        with patch('robotagentbench.simulation.original_backend.bind_original_backend',
                   side_effect=AssertionError('No generic fallback')):
            with self.assertRaises(TaskError) as error:
                run_original_worker(f.launch, f.output, options=NativeRunOptions(0),
                                    sources=tree, workspace=f.workspace)
            self.assertEqual(error.exception.reason, 'special_native_owner_required')
            with self.assertRaisesRegex(RuntimeError, 'handoff returned'):
                run_original_worker(f.launch, f.output, options=NativeRunOptions(0), sources=tree,
                                    workspace=f.workspace, specialized_owner=lambda *args, **kwargs: None)

    def test_frozen_handoff_rejects_wrong_identity_loaded_sources_and_coding_override(self):
        f = AdmissionFixture(self.temp.name)
        f.configure(task='coffee_bag_interference')
        tree = self.fixture.tree()
        callback = lambda *args, **kwargs: self.fail('Invalid handoff must not execute')
        with tree.activated():
            with self.assertRaises(SourceDependencyError):
                run_original_worker(f.launch, f.output, options=NativeRunOptions(0), sources=tree,
                                    workspace=f.workspace, specialized_owner=callback)
        with self.assertRaises(SourceDependencyError):
            run_original_worker(f.launch, f.output, options=NativeRunOptions(0, coding_policy=True),
                sources=tree, workspace=f.workspace, specialized_owner=callback)
        request = json.loads(f.request.read_text())
        request['task_name'] = 'failure_recovery_sofa'
        f.request.write_text(json.dumps(request))
        with self.assertRaises(ValueError):
            run_original_worker(f.launch, f.output, options=NativeRunOptions(0), sources=tree,
                                workspace=f.workspace, specialized_owner=callback)

    def test_generic_worker_keeps_original_binding_active_for_entire_worker_lifetime(self):
        f = AdmissionFixture(self.temp.name)
        tree = self.fixture.tree()
        called = []
        def worker(path, output, *, backend, **kwargs):
            self.assertTrue(tree.active)
            self.assertIs(backend.registry, tree.import_module('short_tasks.registry'))
            self.assertIn('prepare_coffee', backend.registry.TASK_NAMES)
            called.append((path, output))
            raise SystemExit(0)
        with patch('robotagentbench.simulation.worker.run_native_worker', side_effect=worker):
            with self.assertRaises(SystemExit):
                run_original_worker(f.launch, f.output, options=NativeRunOptions(0),
                                    sources=tree, workspace=f.workspace)
        self.assertFalse(tree.active)
        self.assertEqual(called, [(f.launch, f.output)])


if __name__ == '__main__':
    unittest.main()

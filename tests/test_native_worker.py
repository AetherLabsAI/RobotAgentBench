"""Dedicated-worker orchestration with fake CUDA, FDs, native app and process exit."""
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from io import StringIO
import json
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

from robotagentbench.simulation import native_execution, worker
from robotagentbench.simulation.admission import admit_prepared_launch
from robotagentbench.simulation.task_backend import NativeBindings
from robotagentbench.simulation.worker import NativeRunOptions
from admission_fixtures import AdmissionFixture


class LeaseDouble:
    uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
    evidence = {'memory_mib': 0, 'utilization_percent': 0}

    def __init__(self, trace):
        self.trace, self.closed = trace, False

    def configure_environment(self, output):
        self.trace.append('environment')

    def close(self):
        if not self.closed:
            self.trace.append('gpu_lock')
            self.closed = True


class RecordingPipe(StringIO):
    def close(self):
        self.saved = self.getvalue()
        super().close()


def fake_native_imports(app_factory, context):
    modules = {name: ModuleType(name) for name in (
        'isaacsim', 'numpy', 'carb', 'omni', 'omni.usd', 'omni.physx', 'omni.physics',
        'omni.physics.tensors', 'omni.replicator', 'omni.replicator.core', 'pxr', 'PIL')}
    modules['isaacsim'].SimulationApp = app_factory
    modules['omni'].usd = modules['omni.usd']
    modules['omni'].physx = modules['omni.physx']
    modules['omni'].physics = modules['omni.physics']
    modules['omni.physics'].tensors = modules['omni.physics.tensors']
    modules['omni'].replicator = modules['omni.replicator']
    modules['omni.replicator'].core = modules['omni.replicator.core']
    modules['omni.usd'].get_context = lambda: context
    for name in ('Usd', 'UsdGeom', 'UsdPhysics', 'UsdUtils', 'Gf'):
        setattr(modules['pxr'], name, SimpleNamespace())
    modules['PIL'].Image = SimpleNamespace()
    return modules


class NativeWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = AdmissionFixture(self.temp.name)

    def test_invalid_launch_fails_before_gpu_or_output(self):
        f = self.fixture
        f.scene.write_text('changed')
        with patch.object(worker, 'acquire_gpu', side_effect=AssertionError('GPU must not be queried')):
            with self.assertRaises(ValueError):
                worker.run_native_worker(f.launch, f.output, options=NativeRunOptions(0),
                                         backend=f.backend, workspace=f.workspace)
        self.assertFalse(f.output.exists())

    def test_unbound_coding_backend_fails_before_gpu_and_launch(self):
        f = self.fixture
        with patch.object(worker, 'admit_prepared_launch', side_effect=AssertionError('No admission expected')):
            with self.assertRaises(ValueError):
                worker.run_native_worker(f.launch, f.output, options=NativeRunOptions(0, coding_policy=True),
                                         backend=f.backend, workspace=f.workspace)

    def test_special_owner_handoff_cannot_fall_through_to_generic_owner(self):
        f = self.fixture
        f.configure(task='atomic_case')
        f.backend = replace(f.backend, exec_special_owner=lambda *args, **kwargs: None)
        with patch.object(worker, 'acquire_gpu', side_effect=AssertionError('No generic GPU owner')):
            with self.assertRaisesRegex(RuntimeError, 'handoff returned'):
                worker.run_native_worker(f.launch, f.output, options=NativeRunOptions(0),
                                         backend=f.backend, workspace=f.workspace)

    def test_output_creation_failure_releases_gpu_lock(self):
        f = self.fixture
        trace = []
        lease = LeaseDouble(trace)
        with patch.object(worker, 'acquire_gpu', return_value=lease), \
                patch.object(type(f.workspace), 'create_output', side_effect=OSError('injected mkdir failure')):
            with self.assertRaises(OSError):
                worker.run_native_worker(f.launch, f.output, options=NativeRunOptions(0),
                                         backend=f.backend, workspace=f.workspace)
        self.assertEqual(trace, ['gpu_lock'])

    def test_success_ack_is_after_cleanup_and_failure_has_no_success_ack(self):
        for success in (True, False):
            with self.subTest(success=success), TemporaryDirectory() as directory:
                f = AdmissionFixture(directory)
                trace = []
                lease = LeaseDouble(trace)
                pipe = RecordingPipe()
                def execute(admitted, *, completion, **kwargs):
                    completion.write('{"op":"closed"}\n')
                    self.assertEqual(pipe.getvalue(), '')
                    trace.append('app_shutdown')
                    lease.close()
                    return {'status': 'completed' if success else 'error',
                            'source_layers_unchanged': True, 'cleanup_errors': [],
                            'native_shutdown_complete': success}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(worker, 'acquire_gpu', return_value=lease))
                    stack.enter_context(patch.object(worker, 'execute_native', side_effect=execute))
                    stack.enter_context(patch.object(worker.os, 'dup', return_value=99))
                    stack.enter_context(patch.object(worker.os, 'dup2'))
                    stack.enter_context(patch.object(worker.os, 'fdopen', return_value=pipe))
                    def exit_process(code):
                        raise SystemExit(code)
                    stack.enter_context(patch.object(worker.os, '_exit', side_effect=exit_process))
                    with self.assertRaises(SystemExit) as ended:
                        worker.run_native_worker(f.launch, f.output, options=NativeRunOptions(0),
                                                 backend=f.backend, workspace=f.workspace)
                self.assertEqual(ended.exception.code, 0 if success else 1)
                self.assertEqual(trace, ['environment', 'app_shutdown', 'gpu_lock'])
                self.assertEqual(pipe.saved, '{"op":"closed"}\n' if success else '')

    def test_app_configuration_and_stage_open_failure_close_app_and_lock(self):
        f = self.fixture
        admitted = admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)
        f.workspace.create_output(f.output)
        trace, configurations = [], []
        app = SimpleNamespace(close=lambda: trace.append('app_close'))
        def factory(config):
            configurations.append(config)
            return app
        context = SimpleNamespace(open_stage=lambda path: False)
        alignment = SimpleNamespace(apply=lambda *args: None, verify=lambda *args: None,
                                    verify_native_forces=lambda *args: None)
        backend = replace(f.backend, load_native_bindings=lambda: NativeBindings(
            lambda *args: None, lambda *args: None, alignment))
        with patch.dict('sys.modules', fake_native_imports(factory, context)), redirect_stderr(StringIO()):
            report = native_execution.execute_native(admitted, options=NativeRunOptions(0),
                backend=backend, lease=LeaseDouble(trace), completion=StringIO())
        self.assertEqual(trace, ['app_close', 'gpu_lock'])
        self.assertEqual(report['status'], 'error')
        self.assertTrue(report['native_shutdown_complete'])
        self.assertNotIn('task_result', report)
        config = configurations[0]
        self.assertEqual(config['renderer'], 'RaytracedLighting')
        self.assertFalse(config['fast_shutdown'])
        self.assertTrue(config['headless'])
        self.assertEqual((config['active_gpu'], config['physics_gpu']), (0, 0))
        self.assertEqual(config['limit_cpu_threads'], 8)
        saved = json.loads((f.output / 'native_report.json').read_text())
        self.assertEqual(saved['status'], 'error')

    def test_constructor_failure_keeps_shutdown_incomplete_and_releases_lock(self):
        f = self.fixture
        admitted = admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)
        f.workspace.create_output(f.output)
        trace = []
        def broken(config):
            raise RuntimeError('injected app construction failure')
        with patch.dict('sys.modules', fake_native_imports(broken, None)), redirect_stderr(StringIO()):
            report = native_execution.execute_native(admitted, options=NativeRunOptions(0),
                backend=f.backend, lease=LeaseDouble(trace), completion=StringIO())
        self.assertEqual(trace, ['gpu_lock'])
        self.assertFalse(report['native_shutdown_complete'])
        self.assertEqual(report['status'], 'error')

    def test_report_write_failure_still_releases_lock(self):
        f = self.fixture
        admitted = admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)
        f.workspace.create_output(f.output)
        trace = []
        with patch('pathlib.Path.write_text', side_effect=OSError('injected report failure')), \
                redirect_stderr(StringIO()):
            report = native_execution.execute_native(admitted, options=NativeRunOptions(0),
                backend=f.backend, lease=LeaseDouble(trace), completion=StringIO())
        self.assertEqual(trace, ['gpu_lock'])
        self.assertEqual(report['status'], 'error')
        self.assertFalse(report['native_shutdown_complete'])

    def test_options_are_not_silent_mode_or_permission_overrides(self):
        for kwargs in ({'gpu': True}, {'gpu': 0, 'allow_shared_gpu': 1},
                       {'gpu': 0, 'allow_shared_gpu': True, 'allow_idle_utilization': True},
                       {'gpu': 0, 'smoke': True, 'coding_policy': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NativeRunOptions(**kwargs)


if __name__ == '__main__':
    unittest.main()

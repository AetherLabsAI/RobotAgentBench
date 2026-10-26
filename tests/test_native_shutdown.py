"""Failure-injection tests for cleanup order and completion acknowledgement."""
from copy import deepcopy
import hashlib
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from robotagentbench.runtime.completion import CompletionStream
from robotagentbench.simulation.shutdown import NativeResources, shutdown_succeeded


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.layer = Path(self.temp.name) / 'source.usda'
        self.layer.write_text('#usda 1.0\n')
        self.pin = hashlib.sha256(self.layer.read_bytes()).hexdigest()

    def fixture(self, failure=None):
        trace = []
        def action(label):
            def run():
                trace.append(label)
                if label == failure:
                    raise RuntimeError('injected ' + label)
            return run
        resources = NativeResources(
            host=SimpleNamespace(close=action('host')),
            session=SimpleNamespace(close=action('session')),
            simulation=SimpleNamespace(detach_stage=action('physics')),
            collision_guard=SimpleNamespace(close=action('collision_guard')),
            annotators={'head': [SimpleNamespace(detach=action('camera'))]},
            products=[SimpleNamespace(destroy=action('render_product'))],
            app=SimpleNamespace(close=action('app')),
            source_hashes={str(self.layer): self.pin},
            lease=SimpleNamespace(close=action('gpu_lock')))
        report = {'status': 'physics_completed', 'cleanup_errors': [],
                  'native_shutdown_complete': False, 'source_layers_unchanged': None,
                  'task_result': {'opaque_native_scores': {'process': 7, 'final': 3}}}
        return resources, report, trace

    def test_success_preserves_order_and_task_result(self):
        resources, report, trace = self.fixture()
        expected = deepcopy(report['task_result'])
        resources.shutdown(report, lambda: trace.append('checkpoint'))
        self.assertEqual(trace, ['host', 'session', 'physics', 'collision_guard', 'camera',
                                 'render_product', 'checkpoint', 'app', 'gpu_lock', 'checkpoint'])
        self.assertTrue(shutdown_succeeded(report))
        self.assertEqual(report['task_result'], expected)
        resources.shutdown(report, lambda: self.fail('Cleanup must be idempotent'))

    def test_every_resource_failure_still_attempts_remaining_cleanup(self):
        labels = ('host', 'session', 'physics', 'collision_guard', 'camera', 'render_product', 'app', 'gpu_lock')
        for failed in labels:
            with self.subTest(failed=failed):
                resources, report, trace = self.fixture(failed)
                expected = deepcopy(report['task_result'])
                resources.shutdown(report, lambda: trace.append('checkpoint'))
                self.assertEqual([label for label in trace if label != 'checkpoint'], list(labels))
                self.assertFalse(shutdown_succeeded(report))
                self.assertEqual(report['status'], 'error')
                self.assertEqual(report['task_result'], expected)
                self.assertIn(failed, [error['operation'] for error in report['cleanup_errors']])

    def test_missing_or_changed_source_never_skips_app_or_lock_cleanup(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                resources, report, trace = self.fixture()
                self.layer.write_text('changed')
                if missing:
                    self.layer.unlink()
                resources.shutdown(report, lambda: None)
                self.assertFalse(report['source_layers_unchanged'])
                self.assertIn('app', trace)
                self.assertIn('gpu_lock', trace)
                self.assertFalse(shutdown_succeeded(report))
                self.layer.write_text('#usda 1.0\n')

    def test_report_write_failure_never_skips_app_or_lock(self):
        resources, report, trace = self.fixture()
        def fail_report():
            trace.append('failed_checkpoint')
            raise OSError('injected report write failure')
        resources.shutdown(report, fail_report)
        self.assertIn('app', trace)
        self.assertIn('gpu_lock', trace)
        self.assertEqual(trace.count('failed_checkpoint'), 2)
        self.assertFalse(shutdown_succeeded(report))

    def test_partial_startup_cannot_be_reported_as_successful_shutdown(self):
        report = {'status': 'error', 'cleanup_errors': [], 'task_result': None}
        trace = []
        NativeResources(lease=SimpleNamespace(close=lambda: trace.append('gpu_lock'))).shutdown(report, lambda: None)
        self.assertEqual(trace, ['gpu_lock'])
        self.assertFalse(report['native_shutdown_complete'])
        self.assertFalse(shutdown_succeeded(report))
        self.assertIsNone(report['task_result'])

    def test_close_acknowledgement_remains_buffered_on_cleanup_failure(self):
        resources, report, _ = self.fixture('app')
        output = StringIO()
        completion = CompletionStream(output)
        completion.write('{"op":"closed"}\n')
        resources.shutdown(report, lambda: None)
        if shutdown_succeeded(report):
            completion.complete()
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()

"""Mock nvidia-smi responses with real temporary cooperative file locks."""
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from robotagentbench.simulation.gpu import acquire_gpu
from admission_fixtures import AdmissionFixture

GPU = 'GPU-12345678-1234-1234-1234-123456789abc'


def query(*, memory=0, utilization=0, active='', monitor='0 - - - -', existing=''):
    def run(command, **kwargs):
        if '--query-gpu=uuid,memory.used,utilization.gpu' in command:
            return f'{GPU}, {memory}, {utilization}\n'
        if '--query-compute-apps=gpu_uuid,pid' in command:
            return active
        if 'pmon' in command:
            return monitor
        return existing
    return run


class GpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = AdmissionFixture(self.temp.name)

    def acquire(self, **kwargs):
        return acquire_gpu(0, workspace=self.fixture.workspace, **kwargs)

    def test_idle_gpu_is_locked_and_second_owner_cannot_attach(self):
        lease = self.acquire(query=query())
        self.addCleanup(lease.close)
        self.assertEqual(lease.uuid, GPU)
        with self.assertRaises(BlockingIOError):
            self.acquire(query=query())
        lease.close()
        lease.close()
        second = self.acquire(query=query())
        second.close()

    def test_original_occupancy_thresholds_are_preserved(self):
        for memory, utilization, allowed in ((0, 0, True), (32, 0, True),
                                             (33, 0, False), (0, 1, False)):
            with self.subTest(memory=memory, utilization=utilization):
                if allowed:
                    self.acquire(query=query(memory=memory, utilization=utilization)).close()
                else:
                    with self.assertRaises(RuntimeError):
                        self.acquire(query=query(memory=memory, utilization=utilization))

    def test_occupancy_change_after_lock_is_rejected_and_lock_is_released(self):
        calls = []
        def changing(command, **kwargs):
            calls.append(command)
            return f'{GPU}, {0 if len(calls) == 1 else 100}, 0\n'
        with self.assertRaises(RuntimeError):
            self.acquire(query=changing)
        self.assertEqual(len(calls), 2)
        self.acquire(query=query()).close()

    def test_idle_exception_requires_zero_memory_and_no_process_evidence(self):
        self.acquire(query=query(utilization=1), allow_idle_utilization=True).close()
        cases = (query(memory=1, utilization=1), query(utilization=1, active=GPU + ', 123'),
                 query(utilization=1, monitor='0 123 C 1 1'), query(utilization=1, monitor=''))
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(RuntimeError):
                self.acquire(query=candidate, allow_idle_utilization=True)

    def test_shared_gpu_requires_explicit_approval_and_records_occupancy(self):
        lease = self.acquire(query=query(memory=100, utilization=80, existing='existing test process'),
                             allow_shared_gpu=True)
        self.addCleanup(lease.close)
        self.assertTrue(lease.evidence['user_approved_shared_gpu'])
        self.assertEqual(lease.evidence['existing_processes'], 'existing test process')

    def test_mutually_exclusive_and_malformed_approvals_fail_before_query(self):
        def forbidden(*args, **kwargs):
            self.fail('No GPU query should occur')
        with self.assertRaises(ValueError):
            self.acquire(query=forbidden, allow_idle_utilization=True, allow_shared_gpu=True)
        with self.assertRaises(ValueError):
            self.acquire(query=forbidden, allow_shared_gpu='yes')
        with self.assertRaises(ValueError):
            acquire_gpu(True, workspace=self.fixture.workspace, query=forbidden)

    def test_malformed_uuid_and_invalid_occupancy_are_rejected(self):
        for value in ('GPU-../../outside, 0, 0', GPU + ', -1, 0', GPU + ', 0, 101', 'malformed'):
            with self.subTest(value=value), self.assertRaises((RuntimeError, ValueError)):
                self.acquire(query=lambda *args, value=value, **kwargs: value)

    def test_cache_environment_is_explicit_and_does_not_change_unrelated_variables(self):
        lease = self.acquire(query=query())
        self.addCleanup(lease.close)
        with patch.dict(os.environ, {'UNRELATED_TEST_SETTING': 'preserved'}):
            lease.configure_environment(self.fixture.output)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], GPU)
            self.assertEqual(os.environ['XDG_CACHE_HOME'], str(self.fixture.output / 'cache'))
            self.assertEqual(os.environ['UNRELATED_TEST_SETTING'], 'preserved')

    def test_lock_file_cannot_redirect_to_an_unrelated_path(self):
        f = self.fixture
        unrelated = f.root / 'unrelated.txt'
        unrelated.write_text('keep')
        (f.locks / ('gpu-' + GPU + '.lock')).symlink_to(unrelated)
        with self.assertRaises(OSError):
            self.acquire(query=query())
        self.assertEqual(unrelated.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

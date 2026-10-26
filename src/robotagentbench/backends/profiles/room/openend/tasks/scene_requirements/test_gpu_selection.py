import unittest
from gpu_selection import candidates

POLICY = dict(allowed_indices=list(range(6)), max_memory_used_mib=1024, max_utilization_percent=5)

class SelectionTests(unittest.TestCase):
    def test_busy_zero_uses_another_device(self):
        found = candidates('0, GPU-a, 100, 0\n2, GPU-b, 100, 0', 'GPU-a, 12', POLICY)
        self.assertEqual([g['index'] for g in found], [2])

    def test_memory_or_utilization_excludes_graphics_work(self):
        self.assertEqual(candidates('0, GPU-a, 2000, 0\n1, GPU-b, 100, 40', '', POLICY), [])

    def test_unknown_is_not_idle(self):
        self.assertEqual(candidates('0, GPU-a, N/A, 0\n1, GPU-b, 100, N/A', '', POLICY), [])

    def test_allowed_devices_only(self):
        self.assertEqual(candidates('6, GPU-a, 0, 0', '', POLICY), [])

    def test_least_occupied_first(self):
        found = candidates('0, GPU-a, 200, 0\n5, GPU-b, 100, 0', '', POLICY)
        self.assertEqual([g['index'] for g in found], [5, 0])

    def test_exact_uuid_match(self):
        self.assertEqual(len(candidates('0, GPU-a, 0, 0', 'GPU-ab, 12', POLICY)), 1)

if __name__ == '__main__':
    unittest.main()

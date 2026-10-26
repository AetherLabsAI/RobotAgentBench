import unittest
from unittest.mock import patch
from .gpu_admission import admit


class AdmissionTests(unittest.TestCase):
    def run_with(self, index, inventory, xml='<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>', **kwargs):
        with patch('subprocess.check_output', side_effect=[inventory, xml]):
            return admit(index, **kwargs)

    def test_approved_idle_devices(self):
        for index in (0, 1):
            value = self.run_with(index, 'GPU-test, 0, 46068, 0')
            self.assertFalse(value['shared'])
            self.assertTrue(value['no_process_stopped'])

    def test_memory_or_utilization_rejects_not_kills(self):
        for inventory in ('GPU-test, 1, 46068, 0', 'GPU-test, 0, 46068, 1'):
            with self.assertRaisesRegex(RuntimeError, 'no longer idle'):
                self.run_with(0, inventory)

    def test_graphics_or_compute_pid_rejects_even_empty_memory(self):
        xml = '<nvidia_smi_log><gpu><processes><process_info><pid>123</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
        with self.assertRaisesRegex(RuntimeError, 'no longer idle'):
            self.run_with(1, 'GPU-test, 0, 46068, 0', xml)

    def test_no_automatic_device_or_sharing_fallback(self):
        for index in (2, 3, 4, -1, True):
            with self.assertRaises(ValueError), patch('subprocess.check_output') as command:
                admit(index)
            command.assert_not_called()

    def test_gpu1_one_mib_exception_is_explicit_and_narrow(self):
        value = self.run_with(1, 'GPU-test, 1, 46068, 0', authorized_gpu1_residual_1mib=True)
        self.assertTrue(value['idle_gpu1_1mib_exception'])
        for index, inventory in ((0, 'GPU-test, 1, 46068, 0'), (1, 'GPU-test, 2, 46068, 0'),
                                 (1, 'GPU-test, 1, 46068, 1')):
            with self.assertRaises(RuntimeError):
                self.run_with(index, inventory, authorized_gpu1_residual_1mib=True)

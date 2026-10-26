import unittest
import tempfile
import signal
from pathlib import Path
from unittest.mock import patch
from gpu1_reservation import eligible,process_map,reserve_gpu1

class GPU1Tests(unittest.TestCase):
    def test_exact_uuid_and_shared_gpu_guard(self):
        m=process_map('GPU-one, 120\nGPU-two, 120\nGPU-one, 130')
        self.assertFalse(eligible(m,120,'GPU-one'))
        self.assertTrue(eligible(m,130,'GPU-one'))
        self.assertFalse(eligible(m,130,'GPU-on'))
    def test_other_device_policy_never_terminates(self):
        with tempfile.TemporaryDirectory() as d,patch('gpu1_reservation.subprocess.check_output') as query:
            with self.assertRaises(ValueError):reserve_gpu1(dict(allowed_indices=[0],selection='fixed_gpu1_reclaim'),Path(d),Path(d),lambda x:None)
            query.assert_not_called()
    def test_idle_gpu_does_not_signal(self):
        policy=dict(allowed_indices=[1],selection='fixed_gpu1_reclaim')
        with tempfile.TemporaryDirectory() as d,patch('gpu1_reservation.subprocess.check_output',return_value='GPU-one'),patch('gpu1_reservation.processes',return_value={}),patch('gpu1_reservation.inspect',return_value=[dict(index=1,uuid='GPU-one')]),patch('gpu1_reservation.os.pidfd_open',create=True) as signal:
            gpu,lock=reserve_gpu1(policy,Path(d),Path(d),lambda x:None)
            self.assertEqual(gpu['index'],1);signal.assert_not_called();lock.close()
    def test_reclaims_only_verified_gpu1_pid_and_records_it(self):
        policy=dict(allowed_indices=[1],selection='fixed_gpu1_reclaim')
        alive={130:True}
        def ident(pid):return 'start' if alive.get(pid) else None
        def sent(fd,sig):alive[130]=False
        with tempfile.TemporaryDirectory() as d,patch('gpu1_reservation.subprocess.check_output',return_value='GPU-one'),patch('gpu1_reservation.processes',return_value={130:{'GPU-one'},150:{'GPU-two'}}),patch('gpu1_reservation.identity',side_effect=ident),patch('gpu1_reservation.inspect',return_value=[dict(index=1,uuid='GPU-one')]),patch('gpu1_reservation.os.pidfd_open',return_value=90,create=True) as opened,patch('gpu1_reservation.os.close'),patch('gpu1_reservation.signal.pidfd_send_signal',side_effect=sent,create=True) as signaled:
            gpu,lock=reserve_gpu1(policy,Path(d),Path(d),lambda x:None)
            opened.assert_called_once_with(130);signaled.assert_called_once_with(90,signal.SIGTERM)
            self.assertIn('SIGTERM',(Path(d)/'gpu_reclaim.json').read_text());lock.close()
    def test_shared_gpu_process_blocks_without_signaling(self):
        with tempfile.TemporaryDirectory() as d,patch('gpu1_reservation.subprocess.check_output',return_value='GPU-one'),patch('gpu1_reservation.processes',return_value={130:{'GPU-one','GPU-two'}}),patch('gpu1_reservation.identity',return_value='start'),patch('gpu1_reservation.os.pidfd_open',create=True) as opened:
            with self.assertRaisesRegex(RuntimeError,'同时关联'):
                reserve_gpu1(dict(allowed_indices=[1],selection='fixed_gpu1_reclaim'),Path(d),Path(d),lambda x:None)
            opened.assert_not_called()

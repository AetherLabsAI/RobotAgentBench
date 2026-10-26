import unittest
import numpy as np
from fridge_to_microwave_v1.continuous20 import convert


class Conversion(unittest.TestCase):
    def fixture(self,n=7):
        names=['j'+str(i) for i in range(20)]
        return dict(native_target=np.arange(n*20).reshape(n,20),
            timestamp=(np.arange(n)+1)/60,dof_names=np.array(names),
            base_action_body=np.zeros((n,2)),phase=np.array(['a']*3+['b']*(n-3)),
            nav_constraint_mode=np.array(['hold']*n),base_navigation_active=np.ones(n,bool)),dict(
            control_hz=60,action_names=names+['base_forward_m_s','base_yaw_rate_rad_s'])

    def test_exact_sample_and_duration(self):
        log,info=self.fixture();out,r=convert(log,info)
        np.testing.assert_array_equal(out['action'][:,:20],log['native_target'][[0,3,6]])
        self.assertEqual(r['output_frames'],3)
        self.assertAlmostEqual(r['output_duration_s'],.15)
        self.assertLess(r['final_interval_extension_s'],.05)

    def test_dof_mapping(self):
        log,info=self.fixture();log['dof_names']=log['dof_names'][::-1]
        out,_=convert(log,info)
        np.testing.assert_array_equal(out['action'][0,:20],log['native_target'][0,::-1])

    def test_reject_missing_clock(self):
        log,info=self.fixture();log['timestamp'][2]+=.001
        with self.assertRaises(ValueError):convert(log,info)

    def test_reject_dropped_phase(self):
        log,info=self.fixture();log['phase'][1]='c'
        with self.assertRaises(ValueError):convert(log,info)

    def test_reject_nan(self):
        log,info=self.fixture();log['base_action_body'][4,0]=np.nan
        with self.assertRaises(ValueError):convert(log,info)

    def test_reject_drive_and_turn(self):
        log,info=self.fixture();log['base_action_body'][3]=[.1,.1]
        with self.assertRaises(ValueError):convert(log,info)

    def test_excludes_state_teleports(self):
        log,info=self.fixture();log['q']=np.full((7,20),99)
        log['nav_constraint_pose']=np.full((7,7),99)
        out,_=convert(log,info)
        self.assertNotIn('q',out);self.assertNotIn('nav_constraint_pose',out)

if __name__=='__main__':unittest.main()

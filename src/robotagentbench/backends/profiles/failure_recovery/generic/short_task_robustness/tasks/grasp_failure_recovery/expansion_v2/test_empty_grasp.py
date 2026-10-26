import unittest
import numpy as np
from .empty_grasp import draw, fingertip_goal, support_clear_goal, envelope_distance


class EmptyGraspMath(unittest.TestCase):
    def test_seed_reproducible_not_three_fixed_states(self):
        for variant in ('sofa','table_waste_empty_grasp'):
            rows=[draw(variant,s) for s in range(1,101)]
            self.assertEqual(rows,[draw(variant,s) for s in range(1,101)])
            self.assertEqual(len({r['gap_m'] for r in rows}),100)
            self.assertTrue(all(.02<=r['gap_m']<=.05 and not r['orientation_jitter'] for r in rows))
            self.assertEqual({r['miss_sign'] for r in rows},{-1,1})

    def test_exact_requested_envelope_gap_both_sides_and_robot_yaws(self):
        target=np.array([[-.05,-.04,0.],[.05,.04,.13]])
        tips=np.array([[-.02,-.01,-.03],[.02,.01,.03]])
        for angle in (-2.,0.,.7,2.):
            c,s=np.cos(angle),np.sin(angle)
            base=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            for sign in (-1,1):
                req=dict(miss_sign=sign,gap_m=.035)
                goal,axis=fingertip_goal(target,tips,base,base,req)
                actual=tips@goal[:3,:3].T+goal[:3,3]
                gap=float((actual@axis).min()-(target@axis).max())
                self.assertAlmostEqual(gap,.035,places=12)
                np.testing.assert_array_equal(goal[:3,:3],base)

    def test_bad_geometry_not_clipped(self):
        for gap in (0.,.019,.051,float('nan')):
            with self.assertRaises(ValueError):
                fingertip_goal(np.zeros((2,3)),np.ones((2,3)),np.eye(3),np.eye(3),dict(miss_sign=1,gap_m=gap))
        with self.assertRaises(ValueError):
            fingertip_goal(np.zeros((2,3)),np.ones((2,3)),np.zeros((3,3)),np.eye(3),dict(miss_sign=1,gap_m=.03))

    def test_support_clearance_uses_existing_gap_budget_not_new_height_range(self):
        target=np.array([[-.05,-.04,0.],[.05,.04,.13]])
        tips=np.array([[-.02,-.01,-.03],[.02,.01,.03]])
        hand=np.array([[-.03,-.02,-.12],[.03,.02,.05]])
        goal,_=fingertip_goal(target,tips,np.eye(3),np.eye(3),dict(miss_sign=1,gap_m=.03))
        adjusted,receipt=support_clear_goal(goal,target,tips,hand,np.eye(3),maximum_gap=.05)
        self.assertIsNotNone(adjusted)
        self.assertGreater((hand+adjusted[:3,3])[:,2].min(),target[:,2].min())
        self.assertLessEqual(envelope_distance(target,tips+adjusted[:3,3],np.eye(3)),.05)
        np.testing.assert_array_equal(adjusted[:3,:3],goal[:3,:3])
        self.assertIsNone(receipt['extra_clearance_parameter'])

    def test_impossible_fixed_direction_not_relaxed(self):
        target=np.array([[-.05,-.04,0.],[.05,.04,.01]])
        tips=np.array([[-.02,-.01,-.03],[.02,.01,.03]])
        hand=np.array([[-.03,-.02,-.3],[.03,.02,.05]])
        goal,_=fingertip_goal(target,tips,np.eye(3),np.eye(3),dict(miss_sign=1,gap_m=.04))
        adjusted,receipt=support_clear_goal(goal,target,tips,hand,np.eye(3),maximum_gap=.05)
        self.assertIsNone(adjusted)
        self.assertEqual(receipt['reason'],'fixed_direction_cannot_clear_support_within_confirmed_gap')


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from .arm_first_visible_cup_v4 import cup_after_hand,inside_region


class ArmFirstCup(unittest.TestCase):
    def test_place_after_solved_hand_preserves_height_shape_and_gap(self):
        cup=np.array([[-.04,-.04,.451],[.04,.04,.581]])
        tips=np.array([[.1,.2,.49],[.14,.24,.53]])
        original=cup.copy();solved=tips.copy()
        for sign in (-1,1):
            shift,axis=cup_after_hand(cup,tips,np.eye(3),dict(gap_m=.035,miss_sign=sign))
            self.assertEqual(shift[2],0.)
            self.assertAlmostEqual(float((tips@axis).min()-((cup+shift)@axis).max()),.035)
            np.testing.assert_allclose((cup+shift)[1]-(cup+shift)[0],cup[1]-cup[0])
        np.testing.assert_array_equal(cup,original);np.testing.assert_array_equal(tips,solved)

    def test_full_footprint_not_just_pivot_with_transferred_table(self):
        region=dict(u=[1,0],v=[0,1],start=-.5,end=.5,left=-.3,right=.3)
        delta=np.eye(4);delta[3,:3]=[2,3,0]
        self.assertTrue(inside_region([[1.6,2.8,0],[2.4,3.2,0]],delta,region))
        self.assertFalse(inside_region([[1.6,2.8,0],[2.6,3.2,0]],delta,region))

    def test_reject_unsupported_gap(self):
        with self.assertRaises(ValueError):
            cup_after_hand([[0,0,0]],[[1,1,1]],np.eye(3),dict(gap_m=.06,miss_sign=1))


if __name__=='__main__':unittest.main()

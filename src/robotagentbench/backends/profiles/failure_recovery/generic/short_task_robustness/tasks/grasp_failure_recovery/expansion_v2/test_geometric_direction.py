import unittest
import numpy as np
from .geometric_direction import side_direction


class NaturalSideGrasp(unittest.TestCase):
    def test_fixed_natural_axes_with_upper_camera(self):
        inner=np.array([0.,-.01,.15]);outer=np.array([0.,.01,.15]);palm=np.zeros(3)
        camera=np.array([-.05,0.,.03])
        r,e=side_direction(inner,outer,palm,camera)
        np.testing.assert_allclose(r@np.array([0.,0.,1.]),[1.,0.,0.],atol=1e-7)
        self.assertAlmostEqual(float((r@np.array([0.,1.,0.]))[2]),0.)
        self.assertGreater(float((r@camera)[2]),0.)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(r)),1.)
        np.testing.assert_array_equal(inner,[0.,-.01,.15])

    def test_no_fake_direction_for_degenerate_hand(self):
        with self.assertRaises(ValueError):side_direction([0,0,0],[0,0,0],[0,0,0],[1,0,0])
        with self.assertRaises(ValueError):side_direction([0,0,1],[0,0,2],[0,0,0],[1,0,0])


if __name__=='__main__':unittest.main()

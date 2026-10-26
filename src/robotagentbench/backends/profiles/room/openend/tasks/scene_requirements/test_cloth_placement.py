import unittest
try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Run this sampler test on 202 with its existing numpy environment')
from cloth_placement import sample_centres

class ClothPlacementTests(unittest.TestCase):
    def test_four_keep_original_margins(self):
        sizes=np.array([[.04,.05,.03]]*4);cloth=np.array([[0,0,0],[.34,.51,.01]])
        centres=sample_centres(sizes,cloth,np.random.default_rng(42))
        self.assertTrue(np.all(centres[:,:2]-sizes[:,:2]/2>=cloth[0,:2]+.008-1e-9))
        self.assertTrue(np.all(centres[:,:2]+sizes[:,:2]/2<=cloth[1,:2]-.008+1e-9))
        for i in range(4):
            for j in range(i):
                self.assertTrue(np.any(np.abs(centres[i,:2]-centres[j,:2])>=(sizes[i,:2]+sizes[j,:2])/2+.015-1e-9))
        self.assertTrue(np.allclose(centres[:,2]-sizes[:,2]/2,.013))
    def test_impossible_does_not_drop_or_shrink(self):
        with self.assertRaises(ValueError):
            sample_centres([[1,1,1]]*4,[[0,0,0],[.34,.51,.01]],np.random.default_rng(42))

if __name__=='__main__':unittest.main()

"""Task-local regression: existing gap limits reject the historical 63 cm miss."""
import ast
import inspect
import itertools
import unittest
import numpy as np
from .frozen_sofa_session import clearance,build


class FrozenSofa(unittest.TestCase):
    def box(self,lo,hi):return np.array(list(itertools.product(*zip(lo,hi))))
    def test_gap_boundaries_and_no_contact(self):
        points=self.box([0,0,0],[.1,.1,.1])
        for gap,accepted in ((.019,False),(.02,True),(.03,True),(.05,True),(.051,False),(.634,False),(-.001,False)):
            tips=self.box([0,-.02-gap,0],[.1,-gap,.1])
            self.assertEqual(clearance(points,tips,np.eye(3))['accepted'],accepted,gap)
    def test_rotation_invariance(self):
        points=self.box([0,0,0],[.1,.1,.1]);tips=self.box([0,-.05,0],[.1,-.03,.1])
        for angle in (0.,.6,2.1):
            c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            receipt=clearance(points@r.T,tips@r.T,r)
            self.assertTrue(receipt['accepted']);self.assertAlmostEqual(receipt['conservative_3d_envelope_distance_m'],.03)
    def test_no_scene_reinitialization_or_writes(self):
        source=inspect.getsource(build)
        self.assertNotIn('prepare_runtime_scene(',source)
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('Set','set_transforms','set_world_pose','simulate','ImportFromString'))


if __name__=='__main__':unittest.main()

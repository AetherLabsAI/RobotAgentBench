import unittest,importlib.util
if importlib.util.find_spec('pxr') is None:raise unittest.SkipTest('Requires CPU USD')
import numpy as np
from pxr import Usd,UsdGeom,Gf
from station_scene_checks import check,triangles_hit_box

class StationChecks(unittest.TestCase):
    def scene(self,x=0):
        s=Usd.Stage.CreateInMemory()
        for name,pos in [('Coffee',(0,0,1)),('Microwave',(x,0,1))]:
            p=UsdGeom.Cube.Define(s,'/World/'+name);p.CreateSizeAttr(.2);p.AddTranslateOp().Set(Gf.Vec3d(*pos))
        return s
    def audit(self,s):
        cache=UsdGeom.BBoxCache(0,['default','render'])
        def box(path):
            r=cache.ComputeWorldBound(s.GetPrimAtPath(path)).ComputeAlignedRange()
            return np.array([r.GetMin(),r.GetMax()])
        return check(s,[dict(id='coffee',members=['/World/Coffee'],support='/World/Worktop')],box)
    def test_catches_retained_microwave(self):
        r=self.audit(self.scene());self.assertFalse(r['passed']);self.assertEqual(r['failures'][0]['obstacle'],'/World/Microwave')
    def test_separated_and_touching_not_penetration(self):
        self.assertTrue(self.audit(self.scene(.3))['passed']);self.assertTrue(self.audit(self.scene(.2))['passed'])
    def test_nested_transform_is_used(self):
        s=self.scene();root=UsdGeom.Xform.Define(s,'/World/Parent');root.AddTranslateOp().Set(Gf.Vec3d(3,0,0))
        s.RemovePrim('/World/Microwave');p=UsdGeom.Cube.Define(s,'/World/Parent/Microwave');p.CreateSizeAttr(.2);p.AddTranslateOp().Set(Gf.Vec3d(-3,0,1))
        self.assertFalse(self.audit(s)['passed'])
    def test_raised_lip_does_not_fill_empty_space(self):
        box=np.array([[0,0,.923],[.2,.2,.942]])
        triangles=np.array([[[-1,-1,.92],[1,-1,.92],[1,1,.92]],
                            [[-1,.8,.92],[1,.8,.92],[1,.8,.95]]])
        self.assertFalse(triangles_hit_box(triangles,box))
    def test_triangle_inside_target_box_is_rejected(self):
        box=np.array([[0,0,0],[1,1,1]])
        triangles=np.array([[[.1,.1,.5],[.9,.1,.5],[.1,.9,.5]]])
        self.assertTrue(triangles_hit_box(triangles,box))

if __name__=='__main__':unittest.main()

import unittest
try:
    from pxr import Usd, UsdGeom, Gf
    from descendant_transforms import rigid_delta, audit
except ImportError:
    Usd = None

@unittest.skipIf(Usd is None, 'USD runtime required')
class DescendantTransformTests(unittest.TestCase):
    def scene(self, rotated=False, reset=False, scale=1):
        stage=Usd.Stage.CreateInMemory()
        root=UsdGeom.Xform.Define(stage,'/Furniture')
        root.AddTranslateOp().Set((4,2,0))
        child=UsdGeom.Xform.Define(stage,'/Furniture/Child')
        if reset:child.SetResetXformStack(True)
        if rotated:
            child.AddTranslateOp().Set((1,3,0))
            child.AddRotateZOp().Set(90)
        if scale!=1:child.AddScaleOp().Set((scale,1,1))
        mesh=UsdGeom.Mesh.Define(stage,'/Furniture/Child/Mesh')
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,2,0),(0,0,3)])
        return stage

    def test_child_rotation_translation(self):
        delta,evidence=rigid_delta(self.scene(),self.scene(True),'/Furniture')
        self.assertLess(evidence['max_residual_m'],1e-8)
        self.assertEqual(evidence['points'],4)
        self.assertAlmostEqual(delta[0][1],1)

    def test_reset_stack(self):
        _,evidence=rigid_delta(self.scene(),self.scene(True,True),'/Furniture')
        self.assertLess(evidence['max_residual_m'],1e-8)
        result=audit(self.scene(True,True))
        self.assertEqual(result['checked_xformable_nodes'],3)
        self.assertTrue(any(n['reset_stack'] for n in result['authored_nodes']))

    def test_scale_rejected(self):
        with self.assertRaisesRegex(ValueError,'rigid and upright'):
            rigid_delta(self.scene(),self.scene(scale=2),'/Furniture')

if __name__=='__main__':unittest.main()

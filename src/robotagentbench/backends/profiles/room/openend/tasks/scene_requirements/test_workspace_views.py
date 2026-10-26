import unittest
try:
    import numpy as np
    from workspace_views import fit,closeups
except ImportError:
    np=None

@unittest.skipIf(np is None,'numpy runtime required')
class WorkspaceViewTests(unittest.TestCase):
    def test_fit_contains_corners(self):
        import itertools
        for front in ((1,0),(-1,0),(0,1),(0,-1)):
            lo,hi=[-1,-.3,0],[1,.3,2]
            eye,target=map(np.asarray,fit([lo,hi],front,.12))
            forward=target-eye;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
            up=np.cross(right,forward)
            for v in itertools.product(*zip(lo,hi)):
                p=np.array(v)-eye;z=p@forward
                self.assertGreater(z,0)
                self.assertLess(abs(p@right)/z,36/56)
                self.assertLess(abs(p@up)/z,22.5/56)

    def test_shelf_faces_authored_front(self):
        record={'objects':[{'id':'snack_shelf','path':'/s'}],'tasks':[],'overview':{'furniture':[]}}
        views=closeups(record,lambda p:[[-.25,-.6,0],[.25,.6,1.85]] if p=='/s' else None,lambda p,a:(1,0))
        self.assertEqual(len(views),1)
        self.assertEqual(views[0]['label'],'货架正面')
        self.assertGreater(views[0]['eye'][0],views[0]['target'][0])

    def test_degenerate_front_stays_outside(self):
        lo,hi=np.array([-1.,-.5,0]),np.array([1.,.5,2.])
        for front in ((0,0),(1e-16,4e-16),(float('nan'),0)):
            eye,target=map(np.asarray,fit([lo,hi],front,.25))
            self.assertTrue(np.isfinite(eye).all())
            self.assertFalse(np.all(eye>=lo) and np.all(eye<=hi))

    def test_fridge_uses_y_up_geometry_front(self):
        record={'objects':[{'id':'refrigerator','path':'/fridge'}],'tasks':[],'overview':{'furniture':[]}}
        axes=[]
        def front(p,axis):
            axes.append(axis)
            return (1,0) if axis==(0,0,1) else (0,0)
        views=closeups(record,lambda p:[[-.5,-.5,0],[.5,.5,1.8]] if p=='/fridge' else None,front)
        self.assertEqual(axes,[(0,0,1)])
        self.assertGreater(views[0]['eye'][0],.5)

if __name__=='__main__':unittest.main()

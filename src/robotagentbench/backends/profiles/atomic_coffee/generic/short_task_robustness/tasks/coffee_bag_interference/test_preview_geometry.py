import unittest
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from .preview_geometry import robot_visual_boxes,rear_side_normal,rear_entry


class PreviewGeometryTests(unittest.TestCase):
    def test_rear_entry_uses_fixture_space_without_disabling_fixture(self):
        from types import SimpleNamespace
        points=np.array([[-.1,-.1,-.1],[.1,.1,.1]])
        asset=SimpleNamespace(sample=lambda _: (points,None),collision_groups=lambda:{'palm':[0,1]})
        fixture=np.array([[[.3,-1,-1],[.4,1,1]]]);before=fixture.copy()
        entry,info=rear_entry(asset,np.eye(3),np.zeros(3),np.array([1.,0,0]),fixture,[[-.1,-.1,-.1],[.1,.1,.1]])
        np.testing.assert_allclose(entry,[.1,0,0])
        np.testing.assert_array_equal(fixture,before)
        self.assertAlmostEqual(info['available_rear_distance_m'],.2)
        with self.assertRaises(ValueError):
            rear_entry(asset,np.eye(3),np.zeros(3),np.array([1.,0,0]),[[[-1,-1,-1],[1,1,1]]],[[-.1,-.1,-.1],[.1,.1,.1]])

    def test_rear_side_rotates_with_bag_and_station(self):
        for axis in ([1.,0,0],[-1.,0,0],[.7,.7,0],[-.7,-.7,0]):
            n=rear_side_normal(axis,[2,1,1],[1,1,0])
            self.assertGreater(n[0],0)
            self.assertAlmostEqual(np.linalg.norm(n),1.)
        with self.assertRaises(ValueError):rear_side_normal([0,1,0],[2,1,1],[1,1,0])

    def test_filtered_grippers_do_not_remove_native_arm_envelope(self):
        stage=Usd.Stage.CreateInMemory()
        arm='/World/Robot/arm_r_link6';grip='/World/Robot/gripper_r_inner_link4'
        for path in (arm,grip):
            p=UsdGeom.Xform.Define(stage,path).GetPrim();UsdPhysics.RigidBodyAPI.Apply(p)
            UsdGeom.Cube.Define(stage,path+'/Visual')
        native=np.eye(4);native[3,:3]=[10.,0,0]
        boxes=robot_visual_boxes(stage,[grip],{arm:native,grip:np.eye(4)})
        self.assertEqual(len(boxes),1)
        np.testing.assert_allclose(boxes[0],[[9,-1,-1],[11,1,1]])

    def test_missing_native_arm_pose_is_not_replaced_by_usd_guess(self):
        stage=Usd.Stage.CreateInMemory();UsdGeom.Cube.Define(stage,'/World/Robot/arm_r_link6/Visual')
        with self.assertRaises(ValueError):robot_visual_boxes(stage,(),{})


if __name__=='__main__':unittest.main()

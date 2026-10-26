import unittest
from pxr import Usd,UsdGeom,UsdPhysics
from .human_collision import CoffeeHumanPairs


class HumanCollisionTests(unittest.TestCase):
    def setUp(self):
        self.stage=Usd.Stage.CreateInMemory()
        self.robot=['/World/Robot/gripper_'+h+'_'+n for h in ('l','r') for n in ('base_link','inner_link4','outer_link4')]
        self.others=['/World/Robot/arm_l_link7','/World/Robot/arm_r_link6','/World/Robot/wrist_camera_r','/World/Task75/Bag']
        self.human=['/World/Human/Bodies/hand_R','/World/Human/Bodies/thumb_03_R']
        for path in self.robot+self.others+self.human:
            p=UsdGeom.Xform.Define(self.stage,path).GetPrim();UsdPhysics.RigidBodyAPI.Apply(p)
            shape=UsdGeom.Cube.Define(self.stage,path+'/Shape').GetPrim();UsdPhysics.CollisionAPI.Apply(shape)
        self.filter=CoffeeHumanPairs(self.stage,'/World/Human',self.human,self.others[-1])

    def test_only_human_to_grippers_is_filtered_before_release(self):
        self.assertEqual(set(self.filter.grippers),set(self.robot))
        for p in self.robot+self.others:
            self.assertFalse(self.stage.GetPrimAtPath(p).HasAPI(UsdPhysics.FilteredPairsAPI))
            self.assertIsNot(self.stage.GetPrimAtPath(p+'/Shape').GetAttribute('physics:collisionEnabled').Get(),False)
        for path in self.human:
            rel=UsdPhysics.FilteredPairsAPI(self.stage.GetPrimAtPath(path)).GetFilteredPairsRel()
            self.assertEqual(set(map(str,rel.GetTargets())),set(self.robot))

    def test_target_filter_does_not_erase_gripper_filter(self):
        facts=dict(phase='retreat',released=True,constraint_active=False,previous_closure=0.,supported=True)
        self.assertFalse(self.filter.update(**dict(facts,previous_closure=.1)))
        self.assertTrue(self.filter.update(**facts));self.assertFalse(self.filter.update(**facts))
        expected=set(self.robot+[self.others[-1]])
        for path in self.human:
            rel=UsdPhysics.FilteredPairsAPI(self.stage.GetPrimAtPath(path)).GetFilteredPairsRel()
            self.assertEqual(set(map(str,rel.GetTargets())),expected)

    def test_unknown_filter_mutation_fails_closed(self):
        self.filter.relations[0].AddTarget('/World/Robot/arm_r_link6')
        with self.assertRaises(RuntimeError):self.filter.evidence()


if __name__=='__main__':unittest.main()

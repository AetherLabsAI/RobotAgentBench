import unittest
import math
from copy import deepcopy
from .joint_initialization import pair_candidates,direction_check
from .training_domain import contains_origin
from .proximity import CoffeeBagApproachTrigger


class JointTests(unittest.TestCase):
    training=dict(origin_bounds_world_xy_m=[[-.05,-.05],[.05,.05]])
    robot=dict(root_xyz=[-1.,0.,0.],yaw=0.)

    def test_seeded_pairs_stay_in_original_domain(self):
        for seed in range(100):
            candidates=pair_candidates(seed,[.02,-.02,.92],self.training,self.robot)
            self.assertEqual(candidates,pair_candidates(seed,[.02,-.02,.92],self.training,self.robot))
            self.assertEqual(len(candidates),1)
            self.assertEqual(candidates[0]['direction'],'continuous_away_semicircle')
            for p in candidates:
                if not p['admitted_domain']:continue
                self.assertTrue(contains_origin(p['start_origin_m'],self.training))
                self.assertTrue(contains_origin(p['destination_origin_m'],self.training))
                self.assertEqual(p['start_origin_m'][2],.92)
                self.assertTrue(p['initial_direction_check']['allowed'])
                self.assertGreater(p['distance_m'],0)

    def test_initial_start_is_derived_from_endpoint(self):
        p=pair_candidates(3,[.02,-.02,.92],self.training,self.robot)[0]
        self.assertEqual(p['destination_origin_m'],[.02,-.02,.92])
        self.assertLess(p['start_origin_m'][0],p['destination_origin_m'][0])

    def test_continuous_direction_rotates_with_reference_frame(self):
        original=pair_candidates(4,[.02,-.02,.92],self.training,self.robot)[0]
        rotated=pair_candidates(4,[.02,.02,.92],self.training,dict(root_xyz=[0.,-1.,0.],yaw=math.pi/2))[0]
        self.assertEqual(original['sampled_angle_offset_deg'],rotated['sampled_angle_offset_deg'])
        self.assertAlmostEqual(-original['direction_world_xy_unit'][1],rotated['direction_world_xy_unit'][0])
        self.assertAlmostEqual(original['direction_world_xy_unit'][0],rotated['direction_world_xy_unit'][1])

    def test_angles_cover_both_sides_and_are_not_three_fixed_axes(self):
        angles=[pair_candidates(seed,[.02,-.02,.92],self.training,self.robot)[0]['sampled_angle_offset_deg']
                for seed in range(1000)]
        self.assertTrue(all(-90<=a<=90 for a in angles))
        self.assertLess(min(angles),-89.)
        self.assertGreater(max(angles),89.)
        self.assertEqual(len(set(angles)),1000)
        self.assertTrue(all(any(lo<=a<hi for a in angles) for lo,hi in
                            [(-90,-60),(-60,-30),(-30,0),(0,30),(30,60),(60,90)]))

    def test_straight_path_cannot_first_approach_even_if_endpoint_farther(self):
        self.assertFalse(direction_check([1,0,0],[-2,0,0],[0,0,0])['allowed'])
        self.assertFalse(direction_check([1,0,0],[.9,0,0],[0,0,0])['allowed'])
        self.assertTrue(direction_check([1,0,0],[1,.03,0],[0,0,0])['allowed'])
        self.assertTrue(direction_check([1,0,0],[1.02,0,0],[0,0,0])['allowed'])
        self.assertFalse(direction_check([1,0,0],[1,0,0],[0,0,0])['allowed'])

    def test_measured_semicircle_includes_lateral_boundaries_not_front(self):
        for offset,allowed in [(-120,False),(-90,True),(-45,True),(0,True),(45,True),(90,True),(120,False)]:
            angle=math.radians(offset)
            end=[1+.1*math.cos(angle),.1*math.sin(angle),0]
            check=direction_check([1,0,0],end,[0,0,0])
            self.assertEqual(check['allowed'],allowed)
            self.assertAlmostEqual(check['angle_offset_deg'],offset)
        check=direction_check([0,0,.92],[.03,0,.92],[0,0,1.02])
        self.assertFalse(check['allowed']);self.assertFalse(check['direction_defined'])

    def test_activation_locks_direction_even_if_jaw_moves_later(self):
        gate=CoffeeBagApproachTrigger('bag',reservation=dict(planned_bag_origin=[.04,0,0]));gate.arm()
        f=dict(simulation_time_s=0.,target_path='bag',target_bounds_local_m=[[-.01,-.01,-.01],[.01,.01,.01]],
               gripper_centers_target_local_m={'l':[-.1,0,0],'r':[1,1,1]},
               gripper_centers_world_m={'l':[-.1,0,0],'r':[1,1,1]},target_origin_world_m=[0,0,0],
               robot_contacting_target=False,robot_holding_target=False,target_supported=True,contacts_complete=True)
        first=gate.observe(f,route_clear=True)
        self.assertTrue(first['start']);locked=deepcopy(first['direction_check'])
        first['direction_check']['reference_position_m'][0]=100  # caller cannot mutate the lock
        f['simulation_time_s']=1.;f['gripper_centers_world_m']['l']=[.1,0,0]
        f['gripper_centers_target_local_m']['l']=[.1,0,0]
        later=gate.observe(f,route_clear=True)
        self.assertFalse(later['start']);self.assertEqual(later['direction_check'],locked)

    def test_trigger_uses_live_jaw_not_robot_base(self):
        gate=CoffeeBagApproachTrigger('bag',reservation=dict(planned_bag_origin=[.04,0,0]))
        gate.arm()
        f=dict(simulation_time_s=0.,target_path='bag',target_bounds_local_m=[[-.01,-.01,-.01],[.01,.01,.01]],
               gripper_centers_target_local_m={'l':[.1,0,0],'r':[1,1,1]},
               gripper_centers_world_m={'l':[.1,0,0],'r':[1,1,1]},target_origin_world_m=[0,0,0],
               robot_contacting_target=False,robot_holding_target=False,target_supported=True,contacts_complete=True)
        before=deepcopy(f);event=gate.observe(f,route_clear=True)
        self.assertFalse(event['start']);self.assertFalse(event['fired'])
        self.assertEqual(event['reason'],'destination_approaches_measured_jaw')
        self.assertEqual(before,f)
        f['simulation_time_s']=1.;f['gripper_centers_target_local_m']['l']=[-.1,0,0]
        f['gripper_centers_world_m']['l']=[-.1,0,0]
        event=gate.observe(f,route_clear=True)
        self.assertTrue(event['start']);self.assertTrue(event['direction_check']['allowed'])

    def test_invalid_or_missing_positions_fail_closed(self):
        with self.assertRaises(ValueError):pair_candidates(True,[0,0,.92],self.training,self.robot)
        with self.assertRaises(ValueError):pair_candidates(0,[.06,0,.92],self.training,self.robot)
        with self.assertRaises(ValueError):direction_check([0,0,0],[1,0,0],[float('nan'),0,0])

    def test_initial_robot_reference_is_not_reported_as_measured_jaw(self):
        pair=pair_candidates(3,[.02,-.02,.92],self.training,self.robot)[0]
        self.assertEqual(pair['initial_direction_check']['reference_kind'],'original_robot_base')
        self.assertEqual(direction_check([1,0,0],[2,0,0],[0,0,0])['reference_kind'],'measured_triggering_jaw')
        with self.assertRaises(ValueError):direction_check([1,0,0],[2,0,0],[0,0,0],reference_kind='guessed')

    def test_continuous_pair_translates_all_beans(self):
        from unittest.mock import patch
        from pxr import Usd,UsdGeom,Gf
        from evaluation.tasks.coffee_machine.specs import BAG,BEANS
        from .joint_initialization import author_pair
        stage=Usd.Stage.CreateInMemory()
        bag=UsdGeom.Xform.Define(stage,BAG)
        bag.AddTranslateOp().Set(Gf.Vec3d(0,-.02,.92))
        for i in range(100):
            UsdGeom.Xform.Define(stage,BEANS+'/Bean_%03d'%i).AddTranslateOp().Set(Gf.Vec3d(.001*i,-.02,1.))
        def admission(stage,asset,seed,initial,*,pair):
            return dict(admitted=True)
        with patch('short_task_robustness.tasks.coffee_bag_interference.training_domain.domain',return_value=self.training), \
             patch('short_task_robustness.tasks.coffee_bag_interference.preview_geometry.plan',side_effect=admission), \
             patch('short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand.AdaptiveHand',return_value=object()):
            result=author_pair(stage,dict(seed=3,robot_base=self.robot))
        self.assertEqual(result['pair']['direction'],'continuous_away_semicircle')
        self.assertEqual(result['rejected_directions'],[])
        self.assertEqual(result['translated_body_count'],101)
        self.assertLessEqual(result['translation_matrix_max_error'],1e-9)

    def test_blocked_sample_is_not_replaced_with_fixed_left_right(self):
        from unittest.mock import patch
        from pxr import Usd,UsdGeom,Gf
        from evaluation.tasks.coffee_machine.specs import BAG,BEANS
        from .joint_initialization import author_pair
        stage=Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage,BAG).AddTranslateOp().Set(Gf.Vec3d(0,-.02,.92))
        for i in range(100):
            UsdGeom.Xform.Define(stage,BEANS+'/Bean_%03d'%i).AddTranslateOp().Set(Gf.Vec3d(.001*i,-.02,1.))
        with patch('short_task_robustness.tasks.coffee_bag_interference.training_domain.domain',return_value=self.training), \
             patch('short_task_robustness.tasks.coffee_bag_interference.preview_geometry.plan',side_effect=ValueError('fixture blocks sample')) as admission, \
             patch('short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand.AdaptiveHand',return_value=object()):
            with self.assertRaisesRegex(ValueError,'fixture blocks sample'):
                author_pair(stage,dict(seed=3,robot_base=self.robot))
        self.assertEqual(admission.call_count,1)
        self.assertEqual(list(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(BAG)).ExtractTranslation()),[0,-.02,.92])


if __name__=='__main__':unittest.main()

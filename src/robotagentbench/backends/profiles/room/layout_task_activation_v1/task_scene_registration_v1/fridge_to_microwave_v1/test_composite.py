from dataclasses import replace
import unittest
from fridge_to_microwave_v1.composite import Composite,Sample,FridgeFrame,PutFacts


class Tests(unittest.TestCase):
    def setUp(self):
        self.c=Composite('episode','/World/RoomEpisode/FoodBox','beef_box');self.i=-1;self.time=10.
    def sample(self, *, door=0.,held=False,out=False,opened=False,closed=True,inside=False,
               supported=False,interval=True,collision=False,wall=False):
        self.i+=1;self.time+=.05
        f=FridgeFrame('episode',self.c.target_prim,self.i,self.time,door,(0.,door),held,out,collision,interval,True)
        p=PutFacts(self.c.variant,self.c.target_prim,opened,closed,inside,supported,held,False,not held)
        return Sample('episode',self.c.target_prim,self.i,self.time,f,p,wall,False)
    def advance(self,**kwargs):return self.c.observe(self.sample(**kwargs))
    def to_put(self):
        self.advance();self.advance(door=65);self.advance(door=65,held=True,out=True)
        for _ in range(102):r=self.advance(held=True,out=True)
        self.assertEqual(r['phase'],'microwave');return r
    def finish(self):
        self.advance(held=True,out=True,opened=True,closed=False)
        self.advance(out=True,opened=True,closed=False,inside=True,supported=True)
        for _ in range(42):
            r=self.advance(out=True,inside=True,supported=True)
            if r['success']:return r
        self.fail('Hold never completed')
    def test_complete_same_box_left_hand_supported(self):
        self.to_put();r=self.finish();self.assertEqual(r['score'],18);self.assertTrue(r['success'])
        self.assertEqual(len(r['transitions']),1);self.assertFalse(r['transitions'][0]['scene_reset'])
        self.assertEqual(r['source_api_ids'],['26-1','27-1'])
    def test_leftover_variant_same_pipeline(self):
        self.c=Composite('episode',self.c.target_prim,'leftover_box')
        self.to_put();r=self.finish();self.assertEqual(r['source_api_ids'],['26-2','27-2'])
    def test_initial_open_held_is_baseline_not_rejected_or_awarded(self):
        r=self.advance(door=70,held=True,out=True,opened=True,closed=False)
        self.assertEqual(r['score'],0);self.assertEqual(r['phase'],'refrigerator')
    def test_microwave_cannot_score_before_fridge_success(self):
        self.advance();r=self.advance(opened=True,closed=False,inside=True,supported=True)
        self.assertEqual(r['score'],0);self.assertIsNone(r['microwave'])
    def test_wrong_box_rejected_without_consuming_frame(self):
        self.advance();f=self.sample(door=65)
        wrong=replace(f,microwave=replace(f.microwave,payload_instance='/World/OtherBox'))
        before=self.c.summary()
        with self.assertRaises(ValueError):self.c.observe(wrong)
        self.assertEqual(before,self.c.summary());self.assertEqual(self.c.observe(f)['score'],3)
    def test_missing_interval_rejected(self):
        self.advance();f=self.sample();f=replace(f,time_s=f.time_s+1,fridge=replace(f.fridge,time_s=f.time_s+1))
        with self.assertRaises(ValueError):self.c.observe(f)
    def test_conflicting_grasp_rejected(self):
        f=self.sample(held=True)
        with self.assertRaises(ValueError):self.c.observe(replace(f,microwave=replace(f.microwave,held_by_robot=False)))
    def test_take_hold_pause_does_not_count_unobserved_time(self):
        self.advance();self.advance(door=65,held=True,out=True)
        for _ in range(50):self.advance(held=True,out=True)
        self.c.pause();self.time+=20
        r=self.advance(held=True,out=True)
        self.assertEqual(r['phase'],'refrigerator');self.assertEqual(r['refrigerator']['continuous_hold_s'],0)
    def test_put_hold_pause_does_not_count_unobserved_time(self):
        self.to_put();self.advance(held=True,out=True,opened=True,closed=False)
        self.advance(out=True,opened=True,closed=False,inside=True,supported=True)
        for _ in range(20):self.advance(out=True,inside=True,supported=True)
        self.c.pause();self.time+=10
        r=self.advance(out=True,inside=True,supported=True)
        self.assertFalse(r['success']);self.assertEqual(r['microwave']['details']['final_hold_s'],0)
    def test_original_deductions_are_not_removed(self):
        self.advance();self.advance(door=65,collision=True)
        r=self.c.finalize();self.assertEqual(r['score'],0)
        self.assertEqual(r['refrigerator']['collision_deduction'],1)
        self.assertEqual(r['refrigerator']['hold_failure_deduction'],3)
        self.assertEqual(r,self.c.finalize())
    def test_repeated_frame_no_double_reward(self):
        f=self.sample();self.c.observe(f);f=self.sample(door=65)
        r=self.c.observe(f);self.assertEqual(r,self.c.observe(f))
    def test_transition_requests_approved_rates_only(self):
        self.assertEqual(self.c.requested_physics_hz,60)
        self.to_put();self.assertEqual(self.c.requested_physics_hz,240)
    def test_unrelated_variant_rejected(self):
        with self.assertRaises(ValueError):Composite('episode','/World/Box','salad_box')
    def test_put_collision_penalty_preserved(self):
        self.to_put();self.advance(held=True,out=True,opened=True,closed=False,wall=True)
        self.advance(out=True,opened=True,closed=False,inside=True,supported=True)
        for _ in range(42):
            r=self.advance(out=True,inside=True,supported=True)
            if r['success']:break
        self.assertTrue(r['success']);self.assertEqual(r['score'],17)


if __name__=='__main__':unittest.main()

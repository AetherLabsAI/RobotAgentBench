"""Synthetic lifecycle tests, not native manipulation success."""
import unittest
from types import SimpleNamespace as NS
from room_task_runtime_v1 import DualTaskCycle
from test_coffee_hotwake_v1 import Clock


class Tests(unittest.TestCase):
    def setUp(self):
        self.coffee=NS(closed=False,service=NS(consumer=None,controller=NS(dispensing=False),
                       animation=NS(counts=lambda:dict(airborne=0))))
        self.fryer=NS(consumer=None)
        self.clock=Clock()
        def config(x):
            return dict(target=[x,0],center=[x,0],targetPolygon=[[x-.2,-.2],[x+.2,-.2],[x+.2,.2],[x-.2,.2]],
                        wake=1.32,wakeHalfAngle=45,nearWake=0,exit=1.82,preload=2.12,
                        pose=[x-1,0],poseYaw=0,poseRadius=.1,poseYawTolerance=10)
        self.runtime=DualTaskCycle(self.coffee,self.fryer,self.clock,dict(coffee=config(0),fryer=config(4)))

    def tick(self,xy,yaw=0,command=(0,0),**kwargs):
        p=self.runtime.begin_control(dict(xy=xy,yaw=yaw),command_vw=command,base_velocity=[0]*6,**kwargs)
        self.clock.current_time+=.05
        self.runtime.finish_control(completed_substeps=p['substeps'])
        return p

    def test_fryer_only_wakes_and_exits(self):
        self.assertEqual(self.tick([3,0])['physics_hz'],240)
        self.assertFalse(self.runtime.activation['coffee']['active'])
        self.assertTrue(self.runtime.activation['fryer']['overlap'])
        self.assertEqual(self.tick([10,10])['physics_hz'],60)

    def test_heading_not_center_distance(self):
        self.assertEqual(self.tick([-1,0],180)['physics_hz'],60)
        self.assertEqual(self.tick([-1,0],0)['physics_hz'],240)
        self.assertEqual(self.tick([-1,0],180)['activations']['coffee']['reason'],'retained')

    def test_two_scores_are_independent_and_persist(self):
        c=NS(process=NS(ended=False)); f=NS(termination=None)
        self.coffee.service.consumer=c; self.fryer.consumer=f
        for pos in ([3,0],[10,10],[-1,0],[10,10],[3,0]):
            self.assertEqual(self.tick(pos)['physics_hz'],240)
            self.assertIs(self.coffee.service.consumer,c); self.assertIs(self.fryer.consumer,f)
        c.process.ended=True
        self.assertEqual(self.tick([10,10])['physics_hz'],240)
        f.termination='success'
        self.assertEqual(self.tick([10,10])['physics_hz'],60)

    def test_no_score_created_by_clock_layer(self):
        self.tick([3,0]); self.tick([-1,0])
        self.assertIsNone(self.coffee.service.consumer); self.assertIsNone(self.fryer.consumer)

    def test_moving_switch_deferred(self):
        self.assertTrue(self.tick([3,0],command=(.1,0))['switch_pending'])
        self.assertEqual(self.tick([3,0])['physics_hz'],240)

    def test_protected_object_pins_region(self):
        self.assertEqual(self.tick([10,10],flags=dict(fryer=dict(held=True)))['activations']['fryer']['state'],'PINNED')

    def test_overlap_both_regions(self):
        self.runtime.configs['fryer']=self.runtime.configs['coffee'].copy()
        p=self.tick([-1,0])
        self.assertTrue(all(a['overlap'] for a in p['activations'].values()))
        self.assertEqual(p['substeps'],12)

    def test_duplicate_begin_rejected(self):
        self.runtime.begin_control(dict(xy=[3,0],yaw=0),command_vw=[0,0],base_velocity=[0]*6)
        with self.assertRaises(RuntimeError): self.tick([3,0])

    def test_incomplete_control_rejected(self):
        self.runtime.begin_control(dict(xy=[3,0],yaw=0),command_vw=[0,0],base_velocity=[0]*6)
        with self.assertRaises(RuntimeError):self.runtime.finish_control(completed_substeps=3)

    def test_unknown_task_rejected(self):
        with self.assertRaises(ValueError):self.tick([3,0],flags=dict(unknown={}))


if __name__=='__main__':unittest.main()

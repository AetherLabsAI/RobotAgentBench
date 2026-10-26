from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from .proximity import CoffeeBagApproachTrigger, NativeCoffeeBagTriggerFacts

TARGET = '/World/Task75/Bag'


def facts(t=0.,left=(-.101,.1,0.),right=(1,1,1),**changes):
    return dict(simulation_time_s=t,target_path=TARGET,
        target_bounds_local_m=((0.,0.,-.06),(.065,.22,.06)),
        gripper_centers_target_local_m={'l':left,'r':right},
        **dict(dict(robot_contacting_target=False,robot_holding_target=False,
                    target_supported=True,contacts_complete=True),**changes))


class TriggerTests(unittest.TestCase):
    def gate(self):
        gate=CoffeeBagApproachTrigger(TARGET);gate.arm();return gate

    def test_admission_required_and_no_timer_activation(self):
        g=CoffeeBagApproachTrigger(TARGET)
        with self.assertRaises(RuntimeError):g.observe(facts(),route_clear=True)
        g.arm()
        for t in range(100):self.assertFalse(g.observe(facts(t),route_clear=True)['start'])
        with self.assertRaises(RuntimeError):g.arm()

    def test_left_boundary_and_single_trigger(self):
        g=self.gate()
        self.assertFalse(g.observe(facts(left=(-.100001,.1,0.)),route_clear=True)['start'])
        row=g.observe(facts(1,left=(-.1,.1,0.)),route_clear=True)
        self.assertTrue(row['start']);self.assertEqual(row['triggering_hand'],'l')
        self.assertFalse(g.observe(facts(2,left=(-.1,.1,0.)),route_clear=True)['start'])

    def test_right_hand_and_three_dimensional_distance(self):
        g=self.gate()
        self.assertFalse(g.observe(facts(right=(.03,.33,0.)),route_clear=True)['start'])
        row=g.observe(facts(1,right=(.03,.31,0.)),route_clear=True)
        self.assertTrue(row['start']);self.assertEqual(row['triggering_hand'],'r')

    def test_existing_contact_or_holding_prevents_stealing(self):
        for key in ('robot_contacting_target','robot_holding_target'):
            g=self.gate()
            self.assertFalse(g.observe(facts(left=(0,.1,0.),**{key:True}),route_clear=True)['start'])
            self.assertFalse(g.observe(facts(1,left=(0,.1,0.)),route_clear=True)['start'])

    def test_blocked_or_unsupported_does_not_consume_trigger(self):
        for supported,clear in ((True,False),(False,True)):
            g=self.gate()
            self.assertFalse(g.observe(facts(left=(0,.1,0.),target_supported=supported),route_clear=clear)['start'])
            self.assertTrue(g.observe(facts(1,left=(0,.1,0.)),route_clear=True)['start'])

    def test_missing_contact_evidence_fails_closed_without_mutation(self):
        row=facts(contacts_complete=False);before=deepcopy(row)
        with self.assertRaises(ValueError):self.gate().observe(row,route_clear=True)
        self.assertEqual(row,before)

    def test_native_atomic_reader_rotates_measured_jaw_positions(self):
        r=NativeCoffeeBagTriggerFacts.__new__(NativeCoffeeBagTriggerFacts)
        r.target=TARGET;r.bounds=((0,0,-.06),(.065,.22,.06))
        row=dict(time_s=2.,reader_step=3,
            poses={TARGET:dict(position_m=np.array([1.,2.,3.]),quaternion_wxyz=[2**-.5,0,0,2**-.5])},
            contacts={TARGET:dict(robot=False,grasped=False,supports=['counter'])})
        r.reader=SimpleNamespace(latest=row,world=SimpleNamespace(current_time=2.),last_time=2.,step=3)
        def pad(p):return SimpleNamespace(get_world_pose=lambda:(p,[1,0,0,0]))
        # Midpoint is local (-.09,.1,0), 9 cm from the rotated bag box.
        r.pads={'l':[pad([.89,1.91,3.]),pad([.91,1.91,3.])],
                'r':[pad([5,5,5]),pad([5,5,5])]}
        result=self.gate().observe(r.read(),route_clear=True)
        self.assertTrue(result['start']);self.assertAlmostEqual(result['distances_m']['l'],.09)
        r.reader.last_time=1.
        with self.assertRaises(RuntimeError):r.read()

    def test_definition_threshold_matches_runtime(self):
        import json
        from pathlib import Path
        from .proximity import TRIGGER_DISTANCE_M
        d=json.loads(Path(__file__).with_name('definition.json').read_text())
        self.assertEqual(d['trigger']['distance_m'],TRIGGER_DISTANCE_M)

    def test_source_preparation_does_not_trigger_and_shelf_is_unchanged(self):
        from short_task_robustness.tasks.stock_wooden_tray_interference.proximity import TRIGGER_DISTANCE_M
        self.assertEqual(TRIGGER_DISTANCE_M,.25)
        g=self.gate()
        row=g.observe(facts(left=(-.1915,.1,0.)),route_clear=True)
        self.assertFalse(row['start']);self.assertEqual(row['threshold_m'],.1)

    def test_confirmed_relocation_range_and_reproducibility(self):
        from .relocation_profile import sample_destination
        from .training_domain import contains_origin
        training=dict(origin_bounds_world_xy_m=[[-.05,-.05],[.05,.05]])
        start=[.0166,-.0137,.92]
        for seed in range(100):
            proposal=sample_destination(seed,start,training)
            self.assertTrue(0 < proposal['distance_m'] < .10)
            self.assertTrue(contains_origin(proposal['destination_origin_m'],training))
            self.assertEqual(proposal,sample_destination(seed,start,training))
        for seed in (-1,True,None,1.5):
            with self.assertRaises(ValueError):sample_destination(seed,start,training)


if __name__=='__main__':unittest.main()

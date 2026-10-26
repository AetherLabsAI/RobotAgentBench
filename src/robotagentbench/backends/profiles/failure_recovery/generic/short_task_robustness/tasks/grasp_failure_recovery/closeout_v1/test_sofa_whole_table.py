"""Whole-table containment and unchanged original completion prerequisites."""
from copy import deepcopy
from dataclasses import dataclass
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .sofa_whole_table import within_table, plan_metadata, port_class, RULE_VERSION, INSTRUCTION


TABLE = ((-1., -.5), (1., -.5), (1., .5), (-1., .5))


def square(x, y, half=.05):
    return ((x-half,y-half),(x+half,y-half),(x+half,y+half),(x-half,y+half))


class WholeTable(unittest.TestCase):
    def test_both_previously_reserved_halves_are_accepted(self):
        for x,y in ((-.8,.3),(.8,.3),(-.8,-.3),(.8,-.3)):
            self.assertTrue(within_table(square(x,y), [TABLE]))

    def test_overhanging_footprint_rejected_not_just_center(self):
        self.assertFalse(within_table(square(.99,0.), [TABLE]))

    def test_outside_rejected(self):
        self.assertFalse(within_table(square(1.2,0.), [TABLE]))

    def test_original_edge_rule_preserved(self):
        self.assertTrue(within_table(square(.95,0.), [TABLE]))

    def test_no_convex_envelope_over_empty_table_gap(self):
        faces=[((-1.,-.5),(-.2,-.5),(-.2,.5),(-1.,.5)),
               ((.2,-.5),(1.,-.5),(1.,.5),(.2,.5))]
        self.assertFalse(within_table(square(0.,0.), faces))

    def test_rigid_room_transform(self):
        def move(points): return [(2.-y,-3.+x) for x,y in points]
        self.assertTrue(within_table(move(square(-.8,.3)), [move(TABLE)]))
        self.assertFalse(within_table(move(square(1.2,0.)), [move(TABLE)]))

    def test_missing_geometry_is_invalid_not_credited(self):
        for footprint,faces in (([],[TABLE]),(square(0,0),[])):
            with self.assertRaises(ValueError):within_table(footprint,faces)

    def test_only_sofa_metadata_changed(self):
        plan=dict(task_name='failure_recovery_sofa',asset='remote_tv',
                  score_policy={'maximum':7.5},control_hz=20,no_progress_timeout_s=60,
                  physics_policy='original')
        old=deepcopy(plan);plan_metadata(plan)
        self.assertEqual(plan['instruction'],INSTRUCTION)
        self.assertEqual(plan['rule_version'],RULE_VERSION)
        for key in ('score_policy','control_hz','no_progress_timeout_s','physics_policy'):
            self.assertEqual(plan[key],old[key])
        with self.assertRaises(ValueError):plan_metadata(dict(task_name='clear_table_trash',asset='cup'))

    def test_original_current_surface_support_and_release(self):
        from short_tasks.tasks.sofa_to_drawer.transfer import CurrentSurface
        table='/World/Assets/CoffeeTable';binding=SimpleNamespace(destination=table)
        for changes,done,points in (({},True,3),({'held':True},False,3),
                ({'robot_contact':True},False,3),({'supports':frozenset(('/World/Floor',))},False,0),
                ({'on_floor':True},False,0),({'in_zone':False},False,0)):
            f=dict(in_zone=True,on_floor=False,supports=frozenset((table,)),held=False,robot_contact=False)
            f.update(changes);scorer=CurrentSurface()
            self.assertEqual(scorer.observe(SimpleNamespace(**f),binding,enabled=True,final_sample=True),done)
            self.assertEqual(scorer.summary()['earned'],points)
        scorer=CurrentSurface();f=SimpleNamespace(in_zone=True,on_floor=False,
            supports=frozenset((table,)),held=False,robot_contact=False)
        self.assertFalse(scorer.observe(f,binding,enabled=False,final_sample=True))
        self.assertEqual(scorer.summary()['earned'],0)

    def test_native_port_changes_only_same_frame_containment(self):
        from short_tasks.tasks.sofa_to_drawer.native_port import CombinationPort
        @dataclass(frozen=True)
        class Evidence:
            in_zone:bool=False
            held:bool=True
            supports:tuple=('/World/Assets/CoffeeTable',)
        cls=port_class();self.assertIs(cls.__init__,CombinationPort.__init__)
        port=object.__new__(cls)
        port.geometry=SimpleNamespace(footprint=lambda pose:square(.8,.3),top={'faces':[TABLE]})
        original=Evidence()
        with patch.object(CombinationPort,'facts',return_value=original) as parent:
            changed=port.facts({'pose':'same_physical_frame'},True,42)
            parent.assert_called_once_with({'pose':'same_physical_frame'},True,42)
        self.assertTrue(changed.in_zone);self.assertTrue(changed.held)
        self.assertEqual(changed.supports,original.supports)
        self.assertFalse(original.in_zone)
        source=inspect.getsource(cls.facts)
        for forbidden in ('.Set(','.simulate(','.set_transforms(','.set_world_pose('):
            self.assertNotIn(forbidden,source)


if __name__=='__main__':unittest.main()

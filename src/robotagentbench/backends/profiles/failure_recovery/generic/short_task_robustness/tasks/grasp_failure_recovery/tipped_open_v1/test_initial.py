from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from .geometry import below_pose
from .bank import load,validate,choose,checksum

ROOT=Path(__file__).resolve().parent


class AuthoredStart(unittest.TestCase):
    def setUp(self):self.value=load(ROOT/'compile/elevated_open_v2/bank.json')
    def test_eight_checked_profiles(self):
        self.assertEqual(len(self.value['profiles']),8)
        self.assertEqual(len({p['training_provenance']['episode_index'] for p in self.value['profiles']}),8)
        self.assertEqual({p['active_hand'] for p in self.value['profiles']},{'l'})
    def test_new_definition_does_not_require_drop_history(self):
        d=json.loads((ROOT/'definition.json').read_text())
        self.assertFalse(d['physical_history_required']);self.assertEqual(d['initial_score_expected'],0)
        self.assertEqual(d['completion'],'same_cup_released_inside_correct_bin')
    def test_cannot_claim_physical_drop(self):
        v=deepcopy(self.value);v['profiles'][0]['physical_failure_generation_verified']=True;v['content_sha256']=checksum(v)
        with self.assertRaises(ValueError):validate(v)
    def test_open_not_closed(self):
        v=deepcopy(self.value);p=v['profiles'][0]
        for n in p['gripper_opening']['open_coordinates']:
            p['robot_initialization']['positions'][n]=0
            if n in p['robot_initialization']['hold_targets']:p['robot_initialization']['hold_targets'][n]=0
        v['content_sha256']=checksum(v)
        with self.assertRaises(ValueError):validate(v)
    def test_deterministic_and_diverse_choice(self):
        self.assertEqual(choose(self.value,701),choose(self.value,701))
        self.assertEqual(len({choose(self.value,s)['id'] for s in range(1,65)}),8)
    def test_cannot_edit_source_arm(self):
        v=deepcopy(self.value);p=v['profiles'][0];n=next(n for n in p['training_provenance']['source_state18_rad_or_m'] if '_arm_l_' in n)
        p['robot_initialization']['positions'][n]+=.01
        if n in p['robot_initialization']['hold_targets']:p['robot_initialization']['hold_targets'][n]+=.01
        v['content_sha256']=checksum(v)
        with self.assertRaises(ValueError):validate(v)
    def test_preserve_gap_and_table_height(self):
        origin=np.array([0.,0.,.762]);matrix=np.eye(4);matrix[:3,3]=origin
        cup=np.array([[x,y,z] for x in (-.03,.03) for y in (-.03,.03) for z in (.702,.822)])
        delta=np.eye(4);delta[3,2]=.2;table=np.array([[-1.,-1.,.4],[1.,1.,.5]])
        m,p,v,receipt=below_pose(matrix,cup,cup.copy(),delta,table,np.array([.2,.3]))
        self.assertAlmostEqual(p[:,2].min(),.702)
        self.assertAlmostEqual(receipt['support_gap_preserved_m'],.002)
        self.assertTrue(np.allclose((p.min(0)+p.max(0))[:2]/2,[.2,.3]))
        self.assertAlmostEqual(np.ptp(p[:,2]),.06)
        self.assertFalse(receipt['physical_drop_replayed'])
    def test_actual_hand_above_cup(self):
        for p in self.value['profiles']:
            self.assertGreater(p['fit']['hand_lowest_z_m'],p['fit']['cup_highest_z_m'])
            self.assertGreater(p['fit']['hand_lowest_z_m'],p['fit']['upright_source_cup_highest_z_m'])
            self.assertTrue(p['fit']['center_below_fingertips_xy'])
            self.assertFalse(p['validation']['native'])


if __name__=='__main__':unittest.main()

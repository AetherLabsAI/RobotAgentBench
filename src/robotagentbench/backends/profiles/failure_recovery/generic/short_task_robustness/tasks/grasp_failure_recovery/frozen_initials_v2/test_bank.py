"""Portable bank contracts, no dataset/Isaac/optimizer needed."""
import ast
import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from ..frozen_initials_v1.bank import load,select,validate
from ..expansion_v2.empty_grasp import draw

ROOT=Path(__file__).resolve().parent
BANK=ROOT/'compile/cup_training_adapt_v5/bank_v2.json'


def coverage(bank,count=2):
    chosen={p['id']:[] for p in bank['cases']['table_waste_empty_grasp']['profiles']}
    for seed in range(1,10001):
        p=select(bank,'table_waste_empty_grasp',seed,miss_sign=draw('table_waste_empty_grasp',seed)['miss_sign'])
        if len(chosen[p['id']])<count:chosen[p['id']].append(seed)
        if all(len(v)==count for v in chosen.values()):return chosen
    raise ValueError('Bank profile cannot be selected by the real seeded sampler')


class BankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bank=load(BANK);cls.profiles=cls.bank['cases']['table_waste_empty_grasp']['profiles']

    def test_previous_bank_unchanged(self):
        p=ROOT.parent/'frozen_initials_v1/configurations/bank_v1.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'48fda0e41959482ecb750dae079409acb787a15a71a260479e84941440d1b8ee')

    def test_multiple_independent_profiles_each_hand(self):
        for hand in ('l','r'):
            rows=[p for p in self.profiles if p['active_hand']==hand]
            self.assertGreater(len(rows),1)
            vectors={tuple(p['robot_initialization']['positions'][f'idx{2 if hand=="l" else 6}{i}_arm_{hand}_joint{i}'] for i in range(1,8)) for p in rows}
            self.assertEqual(len(vectors),len(rows))

    def test_all_empty_closed(self):
        for p in self.profiles:
            q=p['robot_initialization']['positions'];hand=p['active_hand'];f=[v for n,v in q.items() if '_gripper_'+hand+'_' in n]
            self.assertEqual(len(f),8);self.assertTrue(all(v==0 for v in f))

    def test_training_exact_and_adapted_are_distinguished(self):
        for p in self.profiles:
            pr=p.get('training_provenance')
            if not pr:continue
            q=p['robot_initialization']['positions'];h=p['active_hand']
            actual=[q[f'idx{2 if h=="l" else 6}{i}_arm_{h}_joint{i}'] for i in range(1,8)]
            if pr['active_arm_copied_exactly']:self.assertEqual(actual,pr['measured_active_arm_q_rad'])
            else:
                self.assertTrue(pr['training_seeded_offline_ik']);self.assertEqual(actual,pr['actual_frozen_arm_q_rad'])
                self.assertTrue(pr['ik']['converged'])
            self.assertFalse(p['validation']['native']);self.assertTrue(p['validation']['random_revalidation_required'])

    def test_real_seed_sampler_covers_every_profile(self):
        values=coverage(self.bank);self.assertEqual(len(values),len(self.profiles));self.assertTrue(all(len(v)==2 for v in values.values()))

    def test_content_tampering_rejected(self):
        b=deepcopy(self.bank);b['cases']['table_waste_empty_grasp']['profiles'][0]['active_hand']='x'
        with self.assertRaises(ValueError):validate(b)

    def test_startup_has_no_training_or_solver_calls(self):
        tree=ast.parse((ROOT/'cup.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('solve','least_squares','read_table','np_load') for n in ast.walk(tree)))
        self.assertFalse(self.bank['runtime_training_dataset_required'])


if __name__=='__main__':unittest.main()

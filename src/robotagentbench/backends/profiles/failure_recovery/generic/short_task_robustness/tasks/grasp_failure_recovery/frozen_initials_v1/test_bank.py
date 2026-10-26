import ast
import json
from pathlib import Path
import unittest
from .bank import CASES,SCHEMA,checksum,validate,select


def example():
    q={f'q{i}':float(i) for i in range(34)}
    profiles=[dict(id='cup_'+hand,variant='table_waste_empty_grasp',active_hand=hand,cpu_accepted=True,
        robot_initialization=dict(positions=q,hold_targets=q),supported_miss_signs=[sign]) for hand,sign in (('l',-1),('r',1))]
    v=dict(schema=SCHEMA,cases={n:dict(profiles=profiles if n=='table_waste_empty_grasp' else []) for n in CASES})
    v['content_sha256']=checksum(v);return v


class FrozenBank(unittest.TestCase):
    def test_deterministic_selection_and_no_mutation(self):
        b=example();before=json.dumps(b,sort_keys=True)
        a=select(b,'table_waste_empty_grasp',401,miss_sign=1)
        self.assertEqual(a,select(b,'table_waste_empty_grasp',401,miss_sign=1))
        a['robot_initialization']['positions']['q0']=99
        self.assertEqual(before,json.dumps(b,sort_keys=True))

    def test_hand_coverage_not_mirrored_or_substituted(self):
        b=example();self.assertEqual(select(b,'table_waste_empty_grasp',2,miss_sign=-1)['active_hand'],'l')
        with self.assertRaises(ValueError):select(b,'table_waste_empty_grasp',2,hand='l',miss_sign=1)
        with self.assertRaises(ValueError):select(b,'table_waste_tipped_cup',2)

    def test_hash_seed_and_rejected_pose_guard(self):
        b=example();b['cases']['sofa']['profiles'].append({})
        with self.assertRaises(ValueError):validate(b)
        for seed in (0,True,-1,2**32):
            with self.assertRaises(ValueError):select(example(),'table_waste_empty_grasp',seed)
        b=example();b['cases']['table_waste_empty_grasp']['profiles'][0]['cpu_accepted']=False;b['content_sha256']=checksum(b)
        with self.assertRaises(ValueError):validate(b)

    def test_complete_snapshot_and_exact_hold(self):
        b=example();p=b['cases']['table_waste_empty_grasp']['profiles'][0]
        p['robot_initialization']['positions'].pop('q0');b['content_sha256']=checksum(b)
        with self.assertRaises(ValueError):validate(b)

    def test_online_code_has_no_ik_or_optimizer_call(self):
        tree=ast.parse((Path(__file__).parent/'cup.py').read_text())
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
            for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn('solve',calls);self.assertNotIn('least_squares',calls)


if __name__=='__main__':unittest.main()

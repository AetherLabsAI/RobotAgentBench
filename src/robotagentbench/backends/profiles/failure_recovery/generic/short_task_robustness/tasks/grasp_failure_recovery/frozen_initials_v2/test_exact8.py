import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from .exact8 import IDS,DEFAULT_BANK,checked_bank,selection
from ..frozen_initials_v1.bank import load,checksum
from ..expansion_v2.empty_grasp import draw

ROOT=Path(__file__).resolve().parent


def coverage(count=2):
    bank=checked_bank();out={key:[] for key in IDS}
    for seed in range(1,10001):
        profile,request=selection(bank,seed)
        if len(out[profile['id']])<count:out[profile['id']].append(seed)
        if all(len(v)==count for v in out.values()):return out
    raise ValueError('Unreachable approved profile')


class ExactEightTest(unittest.TestCase):
    def test_exact_eight_only(self):
        b=checked_bank();p=b['cases']['table_waste_empty_grasp']['profiles']
        self.assertEqual(len(p),8);self.assertEqual({r['id'] for r in p},set(IDS))
        self.assertEqual({r['active_hand'] for r in p},{'l'})

    def test_profiles_are_bit_for_bit_unchanged(self):
        old=load(ROOT/'compile/cup_training_adapt_v5/bank_v2.json')
        old_by_id={p['id']:p for p in old['cases']['table_waste_empty_grasp']['profiles']}
        new=checked_bank()
        for p in new['cases']['table_waste_empty_grasp']['profiles']:self.assertEqual(p,old_by_id[p['id']])
        for k,v in new['cases'].items():
            if k!='table_waste_empty_grasp':self.assertEqual(v,old['cases'][k])

    def test_legacy_bank_stays_immutable(self):
        path=ROOT/'compile/cup_training_adapt_v5/bank_v2.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'1c5fda4175aecef4ad7e91c9084ff57f7094311ca3b8353618e1c01ff0ec6a52')

    def test_existing_protected_scan_passes_all_eight(self):
        rows=json.loads((ROOT/'verification/protected_pairs_v1/report.json').read_text())['rows']
        scans={r['profile']:r['scan']['overlap'] for r in rows}
        self.assertTrue(all(not scans[key] for key in IDS))

    def test_every_seed_has_a_profile_without_sign_resampling(self):
        bank=checked_bank();gaps=set();seen=set();negative_original_sign=False
        for seed in range(1,1001):
            profile,request=selection(bank,seed);original=draw('table_waste_empty_grasp',seed)
            negative_original_sign|=original['miss_sign']==-1
            self.assertIn(request['miss_sign'],profile['supported_miss_signs'])
            self.assertEqual(request['gap_m'],original['gap_m']);self.assertGreaterEqual(request['gap_m'],.02)
            self.assertLessEqual(request['gap_m'],.05);seen.add(profile['id']);gaps.add(request['gap_m'])
            self.assertEqual(selection(bank,seed),(profile,request))
        self.assertEqual(seen,set(IDS));self.assertEqual(len(gaps),1000);self.assertTrue(negative_original_sign)

    def test_all_eight_reachable_twice(self):self.assertTrue(all(len(v)==2 for v in coverage().values()))

    def test_excluded_pose_cannot_sneak_back(self):
        b=deepcopy(checked_bank());b['cases']['table_waste_empty_grasp']['profiles'][0]['training_provenance']['active_arm_copied_exactly']=False
        self.assertNotEqual(b['content_sha256'],checksum(b))

    def test_startup_has_no_solver(self):
        tree=ast.parse((ROOT/'exact8.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ('solve','least_squares','read_table') for n in ast.walk(tree)))

    def test_default_candidate_entry(self):
        source=(ROOT.parent/'expansion_v2/candidate_runner.py').read_text()
        self.assertIn("module='grasp_failure_recovery.frozen_initials_v2.exact8'",source)
        self.assertIn("if a.configuration_key is not None:",source)
        self.assertIn("a.bank_key='cup_training_exact8_v6'",(ROOT/'runner.py').read_text())


if __name__=='__main__':unittest.main()

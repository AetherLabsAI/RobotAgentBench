"""Pure contracts; no simulator, source dataset or online optimizer."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from . import bank,adapted_bank
from ..frozen_initials_v1.bank import checksum


ROOT=Path(__file__).resolve().parent


def adapted_fixture():
    # Synthetic contract-only record, never exported as a physical pose.
    raw=json.loads((ROOT/'banks/remaining_exact_training_v1/bank.json').read_text())
    p=deepcopy(raw['cases']['shelf_drink']['profiles'][0]);p['variant']='sofa';p['id']='synthetic_contract_only'
    pr=p['training_provenance'];pr['active_arm_copied_exactly']=False;pr['offline_ik_adapted']=True
    q=p['robot_initialization']['positions'];h=p['robot_initialization']['hold_targets']
    names=[f'idx21_arm_l_joint1',*[f'idx2{i}_arm_l_joint{i}' for i in range(2,8)]]
    pr['active_arm_delta_rad']={n:0. for n in names}
    q[names[0]]+=.01
    if names[0] in h:h[names[0]]=q[names[0]]
    pr['active_arm_delta_rad'][names[0]]=.01
    p['provenance_kind']='training_active_arm_adapted_offline_NOT_raw_training_pose'
    p['fit']['ik']={'converged':True}
    v=dict(schema='failure_recovery.offline_adapted_sofa_pool.v1',profiles=[p],online_ik_required=False,
        training_dataset_required_at_startup=False)
    v['content_sha256']=adapted_bank.checksum(v);return v


class Contracts(unittest.TestCase):
    def setUp(self):self.raw=json.loads((ROOT/'banks/remaining_exact_training_v1/bank.json').read_text())
    def test_actual_raw_pool(self):
        bank.validate(self.raw)
        self.assertEqual([len(self.raw['cases'][v]['profiles']) for v in ('shelf_drink','coffee_bag','sofa')],[8,8,0])
    def test_no_eligible_fallback(self):
        with self.assertRaises(ValueError):bank.choose(self.raw,'shelf_drink',301,set())
    def test_deterministic_choice(self):
        ids={p['id'] for p in self.raw['cases']['coffee_bag']['profiles']}
        self.assertEqual(bank.choose(self.raw,'coffee_bag',14,ids),bank.choose(self.raw,'coffee_bag',14,ids))
    def test_raw_arm_edit_rejected(self):
        p=self.raw['cases']['shelf_drink']['profiles'][0];p['robot_initialization']['positions']['idx21_arm_l_joint1']+=.01
        self.raw['content_sha256']=checksum(self.raw)
        with self.assertRaises(ValueError):bank.validate(self.raw)
    def test_adjacent_frame_not_new_pose(self):
        rows=self.raw['cases']['shelf_drink']['profiles'];rows[1]['training_provenance']['episode_index']=rows[0]['training_provenance']['episode_index']
        self.raw['content_sha256']=checksum(self.raw)
        with self.assertRaises(ValueError):bank.validate(self.raw)
    def test_paper_pool_unchanged(self):
        old=json.loads((ROOT.parent/'frozen_initials_v2/compile/cup_training_exact8_v6/bank_v2.json').read_text())
        self.assertEqual(self.raw['cases']['table_waste_empty_grasp'],old['cases']['table_waste_empty_grasp'])
    def test_valid_adapted_label(self):adapted_bank.validate(adapted_fixture())
    def test_adapted_must_not_claim_raw(self):
        v=adapted_fixture();v['profiles'][0]['training_provenance']['active_arm_copied_exactly']=True;v['content_sha256']=adapted_bank.checksum(v)
        with self.assertRaises(ValueError):adapted_bank.validate(v)
    def test_adapted_nonactive_change_rejected(self):
        v=adapted_fixture();p=v['profiles'][0];n=next(n for n in p['training_provenance']['source_state18_rad_or_m'] if '_head_' in n)
        p['robot_initialization']['positions'][n]+=.1
        if n in p['robot_initialization']['hold_targets']:p['robot_initialization']['hold_targets'][n]+=.1
        v['content_sha256']=adapted_bank.checksum(v)
        with self.assertRaises(ValueError):adapted_bank.validate(v)
    def test_adapted_no_runtime_fallback(self):
        with self.assertRaises(ValueError):adapted_bank.select(adapted_fixture(),14,set())
    def test_adapted_reject_native_inheritance(self):
        v=adapted_fixture();v['profiles'][0]['validation']['native']=True;v['content_sha256']=adapted_bank.checksum(v)
        with self.assertRaises(ValueError):adapted_bank.validate(v)
    def test_actual_sofa_eight_offline(self):
        v=adapted_bank.load(ROOT/'compile/sofa_offline_adapt_v3/pool.json')
        self.assertEqual(len(v['profiles']),8)
        self.assertEqual({p['active_hand'] for p in v['profiles']},{'l'})
        self.assertEqual(len({p['training_provenance']['episode_index'] for p in v['profiles']}),8)
        self.assertFalse(v['native_physics_tested'])
        self.assertTrue(v['physics_unchanged'])
    def test_actual_sofa_random_selection(self):
        v=adapted_bank.load(ROOT/'compile/sofa_offline_adapt_v3/pool.json');ids={p['id'] for p in v['profiles']}
        self.assertEqual(len({adapted_bank.select(v,seed,ids)['id'] for seed in range(1,65)}),8)


if __name__=='__main__':unittest.main()

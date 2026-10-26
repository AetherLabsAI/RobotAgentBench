"""One-time immutable selection; no training/physics/solver dependency."""
from copy import deepcopy
import json
from ..common import HERE,owned,digest,write_json
from ..frozen_initials_v1.bank import load,checksum,validate
from .exact8 import IDS,checked_bank,DEFAULT_BANK


def main():
    old=HERE/'frozen_initials_v2/compile/cup_training_adapt_v5/bank_v2.json'
    scan=HERE/'frozen_initials_v2/verification/protected_pairs_v1/report.json'
    out=owned(DEFAULT_BANK.parent,fresh=True);out.mkdir(parents=True)
    bank=deepcopy(load(old));profiles=bank['cases']['table_waste_empty_grasp']['profiles']
    keep=[p for p in profiles if p['id'] in IDS];reject=[p['id'] for p in profiles if p['id'] not in IDS]
    if len(keep)!=8:raise ValueError('Expected eight original profiles')
    scans={r['profile']:r['scan']['overlap'] for r in json.loads(scan.read_text())['rows']}
    if any(scans[p['id']] for p in keep):raise ValueError('Protected overlap in selected profile')
    bank['cases']['table_waste_empty_grasp'].update(profiles=keep,hand_coverage=['l'])
    bank['selection_revision']='exact8_training_only_v6'
    bank['selection_receipt']=dict(user_decision='Keep only the eight directly copied training arm/body snapshots',
        parent_sha256=digest(old),excluded=reject,protected_scan_sha256=digest(scan),
        unchanged_profiles=True,training_dataset_read=False,solver_called=False,
        runtime_side_policy='selected_frozen_profile_prevalidated_side',new_episode_native_acceptance_inherited=False)
    bank['historical_training_expansion']=bank.pop('training_expansion')
    bank['historical_offline_right_adaptation']=bank.pop('offline_right_adaptation')
    bank['training_expansion']=dict(accepted_per_hand={'l':8,'r':0},profiles=list(IDS),active_arm_copied_exactly=True)
    bank['content_sha256']=checksum(bank);validate(bank);write_json(DEFAULT_BANK,bank);checked_bank(DEFAULT_BANK)
    write_json(out/'selection.json',dict(selected=list(IDS),excluded=reject,bank_sha256=digest(DEFAULT_BANK),
        profiles_count=8,hands={'l':8,'r':0},source_bank_sha256=digest(old),unchanged_other_four_cases=True,
        runtime_training_dataset_required=False,online_ik_calls=0))


if __name__=='__main__':main()

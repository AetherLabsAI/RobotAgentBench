"""Combine independently checked case pools, without editing any old bank."""
import argparse
from copy import deepcopy
import json
from ..common import HERE,owned,digest,write_json
from ..frozen_initials_v1.bank import load as old_load,checksum
from .bank import validate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shelf-key',required=True);p.add_argument('--bag-key',required=True)
    p.add_argument('--sofa-key',required=True);p.add_argument('--key',required=True);a=p.parse_args()
    for key in (a.shelf_key,a.bag_key,a.sofa_key,a.key):
        if not key.replace('_','').isalnum():raise ValueError('Own simple keys required')
    old=HERE/'frozen_initials_v2/compile/cup_training_exact8_v6/bank_v2.json';bank=deepcopy(old_load(old))
    out=owned(HERE/'frozen_initials_v3/banks'/a.key,fresh=True);out.mkdir(parents=True)
    sources={};inventories={};pools={}
    invkeys={'shelf_drink':'shelf_training55_allphases_v2','coffee_bag':'bag_training50_v1','sofa':'sofa_training58_allphases_v2'}
    for variant,key in (('shelf_drink',a.shelf_key),('coffee_bag',a.bag_key),('sofa',a.sofa_key)):
        path=HERE/'frozen_initials_v3/compile'/key/'pool.json';data=json.loads(path.read_text());pools[variant]=data
        inventory=HERE/'frozen_initials_v3/mine'/invkeys[variant]/'inventory.json';inv=json.loads(inventory.read_text())
        if digest(inventory)!=data['inventory_sha256']:raise ValueError('Pool/inventory identity mismatch')
        by_id={r['id']:r for r in inv['candidates']}
        profiles=deepcopy(data['profiles'])
        for profile in profiles:
            source=by_id[profile['id']];pr=profile['training_provenance']
            pr['source_state18_rad_or_m']={n:v for n,v in source['measured_state'].items() if '_arm_' in n or '_head_' in n or '_body_' in n}
        bank['cases'][variant]=dict(profiles=profiles,online_ik_fallback=False,hand_coverage=sorted({p['active_hand'] for p in profiles}),
            fast_scene_generator_connected=True,runtime_registered=False,
            status='cpu_training_pool_native_pending' if profiles else 'blocked_no_compatible_exact_training_pose')
        sources[str(path)]=digest(path);inventories[variant]=digest(inventory)
    bank.update(selection_revision='remaining_exact_training_v3',parent_bank_sha256=digest(old),
        case_pool_sources=sources,source_inventory_sha256=inventories,runtime_training_dataset_required=False,
        online_ik_fallback=False,new_banks_not_published_to_shared_entry=True)
    bank['content_sha256']=checksum(bank);validate(bank);write_json(out/'bank.json',bank)
    write_json(out/'summary.json',dict(counts={k:len(v['profiles']) for k,v in bank['cases'].items()},
        hands={k:v.get('hand_coverage') for k,v in bank['cases'].items()},bank_sha256=digest(out/'bank.json'),
        native_tested=False,shared_entry_changed=False,paper_cup_eight_profiles_unchanged=True,
        tipped_cup='blocked: physical grasp/lift/early-release/rest witnesses still required'))

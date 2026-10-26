"""Task-local inventory adapter; no USD edits, runtime registration or physics.

Three or four access candidates are retained symmetrically. The actual scene builder
must run the existing geometry sampler before declaring this plan runnable.
"""
import json
import random
from pathlib import Path
from openend.tasks.scene_requirements.shelf_inventory_policy import choose_inventory
from .expanded_catalog import POOLS, LABELS, NEW_PRODUCTS
from .expanded_placement import POLICY
from .initial_state import initial_for_question
from .bank import BANK_DIR
from .build_expanded_bank import PRODUCTS, satisfies

HERE=Path(__file__).resolve().parent
INVENTORY_POLICY='context_shelf_scene_unique.v5'
COUNT_RANGES={'tier3':(6,8),'tier2':(6,8)}

def validate_stocked_answer(acceptable, rows):
    """Count matching instances, not merely matching product categories."""
    matches=[o for o in rows if o['product_family'] in acceptable]
    if len(matches)!=1:raise ValueError('One and only one correct instance required')
    return matches[0]

def unique_inventory(seed, required, excluded=()):
    """Approved ranges; sample without replacement, retaining source gaps."""
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer seed required')
    required=set(required)
    excluded=set(excluded)
    if not excluded<=set(LABELS) or required & excluded:raise ValueError('Invalid excluded products')
    if not required<=set(LABELS):raise ValueError('Unknown required product')
    layers=[]
    for sid,pool in POOLS.items():
        pool=tuple(p for p in pool if p not in excluded)
        low,high=COUNT_RANGES[sid]
        high=min(high,len(pool))
        if high<low:raise ValueError('Not enough unique distractors after excluding other valid answers')
        needed=required.intersection(pool)
        if len(needed)>low:raise ValueError('Required inventory exceeds minimum admitted count')
        rng=random.Random(f'{INVENTORY_POLICY}:{seed}:{sid}')
        count=rng.randint(low,high)
        names=choose_inventory(rng,pool,needed,count)
        if len(names)!=count or len(set(names))!=count:
            raise ValueError('Unique source inventory invariant failed')
        objects=[dict(instance_id=f'shelf/{sid}/{name}/01',product_family=name,
            label=LABELS[name],support_id=sid,
            asset_scope='task_expansion' if name in NEW_PRODUCTS else 'all_assets',
            asset_relative_path=f'{name}/asset.usda',required_instance=name in needed)
            for name in names]
        layers.append(dict(support_id=sid,count=count,drawn_count=count,
            count_range=list(COUNT_RANGES[sid]),feasible_count_range=[low,high],count_raised_for_required_categories=False,
            required_categories=sorted(needed),required_counts={p:1 for p in sorted(needed)},
            minimum_footprint_gap_m=.085 if count<=5 else .055,objects=objects))
    all_names=[o['product_family'] for l in layers for o in l['objects']]
    if len(set(all_names))!=len(all_names) or not required<=set(all_names):
        raise ValueError('Whole-shelf uniqueness or required coverage failed')
    return dict(policy=INVENTORY_POLICY,status='inventory_plan_ready',seed=seed,
        layers=layers,required_categories=sorted(required),excluded_products=sorted(excluded),required_counts={p:1 for p in sorted(required)},
        all_products_unique=True,task_product_background_duplicates=False,
        required_category_coverage_passed=True,poses_sampled=False,usd_written=False,runtime_ready=False,
        placement_contract=dict(full_footprint_on_support=True,
            missing_required_action='reject_scene_and_resample_never_silently_drop',
            inventory_shortfall_action='reject_scene_and_resample',
            geometry_collision_visibility_and_reachability='not_checked'))

def plan_inventory(question_id, seed):
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer seed required')
    bank=json.loads((BANK_DIR/'answers_private.json').read_text())
    matches=[a for a in bank['answers'] if a['id']==question_id]
    if len(matches)!=1:raise ValueError('Unique question ID required')
    answer=matches[0]
    acceptable=sorted(p for p in PRODUCTS if PRODUCTS[p][0]==answer['kind'] and satisfies(p,answer['criteria']))
    if acceptable!=answer['acceptable_products'] or not acceptable:
        raise ValueError('Answer-set evidence mismatch')
    target=answer['target'] or random.Random(f'answer:{seed}:{question_id}').choice(acceptable)
    excluded=set(acceptable)-{target}
    required=answer['required_products']
    if not required:
        foils=sorted(p for p in PRODUCTS if PRODUCTS[p][0]==answer['kind'] and p not in acceptable)
        required=sorted([target]+random.Random(f'foils:{seed}:{question_id}').sample(foils,2))
    if len(required)!=3 or len(set(required))!=3 or target not in required:
        raise ValueError('Three distinct compared products including correct target required')
    inventory=unique_inventory(seed,required,excluded)
    all_rows=[o for layer in inventory['layers'] for o in layer['objects']]
    # All compared alternatives share access rules. A fourth candidate, when drawn,
    # is selected without consulting which compared alternative is correct.
    access_rng=random.Random(f'access:{seed}:{sorted(required)}')
    candidates=list(sorted(required))
    if access_rng.randint(3,4)==4:
        extras=sorted(o['product_family'] for o in all_rows
            if PRODUCTS[o['product_family']][0]==answer['kind'] and o['product_family'] not in candidates)
        candidates.append(access_rng.choice(extras))
    reservations=[dict(id=f'placement_item_{i}',type='snack_drink_to_tray',product=p)
        for i,p in enumerate(sorted(candidates))]
    match=validate_stocked_answer(acceptable,all_rows)
    return dict(question_id=question_id,seed=seed,inventory=inventory,
        target_instance_id=match['instance_id'],target_product=target,
        acceptable_products=acceptable,access_candidates=sorted(candidates),
        placement_policy=dict(POLICY),geometry_only_reservations=reservations,
        robot_initial=initial_for_question(question_id),
        scoring_task_count=1,wrong_touch_scope='all_stocked_product_instances',
        accessibility_scope='three_or_four_same_kind_candidates',background_access_guaranteed=False,
        source='user-approved unique inventory; existing shelf short-task geometry and gap policy',
        shared_runtime_registered=False,geometry_checked=False,native_physics_validated=False,
        caveat='Run stock_wooden_tray.sample_stock with these reservations before scene admission; never treat inventory-only success as a valid pose.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('question_id');p.add_argument('--seed',required=True,type=int)
    args=p.parse_args();print(json.dumps(plan_inventory(args.question_id,args.seed),ensure_ascii=False,indent=2))

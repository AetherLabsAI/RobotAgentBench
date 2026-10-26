"""Extended catalog, unchanged support/gap/lane checks and packing budget.

New snack headings inherit the same-format source product convention; these
are not newly fitted demonstration distributions. Drink fronts use real UVs.
"""
from collections import Counter
from pathlib import Path
import copy
import json
import numpy as np
from short_tasks.tasks.stock_wooden_tray import generation as base

HERE = Path(__file__).resolve().parent
POLICY = {'task_name':'select_shelf','version':'three_four_candidates_rear204.v6'}
MAX_REASONING_ORIGIN_DEPTH_M = 0.204

def rules():
    rule = copy.deepcopy(base.rules())
    # Reasoning-only operation box: rear edge 28.4 -> 20.4 cm.
    # Keep front edge, lateral bounds, source station and background rules.
    rule['origin_depth_from_front_m'][1] = MAX_REASONING_ORIGIN_DEPTH_M
    for new, donor in [('lindt','skit_littles'),('pringles_green','pring_bbq'),('red_bull_pink','starbucks')]:
        value = copy.deepcopy(rule['source_yaw_audit']['products'][donor])
        value['range_source'] = {'inherited_convention_from':donor,'new_product_data_fit':False}
        rule['source_yaw_audit']['products'][new] = value
    rule['snack_target_angle_exclusion']['products'] += ['lindt','pringles_green']
    rule['drink_fronts']['products'].update(json.loads((HERE/'expanded_assets/fronts.json').read_text()))
    return rule

def validate_policy(policy, requests):
    assert policy == POLICY
    assert 3 <= len(requests) <= 4 and len({r['id'] for r in requests})==len(requests)
    known = rules()['source_yaw_audit']['products']
    assert all(r.get('type')=='snack_drink_to_tray' and r['product'] in known for r in requests)

def sample_stock(policy, requests, seed, catalog, supports, fixtures, counts,
                 pools, pose_item, separated, planned, sample_scene):
    validate_policy(policy,requests)
    rule = rules()
    targets = Counter(r['product'] for r in requests)
    inventory = Counter(p for names in planned.values() for p in names)
    assert set(targets)<=set(inventory) and set(targets.values())=={1} and set(inventory.values())=={1}
    assert all(sid in rule['source_yaw_audit']['products'][p]['tiers'] for sid,names in planned.items() for p in names)
    def allowed(item,placed):
        return base.candidate_allowed(item,placed,targets,rule)
    rng = np.random.default_rng(seed)
    headings = base.TargetHeadingRanges(sorted(set(targets)|set(rule['drink_fronts']['products'])),rule,rng)
    placed,metrics = sample_scene(rng,catalog,supports,fixtures,counts,pools,pose_item,separated,
        planned_inventory=planned,candidate_allowed=allowed,priority_products=frozenset(targets),heading_ranges=headings)
    assert Counter(row['product_family'] for row,_,_ in placed)==inventory
    assert all(allowed(item,placed[:i]+placed[i+1:]) for i,item in enumerate(placed))
    flags=[row['product_family'] in targets for row,_,_ in placed]
    assert flags==sorted(flags,reverse=True)
    lanes=[dict(product=row['product_family'],support_id=row['support_id'],
        polygon_xy_m=[list(p) for p in base.lane_polygon(fp.bounds,rule).exterior.coords])
        for row,_,fp in placed if row['product_family'] in targets]
    receipt=dict(policy=POLICY,lanes=lanes,inventory_preserved=True,final_geometry_checked=True,
        origin_depth_from_front_m=rule['origin_depth_from_front_m'],
        origin_x_m=rule['origin_x_m'],shelf_front_y_m=rule['shelf_front_y_m'],
        accessibility_scope='three_or_four_same_kind_candidates', privileged_candidate_subset=True,
        background_access_guaranteed=False,
        robot_reachability_verified=False, native_contact_verified=False,
        native_physics_validated=False,source_gap_and_lane_rules_unchanged=True,
        added_snack_heading_convention={'lindt':'skit_littles','pringles_green':'pring_bbq'},
        source_rule_files=[str(base.RULE_FILE),str(base.FRONT_FILE)])
    metrics.update(ordinary_shelf_policy=POLICY['version'],uses_target=True,
        uses_robot=False,reserves_grasp_corridor=True)
    return placed,metrics,receipt

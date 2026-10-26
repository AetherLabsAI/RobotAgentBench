"""Ordinary stock_wooden_tray: one common range and a trapezoid per target.

Task-owned policy; shared packing still owns support, spacing, fixtures, counts,
heading proposals and retry budget. No robot control, physics or score changes.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
from openend.tasks.scene_requirements import shelf_operation_region as region

TASK = 'stock_wooden_tray'
RULE_FILE = Path(__file__).with_name('placement_policy.json')
FRONT_FILE = Path(__file__).with_name('drink_fronts.json')
POLICY = {'task_name': TASK, 'version': 'common_range_trapezoid.v1'}


def rules():
    value = json.loads(RULE_FILE.read_text())
    if value['task_name'] != TASK or value['version'] != POLICY['version']:
        raise ValueError('Unknown ordinary shelf placement rule')
    if value['origin_depth_from_front_m'][1] != region.MAX_ORIGIN_DEPTH_M:
        raise ValueError('Shelf task rear boundary differs from shared operation region')
    value['drink_fronts'] = json.loads(FRONT_FILE.read_text())
    return value


def heading_range(name, rule):
    front = rule['drink_fronts']['products'].get(name)
    if front is not None:
        return [front['wrapper_yaw_deg'], front['wrapper_yaw_deg']]
    return rule['source_yaw_audit']['products'][name]['yaw_deg']


def heading_intervals(name, rule):
    lo, hi = heading_range(name, rule)
    exclusion = rule['snack_target_angle_exclusion']
    if name not in exclusion['products']:
        return [[lo, hi]]
    a, b = exclusion['closed_interval_deg']
    # The excluded interval includes both endpoints.
    return [[lo, math.nextafter(a, -math.inf)],
            [math.nextafter(b, math.inf), hi]]


class TargetHeadingRanges(dict):
    """Existing sampler mapping: choose a side on each object's heading lookup.

    Membership is side-effect free. The shared sampler then draws uniformly
    within the returned interval before computing the rotated footprint.
    Repeated products and packing retries draw independently from the same seed.
    """
    def __init__(self, names, rule, rng):
        super().__init__((name, heading_range(name, rule)) for name in names)
        self.rule, self.rng = rule, rng

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        intervals = heading_intervals(name, self.rule)
        return intervals[int(self.rng.integers(len(intervals)))] if len(intervals) > 1 else intervals[0]


def drink_faces_front(row, rule):
    front = rule['drink_fronts']['products'].get(row['product_family'])
    if front is None:
        return True
    yaw = float(row['yaw_deg'])
    error = (yaw - front['wrapper_yaw_deg'] + 180) % 360 - 180
    return math.isfinite(yaw) and abs(error) < 1e-6


def validate_policy(policy, requests):
    if policy != POLICY:
        raise ValueError('Ordinary shelf requires the single current placement policy')
    if not 1 <= len(requests) <= 4 or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('Expected 1-4 distinct ordinary shelf task instances')
    known = rules()['source_yaw_audit']['products']
    if any(r.get('type') != 'snack_drink_to_tray' or r.get('product') not in known for r in requests):
        raise ValueError('Unknown ordinary shelf product or task type')


def validate_output(source, root):
    original = Path(source).absolute()
    source = original.resolve()
    long_cache = Path(root).resolve()/'short_tasks/agent_probe_v1/long_task_eval_setup_v1/generated_rooms'
    if original.is_relative_to(long_cache) or source.is_relative_to(long_cache):
        if original != source or long_cache.resolve() != long_cache:
            raise ValueError('Long Task generation must not use symbolic links')
    allowed = [
        (Path(root)/'openend/tasks/scene_requirements/generated_rooms').resolve(),
        (Path(__file__).parent/'generated_rooms').resolve(),
        Path('/dev/shm/stock_wooden_tray_initials_20261012/generated_rooms'),
        long_cache,
    ]
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import root
    temporary = root()
    if temporary is not None:
        allowed.append(temporary/'generated_rooms')
    if source.parent.parent not in allowed:
        raise ValueError('Ordinary shelf output must be a fresh owned generated-room folder')


def target_allowed(row, rule=None):
    rule = rules() if rule is None else rule
    product = rule['source_yaw_audit']['products'].get(row['product_family'])
    if product is None or row['support_id'] not in product['tiers']:
        return False
    x, y = row['xyz_m'][:2]
    yaw = row['yaw_deg']
    if not all(math.isfinite(float(v)) for v in (x, y, yaw)):
        return False
    depth = y - rule['shelf_front_y_m']
    a, b = rule['origin_x_m']
    c, d = rule['origin_depth_from_front_m']
    lo, hi = heading_range(row['product_family'], rule)
    facing_ok = (drink_faces_front(row, rule) if row['product_family'] in rule['drink_fronts']['products']
                 else any(a <= yaw <= b for a,b in heading_intervals(row['product_family'], rule)))
    return a <= x <= b and c <= depth <= d and facing_ok


def lane_polygon(bounds, rule=None):
    from shapely.geometry import Polygon
    rule = rules() if rule is None else rule
    x0, y0, x1, y1 = map(float, bounds)
    front = rule['shelf_front_y_m']
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x0 >= x1 or y0 >= y1 or y1 <= front:
        raise ValueError('Invalid target footprint')
    near = rule['target_side_margin_m']
    wide = rule['front_side_margin_m']
    return Polygon([(x0-wide, front), (x1+wide, front),
                    (x1+near, y1), (x0-near, y1)])


def clear_of_lane(lane, footprint):
    from shapely.geometry import box
    # Full rotated footprint AABB, not centre point or an inscribed circle.
    return lane.intersection(box(*footprint.bounds)).area <= 1e-12


def candidate_allowed(item, placed, target_products, rule=None):
    rule = rules() if rule is None else rule
    row, _, footprint = item
    if not drink_faces_front(row, rule):
        return False
    is_target = row['product_family'] in target_products
    if is_target and not target_allowed(row, rule):
        return False
    proposed_lane = lane_polygon(footprint.bounds, rule) if is_target else None
    for other_row, _, other_fp in placed:
        if other_row['support_id'] != row['support_id']:
            continue
        if proposed_lane is not None and not clear_of_lane(proposed_lane, other_fp):
            return False
        if other_row['product_family'] in target_products:
            if not clear_of_lane(lane_polygon(other_fp.bounds, rule), footprint):
                return False
    return True


def sample_stock(policy, requests, seed, catalog, supports, fixtures, counts,
                 pools, pose_item, separated, planned, sample_scene):
    import numpy as np
    validate_policy(policy, requests)
    rule = rules()
    targets = Counter(r['product'] for r in requests)
    inventory = Counter(name for names in planned.values() for name in names)
    if any(inventory[p] != count for p, count in targets.items()):
        raise ValueError('Target product inventory must equal requested count; no ambiguous extras')
    for sid, names in planned.items():
        if any(sid not in rule['source_yaw_audit']['products'][p]['tiers']
               for p in names if p in targets):
            raise ValueError('Target on wrong shelf tier')
    def allowed(item, placed):
        return candidate_allowed(item, placed, targets, rule)
    # Place every requested target first; its lane is then reserved while
    # backgrounds are packed. Symmetric checks still protect multiple targets.
    rng = np.random.default_rng(seed)
    headings = TargetHeadingRanges(sorted(set(targets) | set(rule['drink_fronts']['products'])), rule, rng)
    placed, metrics = sample_scene(rng, catalog, supports,
        fixtures, counts, pools, pose_item, separated, planned_inventory=planned,
        candidate_allowed=allowed, priority_products=frozenset(targets),
        heading_ranges=headings)
    flags=[row['product_family'] in targets for row,_,_ in placed]
    if flags != sorted(flags,reverse=True):
        raise ValueError('Sampler did not place all targets before backgrounds')
    if Counter(row['product_family'] for row, _, _ in placed) != inventory:
        raise ValueError('Constrained packing changed the inventory')
    for i, item in enumerate(placed):
        if not allowed(item, placed[:i] + placed[i+1:]):
            raise ValueError('Final shelf arrangement violates placement rule')
    lanes = [dict(product=row['product_family'], wrapper_path=row['prim_path'],
                  support_id=row['support_id'], origin_xy_m=row['xyz_m'][:2],
                  yaw_deg=row['yaw_deg'], target_bounds_xy_m=list(fp.bounds),
                  polygon_xy_m=[list(p) for p in lane_polygon(fp.bounds, rule).exterior.coords])
             for row, _, fp in placed if row['product_family'] in targets]
    receipt = dict(policy=dict(POLICY), coordinate_frame=rule['coordinate_frame'],
        seed=seed, target_count=sum(targets.values()), lanes=lanes,
        origin_x_m=rule['origin_x_m'], origin_depth_from_front_m=rule['origin_depth_from_front_m'],
        shelf_front_y_m=rule['shelf_front_y_m'],
        target_side_margin_m=rule['target_side_margin_m'],
        front_side_margin_m=rule['front_side_margin_m'],
        inventory_preserved=True, final_geometry_checked=True,
        packing_order='targets_then_backgrounds',
        snack_target_angle_exclusion=rule['snack_target_angle_exclusion'],
        actual_placement_order=[row['product_family'] for row,_,_ in placed],
        drink_orientation_policy=rule['drink_fronts']['policy'],
        drink_front_direction_shelf_xy=rule['drink_fronts']['shelf_front_direction_xy'],
        drink_headings=[dict(product=row['product_family'], wrapper_path=row['prim_path'],
                            wrapper_yaw_deg=row['yaw_deg'], is_target=row['product_family'] in targets)
                        for row,_,_ in placed if row['product_family'] in rule['drink_fronts']['products']],
        native_physics_validated=False,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), RULE_FILE, FRONT_FILE, region.SOURCE)})
    metrics.update(ordinary_shelf_policy=POLICY['version'], uses_target=True,
                   uses_robot=False, reserves_grasp_corridor=True)
    return placed, metrics, receipt

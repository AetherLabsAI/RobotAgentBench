"""Approved rear cap for NEW shelf episodes; not a reachability certificate."""
import math
from pathlib import Path

POLICY = 'shelf_rear_boundary_20261014'
MAX_ORIGIN_DEPTH_M = 0.284
SOURCE = Path(__file__)


def cap_bounds(bounds):
    """Intersect an existing depth interval, preserving its front boundary."""
    low, high = map(float, bounds)
    high = min(high, MAX_ORIGIN_DEPTH_M)
    if low > high:
        raise ValueError('Source shelf range lies beyond the approved rear boundary')
    return [low, high]


def target_filter(target_products, supports):
    """Only task targets are capped. Existing support/spacing checks still apply."""
    targets = frozenset(target_products)
    front = {sid: min(float(p[1]) for p in s['polygon_xy_m'])
             for sid, s in supports.items()}

    def allowed(item, placed):
        row = item[0]
        if row['product_family'] not in targets:
            return True
        depth = float(row['xyz_m'][1]) - front[row['support_id']]
        return math.isfinite(depth) and depth <= MAX_ORIGIN_DEPTH_M + 1e-9
    return allowed


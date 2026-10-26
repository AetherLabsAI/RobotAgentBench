"""Source-observed asset-origin bounds, not a robot reachability certificate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from openend.tasks.scene_requirements.shelf_operation_region import cap_bounds

SOURCE = Path(__file__).with_name('source_product_ranges.json')


def contract(product, shelf_box):
    data = json.loads(SOURCE.read_text())
    spec = deepcopy(data['products'][product])
    spec['inward_m'] = cap_bounds(spec['inward_m'])
    if spec['tiers'] != ['tier3']:
        raise ValueError('Only recorded drink-tier ranges are supported')
    box = np.asarray(shelf_box, dtype=float)
    return dict(policy=data['policy'], product=product, bounds=deepcopy(spec),
                front_midpoint_local_m=[float(box[:, 0].mean()), float(box[0, 1])],
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                audit_sha256=data['audit_sha256'], robot_reachability_verified=False)


def contains(rule, origin_xy, yaw_deg):
    xy = np.asarray(origin_xy) - rule['front_midpoint_local_m']
    values = [*xy, (float(yaw_deg) + 180.) % 360. - 180.]
    return all(np.isfinite(v) and low - 1e-9 <= v <= high + 1e-9
               for v, (low, high) in zip(values, [rule['bounds'][k]
                       for k in ('lateral_m', 'inward_m', 'yaw_deg')]))


def destination_allowed(rule, original_origin_xy, original_box, destination_center, yaw):
    # The route uses AABB centers; the data use asset origins. Do not confuse them.
    delta = np.asarray(destination_center) - np.asarray(original_box).mean(0)[:2]
    return contains(rule, np.asarray(original_origin_xy) + delta, yaw)


def validate_record(reservation, shelf_box, origin_xy=None, yaw_deg=None):
    stored = reservation['source_product_range']
    expected = contract(reservation['product'], shelf_box)
    key='front_midpoint_local_m'
    if (set(stored) != set(expected)
            or any(stored[k] != expected[k] for k in expected if k != key)
            or not np.allclose(stored[key],expected[key],atol=1e-7,rtol=0)):
        raise ValueError('Source product range or shelf frame changed')
    pose = reservation['target_initial_pose']
    xy, yaw = pose['xyz_m'][:2], pose['yaw_deg']
    if origin_xy is not None and not np.allclose(origin_xy, xy, atol=1e-7, rtol=0):
        raise ValueError('Target asset origin changed')
    if yaw_deg is not None and abs((yaw_deg-yaw+180.) % 360. - 180.) > 1e-5:
        raise ValueError('Target asset heading changed')
    if not contains(stored, xy, yaw):
        raise ValueError('Initial target outside source product range')
    if not destination_allowed(stored, xy, reservation['initial_target_bounds'],
                               reservation['destination_xy'], yaw):
        raise ValueError('Relocated target outside source product range')

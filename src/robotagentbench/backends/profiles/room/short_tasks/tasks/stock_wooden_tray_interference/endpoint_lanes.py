"""Two robot-access trapezoids, separate from the human's swept corridor.

Reuse ordinary shelf geometry/clearance values without changing its sampler.
Records are in shelf-local metres; they are reservations, never colliders.
"""
from copy import deepcopy
import numpy as np

SCHEMA = 'shelf_interference.endpoint_trapezoids.v1'
RULE_KEYS = ('shelf_front_y_m', 'target_side_margin_m', 'front_side_margin_m')


def current_rule():
    from short_tasks.tasks.stock_wooden_tray.generation import rules
    source = rules()
    if source['coordinate_frame'] != 'shelf_local_m':
        raise ValueError('Ordinary grasp lanes must use shelf-local metres')
    return {key: source[key] for key in RULE_KEYS}


def build_endpoint_lanes(source_bounds, destination_bounds, rule=None):
    from short_tasks.tasks.stock_wooden_tray.generation import lane_polygon
    from .reservation import _bounds
    rule = current_rule() if rule is None else deepcopy(rule)
    lanes = []
    for role, bounds in (('source', source_bounds), ('destination', destination_bounds)):
        lo, hi = _bounds(bounds)
        footprint = [lo[0], lo[1], hi[0], hi[1]]
        lanes.append(dict(role=role, target_bounds_xy_m=footprint,
            height_interval_m=[lo[2], hi[2]],
            polygon_xy_m=[list(p) for p in lane_polygon(footprint, rule).exterior.coords]))
    return dict(schema=SCHEMA, coordinate_frame='shelf_local_m', rule=rule,
                lanes=lanes, scoring_target=False)


def validate_endpoint_lanes(reservation):
    """No silent upgrade/replanning of old rooms missing this evidence."""
    saved = reservation.get('endpoint_grasp_lanes')
    if not isinstance(saved, dict):
        raise ValueError('Endpoint grasp trapezoids missing; regenerate this episode')
    expected = build_endpoint_lanes(reservation['initial_target_bounds'],
                                    reservation['reserved_bounds'])
    for key in ('schema', 'coordinate_frame', 'rule', 'scoring_target'):
        if saved.get(key) != expected[key]:
            raise ValueError('Endpoint grasp trapezoid rule changed')
    if len(saved.get('lanes', [])) != 2:
        raise ValueError('Both endpoint grasp trapezoids are required')
    for actual, wanted in zip(saved['lanes'], expected['lanes']):
        if actual.get('role') != wanted['role']:
            raise ValueError('Endpoint grasp trapezoid identity changed')
        for key in ('target_bounds_xy_m', 'height_interval_m', 'polygon_xy_m'):
            a, b = np.asarray(actual.get(key), dtype=float), np.asarray(wanted[key])
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-7, rtol=0):
                raise ValueError('Endpoint grasp trapezoid geometry changed')
    return saved


def clear_of_endpoint_lanes(reservation, candidate_bounds):
    from shapely.geometry import Polygon, box
    from short_tasks.tasks.stock_wooden_tray.generation import clear_of_lane
    from .reservation import _bounds
    record = reservation.get('endpoint_grasp_lanes')
    if (not isinstance(record, dict) or record.get('schema') != SCHEMA
            or record.get('coordinate_frame') != 'shelf_local_m'
            or [x.get('role') for x in record.get('lanes', [])] != ['source', 'destination']):
        raise ValueError('Both endpoint grasp trapezoids are required; regenerate this episode')
    lo, hi = _bounds(candidate_bounds)
    footprint = box(lo[0], lo[1], hi[0], hi[1])
    for lane in record['lanes']:
        z0, z1 = lane['height_interval_m']
        # Other shelf levels are not blocked by a 2-D projection alone.
        if lo[2] < z1 and hi[2] > z0:
            if not clear_of_lane(Polygon(lane['polygon_xy_m']), footprint):
                return False
    return True

"""New initials only: centered proposals, reserve both sides, select once.

No new placement bounds, motion times, offsets or physical parameters. Every
route is admitted by the existing full hand/carry/retreat geometry checks.
"""
from copy import deepcopy
import math
import numpy as np

SCHEMA = 'shelf_interference.bilateral_routes.v1'


def centered_x(rng, *, original_xy, target_bounds, shelf_bounds, source_rule):
    """Triangular preference for shelf center inside the ORIGINAL feasible span.

    Bounds use asset origins for the data and full measured geometry for the
    shelf. Depth and heading are not changed. No arbitrary central strip.
    """
    origin = np.asarray(original_xy, dtype=float)
    target, shelf = np.asarray(target_bounds), np.asarray(shelf_bounds)
    front_x = source_rule['front_midpoint_local_m'][0]
    lo, hi = np.asarray(source_rule['bounds']['lateral_m']) + front_x
    lo = max(float(lo), float(shelf[0, 0] - (target[0, 0] - origin[0])))
    hi = min(float(hi), float(shelf[1, 0] - (target[1, 0] - origin[0])))
    if not math.isfinite(lo + hi) or lo >= hi:
        raise ValueError('No source-bounded lateral span for this target')
    mode = min(hi, max(lo, float(shelf[:, 0].mean())))
    return float(rng.triangular(lo, mode, hi))


def combine(left, right):
    """One scoring target; two complete immutable alternative reservations."""
    for key in ('target_path', 'shelf_level_path', 'seed', 'initial_target_bounds',
                'hand_orientation', 'relocation_distance'):
        if left[key] != right[key]:
            raise ValueError('Bilateral routes disagree on target/grip: ' + key)
    if not np.allclose(np.asarray(left['waypoints'])[:3],
                       np.asarray(right['waypoints'])[:3], atol=1e-9, rtol=0):
        raise ValueError('Both routes must share the same rear entry and grasp')
    root = deepcopy(left)
    # The unchanged shared asset copier binds target_path/target_wrapper on the
    # root after copying. Derive these two identities from that bound root;
    # never embed stale pre-copy identities in child routes.
    def geometry(r):
        return {k:deepcopy(v) for k,v in r.items()
                if k not in ('target_path','target_wrapper')}
    root['bilateral_routes'] = dict(schema=SCHEMA, coordinate_frame='shelf_local_m',
        selection_at='native_proximity', runtime_replanning=False,
        routes=[dict(side='left', reservation=geometry(left)),
                dict(side='right', reservation=geometry(right))])
    reservations(root)
    return root


def reservations(root):
    """Fail closed on partial, nested or mismatched route sets."""
    pool = root.get('bilateral_routes')
    if not isinstance(pool, dict) or pool.get('schema') != SCHEMA:
        raise ValueError('Explicit bilateral route schema required')
    if (pool.get('coordinate_frame') != 'shelf_local_m'
            or pool.get('selection_at') != 'native_proximity'
            or pool.get('runtime_replanning') is not False):
        raise ValueError('Invalid frozen bilateral selection contract')
    rows = pool.get('routes', [])
    if [r.get('side') for r in rows] != ['left', 'right']:
        raise ValueError('Both left and right reservations are required')
    center = np.asarray(root['initial_target_bounds']).mean(0)
    clean = deepcopy(root)
    clean.pop('bilateral_routes')
    derived=[]
    for row in rows:
        r=deepcopy(row['reservation'])
        if any(k in r for k in ('target_path','target_wrapper')):
            raise ValueError('Child route must inherit the copied target identity')
        for key in ('target_path','target_wrapper'):
            if key in root:r[key]=root[key]
        derived.append(r)
    if clean != derived[0]:
        raise ValueError('Compatibility route does not match the frozen left route')
    for r, sign in zip(derived, (-1, 1)):
        if 'bilateral_routes' in r:
            raise ValueError('Nested route options are forbidden')
        for key in ('target_path', 'shelf_level_path', 'seed', 'initial_target_bounds',
                    'hand_orientation', 'relocation_distance'):
            if r[key] != root[key]:
                raise ValueError('Route changed frozen target/grip: ' + key)
        delta = np.asarray(r['destination_xy']) - center[:2]
        if sign * delta[0] <= 0 or not np.isclose(delta[1], 0., atol=1e-9, rtol=0):
            raise ValueError('Expected opposite, same-depth lateral endpoints')
        if not np.allclose(np.asarray(r['waypoints'])[:3],
                           np.asarray(root['waypoints'])[:3], atol=1e-9, rtol=0):
            raise ValueError('Reserved route changed the common entry/grasp')
    return derived


def choose(root, *, evaluate):
    """Select among frozen routes only. Caller supplies fresh native checks.

    Rejections are diagnostic evidence, never a request to move stock or invent
    another endpoint. The selected reservation is deep-copied once by caller.
    """
    routes = reservations(root)
    accepted, rejected = [], []
    for side, route in zip(('left', 'right'), routes):
        try:
            evidence = evaluate(route)
        except ValueError as exc:
            rejected.append(dict(side=side, reason=str(exc)))
            continue
        away = evidence['away_from_gripper']
        gain = min(away['after_m'].values()) - min(away['before_m'].values())
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError('Native admission returned a non-away route')
        accepted.append((gain, side, route, evidence))
    if not accepted:
        return None, dict(selected=None, rejected=rejected, runtime_replanning=False)
    _, side, route, evidence = max(accepted, key=lambda row: row[0])
    return deepcopy(route), dict(selected=side, rejected=rejected,
        admitted=[s for _, s, _, _ in accepted], selected_evidence=evidence,
        runtime_replanning=False)

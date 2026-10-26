"""Approved cap XY domain for four sofa targets; no USD or physics writes.

The caller supplies the selected asset's actual, already-oriented mesh footprint
relative to its origin, expressed in canonical sofa UV. Returned translations
must be rigid-mapped into the verified layout by the scene initializer.
This does not set orientation, height, robot posture or score.
"""
import hashlib
import json
import math
from pathlib import Path
import random

POLICY = Path(__file__).with_name('position_policy.json')


def policy():
    return json.loads(POLICY.read_text())


def bounds(points):
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError('Complete oriented XY footprint vertices required')
    if any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                              for v in p) for p in points):
        raise ValueError('Finite footprint vertices required')
    lo = [min(p[i] for p in points) for i in (0, 1)]
    hi = [max(p[i] for p in points) for i in (0, 1)]
    if any(lo[i] >= hi[i] for i in (0, 1)):
        raise ValueError('Nonzero footprint area required')
    return lo, hi


def fits(points):
    lo, hi = bounds(points)
    p = policy(); lower, upper = p['footprint_bounds_uv_m']
    if any(lo[i] < lower[i]-1e-9 or hi[i] > upper[i]+1e-9 for i in (0, 1)):
        return False
    pad = p['structural_padding_m']
    for obstacle in p['obstacles']:
        a, b, c, d = obstacle['bounds']
        if not (hi[0]+pad <= a+1e-9 or lo[0]-pad >= c-1e-9
                or hi[1]+pad <= b+1e-9 or lo[1]-pad >= d-1e-9):
            return False
    return True


def sample_positions(asset, seed, oriented_footprint_uv, *, count):
    """Uniform feasible-origin proposals for the complete footprint, not center.

    No room obstacle/support claim: the caller must still check actual geometry,
    other objects, robot feasibility and native settling before execution.
    """
    p = policy()
    if asset not in p['assets']:
        raise ValueError('Unsupported sofa-to-table asset')
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Unsigned nonzero 32-bit seed required')
    if type(count) is not int or count < 1:
        raise ValueError('Positive proposal count required')
    lo, hi = bounds(oriented_footprint_uv)
    lower, upper = p['footprint_bounds_uv_m']
    start = [lower[i]-lo[i] for i in (0, 1)]
    end = [upper[i]-hi[i] for i in (0, 1)]
    if any(start[i] > end[i] for i in (0, 1)):
        raise ValueError('Target footprint cannot fit approved cap domain; no widening')
    digest = hashlib.sha256(f'sofa-to-coffee-table-position-v1:{asset}:{seed}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest, 'big'))
    result = []
    for index in range(count):
        uv = [rng.uniform(start[i], end[i]) for i in (0, 1)]
        moved = [[v[i]+uv[i] for i in (0, 1)] for v in oriented_footprint_uv]
        if not fits(moved):
            raise ValueError('Policy obstacle/domain inconsistency; no unsafe fallback')
        result.append(dict(asset=asset, seed=seed, candidate_index=index,
            origin_uv_m=uv, footprint_uv=moved, position_domain=p['domain'],
            native_physics_validated=False))
    return result

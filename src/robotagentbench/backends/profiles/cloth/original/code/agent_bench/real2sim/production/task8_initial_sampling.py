"""Seeded, footprint-safe initial sampling for the Task8 owner producer."""
from collections import Counter
import numpy as np


def sample_assets(pool, count, rng):
    pool = list(dict.fromkeys(pool))
    if count not in (1, 2, 3) or len(pool) * 2 < count:
        raise ValueError("Need 1–3 targets and enough identities for a two-copy cap")
    selected = []
    for _ in range(count):
        counts = Counter(selected)
        choices = [asset for asset in pool if counts[asset] < 2]
        selected.append(str(rng.choice(choices)))
    return selected


def sample_centres(sizes, cloth_bounds, rng, margin=.008, gap=.015):
    """Enumerate axis separation constraints for at most three real footprints.

    Sample continuous coordinates within each feasible constraint system. This
    also handles tight book compositions without reverting to fixed grid poses.
    """
    from itertools import combinations, product
    sizes = np.asarray(sizes, dtype=float)
    cloth = np.asarray(cloth_bounds, dtype=float)
    n = len(sizes)
    if n not in (1, 2, 3) or sizes.shape != (n, 3) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('Invalid target sizes')
    low = cloth[0, :2] + sizes[:, :2]/2 + margin
    high = cloth[1, :2] - sizes[:, :2]/2 - margin
    pairs = list(combinations(range(n), 2))
    patterns = list(product(range(4), repeat=len(pairs)))
    rng.shuffle(patterns)
    def tighten(lo, hi, edges):
        for _ in range(n + 1):
            for a, b, axis, distance in edges:
                lo[b, axis] = max(lo[b, axis], lo[a, axis] + distance)
                hi[a, axis] = min(hi[a, axis], hi[b, axis] - distance)
            if np.any(lo > hi + 1e-10):
                return False
        return True
    for pattern in patterns:
        edges = []
        for (i, j), relation in zip(pairs, pattern):
            axis = relation // 2
            a, b = (i, j) if relation % 2 == 0 else (j, i)
            edges.append((a, b, axis, (sizes[a, axis] + sizes[b, axis])/2 + gap))
        lo, hi = low.copy(), high.copy()
        if not tighten(lo, hi, edges):
            continue
        for i in rng.permutation(n):
            for axis in rng.permutation(2):
                value = rng.uniform(lo[i, axis], max(lo[i, axis], hi[i, axis]))
                lo[i, axis] = hi[i, axis] = value
                if not tighten(lo, hi, edges):
                    break
            else:
                continue
            break
        else:
            return np.column_stack((lo, cloth[1, 2] + .003 + sizes[:, 2]/2))
    raise ValueError('Requested composition cannot fit on cloth without overlap')

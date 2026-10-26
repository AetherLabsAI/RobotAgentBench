"""Task-local extraction of the reviewed source's 1–3-book sampler.

Source: source_randomization_16/task8_initial_sampling.py (hash in review).
No source producer import, physics edits, clipping or rejection reseeding.
"""
from itertools import combinations, product
import numpy as np


def sample_centres(sizes, cloth_bounds, rng):
    sizes, cloth = np.asarray(sizes, float), np.asarray(cloth_bounds, float)
    n = len(sizes)
    if (n not in (1, 2, 3) or sizes.shape != (n, 3) or cloth.shape != (2, 3)
            or not np.isfinite(sizes).all() or not np.isfinite(cloth).all()
            or np.any(sizes <= 0) or np.any(cloth[0] > cloth[1])):
        raise ValueError('Invalid book composition or cloth bounds')
    low, high = cloth[0, :2]+sizes[:, :2]/2+.008, cloth[1, :2]-sizes[:, :2]/2-.008
    pairs = list(combinations(range(n), 2))
    patterns = list(product(range(4), repeat=len(pairs)))
    rng.shuffle(patterns)

    def tighten(lo, hi, edges):
        for _ in range(n+1):
            for a, b, axis, distance in edges:
                lo[b, axis] = max(lo[b, axis], lo[a, axis]+distance)
                hi[a, axis] = min(hi[a, axis], hi[b, axis]-distance)
            if np.any(lo > hi+1e-10):
                return False
        return True

    for pattern in patterns:
        edges = []
        for (i, j), relation in zip(pairs, pattern):
            axis = relation//2
            a, b = (i, j) if relation % 2 == 0 else (j, i)
            edges.append((a, b, axis, (sizes[a, axis]+sizes[b, axis])/2+.015))
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
            return np.column_stack((lo, cloth[1, 2]+.003+sizes[:, 2]/2))
    raise ValueError('Books do not fit on cloth without overlap')

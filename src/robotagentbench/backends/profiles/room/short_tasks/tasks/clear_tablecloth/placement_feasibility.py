"""Cheap cached preflight for the canonical cloth; composer checks actual geometry."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from openend.tasks.scene_requirements.cloth_placement import sample_centres

@lru_cache(maxsize=1)
def _dimensions():
    return json.loads(Path(__file__).with_name('placement_sizes.json').read_text())

@lru_cache(maxsize=8192)
def _can_place(categories):
    data=_dimensions()
    if not 1 <= len(categories) <= 4:
        return False
    sizes=[data['sizes_m'][name] for name in categories]
    try:
        sample_centres(sizes,data['cloth_bounds_m'],np.random.default_rng(0))
    except ValueError:
        return False
    return True

def can_place(categories):
    return _can_place(tuple(sorted(categories)))

def sample_categories(rng,pool,n):
    # Keep the requested count, RNG stream, target identity and original gaps.
    for _ in range(128):
        values=rng.sample(pool,n)
        if can_place(values):
            return values
    raise ValueError('No feasible cloth target combination found; target count was not reduced')

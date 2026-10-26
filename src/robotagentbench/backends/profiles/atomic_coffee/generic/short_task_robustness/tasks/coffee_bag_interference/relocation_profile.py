"""Training-domain endpoint sampling; distance follows the feasible endpoint."""
import math
import random
from .training_domain import contains_origin

from .motion import DURATION_S as MOVEMENT_DURATION_S


def sample_destination(seed,start_origin,training_domain):
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative integer seed required')
    if len(start_origin)!=3 or not all(math.isfinite(v) for v in start_origin):
        raise ValueError('Finite original bag root required')
    if not contains_origin(start_origin,training_domain):
        raise ValueError('Start must be inside the original training domain')
    lower,upper=training_domain['origin_bounds_world_xy_m']
    rng=random.Random(seed)
    destination=[rng.uniform(lower[i],upper[i]) for i in range(2)]+[float(start_origin[2])]
    distance=math.hypot(destination[0]-start_origin[0],destination[1]-start_origin[1])
    if distance==0.:raise ValueError('Sampled unchanged position: no interference, no silent resampling')
    return dict(seed=seed,destination_origin_m=destination,distance_m=distance,
                sampling='one uniform XY endpoint in original bag training domain',
                minimum_distance_m=None,height_preserved=True)

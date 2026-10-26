"""Reuse the atomic 1-2 initial sampler without scene or parameter writes."""
import math

YAW_RANGE_DEG = (0.,90.)


def validate_yaw(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0. <= value <= 90.:
        raise ValueError('Coffee-bag yaw must be within approved nominal-frame 0–90 degrees')
    return float(value)


def sample_profile(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative integer seed required')
    from evaluation.tasks.coffee_machine.randomization import sample_pour
    # Keep the original draw order and source angle-coverage seed selection.
    # The source uses uniform(0,90); no scene edits or success-history writes.
    profile = sample_pour(seed)
    return dict(source_atomic_api='1-2', original_proposal=profile,
                bag_yaw_deg=validate_yaw(profile['bag_yaw_deg']),
                bag_yaw_range_deg=list(YAW_RANGE_DEG),
                reference='atomic nominal bag orientation, rotation about world vertical',
                seed=seed, episode_seed=profile['episode_seed'],
                robot_randomization=False, scene_written=False,
                physics_parameters_written=False)

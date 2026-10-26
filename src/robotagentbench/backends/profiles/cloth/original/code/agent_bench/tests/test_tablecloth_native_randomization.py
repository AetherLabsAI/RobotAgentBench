import numpy as np

from experiments.physics_fluid.tablecloth_native_randomization import sample_reset
from real2sim.production.task_randomization.benchmark_8_10.inject_robot_initial_state import (
    sample_initial_perturbation,
)


def test_full_robot_sampler_and_missing_object_sampler_are_explicit():
    for seed in (0, 12, 2026092100):
        for workflow in ('remove', 'spread'):
            draw = sample_reset(workflow, seed)
            assert draw['reference_robot_draw'] == sample_initial_perturbation(seed)
            assert draw['applied'] is False and draw['actual_readback'] is None
            obj = draw['interaction_object_pose']
            assert obj['required'] is True and obj['sampled'] is False
            assert obj['actual_readback'] is None and obj['blocker']
            assert 'user_exception' not in draw

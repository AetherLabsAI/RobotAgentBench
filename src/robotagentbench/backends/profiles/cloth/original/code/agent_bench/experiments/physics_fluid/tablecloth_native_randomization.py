"""Plan the user-requested cloth reset variation; plans are not physics data.

All applicable owner ranges are required. The historical clean-cloth exemption
is superseded. Missing object samplers block production, not silently fix poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from real2sim.production.task_randomization.benchmark_8_10.inject_robot_initial_state import (
    sample_initial_perturbation,
)


def sample_reset(workflow, seed):
    if workflow not in ('remove', 'spread'):
        raise ValueError('Expected remove or spread workflow')
    # Preserve the delivered owner sampler, including its nonzero base centre
    # and complete ranges; do not rewrite its identity to masquerade as 8.7.
    robot = sample_initial_perturbation(seed)
    return dict(workflow=workflow, seed=int(seed), reference_robot_draw=robot,
        robot_sampler='real2sim/production/task_randomization/benchmark_8_10/inject_robot_initial_state.py',
        robot_sampler_reuse='same delivered full robot range for both workflows',
        interaction_object_pose=dict(required=True, owner_sampler=None,
            sampled=False, applied=False, actual_readback=None,
            blocker='current full-range object sampler must be delivered and connected'),
        applied=False, actual_readback=None)


def build_batch(seed, count=50):
    from real2sim.production import appearance, house_layout
    from real2sim.production.benchmark_eval_registry import resolve

    if not 1 <= count <= 50:
        raise ValueError('Count must be 1..50 per workflow')
    rows = []
    for workflow, task_ids in (('remove', ['8.7', '8.8']), ('spread', ['8.9', '8.10'])):
        # Catalog task identities stay independent. A row is a continuous
        # workflow request, not a second definition of either Sheet task.
        tasks = [resolve(t)['task'] for t in task_ids]
        for ordinal in range(count):
            row_seed = seed+ordinal+(0 if workflow == 'remove' else 50)
            layout = house_layout.scheduled_layout_for_core(tasks[0]['key'], ordinal, seed=seed)
            rows.append(dict(workflow=workflow, task_ids=task_ids, seed=row_seed,
                layout_id=layout.layout_id, reset=sample_reset(workflow, row_seed),
                appearance=appearance.build_plan(task_id=tasks[0]['key'], gen_seed=row_seed),
                physical_distractors=dict(required=True, owner='real2sim/production/prepare_physical_distractors.py',
                                          sampled=False, applied=False),
                physics_validated=False, randomization_readback_verified=False,
                eligible_for_production=False, status='planned_only'))
    return dict(schema='real2sim.tablecloth_workflow_plan.v1', rows=rows,
                count_per_workflow=count, executable_production_ready=False,
                missing_validation=['complete friction-only workflow', 'full-range interaction-object sampler', 'all-axis physical readback',
                                    'owner task success gates', 'rendered actual scene state'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = build_batch(args.seed, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(f'{len(plan["rows"])} planned rows; no physics success or randomized count claimed')


if __name__ == '__main__':
    main()

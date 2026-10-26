"""Thin launcher of the public runner for one bound source trajectory, with explicit cold-start wait.

No custom scoring, worker logic, simulation timeout or physical override: the score is the registered
`coffee_machine.process.v1` server evaluation. Per-step evidence is written by the task side
(`coffee_detection/physical_facts.jsonl` and `control_facts.jsonl` in the episode directory).
"""
import argparse
import json
import math
import os
from pathlib import Path

from arena_api.action_sources import TrajectoryActionSource
from arena_api.client import ArenaClient
from arena_api.evaluate import evaluate_task
from arena_api.task_evaluation import load_tasks, TaskDefinition
from arena_api import trajectory

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-url', default='http://127.0.0.1:8180')
    parser.add_argument('--reset-wait-s', type=float, default=900.)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--wall-watchdog-s', type=float, default=7200.)
    parser.add_argument('--api', required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((TASK / 'artifacts').resolve()):
        parser.error('Output must be in this task artifacts directory')
    for name, value in (('Reset wait', args.reset_wait_s), ('Wall watchdog', args.wall_watchdog_s)):
        if not math.isfinite(value) or value <= 0:
            parser.error(name + ' must be finite and positive')
    launch = output.with_suffix('.launch.json')
    if output.exists() or launch.exists():
        parser.error('Use a fresh output and launch record')
    recorded, _ = trajectory.load_trajectory(args.trajectory)
    reset = recorded['reset']
    if reset['task_id'] != args.api or reset['seed'] != args.seed:
        parser.error('Trajectory belongs to another API or seed')
    task = load_tasks(REPO / 'configs/tasks.json')[args.api]
    task = TaskDefinition.model_validate(dict(
        task.model_dump(), seed=args.seed,
        randomization=dict(appearance=False, layout=False, microwave_pose=False, left_arm_pose=False),
        task_parameters=dict(reset['task_parameters'])))
    output.parent.mkdir(parents=True, exist_ok=True)
    launch.write_text(json.dumps(dict(
        task_id=task.task_id, env_url=args.env_url, trajectory=str(args.trajectory.resolve()), reset_wait_s=args.reset_wait_s,
        task_timeout_sim_s=task.timeout_s, wall_watchdog_s=args.wall_watchdog_s, evaluator_id=task.evaluator_id,
        task_parameters=task.task_parameters, runner='arena_api.evaluate.evaluate_task',
        step_evidence='episodes/<episode_id>/coffee_detection/{physical_facts,control_facts}.jsonl',
        reason='Allow cold-start response without changing task simulation time or physics'), indent=2) + '\n')
    report = evaluate_task(task, output=output, wall_watchdog_s=args.wall_watchdog_s,
                           env_factory=lambda: ArenaClient(args.env_url, token=os.environ.get('ARENA_API_TOKEN'),
                                                           timeout=240, reset_timeout=args.reset_wait_s),
                           action_source=TrajectoryActionSource(args.trajectory))
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(2 if report['run_status'] != 'completed' else 0 if report['success'] is True else 1)


if __name__ == '__main__':
    main()

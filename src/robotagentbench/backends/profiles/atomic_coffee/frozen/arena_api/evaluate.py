"""Unified task -> model OR recorded actions -> public API -> evaluation runner."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from .client import ArenaClient
from .closed_loop import run
from .action_sources import ModelActionSource, TrajectoryActionSource
from .policy_contract import digest
from .policy_ws import BlackboxClient
from .task_evaluation import EVALUATORS, TaskDefinition, TaskMonitor, load_tasks


def write_json(path, value):
    # Reports are always under a fresh run directory; atomic replacement keeps
    # a readable last snapshot if interrupted during report serialization.
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def evaluate_task(task: TaskDefinition, *, output: Path, env_factory, policy_factory=None,
                  action_source=None,
                  service_label=None, evaluator=None, evaluator_id=None, wall_watchdog_s=None):
    """Factories are context managers; connect/init errors also produce a report.

    evaluator injection is a trusted local extension/test seam, not a CLI or
    network code-loading mechanism. Recorded actions use the same server scorer.
    """
    if (policy_factory is None) == (action_source is None):
        raise ValueError('Provide exactly one policy_factory or action_source')
    source_kind = 'model' if policy_factory is not None else action_source.kind
    if evaluator is not None and (not evaluator_id or evaluator_id == task.evaluator_id):
        raise ValueError("A custom evaluator requires its own explicit versioned ID")
    if evaluator is None:
        if evaluator_id is not None:
            raise ValueError("evaluator_id override requires a matching evaluator")
        if task.evaluator_id not in EVALUATORS:
            raise ValueError("Unknown evaluator; no silent timeout-only fallback")
        evaluator = EVALUATORS[task.evaluator_id]
    if wall_watchdog_s is None:
        wall_watchdog_s = max(900, task.timeout_s * 30 + 300)
    if isinstance(wall_watchdog_s, bool) or not isinstance(wall_watchdog_s, (int, float)) or not math.isfinite(wall_watchdog_s) or wall_watchdog_s <= 0:
        raise ValueError("Wall watchdog must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    definition = task.model_dump(mode="json")
    manifest = {"schema_version": "arena.task_run.v1", "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": definition, "task_sha256": digest(definition),
        "evaluator_id": evaluator_id or task.evaluator_id, "service_label": service_label,
        "action_source": source_kind,
        "task_reset_validated": False, "wall_watchdog_s": wall_watchdog_s,
        "timeout_origin": "initial observation after scene load and settle"}
    write_json(output / "manifest.json", manifest)
    report = {"schema_version": "arena.task_result.v1", "run_id": manifest["run_id"],
        "task_id": task.task_id, "task_sha256": manifest["task_sha256"],
        "task_parameters": task.task_parameters,
        "action_source": source_kind,
        "evaluator_id": manifest["evaluator_id"], "run_status": "running",
        "episode_id": None, "artifacts": {"trace": "episode/trace.json",
                                       "events": "evaluation_events.jsonl"}}
    trace = {}

    def event(value):
        with (output / "evaluation_events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        report.update(monitor.summary())
        report.update(episode_id=str(monitor.episode_id), executed_steps=monitor.last_step)
        write_json(output / "report.json", report)

    monitor = TaskMonitor(task, evaluator=evaluator, record_event=event)
    report.update(monitor.summary())
    write_json(output / "report.json", report)
    try:
        with ExitStack() as stack:
            env = stack.enter_context(env_factory())
            source = (ModelActionSource(stack.enter_context(policy_factory()))
                      if policy_factory is not None else action_source)
            trace = run(env, action_source=source, instruction=task.instruction, output=output / "episode",
                max_steps=task.timeout_steps, prefix_steps=task.prefix_steps, seed=task.seed,
                deadline_s=manifest['wall_watchdog_s'], service_label=service_label, on_observation=monitor.observe,
                environment_task_id=task.environment_task_id, randomization=task.randomization.model_dump(),
                task_parameters=task.task_parameters)
            if monitor.verdict == "undetermined" and not (
                    trace.get('termination_reason') == 'source_exhausted' and
                    task.evaluator_id == 'server.v1' and evaluator_id is None):
                raise RuntimeError("Execution ended without a task verdict")
            if trace.get("cleanup") != {"environment": "closed", source.cleanup_key: "closed"}:
                raise RuntimeError("Episode cleanup was not confirmed")
        report["run_status"] = "completed"
    except KeyboardInterrupt:
        report.update(run_status="interrupted", error={"type": "KeyboardInterrupt"})
    except Exception as exc:
        # Preserve the failure category, not provider internals or credentials.
        report.update(run_status="error", error={"type": type(exc).__name__})
    finally:
        trace_path = output / "episode" / "trace.json"
        if trace_path.is_file():
            try:
                trace = json.loads(trace_path.read_text())
            except (OSError, ValueError):
                report.update(run_status="error", trace_error="unreadable_trace")
        report.update(monitor.summary())
        final = trace.get('result', {})
        if task.evaluator_id == 'server.v1' and evaluator_id is None and report['run_status'] == 'completed':
            if not final.get('evaluation_valid') or final.get('scoring_status') != 'finalized':
                report.update(run_status='error', evaluation_valid=False, success=None, score=None)
            else:
                report.update(score=final['score'], success=final['success'], evaluation=final['evaluation'],
                              verdict='success' if final['success'] is True else 'failure',
                              scoring_status='finalized', evaluation_valid=True,
                              task_reset_validated=True, termination_reason=final['termination_reason'])
        if report['run_status'] != 'completed':
            report['last_observed_evaluation'] = monitor.summary()
            report.update(evaluation_valid=False, success=None, score=None,
                          scoring_status='invalid', verdict='undetermined',
                          termination_reason='run_' + report['run_status'])
        report.update(episode_id=trace.get("episode_id"),
            executed_steps=len(trace.get("steps", [])),
            inferences=len(trace.get("calls", [])) if source_kind == 'model' else 0,
            action_batches=len(trace.get("calls", [])),
            runner_stop_reason=trace.get('termination_reason'),
            cleanup=trace.get("cleanup", {}),
            closed_loop_transport_pass=trace.get("closed_loop_transport_pass", False),
            elapsed_wall_s=time.monotonic() - started)
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("configs/tasks.json"))
    parser.add_argument("--task", help="Task ID in the registry")
    parser.add_argument('--task-parameters-json', help='Task-owned parameter object; see /v1/tasks')
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--timeout-s", type=float, help="Explicit per-run task timeout override, recorded in manifest")
    parser.add_argument("--env-url", default="http://127.0.0.1:8093")
    parser.add_argument("--policy-url")
    parser.add_argument('--source', choices=('model', 'trajectory'), default='model')
    parser.add_argument('--trajectory', type=Path, help='Portable arena.recorded_actions.v1 JSON')
    parser.add_argument("--service-label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        tasks = load_tasks(args.tasks)
        if args.list_tasks:
            print(json.dumps([t.model_dump(mode="json") for t in tasks.values()], indent=2))
            return
        if not args.task or args.task not in tasks or args.output is None:
            parser.error('Provide a registered --task and fresh --output directory')
        if args.source == 'model' and (not args.policy_url or args.trajectory):
            parser.error('--source model requires --policy-url and forbids --trajectory')
        if args.source == 'trajectory' and (not args.trajectory or args.policy_url):
            parser.error('--source trajectory requires --trajectory and forbids --policy-url')
        task = tasks[args.task]
        if args.task_parameters_json is not None:
            from .trajectory import _unique
            from .task_registry import default_registry
            parameters = json.loads(args.task_parameters_json, object_pairs_hook=_unique)
            registration = default_registry().get(task.environment_task_id)
            instruction = task.instruction
            if instruction == registration.resolve_instruction('', task.task_parameters):
                instruction = registration.resolve_instruction('', registration.resolve_parameters(parameters))
            task = TaskDefinition.model_validate(dict(task.model_dump(), task_parameters=parameters,
                                                      instruction=instruction))
        if args.timeout_s is not None:
            task = TaskDefinition.model_validate(dict(task.model_dump(), timeout_s=args.timeout_s))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"TASK {task.task_id}: timeout={task.timeout_s}s simulation time; "
          f"environment={task.environment_task_id}, evaluator={task.evaluator_id}", flush=True)
    try:
        report = evaluate_task(task, output=args.output, service_label=args.service_label,
            env_factory=lambda: ArenaClient(args.env_url, token=os.environ.get("ARENA_API_TOKEN"), timeout=240),
            policy_factory=(lambda: BlackboxClient(args.policy_url, token=os.environ.get("ARENA_POLICY_TOKEN")))
                           if args.source == 'model' else None,
            action_source=TrajectoryActionSource(args.trajectory) if args.source == 'trajectory' else None)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report["run_status"] == "interrupted":
        raise SystemExit(130)
    if report["run_status"] != "completed":
        raise SystemExit(2)
    raise SystemExit(0 if report["success"] is True else 1)


if __name__ == "__main__":
    main()

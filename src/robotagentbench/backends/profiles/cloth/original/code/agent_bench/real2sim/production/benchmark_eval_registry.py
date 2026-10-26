#!/usr/bin/env python3
"""Sheet-backed benchmark catalog and runtime component inspection.

The projected Sheet rows are the only task set. Runtime code is either
discovered below the task's
conventional directory or supplied by one of the shared Kitchen/recovered
component contracts. Availability is computed when ``preflight`` runs; it is
not copied from another task list.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from air_fryer_task_contract import preflight as air_fryer_preflight  # noqa: E402
from dining_task_contract import preflight as dining_preflight  # noqa: E402
from house_task_environment import resolve as resolve_house_environment  # noqa: E402
from kitchen_task_components import (  # noqa: E402
    COMPONENTS as KITCHEN_COMPONENTS,
    preflight_component as kitchen_component_preflight,
    resolve_component as resolve_kitchen_component,
)
from recovered_task_contract import (  # noqa: E402
    TASKS as RECOVERED_COMPONENTS,
    preflight as recovered_preflight,
)
from task_catalog_projection import project_sheet  # noqa: E402


SHEET_PATH = HERE / "google_sheet_tasks_current.json"
SCHEMA = "real2sim.benchmark_catalog.v4"
DINING_FAMILIES = frozenset({1, 2, 9, 10})
OWNER_DEFERRED_FAMILIES = frozenset({8})
OWNER_DEFERRED_STATUS = "deferred_by_owner"
VISUAL_IMPLEMENTATIONS = {
    **{
        task: {
            "adapter_id": "family4_canonical_state_capture_v1",
            "adapter_path": (
                "real2sim/production/"
                "family4_task_owned_visual_adapter.py"
            ),
            "source_dataset": (
                "dataset/sim_episodes_repro/task_owned_50x4/"
                f"benchmark_{task.replace('.', '_')}"
            ),
        }
        for task in (
            "4.1", "4.2", "4.3", "4.7", "4.8", "4.9", "4.13", "4.14"
        )
    },
    **{
        task: {
            "adapter_id": "coffee_table_drawer_state_capture_v2",
            "adapter_path": (
                "real2sim/production/"
                "task6_drawer_task_owned_visual_adapter.py"
            ),
            "source_dataset": (
                "real2sim/production/action_templates"
            ),
        }
        for task in ("4.4", "4.10")
    },
    "3.1": {
        "adapter_id": "shelf_state_capture_v1",
        "adapter_path": (
            "real2sim/production/benchmark_3_29_task_owned_visual_adapter.py"
        ),
        "source_dataset": (
            "dataset/sim_episodes_repro/task_owned_50x4/benchmark_3_1"
        ),
    },
    "3.29": {
        "adapter_id": "benchmark_3_29_state_capture_v1",
        "adapter_path": (
            "real2sim/production/"
            "benchmark_3_29_task_owned_visual_adapter.py"
        ),
        "source_dataset": (
            "dataset/sim_episodes_repro/task_owned_50x4/benchmark_3_29"
        ),
    },
    "6.5": {
        "adapter_id": "task6_drawer_state_capture_v1",
        "adapter_path": (
            "real2sim/production/"
            "task6_drawer_task_owned_visual_adapter.py"
        ),
        "source_dataset": (
            "dataset/sim_episodes_repro/"
            "task6_drawer_task_owned_sources_v1"
        ),
    },
    "6.15": {
        "adapter_id": "task6_drawer_state_capture_v1",
        "adapter_path": (
            "real2sim/production/"
            "task6_drawer_task_owned_visual_adapter.py"
        ),
        "source_dataset": (
            "dataset/sim_episodes_repro/"
            "task6_drawer_task_owned_sources_v1"
        ),
    },
    **{
        task: {
            "adapter_id": "family7_state_capture_v1",
            "adapter_path": (
                "real2sim/production/"
                "family7_task_owned_visual_adapter.py"
            ),
            "source_dataset": (
                "dataset/sim_episodes_repro/"
                "family7_code_control_single_target_tray_v54"
            ),
        }
        for task in ("7.2", "7.4", "7.8", "7.10")
    },
}

# These are implementation links, not a second task registry.  The Sheet task
# remains the sole identity and semantic source; this table only connects an
# already delivered owner scorer to that identity.  A task omitted here and
# without a shared Kitchen/recovered checker is deliberately fail-closed and
# returned to its Sheet owner.
OWNER_SUCCESS_IMPLEMENTATIONS = {
    **{task: {
        "checker": "real2sim/production/check_task8_clear.py:score",
        "adapter": "tablecloth_clear",
        "evidence_schema": "real2sim.task8_clear_score.v1",
        "required_artifacts": ["action.npz", "full_state.npz", "result.json", "contacts.json"],
        "parameters": {},
    } for task in ("8.2", "8.4", "8.6")},
    **{
        task: {
            "checker": "real2sim/production/score_family4_drawer_pick_episode.py:score",
            "adapter": "family4_drawer_pick",
            "evidence_schema": "real2sim.family4_drawer_pick_success.v1",
            "required_artifacts": [
                "action.npz", "full_state.npz", "pick_contact_trace.npz", "scene_setup.json"
            ],
            "parameters": {"expected_action_kind": ["controller_input", "executed_commands"]},
        }
        for task in ("4.4", "4.10")
    },
    **{task: {
        "checker": "real2sim/production/score_shelf_pick_episode.py:score",
        "adapter": "shelf_pick",
        "evidence_schema": "real2sim.shelf_pick_score.v1",
        "required_artifacts": ["action.npz", "full_state.npz", "pick_contact_trace.npz"],
        "parameters": {},
    } for task in ("3.1", "3.7", "3.11")},
    "8.10": {
        "checker": "real2sim/production/task_randomization/benchmark_8_10/check_success.py:check_success",
        "adapter": "tablecloth_spread",
        "evidence_schema": "real2sim.tablecloth_spread_score.v1",
        "required_artifacts": ["action.npz", "full_state.npz", "tablecloth_evidence.json"],
        "parameters": {},
    },
    "3.29": {
        "checker": (
            "real2sim/production/score_pick_episode.py:score_pick_episode"
        ),
        "adapter": "livingroom_pick",
        "evidence_schema": "real2sim.livingroom_pick_score.v1",
        "required_artifacts": ["full_state.npz", "pick_contact_trace.npz"],
        "parameters": {"runtime_task_id": "38"},
    },
    **{
        task: {
            "checker": (
                "real2sim/production/score_pick_episode.py:score_pick_episode"
            ),
            "adapter": "livingroom_pick",
            "evidence_schema": "real2sim.livingroom_pick_score.v1",
            "required_artifacts": [
                "action.npz", "full_state.npz", "pick_contact_trace.npz"
            ],
            "parameters": {"runtime_task_id": runtime_task},
        }
        for task, runtime_task in {
            # Delivered shelf pick aliases: each selects its existing
            # object-bound physical lift/contact/retained-grasp checker.
            "3.3": "6", "3.5": "22", "3.9": "47", "3.13": "8",
            "3.15": "10", "3.19": "35", "3.21": "36", "3.23": "40",
            "3.25": "37", "3.27": "39",
            "4.1": "91",
            "4.2": "57",
            # L2's delivered 405 target/station binding is now integrated.
            "4.3": "405",
            "4.7": "53",
            "4.8": "55",
            "4.9": "60",
            "4.13": "31",
            # The current Sheet task is the remote on the small side table.
            # Runtime Task 61 is the superseded coffee-table placement; using
            # it here would certify a different task definition.
            "4.14": "109",
            # The Sheet's coffee-table wallet uses 55; sofa alias 53 does
            # not match this source region.
            "5.9": "55",
            "6.4": "11",
            "6.6": "32",
            "6.8": "31",
            "6.11": "33",
            "6.14": "53",
            "6.17": "91",
            "7.13": "55",
            "7.15": "57",
            "7.17": "109",
        }.items()
    },
    "6.1": {
        "checker": "real2sim/production/score_pick_episode.py:score_pick_episode",
        "adapter": "livingroom_pick",
        "evidence_schema": "real2sim.livingroom_pick_score.v1",
        "required_artifacts": ["action.npz", "full_state.npz", "pick_contact_trace.npz"],
        "parameters": {"runtime_task_ids": ["12", "13"]},
    },
    **{
        task: {
            "checker": "real2sim/production/score_place_episode.py:score_place_episode",
            "adapter": "livingroom_place",
            "evidence_schema": "real2sim.strict_place_state_score.v1",
            "required_artifacts": ["action.npz", "full_state.npz", "place_metrics.npz"],
            "parameters": {"runtime_task_id": runtime_task},
        }
        # Delivered L1 place implementations. The Sheet owns the task;
        # these labels select existing object/destination-bound scorers.
        # Historical score.json files are never used as a verdict.
        for task, runtime_task in {
            "3.2": "88", "3.4": "14", "3.6": "17", "3.8": "89",
            "3.10": "86", "3.12": "87", "3.14": "19", "3.16": "18",
            "3.20": "43", "3.22": "42", "3.24": "45", "3.26": "41", "3.28": "46",
            "3.30": "44",
            # Delivered tote-place implementations: held umbrella, wallet
            # and bottle respectively. Their destination and target remain
            # checked from the raw place metrics and captured state.
            "5.2": "71", "5.10": "74", "5.12": "73",
        }.items()
    },
    **{
        task: {
            "checker": "real2sim/production/score_place_episode.py:score_place_episode",
            "adapter": "livingroom_place",
            "evidence_schema": "real2sim.strict_place_state_score.v1",
            "required_artifacts": ["action.npz", "full_state.npz", "place_metrics.npz"],
            "parameters": {"runtime_task_ids": runtime_tasks},
        }
        for task, runtime_tasks in {"6.2": ["20", "27"], "6.16": ["74"]}.items()
    },
    **{
        task: {
            "checker": (
                "real2sim/production/score_task6_code_control.py:score"
            ),
            "adapter": "task6_place",
            "evidence_schema": "real2sim.task6_code_control_score.v1",
            "required_artifacts": [
                "action.npz", "full_state.npz", "place_metrics.npz"
            ],
        }
        for task in ("6.3", "6.5", "6.7", "6.9", "6.10", "6.15")
    },
    **{
        task: {
            "checker": (
                "real2sim/teleop/score_family7_code_control.py:score"
            ),
            "adapter": "family7_e2e",
            "evidence_schema": "real2sim.family7_code_control_score.v1",
            "required_artifacts": [
                "action.npz", "full_state.npz", "pick_contact_trace.npz"
            ],
            "parameters": {"mode": mode},
        }
        for task, mode in {
            "7.2": "paper",
            "7.4": "snack",
            "7.8": "can",
            "7.10": "cup",
        }.items()
    },
}

OWNER_SUCCESS_BLOCKERS = {
    "4.3": (
        "owner must bind benchmark 4.3 to a current runtime identity and "
        "deliver a scorer that the current production contract accepts"
    ),
}

DEFAULT_METRIC_ARTIFACTS = [
    "task_metrics.npz",
    "metrics.npz",
    "task_metrics.json",
    "metrics.json",
]
DEFAULT_RANDOMIZATION_AUDITS = [
    "randomization_audit.json",
    "distractor_audit.json",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _reference_ids(*values: Any) -> list[str]:
    """Collect optional real-recording labels without making them identity."""
    references: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            value = value.get(
                "real_data_reference_ids",
                value.get("canonical_real_data_reference_ids", ()),
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            references.update(str(item) for item in value)
    return sorted(references, key=lambda item: (not item.isdigit(), item))


def _implementation(benchmark: str) -> dict[str, Any]:
    """Discover the task's Git-owned components by one path convention."""
    task_key = "benchmark_" + benchmark.replace(".", "_")
    directory = HERE / "task_randomization" / task_key
    result: dict[str, Any] = {}
    for role, path in {
        "randomization": directory / "inject_robot_initial_state.py",
        "producer": directory / "produce_cores.py",
        "policy": directory / "policy.json",
    }.items():
        if path.is_file():
            result[role] = str(path.relative_to(REPO))
    visual = VISUAL_IMPLEMENTATIONS.get(benchmark)
    if visual is not None:
        result["visual"] = {
            **visual,
            "prepare_callable": "prepare_visual_scene",
            "close_callable": "close_visual_scene",
            "state_renderer_path": (
                "real2sim/production/state_driven_visual_renderer.py"
            ),
        }
    return result


def _runtime_handler(
    benchmark: str, implementation: Mapping[str, Any]
) -> dict[str, Any]:
    if benchmark in KITCHEN_COMPONENTS:
        kitchen = resolve_kitchen_component(benchmark)
        return {
            "kind": "canonical_house_kitchen_pipeline",
            "group": kitchen["group"],
            "pipeline": kitchen["pipeline"],
            "initializer": kitchen["initializer_entrypoint"],
            "checker": kitchen["checker"],
            "metrics_schema": kitchen["metrics_schema"],
        }
    if benchmark in RECOVERED_COMPONENTS:
        return {
            "kind": "recovered_house_pipeline",
            "initializer": (
                "real2sim/production/recovered_task_contract.py:"
                "prepare_rollout"
            ),
            "checker": (
                "real2sim/production/"
                "recovered_task_check_success.py:check_task"
            ),
            "metrics_schema": "real2sim.recovered_task_metrics.v1",
        }
    if implementation:
        return {
            "kind": "owner_component_discovered",
            "components": dict(implementation),
        }
    return {
        "kind": "owner_delivery_pending",
        "description": (
            "task semantics are complete; no central action/state component "
            "has been delivered at the conventional path"
        ),
    }


def _success_contract(
    benchmark: str,
    handler: Mapping[str, Any],
    *,
    owner: str,
) -> dict[str, Any]:
    """Return the one task-success interface attached to the Sheet task.

    Different tasks keep different physical conditions.  This contract only
    standardizes how central production locates and invokes the owner checker.
    Missing or ambiguous owner logic is an explicit unverified state, never a
    default success rule.
    """
    delivered = OWNER_SUCCESS_IMPLEMENTATIONS.get(benchmark)
    checker = str(handler.get("checker", "")) or None
    metrics_schema = str(handler.get("metrics_schema", "")) or None
    if delivered is not None:
        return {
            "schema": "real2sim.task_success_contract.v1",
            "status": "registered",
            "checker": delivered["checker"],
            "adapter": delivered["adapter"],
            "evidence_schema": delivered["evidence_schema"],
            "required_artifacts": list(delivered["required_artifacts"]),
            "parameters": dict(delivered.get("parameters", {})),
            "environment_scope": "canonical_house_default",
            "result_schema": "real2sim.task_success_result.v1",
        }
    if checker is not None:
        return {
            "schema": "real2sim.task_success_contract.v1",
            "status": "registered",
            "checker": checker,
            "adapter": "metrics",
            "evidence_schema": metrics_schema,
            "required_artifact_alternatives": {
                "metrics": list(DEFAULT_METRIC_ARTIFACTS),
                "randomization_audit": list(DEFAULT_RANDOMIZATION_AUDITS),
            },
            "parameters": {
                "call_style": (
                    "metrics_only"
                    if checker.endswith(":check_task75_pour")
                    else "task_and_metrics"
                )
            },
            "environment_scope": "canonical_house_default",
            "result_schema": "real2sim.task_success_result.v1",
        }
    reason = OWNER_SUCCESS_BLOCKERS.get(
        benchmark,
        "owner must deliver and register an explicit task success checker",
    )
    return {
        "schema": "real2sim.task_success_contract.v1",
        "status": "owner_action_required",
        "checker": None,
        "adapter": None,
        "evidence_schema": None,
        "required_artifacts": [],
        "parameters": {},
        "environment_scope": "canonical_house_default",
        "result_schema": "real2sim.task_success_result.v1",
        "owner": owner,
        "owner_requirement": reason,
    }


def _real_references(benchmark: str) -> list[str]:
    if benchmark in KITCHEN_COMPONENTS:
        return _reference_ids(resolve_kitchen_component(benchmark))
    if benchmark in RECOVERED_COMPONENTS:
        return _reference_ids(RECOVERED_COMPONENTS[benchmark])
    return []


def build_catalog() -> dict[str, dict[str, Any]]:
    sheet = project_sheet(_json(SHEET_PATH))
    catalog: dict[str, dict[str, Any]] = {}
    for source in sheet.get("tasks", ()):
        benchmark = str(source["benchmark_subtask"])
        family = int(source["benchmark_family"])
        if benchmark in catalog:
            raise ValueError(f"duplicate benchmark task {benchmark}")
        implementation = _implementation(benchmark)
        task = {
            "id": benchmark,
            "key": "benchmark_" + benchmark.replace(".", "_"),
            "family": family,
            "name": str(source.get("task_name", "")),
            "instruction": str(source.get("action", "")),
            "action": str(source.get("action", "")),
            "target_or_destination": str(
                source.get("object_or_destination", "")
            ),
            "owner": str(source.get("owner", "")),
            "source_sheet_rows": list(
                source.get("source_sheet_rows", (benchmark,))
            ),
            "granularity": str(
                source.get("task_granularity", "sheet_row")
            ),
            "sheet_progress": {
                key: source.get(key)
                for key in (
                    "planned",
                    "real_collected",
                    "real_candidates",
                    "sim_requested",
                    "sim_existing",
                    "sim_status",
                    "ddl",
                )
            },
            "implementation": implementation,
        }
        handler = _runtime_handler(benchmark, implementation)
        task["runtime"] = {
            "handler": handler,
            "environment": resolve_house_environment(task),
            "success": _success_contract(
                benchmark, handler, owner=task["owner"]
            ),
        }
        deferred = family in OWNER_DEFERRED_FAMILIES and benchmark not in {"8.2", "8.4", "8.6"}
        catalog[benchmark] = {
            "schema": "real2sim.benchmark_task.v4",
            "task": task,
            "defined_in_sheet": True,
            "owner_scope_status": (
                OWNER_DEFERRED_STATUS if deferred else "active"
            ),
            "in_current_owner_scope": not deferred,
            "real_data": {
                "required": False,
                "role": "optional_mother_or_reference",
                "legacy_recording_labels": _real_references(benchmark),
            },
        }
    expected = int(sheet.get("task_count", 0))
    if len(catalog) != expected:
        raise ValueError(
            f"Sheet projection contains {expected} tasks, got {len(catalog)}"
        )
    return catalog


CATALOG = build_catalog()


def resolve(benchmark_subtask: str) -> dict[str, Any]:
    task = str(benchmark_subtask)
    if task not in CATALOG:
        raise KeyError(f"unknown benchmark task {task}")
    return dict(CATALOG[task])


def _shared_runtime_preflight(
    task_id: str,
    episode_destination: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if task_id in KITCHEN_COMPONENTS:
        primary = dict(kitchen_component_preflight(task_id))
        blockers = list(primary.get("blockers", ()))
        missing = list(primary.get("missing_paths", ()))
        group = resolve_kitchen_component(task_id)["group"]
        if group in {"microwave_fridge", "serve_meal"}:
            detail = dining_preflight(task_id)
            blockers.extend(detail.get("blockers", ()))
            missing.extend(detail.get("missing_paths", ()))
        if group == "air_fryer":
            detail = air_fryer_preflight(task_id)
            blockers.extend(detail.get("blockers", ()))
            missing.extend(detail.get("missing_paths", ()))
        primary["blockers"] = list(dict.fromkeys(map(str, blockers)))
        primary["missing_paths"] = list(dict.fromkeys(map(str, missing)))
        return primary
    if task_id in RECOVERED_COMPONENTS:
        return dict(recovered_preflight(
            task_id, episode_destination=(
                dict(episode_destination)
                if episode_destination is not None else None
            ),
        ))
    return None


def preflight(
    benchmark_subtask: str,
    *,
    episode_destination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect runtime availability without changing the task definition."""
    task_id = str(benchmark_subtask)
    entry = resolve(task_id)
    task = entry["task"]
    success_contract = task["runtime"]["success"]
    success_ready = success_contract["status"] == "registered"
    component = _shared_runtime_preflight(task_id, episode_destination)
    blockers: list[str] = []
    missing: list[str] = []
    source_revision = None
    runtime_materialized = False
    pipeline_ready = False
    schedulable = False
    runtime_ready = False

    if component is not None:
        blockers.extend(map(str, component.get("blockers", ())))
        missing.extend(map(str, component.get("missing_paths", ())))
        task_owned_producer = component.get("task_owned_producer")
        source_revision = component.get("source_revision")
        runtime_materialized = bool(component.get("runtime_materialized"))
        pipeline_ready = bool(
            component.get("pipeline_ready_for_materialization")
        )
        runtime_ready = bool(
            component.get("runtime_ready", pipeline_ready)
        )
        schedulable = bool(component.get("schedulable")) and not missing
        if (
            component.get("episode_destination_required") is True
            and component.get("episode_destination") is None
        ):
            decision = "episode_destination_required"
        elif (
            isinstance(task_owned_producer, Mapping)
            and task_owned_producer.get("action_state_production_ready")
            is not True
        ):
            blockers.extend(map(str, task_owned_producer.get("blockers", ())))
            decision = "canary_required"
        elif schedulable:
            decision = "run"
        elif pipeline_ready and not missing:
            decision = "requires_runtime_materialization"
        else:
            decision = "blocked"
    elif task["implementation"]:
        runtime_ready = True
        if success_ready:
            decision = "owner_component_available"
            blockers.append(
                "owner action/state component is discoverable; central "
                "same-action replay and the registered current checker still "
                "decide acceptance"
            )
        else:
            decision = "owner_checker_required"
            blockers.append(str(success_contract["owner_requirement"]))
    else:
        decision = "owner_delivery_pending"
        blockers.append(
            "task is defined, but no owner action/state component has been "
            "delivered at the conventional path"
        )

    if missing:
        blockers.append("missing runtime paths: " + ", ".join(missing))
    blockers = list(dict.fromkeys(blockers))
    work_queue_decision = (
        "deferred"
        if entry["owner_scope_status"] == OWNER_DEFERRED_STATUS
        else decision
    )
    return {
        "schema": "real2sim.benchmark_runtime_preflight.v3",
        "benchmark_subtask": task_id,
        "defined": True,
        "decision": decision,
        "work_queue_decision": work_queue_decision,
        "runtime_ready": runtime_ready,
        "success_ready": success_ready,
        "success_contract": success_contract,
        "schedulable": schedulable,
        "pipeline_ready_for_materialization": pipeline_ready,
        "runtime_materialized": runtime_materialized,
        "in_current_owner_scope": entry["in_current_owner_scope"],
        "owner_scope_status": entry["owner_scope_status"],
        "blockers": blockers,
        "missing_paths": missing,
        "source_revision": source_revision,
        "task": task,
        "handler": task["runtime"]["handler"],
        "environment": resolve_house_environment({
            **task,
            **({"episode_destination": dict(episode_destination)}
               if episode_destination is not None else {}),
        }),
        "episode_destination_required": bool(
            component and component.get("episode_destination_required")
        ),
        "episode_destination": (
            component.get("episode_destination") if component else None
        ),
        "task_owned_producer": (
            component.get("task_owned_producer") if component else None
        ),
    }


def prepare_rollout(
    benchmark_subtask: str, *, rollout_index: int,
    episode_destination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a supported shared pipeline; owner code stays owner-run."""
    task_id = str(benchmark_subtask)
    decision = preflight(
        task_id, episode_destination=episode_destination,
    )
    if not decision["in_current_owner_scope"]:
        return {**decision, "decision": "deferred"}
    if decision["decision"] not in {
        "run",
        "requires_runtime_materialization",
    }:
        return decision
    if task_id in KITCHEN_COMPONENTS:
        entrypoint = str(decision["handler"]["initializer"])
        module_path, function_name = entrypoint.rsplit(":", 1)
        if not module_path.endswith(".py"):
            raise ValueError(f"invalid Kitchen initializer: {entrypoint}")
        module = importlib.import_module(module_path[:-3].replace("/", "."))
        return getattr(module, function_name)(
            task_id, rollout_index=int(rollout_index)
        )
    if task_id in RECOVERED_COMPONENTS:
        try:
            from .recovered_task_contract import prepare_rollout as prepare
        except ImportError:
            from recovered_task_contract import prepare_rollout as prepare
        return prepare(
            task_id,
            rollout_index=int(rollout_index),
            episode_destination=(
                dict(episode_destination)
                if episode_destination is not None else None
            ),
        )
    return decision


def summary() -> dict[str, Any]:
    decisions = {task: preflight(task) for task in CATALOG}
    counts: dict[str, int] = {}
    for row in decisions.values():
        decision = str(row["decision"])
        counts[decision] = counts.get(decision, 0) + 1
    deferred = sorted(
        (
            task
            for task, entry in CATALOG.items()
            if entry["owner_scope_status"] == OWNER_DEFERRED_STATUS
        ),
        key=lambda value: tuple(map(int, value.split("."))),
    )
    return {
        "schema": SCHEMA,
        "task_count": len(CATALOG),
        "runtime_decisions": dict(sorted(counts.items())),
        "deferred_tasks": deferred,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--rollout-index", type=int)
    args = parser.parse_args()
    if args.task is None:
        payload = summary()
    elif args.rollout_index is None:
        payload = preflight(args.task)
    else:
        payload = prepare_rollout(args.task, rollout_index=args.rollout_index)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

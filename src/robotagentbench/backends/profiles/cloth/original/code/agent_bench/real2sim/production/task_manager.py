#!/usr/bin/env python3
"""Inspect benchmark tasks, owner implementations, and trajectories.

The benchmark catalog is the only task identity source.  This module does not
maintain another registry: it reads ``benchmark_eval_registry.CATALOG`` and
discovers task-owned implementation files by their benchmark ID.  Discovery
does not register, freeze, or approve the contents of an owner's script.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import benchmark_eval_registry
    from . import task_data_compatibility
    from . import task_success
    from . import trajectory_trim
    from .external_kitchen_producers import for_task as external_producers_for_task
    from .house_episode import compose_episode, materialize_episode
except ImportError:
    import benchmark_eval_registry
    import task_data_compatibility
    import task_success
    import trajectory_trim
    from external_kitchen_producers import for_task as external_producers_for_task
    from house_episode import compose_episode, materialize_episode


REPO = Path(__file__).resolve().parents[2]
PRODUCTION = Path(__file__).resolve().parent
CATALOG = benchmark_eval_registry.CATALOG
SCHEMA = "real2sim.task_manager.v1"
TASK_DIRECTORY = re.compile(r"benchmark_(\d+)_(\d+)(?:_(.+))?")
TASK_ALIASES = re.compile(
    r"^(?:benchmark[-_])?(\d+)[._-](\d+)$",
    re.IGNORECASE,
)
SOURCE_SUFFIXES = frozenset({".py", ".sh", ".json", ".md", ".yaml", ".yml"})
CHANGE_DISPOSITIONS = {
    "metadata": {
        "physics_replay": False,
        "action": "reuse",
        "state": "reuse",
        "visual": "reuse",
        "meaning": "task identity, provenance, documentation, or source layout only",
    },
    "visual": {
        "physics_replay": False,
        "action": "reuse",
        "state": "reuse",
        "visual": "rerender",
        "meaning": "appearance, lighting, camera, or renderer output changed",
    },
    "physics": {
        "physics_replay": True,
        "action": "replay_same_action",
        "state": "replace_with_replay_state_on_pass",
        "visual": "rerender_if_state_changed",
        "meaning": "scene physics, robot, controller, or collision behavior may have changed",
    },
    "definition": {
        "physics_replay": False,
        "action": "regenerate",
        "state": "regenerate",
        "visual": "regenerate",
        "meaning": "instruction semantics, success condition, or action/state schema changed",
    },
}


def task_sort_key(task_id: str) -> tuple[int, int]:
    family, subtask = normalize_task_id(task_id).split(".")
    return int(family), int(subtask)


def normalize_task_id(value: Any) -> str:
    """Normalize 3.1, 3_1, benchmark_3_1, and benchmark-3.1 to 3.1."""
    text = str(value).strip()
    match = TASK_ALIASES.fullmatch(text)
    if not match:
        raise ValueError(f"invalid benchmark task ID: {value!r}")
    return f"{int(match.group(1))}.{int(match.group(2))}"


def implementation_paths(
    task_id: str,
    *,
    source_root: Path = REPO,
) -> dict[str, Any]:
    """Discover conventional components without creating another registry."""
    task_id = normalize_task_id(task_id)
    family, subtask = task_id.split(".")
    production = source_root / "real2sim" / "production"
    task_directory = production / "task_randomization" / (
        f"benchmark_{family}_{subtask}"
    )
    names = {
        "randomization": task_directory / "inject_robot_initial_state.py",
        "producer": task_directory / "produce_cores.py",
        "policy": task_directory / "policy.json",
        "canary": task_directory / "canary.json",
        "canary_artifact": task_directory / "canary_artifact.json",
        "visual_adapter": production / (
            f"benchmark_{family}_{subtask}_task_owned_visual_adapter.py"
        ),
    }
    found = {
        role: str(path.relative_to(source_root))
        for role, path in names.items()
        if path.is_file()
    }
    declared = dict(
        CATALOG.get(task_id, {}).get("task", {}).get("implementation", {})
    )
    visual = declared.get("visual")
    if isinstance(visual, Mapping):
        adapter = source_root / str(visual.get("adapter_path", ""))
        renderer = source_root / str(visual.get("state_renderer_path", ""))
        if adapter.is_file() and renderer.is_file():
            found["visual_adapter"] = str(adapter.relative_to(source_root))
            found["state_renderer"] = str(renderer.relative_to(source_root))
    return {
        "task_directory": (
            str(task_directory.relative_to(source_root))
            if task_directory.is_dir() else None
        ),
        "components": found,
        "component_count": len(found),
        "catalog_configuration": declared,
        "external_producers": external_producers_for_task(task_id),
    }


def describe_task(task_id: str, *, source_root: Path = REPO) -> dict[str, Any]:
    task_id = normalize_task_id(task_id)
    entry = benchmark_eval_registry.resolve(task_id)
    runtime = benchmark_eval_registry.preflight(task_id)
    return {
        "schema": SCHEMA,
        "task": entry["task"],
        "status": {
            "defined_in_sheet": entry["defined_in_sheet"],
            "in_current_owner_scope": entry["in_current_owner_scope"],
            "owner_scope_status": entry["owner_scope_status"],
            "runtime_decision": runtime["decision"],
            "work_queue_decision": runtime["work_queue_decision"],
            "success_registration": entry["task"]["runtime"]["success"][
                "status"
            ],
        },
        "implementation": implementation_paths(task_id, source_root=source_root),
        "environment": entry["task"]["runtime"]["environment"],
        "management_boundary": {
            "catalog": "Google Sheet task identity and owner",
            "owner_code": "owner-managed; discovered by path convention",
            "acceptance": "current task semantics plus runtime/checker evidence",
        },
        "real_data": entry.get("real_data", {}),
        "runtime_preflight": runtime,
    }


def list_tasks(*, family: int | None = None, source_root: Path = REPO) -> dict[str, Any]:
    rows = []
    for task_id in sorted(CATALOG, key=task_sort_key):
        entry = CATALOG[task_id]
        if family is not None and entry["task"]["family"] != family:
            continue
        implementation = implementation_paths(task_id, source_root=source_root)
        runtime = benchmark_eval_registry.preflight(task_id)
        rows.append({
            "id": task_id,
            "name": entry["task"]["name"],
            "action": entry["task"]["action"],
            "target_or_destination": entry["task"]["target_or_destination"],
            "owner": entry["task"].get("owner", ""),
            "runtime_decision": runtime["decision"],
            "work_queue_decision": runtime["work_queue_decision"],
            "success_registration": entry["task"]["runtime"]["success"][
                "status"
            ],
            "success_checker": entry["task"]["runtime"]["success"].get(
                "checker"
            ),
            "implementation_components": implementation["component_count"],
            "external_producer_links": len(
                implementation["external_producers"]),
        })
    return {
        "schema": SCHEMA,
        "task_count": len(rows),
        "family": family,
        "tasks": rows,
    }


def change_impact(kind: str) -> dict[str, Any]:
    if kind not in CHANGE_DISPOSITIONS:
        raise ValueError(f"unknown change kind {kind!r}")
    return {
        "schema": SCHEMA,
        "change_kind": kind,
        **CHANGE_DISPOSITIONS[kind],
    }


def _task_from_directory(path: Path) -> tuple[str, str]:
    match = TASK_DIRECTORY.fullmatch(path.name)
    if not match:
        raise ValueError(f"not a benchmark task directory: {path}")
    task_id = f"{int(match.group(1))}.{int(match.group(2))}"
    return task_id, match.group(3) or ""


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _episode_metadata(episode: Path) -> dict[str, Any]:
    identities: set[str] = set()
    initial_state: dict[str, Any] = {}
    perturbation: dict[str, bool] = {}
    for name in ("result.json", "validation.json", "score.json", "inject_meta.json"):
        value = _json_object(episode / name)
        if value is None:
            continue
        for key, child in _walk(value):
            lowered = key.lower()
            if lowered in {"task_key", "benchmark_task", "benchmark_subtask"}:
                try:
                    identities.add(normalize_task_id(child))
                except ValueError:
                    pass
            if lowered in {
                "applied_arm_initial_state",
                "applied_robot_arm_initial_state",
            }:
                initial_state.setdefault("arm", child)
            if lowered in {
                "applied_base_initial_state",
                "applied_robot_base_initial_state",
            }:
                initial_state.setdefault("base", child)
            if lowered in {"arm_delta", "robot_arm_initial_state_delta"}:
                perturbation.setdefault("arm", _contains_nonzero(child))
            if lowered in {
                "base_delta_xy_m",
                "robot_base_initial_state_delta",
            }:
                perturbation.setdefault("base", _contains_nonzero(child))
    return {
        "identities": identities,
        "initial_states": (
            [[initial_state.get("arm"), initial_state.get("base")]]
            if initial_state else []
        ),
        "nonzero_arm_perturbation": perturbation.get("arm"),
        "nonzero_base_perturbation": perturbation.get("base"),
    }


def _contains_nonzero(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_nonzero(item) for item in value)
    try:
        return abs(float(value)) > 0.0
    except (TypeError, ValueError):
        return False


def _distinct_values(values: Sequence[Any]) -> int:
    representatives: list[Any] = []
    for value in values:
        if not any(value == existing for existing in representatives):
            representatives.append(value)
    return len(representatives)


def _npz_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import numpy as np

    if set(left) != set(right):
        return False
    return all(np.array_equal(left[key], right[key]) for key in left)


def _distinct_npz(paths: Sequence[Path]) -> int:
    """Count exact array-distinct files by direct array comparison."""
    import numpy as np

    representatives: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            value = {key: archive[key] for key in archive.files}
        if not any(_npz_equal(value, existing) for existing in representatives):
            representatives.append(value)
    return len(representatives)


def _disposition(*, kind: str, relabel: bool) -> str:
    if kind == "metadata":
        return "reuse_after_metadata_relabel" if relabel else "reuse"
    if kind == "visual":
        return (
            "reuse_action_state_after_metadata_relabel_and_rerender"
            if relabel else "reuse_action_state_and_rerender"
        )
    if kind == "physics":
        return (
            "metadata_relabel_then_same_action_physics_replay"
            if relabel else "same_action_physics_replay"
        )
    return "regenerate_under_new_task_definition"


def audit_trajectory_root(
    root: Path,
    *,
    expected_count: int = 50,
    task_map: Mapping[str, str] | None = None,
    change_kind: str = "metadata",
    compare_actions: bool = False,
) -> dict[str, Any]:
    """Audit action/state compatibility using direct array comparisons."""
    root = root.resolve()
    task_map = {
        normalize_task_id(source): normalize_task_id(target)
        for source, target in (task_map or {}).items()
    }
    impact = change_impact(change_kind)
    rows = []
    skipped_noncanonical_directories = []
    for task_directory in sorted(root.glob("benchmark_*"), key=lambda path: path.name):
        if not task_directory.is_dir() or not TASK_DIRECTORY.fullmatch(task_directory.name):
            continue
        source_task, variant = _task_from_directory(task_directory)
        if variant:
            skipped_noncanonical_directories.append(task_directory.name)
            continue
        target_task = task_map.get(source_task, source_task)
        trajectory_root = task_directory / "trajectories"
        episodes = sorted(path for path in trajectory_root.glob("episode_*") if path.is_dir())
        complete = []
        strict_evidence = 0
        task_success_counts: dict[str, int] = {
            "passed": 0, "failed": 0, "unverified": 0,
        }
        task_success_reasons: dict[str, int] = {}
        triple_video = 0
        identities: set[str] = set()
        initial_states: list[Any] = []
        nonzero_arm_perturbation = 0
        nonzero_base_perturbation = 0
        arm_perturbation_evidence = 0
        base_perturbation_evidence = 0
        for episode in episodes:
            action = episode / "action.npz"
            state = episode / "full_state.npz"
            if action.is_file() and state.is_file():
                complete.append(episode)
            try:
                current_success = task_success.evaluate_task_success(
                    source_task, episode
                )
            except (KeyError, TypeError, ValueError) as exc:
                current_success = {
                    "status": "unverified",
                    "reasons": [f"current task success evaluation failed: {exc}"],
                }
            current_status = str(current_success.get("status", "unverified"))
            if current_status not in task_success_counts:
                current_status = "unverified"
            task_success_counts[current_status] += 1
            strict_evidence += int(current_status == "passed")
            for reason in current_success.get("reasons", ()):
                text = str(reason)
                task_success_reasons[text] = task_success_reasons.get(text, 0) + 1
            videos = episode / "videos"
            triple_video += int(all(
                (videos / name).is_file()
                for name in ("head.mp4", "hand_left.mp4", "hand_right.mp4")
            ))
            metadata = _episode_metadata(episode)
            identities.update(metadata["identities"])
            initial_states.extend(metadata["initial_states"])
            if metadata["nonzero_arm_perturbation"] is not None:
                arm_perturbation_evidence += 1
                nonzero_arm_perturbation += int(
                    metadata["nonzero_arm_perturbation"]
                )
            if metadata["nonzero_base_perturbation"] is not None:
                base_perturbation_evidence += 1
                nonzero_base_perturbation += int(
                    metadata["nonzero_base_perturbation"]
                )

        identity_conflicts = sorted(identities - {source_task}, key=task_sort_key)
        target_defined = target_task in CATALOG
        relabel = target_task != source_task
        distinct_initial_states = (
            _distinct_values(initial_states) if initial_states else None
        )
        action_paths = [episode / "action.npz" for episode in complete]
        distinct_actions = (
            _distinct_npz(action_paths) if compare_actions else None
        )
        design_issues = []
        if len(complete) < expected_count:
            design_issues.append("insufficient_action_state_cores")
        if strict_evidence < expected_count:
            design_issues.append("insufficient_current_checker_evidence")
        if len(initial_states) < expected_count:
            design_issues.append("insufficient_robot_initial_state_evidence")
        elif distinct_initial_states != len(initial_states):
            design_issues.append("duplicate_robot_initial_states")
        if arm_perturbation_evidence < expected_count:
            design_issues.append("insufficient_arm_perturbation_evidence")
        elif nonzero_arm_perturbation < arm_perturbation_evidence:
            design_issues.append("zero_arm_perturbation_present")
        if base_perturbation_evidence < expected_count:
            design_issues.append("insufficient_base_perturbation_evidence")
        elif nonzero_base_perturbation < base_perturbation_evidence:
            design_issues.append("zero_base_perturbation_present")
        if compare_actions and distinct_actions != len(action_paths):
            design_issues.append("duplicate_actions")
        if not target_defined:
            status = "no_current_task"
            disposition = "hold_until_task_definition_exists"
        elif identity_conflicts:
            status = "identity_conflict"
            disposition = "review_identity_before_reuse"
        elif len(complete) < expected_count:
            status = "incomplete"
            disposition = _disposition(kind=change_kind, relabel=relabel)
        elif strict_evidence < expected_count:
            status = (
                "needs_metadata_relabel_and_validation"
                if relabel else "needs_current_validation"
            )
            disposition = (
                _disposition(kind=change_kind, relabel=relabel)
                + "_then_run_current_checker"
            )
        elif relabel:
            status = "needs_metadata_relabel"
            disposition = _disposition(kind=change_kind, relabel=True)
        else:
            status = "complete"
            disposition = _disposition(kind=change_kind, relabel=False)

        row = {
            "source_task": source_task,
            "target_task": target_task,
            "directory": task_directory.name,
            "variant": variant,
            "status": status,
            "trajectory_disposition": disposition,
            "target_defined": target_defined,
            "episode_count": len(episodes),
            "complete_action_state_count": len(complete),
            "strict_evidence_count": strict_evidence,
            "task_success_counts": task_success_counts,
            "task_success_reason_counts": dict(sorted(
                task_success_reasons.items()
            )),
            "triple_video_count": triple_video,
            "visual_evidence_complete": triple_video >= expected_count,
            "expected_count": expected_count,
            "missing_episode_count": max(0, expected_count - len(complete)),
            "metadata_task_ids": sorted(identities, key=task_sort_key),
            "initial_state_evidence_count": len(initial_states),
            "distinct_initial_state_count": distinct_initial_states,
            "arm_perturbation_evidence_count": arm_perturbation_evidence,
            "nonzero_arm_perturbation_count": nonzero_arm_perturbation,
            "base_perturbation_evidence_count": base_perturbation_evidence,
            "nonzero_base_perturbation_count": nonzero_base_perturbation,
            "physics_core_design_issues": design_issues,
            "physics_core_conforms": not design_issues,
            "design_conformance": (
                "pass"
                if not design_issues and compare_actions
                else ("fail" if design_issues else "action_uniqueness_not_checked")
            ),
            "trajectory_ready_after_listed_disposition": bool(
                target_defined
                and not identity_conflicts
                and len(complete) >= expected_count
                and strict_evidence >= expected_count
                and not design_issues
                and change_kind not in {"physics", "definition"}
            ),
        }
        if compare_actions:
            row["directly_compared_action_count"] = len(action_paths)
            row["distinct_action_count"] = distinct_actions
        rows.append(row)

    return {
        "schema": "real2sim.trajectory_compatibility_audit.v1",
        "root": str(root),
        "change": impact,
        "task_map": task_map,
        "skipped_noncanonical_directories": skipped_noncanonical_directories,
        "summary": {
            "task_directories": len(rows),
            "complete": sum(row["status"] == "complete" for row in rows),
            "needs_metadata_relabel": sum(
                row["status"] == "needs_metadata_relabel" for row in rows
            ),
            "needs_current_validation": sum(
                row["status"] == "needs_current_validation" for row in rows
            ),
            "needs_metadata_relabel_and_validation": sum(
                row["status"] == "needs_metadata_relabel_and_validation"
                for row in rows
            ),
            "incomplete": sum(row["status"] == "incomplete" for row in rows),
            "identity_conflict": sum(
                row["status"] == "identity_conflict" for row in rows
            ),
            "no_current_task": sum(
                row["status"] == "no_current_task" for row in rows
            ),
            "physics_core_conforms": sum(
                row["physics_core_conforms"] for row in rows
            ),
            "visual_evidence_complete": sum(
                row["visual_evidence_complete"] for row in rows
            ),
        },
        "tasks": rows,
    }


def _source_issues(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ["unreadable"]
    issues = []
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError:
            issues.append("syntax_error")
    elif path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError:
            issues.append("invalid_json")
    if re.search(r"/(?:home|mnt|data)/[^\s'\"]+", text):
        issues.append("absolute_machine_path")
    if re.search(r"small[_ -]?task", text, re.IGNORECASE):
        issues.append("legacy_real_data_task_identity")
    if re.search(
        r"task_(?:initial_state_randomization|owned_visual_adapter)_registry",
        text,
    ):
        issues.append("separate_extension_registry")
    return sorted(set(issues))


def audit_handoff(
    root: Path,
    *,
    task_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Statically assess a pulled colleague handoff; never import its code."""
    root = root.resolve()
    task_map = {
        normalize_task_id(source): normalize_task_id(target)
        for source, target in (task_map or {}).items()
    }
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
    ]
    issue_counts: dict[str, int] = {}
    task_files: dict[str, list[dict[str, Any]]] = {}
    unassigned = 0
    for path in files:
        relative = path.relative_to(root)
        issues = _source_issues(path)
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        matches = list(re.finditer(r"benchmark_(\d+)_(\d+)", str(relative)))
        if not matches:
            unassigned += 1
            continue
        match = matches[-1]
        task_id = f"{int(match.group(1))}.{int(match.group(2))}"
        task_files.setdefault(task_id, []).append({
            "path": str(relative),
            "issues": issues,
        })

    rows = []
    for source_task in sorted(task_files, key=task_sort_key):
        target_task = task_map.get(source_task, source_task)
        components = task_files[source_task]
        issues = sorted({issue for row in components for issue in row["issues"]})
        if target_task not in CATALOG:
            disposition = "no_current_task"
        elif target_task != source_task:
            disposition = "relabel_and_adapt_before_merge"
        elif issues:
            disposition = "adapt_before_merge"
        else:
            disposition = "static_interface_compatible"
        rows.append({
            "source_task": source_task,
            "target_task": target_task,
            "target_defined": target_task in CATALOG,
            "file_count": len(components),
            "issues": issues,
            "disposition": disposition,
            "files": components,
        })
    return {
        "schema": "real2sim.colleague_handoff_audit.v1",
        "root": str(root),
        "file_count": len(files),
        "task_file_count": sum(len(rows) for rows in task_files.values()),
        "unassigned_file_count": unassigned,
        "issue_counts": dict(sorted(issue_counts.items())),
        "summary": {
            disposition: sum(row["disposition"] == disposition for row in rows)
            for disposition in (
                "static_interface_compatible",
                "adapt_before_merge",
                "relabel_and_adapt_before_merge",
                "no_current_task",
            )
        },
        "tasks": rows,
    }


def _task_map(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator:
            raise ValueError(f"task mapping must be SOURCE=TARGET, got {value!r}")
        result[normalize_task_id(source)] = normalize_task_id(target)
    return result


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def _json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--family", type=int)
    list_parser.add_argument("--source-root", type=Path, default=REPO)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("task")
    show_parser.add_argument("--source-root", type=Path, default=REPO)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("task")

    success_coverage_parser = subparsers.add_parser("success-coverage")
    success_coverage_parser.add_argument("--output", type=Path)

    check_success_parser = subparsers.add_parser("check-success")
    check_success_parser.add_argument("task")
    check_success_parser.add_argument("episode", type=Path)
    check_success_parser.add_argument("--write-result", action="store_true")
    check_success_parser.add_argument("--output", type=Path)

    rollout_stop_parser = subparsers.add_parser("rollout-stop")
    rollout_stop_parser.add_argument("task")
    rollout_stop_parser.add_argument("episode", type=Path)
    rollout_stop_parser.add_argument("--output", type=Path)

    audit_success_parser = subparsers.add_parser("audit-success")
    audit_success_parser.add_argument("root", type=Path)
    audit_success_parser.add_argument("--output", type=Path)

    manifest_success_parser = subparsers.add_parser("audit-success-manifest")
    manifest_success_parser.add_argument("manifest", type=Path)
    manifest_success_parser.add_argument("--workers", type=int, default=4)
    manifest_success_parser.add_argument("--output", type=Path)

    check_compatibility_parser = subparsers.add_parser("check-compatibility")
    check_compatibility_parser.add_argument("task")
    check_compatibility_parser.add_argument("episode", type=Path)
    check_compatibility_parser.add_argument("--output", type=Path)

    audit_compatibility_parser = subparsers.add_parser(
        "audit-compatibility"
    )
    audit_compatibility_parser.add_argument("root", type=Path)
    audit_compatibility_parser.add_argument("--output", type=Path)

    audit_workspace_data_parser = subparsers.add_parser(
        "audit-workspace-data"
    )
    audit_workspace_data_parser.add_argument("roots", nargs="+", type=Path)
    audit_workspace_data_parser.add_argument("--output", type=Path)

    audit_remote_data_parser = subparsers.add_parser("audit-remote-data")
    audit_remote_data_parser.add_argument("host")
    audit_remote_data_parser.add_argument("root")
    audit_remote_data_parser.add_argument("--source-checkout")
    audit_remote_data_parser.add_argument("--output", type=Path)

    trim_plan_parser = subparsers.add_parser("trim-plan")
    trim_plan_parser.add_argument("task")
    trim_plan_parser.add_argument("episode", type=Path)
    trim_plan_parser.add_argument("--output", type=Path)

    trim_apply_parser = subparsers.add_parser("trim-apply")
    trim_apply_parser.add_argument("task")
    trim_apply_parser.add_argument("episode", type=Path)
    trim_apply_parser.add_argument("destination", type=Path)
    trim_apply_parser.add_argument("--output", type=Path)

    impact_parser = subparsers.add_parser("impact")
    impact_parser.add_argument("kind", choices=tuple(CHANGE_DISPOSITIONS))

    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("tasks", nargs="+")
    compose_parser.add_argument("--contributions", type=Path)
    compose_parser.add_argument(
        "--destinations", type=Path,
        help="JSON task-id to episode-selected destination mapping",
    )
    compose_parser.add_argument("--layout", default="canonical")
    compose_parser.add_argument("--output", type=Path)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("plan", type=Path)
    materialize_parser.add_argument("--session", required=True)
    materialize_parser.add_argument("--recover-stale", action="store_true")

    trajectory_parser = subparsers.add_parser("audit-trajectories")
    trajectory_parser.add_argument("root", type=Path)
    trajectory_parser.add_argument("--expected-count", type=int, default=50)
    trajectory_parser.add_argument("--map", action="append", default=[])
    trajectory_parser.add_argument(
        "--change-kind", choices=tuple(CHANGE_DISPOSITIONS), default="metadata"
    )
    trajectory_parser.add_argument("--compare-actions", action="store_true")
    trajectory_parser.add_argument("--output", type=Path)

    handoff_parser = subparsers.add_parser("audit-handoff")
    handoff_parser.add_argument("root", type=Path)
    handoff_parser.add_argument("--map", action="append", default=[])
    handoff_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "list":
        payload = list_tasks(family=args.family, source_root=args.source_root)
        output = None
    elif args.command == "show":
        payload = describe_task(args.task, source_root=args.source_root)
        output = None
    elif args.command == "check":
        payload = benchmark_eval_registry.preflight(normalize_task_id(args.task))
        output = None
    elif args.command == "success-coverage":
        payload = task_success.coverage_report()
        output = args.output
    elif args.command == "check-success":
        payload = task_success.evaluate_task_success(
            normalize_task_id(args.task),
            args.episode,
            write_result=args.write_result,
        )
        output = args.output
    elif args.command == "rollout-stop":
        payload = task_success.evaluate_rollout_stop(
            normalize_task_id(args.task), args.episode
        )
        output = args.output
    elif args.command == "audit-success":
        payload = task_success.audit_root(args.root)
        output = args.output
    elif args.command == "audit-success-manifest":
        payload = task_success.audit_manifest(args.manifest, workers=args.workers)
        output = args.output
    elif args.command == "check-compatibility":
        payload = task_data_compatibility.evaluate_episode_compatibility(
            normalize_task_id(args.task), args.episode
        )
        output = args.output
    elif args.command == "audit-compatibility":
        payload = task_data_compatibility.audit_root(args.root)
        output = args.output
    elif args.command == "audit-workspace-data":
        payload = task_data_compatibility.audit_workspace(args.roots)
        output = args.output
    elif args.command == "audit-remote-data":
        payload = task_data_compatibility.audit_remote_task_owned_root(
            args.host,
            args.root,
            source_checkout=args.source_checkout,
        )
        output = args.output
    elif args.command == "trim-plan":
        payload = trajectory_trim.plan_trim(
            normalize_task_id(args.task), args.episode
        )
        output = args.output
    elif args.command == "trim-apply":
        payload = trajectory_trim.apply_trim(
            normalize_task_id(args.task), args.episode, args.destination
        )
        output = args.output
    elif args.command == "impact":
        payload = change_impact(args.kind)
        output = None
    elif args.command == "compose":
        payload = compose_episode(
            [normalize_task_id(task) for task in args.tasks],
            contributions=_json_mapping(args.contributions),
            episode_destinations=_json_mapping(args.destinations),
            layout=args.layout,
        )
        output = args.output
    elif args.command == "materialize":
        payload = materialize_episode(
            _json_mapping(args.plan),
            session_name=args.session,
            recover_stale=args.recover_stale,
        )
        output = None
    elif args.command == "audit-trajectories":
        payload = audit_trajectory_root(
            args.root,
            expected_count=args.expected_count,
            task_map=_task_map(args.map),
            change_kind=args.change_kind,
            compare_actions=args.compare_actions,
        )
        output = args.output
    else:
        payload = audit_handoff(args.root, task_map=_task_map(args.map))
        output = args.output
    _emit(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One fail-closed task-success entrypoint for all Sheet benchmark tasks.

Task semantics remain task-specific and owner-managed.  This module resolves
the checker attached to the sole Sheet-backed catalog entry, validates only
the minimum identity/action alignment needed for semantic scoring, invokes
that checker, and normalizes the result to passed/failed/unverified.  Default
environment and rollout-start observations are reported as diagnostics rather
than extra success gates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "real2sim.task_success_result.v1"
COVERAGE_SCHEMA = "real2sim.task_success_coverage.v1"
AUDIT_SCHEMA = "real2sim.task_success_audit.v1"
ROLLOUT_SCHEMA = "real2sim.rollout_stop_decision.v1"
VALID_STATUSES = frozenset({"passed", "failed", "unverified"})
DEFAULT_ENVIRONMENT = "canonical_house_default"
START_SETTLE_SECONDS = 0.20
MAX_INITIAL_JOINT_SPEED = 0.25
MAX_INITIAL_TARGET_LINEAR_SPEED = 0.10
INITIAL_POSITION_TOLERANCE = 0.01
INITIAL_BASE_POSITION_TOLERANCE_M = 0.02
INITIAL_BASE_YAW_TOLERANCE_DEG = 2.0


class EvidenceUnavailable(ValueError):
    """The current checker cannot make a semantic pass/fail decision."""


def _catalog():
    try:
        from .benchmark_eval_registry import CATALOG
    except ImportError:
        from benchmark_eval_registry import CATALOG
    return CATALOG


def _task_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("benchmark_"):
        text = text[len("benchmark_"):].replace("_", ".")
    family, separator, subtask = text.partition(".")
    if not separator or not family.isdigit() or not subtask.isdigit():
        raise ValueError(f"invalid benchmark task ID: {value!r}")
    return f"{int(family)}.{int(subtask)}"


def _task_key(task_id: str) -> str:
    return "benchmark_" + _task_id(task_id).replace(".", "_")


def _json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise EvidenceUnavailable(f"missing {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceUnavailable(f"unreadable {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceUnavailable(f"{path.name} must contain one object")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise EvidenceUnavailable(f"missing {path.name}")
    try:
        with np.load(path, allow_pickle=False) as source:
            return {name: np.asarray(source[name]) for name in source.files}
    except (OSError, ValueError) as exc:
        raise EvidenceUnavailable(f"unreadable {path.name}: {exc}") from exc


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise EvidenceUnavailable("expected one scalar value")
    return array.item()


def _entrypoint(value: str):
    try:
        module_path, function_name = str(value).rsplit(":", 1)
    except ValueError as exc:
        raise EvidenceUnavailable(f"invalid checker entrypoint {value!r}") from exc
    if not module_path.endswith(".py"):
        raise EvidenceUnavailable(f"checker must be a repository Python path: {value}")
    module_name = module_path[:-3].replace("/", ".")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise EvidenceUnavailable(f"checker cannot be loaded: {value}: {exc}") from exc
    if not callable(function):
        raise EvidenceUnavailable(f"checker is not callable: {value}")
    return function


def success_contract(benchmark_subtask: str) -> dict[str, Any]:
    task_id = _task_id(benchmark_subtask)
    try:
        value = _catalog()[task_id]["task"]["runtime"]["success"]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark task {task_id}") from exc
    return dict(value)


def coverage_report() -> dict[str, Any]:
    catalog = _catalog()
    rows = []
    counts: Counter[str] = Counter()
    for task_id in sorted(
        catalog, key=lambda value: tuple(map(int, value.split(".")))
    ):
        task = catalog[task_id]["task"]
        contract = task["runtime"]["success"]
        status = str(contract["status"])
        counts[status] += 1
        rows.append({
            "benchmark_subtask": task_id,
            "task_key": task["key"],
            "owner": task.get("owner", ""),
            "status": status,
            "checker": contract.get("checker"),
            "evidence_schema": contract.get("evidence_schema"),
            "owner_requirement": contract.get("owner_requirement"),
        })
    owner_actions: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] != "owner_action_required":
            continue
        owner_actions.setdefault(str(row["owner"] or "unassigned"), []).append(
            str(row["benchmark_subtask"])
        )
    return {
        "schema": COVERAGE_SCHEMA,
        "task_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "all_tasks_have_success_interface": len(rows) == len(catalog),
        "all_tasks_have_registered_checker": (
            counts.get("registered", 0) == len(rows)
        ),
        "owner_action_groups": [
            {
                "owner": owner,
                "task_count": len(tasks),
                "benchmark_subtasks": tasks,
            }
            for owner, tasks in sorted(owner_actions.items())
        ],
        "tasks": rows,
    }


def _identity_gate(
    task_id: str,
    episode: Mapping[str, Any],
    result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_key = _task_key(task_id)
    claims: list[str] = []
    sources: list[Mapping[str, Any]] = [episode, result]
    config = result.get("config")
    if isinstance(config, Mapping):
        sources.append(config)
        request = config.get("task_owned_request")
        if isinstance(request, Mapping):
            sources.append(request)
    for source in sources:
        for field in ("benchmark_subtask", "benchmark_task_id"):
            if source.get(field) not in (None, ""):
                try:
                    claims.append(_task_id(str(source[field])))
                except ValueError:
                    claims.append(f"invalid:{source[field]}")
        for field in ("task_key", "benchmark_task_key"):
            if source.get(field) not in (None, ""):
                value = str(source[field])
                claims.append(task_id if value == expected_key else f"invalid:{value}")
    parameters = contract.get("parameters")
    expected_runtime_task_id = (
        str(parameters.get("runtime_task_id") or "").strip()
        if isinstance(parameters, Mapping) else ""
    )
    expected_runtime_task_ids = set(
        map(str, parameters.get("runtime_task_ids", ()))
    ) if isinstance(parameters, Mapping) else set()
    if expected_runtime_task_id:
        expected_runtime_task_ids.add(expected_runtime_task_id)
    runtime_claims: list[str] = []
    if expected_runtime_task_ids:
        # ``task_id`` is a runtime identity in the living-room episode schema.
        # Elsewhere only the unambiguous runtime/small-task field names count;
        # a generic result.task_id may instead mean the Sheet task.
        for field in ("task_id", "runtime_task_id", "small_task_id"):
            if episode.get(field) not in (None, ""):
                runtime_claims.append(str(episode[field]).strip())
        for source in sources[1:]:
            for field in ("runtime_task_id", "small_task_id"):
                if source.get(field) not in (None, ""):
                    runtime_claims.append(str(source[field]).strip())
    if not claims:
        return {
            "status": "unverified",
            "passed": None,
            "reasons": ["episode has no direct Sheet benchmark identity"],
            "claims": [],
            "expected_runtime_task_id": expected_runtime_task_id or None,
            "runtime_claims": runtime_claims,
        }
    mismatches = sorted({claim for claim in claims if claim != task_id})
    runtime_mismatches = sorted({
        claim for claim in runtime_claims
        if claim not in expected_runtime_task_ids
    })
    reasons = []
    if mismatches:
        reasons.append(
            "episode benchmark identity mismatch: " + ", ".join(mismatches))
    if runtime_mismatches:
        reasons.append(
            "episode runtime task identity mismatch: "
            + ", ".join(runtime_mismatches)
        )
    if mismatches or runtime_mismatches:
        status, passed = "failed", False
    else:
        status, passed = "passed", True
    return {
        "status": status,
        "passed": passed,
        "reasons": reasons,
        "claims": claims,
        "expected_runtime_task_id": expected_runtime_task_id or None,
        "runtime_claims": runtime_claims,
    }


def _environment_gate(
    episode: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    failures: list[str] = []
    declared_scopes = {
        str(source.get("environment_scope"))
        for source in (episode, result)
        if source.get("environment_scope") not in (None, "")
    }
    if not declared_scopes:
        reasons.append("episode does not declare the canonical default environment")
    elif declared_scopes != {DEFAULT_ENVIRONMENT}:
        reasons.append(
            "episode environment is not the canonical default: "
            + ", ".join(sorted(declared_scopes))
        )
    for source in (episode, result):
        if source.get("layout_overlay_applied") is True:
            reasons.append("layout overlay is not physical default-environment evidence")
        layout = source.get("house_layout_id", source.get("layout_id"))
        if layout not in (None, "", "canonical", "default"):
            reasons.append(f"non-default physical layout {layout!r} is not supported")
        if source.get("furniture_moved") is True:
            failures.append("furniture moved during the rollout")
        if source.get("object_pose_or_velocity_writes") is True:
            failures.append("rollout wrote object pose or velocity directly")
    if failures:
        status, passed = "failed", False
    elif reasons:
        status, passed = "unverified", None
    else:
        status, passed = "passed", True
    return {
        "status": status,
        "passed": passed,
        "environment_scope": DEFAULT_ENVIRONMENT,
        "reasons": reasons + failures,
    }


def _action_frames(action: Mapping[str, np.ndarray]) -> int:
    counts = {
        int(value.shape[0])
        for value in action.values()
        if np.asarray(value).ndim >= 1
    }
    if len(counts) != 1:
        raise EvidenceUnavailable("action channels do not share one frame count")
    frames = counts.pop()
    if frames < 2:
        raise EvidenceUnavailable("trajectory must contain at least two frames")
    return frames


def _action_state_integrity_gate(directory: Path) -> dict[str, Any]:
    """Check only the minimum alignment needed to interpret the action."""
    try:
        action = _npz(directory / "action.npz")
        state = _npz(directory / "full_state.npz")
        frames = _action_frames(action)
        checks: dict[str, bool] = {}
        if "frame_index" in state:
            frame_index = np.asarray(state["frame_index"])
            checks["action_state_frame_count_match"] = bool(
                frame_index.shape == (frames,)
                and np.array_equal(frame_index, np.arange(frames))
            )
        else:
            raise EvidenceUnavailable("full_state.npz has no frame_index")
        if "fps" not in action or "fps" not in state:
            raise EvidenceUnavailable(
                "action/full_state sampling frequency is missing"
            )
        action_fps = float(_scalar(action["fps"]))
        state_fps = float(_scalar(state["fps"]))
        checks["action_state_fps_match"] = bool(
            np.isfinite(action_fps)
            and action_fps > 0
            and np.isclose(action_fps, state_fps, atol=1.0e-6, rtol=0.0)
        )
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "status": "failed" if failed else "passed",
            "passed": not failed,
            "reasons": [f"failed action/state gate: {name}" for name in failed],
            "checks": checks,
        }
    except EvidenceUnavailable as exc:
        return {
            "status": "unverified",
            "passed": None,
            "reasons": [str(exc)],
            "checks": {},
        }


def _angle_error_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _yaw_degrees(wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(wxyz, dtype=float)
    return float(np.degrees(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )))


def _planned_base_xyyaw(
    result: Mapping[str, Any], state: Mapping[str, np.ndarray]
) -> np.ndarray:
    evidence = result.get("task_initial_state_randomization", {})
    if isinstance(evidence, Mapping):
        for key in (
            "applied_robot_base_initial_state",
            "applied_base_initial_state",
        ):
            planned = evidence.get(key)
            if planned is None:
                continue
            value = np.asarray(planned, dtype=float)
            if value.shape == (3,) and np.isfinite(value).all():
                # Task-owned initial-state evidence stores x/y/yaw in the
                # simulator's native metres/radians convention.  The explicit
                # full-state fallback below is the only degrees-valued field.
                value = value.copy()
                value[2] = np.degrees(value[2])
                return value
    if "robot_initial_base_xyyaw_deg" in state:
        value = np.asarray(state["robot_initial_base_xyyaw_deg"], dtype=float)
        if value.shape == (3,) and np.isfinite(value).all():
            return value
    raise EvidenceUnavailable(
        "planned initial robot base XY/yaw evidence is missing"
    )


def _initial_state_gate(
    directory: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        action = _npz(directory / "action.npz")
        state = _npz(directory / "full_state.npz")
        frames = _action_frames(action)
        required = {
            "fps",
            "frame_index",
            "current_robot_base_xy_m",
            "current_robot_orient_wxyz",
            "robot_initial_joint_position_rad_or_m",
            "robot_joint_position_rad_or_m",
            "robot_joint_velocity_radps_or_mps",
            "rigid_body_paths",
            "target_prim",
            "prop_velocity_mps_radps",
        }
        missing = sorted(required - set(state))
        if missing:
            raise EvidenceUnavailable(
                "initial-position evidence is missing: " + ", ".join(missing)
            )
        fps = float(_scalar(state["fps"]))
        if not np.isfinite(fps) or fps <= 0:
            raise EvidenceUnavailable("state fps is invalid")
        frame_index = np.asarray(state["frame_index"])
        if frame_index.shape != (frames,):
            raise EvidenceUnavailable("action/state frame counts disagree")
        base = np.asarray(state["current_robot_base_xy_m"], dtype=float)
        if base.shape != (2,) or not np.isfinite(base).all():
            raise EvidenceUnavailable("captured initial robot base XY is missing")
        orientation = np.asarray(
            state["current_robot_orient_wxyz"], dtype=float
        )
        if orientation.shape != (4,) or not np.isfinite(orientation).all():
            raise EvidenceUnavailable("captured initial robot base yaw is missing")
        orientation_norm = float(np.linalg.norm(orientation))
        if not np.isclose(orientation_norm, 1.0, atol=1.0e-3, rtol=0.0):
            raise EvidenceUnavailable("captured initial robot orientation is invalid")
        planned_base = _planned_base_xyyaw(result, state)
        base_position_error = float(np.linalg.norm(base - planned_base[:2]))
        base_yaw_error = _angle_error_degrees(
            _yaw_degrees(orientation), float(planned_base[2])
        )
        initial = np.asarray(
            state["robot_initial_joint_position_rad_or_m"], dtype=float
        )
        joint = np.asarray(state["robot_joint_position_rad_or_m"], dtype=float)
        velocity = np.asarray(
            state["robot_joint_velocity_radps_or_mps"], dtype=float
        )
        if (
            joint.ndim != 2
            or velocity.shape != joint.shape
            or joint.shape[0] != frames
            or initial.shape != joint.shape[1:]
            or not np.isfinite(initial).all()
            or not np.isfinite(joint).all()
            or not np.isfinite(velocity).all()
        ):
            raise EvidenceUnavailable("captured robot initial-state arrays are invalid")
        initial_error = float(np.max(np.abs(joint[0] - initial)))
        window = max(2, min(frames, int(np.ceil(START_SETTLE_SECONDS * fps))))
        initial_joint_speed = float(np.max(np.abs(velocity[:window])))

        paths = [str(value) for value in state["rigid_body_paths"].tolist()]
        target = str(_scalar(state["target_prim"]))
        if paths.count(target) != 1:
            raise EvidenceUnavailable("target is not one captured rigid body")
        prop_velocity = np.asarray(state["prop_velocity_mps_radps"], dtype=float)
        if prop_velocity.shape[:2] != (frames, len(paths)) or prop_velocity.shape[2] < 3:
            raise EvidenceUnavailable("captured target velocity array is invalid")
        target_index = paths.index(target)
        initial_target_speed = float(np.max(np.linalg.norm(
            prop_velocity[:window, target_index, :3], axis=1
        )))
        checks = {
            "action_state_frame_count_match": True,
            "initial_robot_base_position_reached": (
                base_position_error <= INITIAL_BASE_POSITION_TOLERANCE_M
            ),
            "initial_robot_base_yaw_reached": (
                base_yaw_error <= INITIAL_BASE_YAW_TOLERANCE_DEG
            ),
            "initial_joint_position_reached": (
                initial_error <= INITIAL_POSITION_TOLERANCE
            ),
            "initial_robot_motion_settled": (
                initial_joint_speed <= MAX_INITIAL_JOINT_SPEED
            ),
            "initial_target_motion_settled": (
                initial_target_speed <= MAX_INITIAL_TARGET_LINEAR_SPEED
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "status": "failed" if failed else "passed",
            "passed": not failed,
            "reasons": [f"failed rollout-start gate: {name}" for name in failed],
            "checks": checks,
            "metrics": {
                "settle_window_frames": window,
                "initial_base_position_error_m": base_position_error,
                "initial_base_yaw_error_deg": base_yaw_error,
                "initial_joint_position_error": initial_error,
                "initial_max_joint_speed": initial_joint_speed,
                "initial_max_target_linear_speed": initial_target_speed,
            },
            "thresholds": {
                "initial_joint_position_tolerance": INITIAL_POSITION_TOLERANCE,
                "initial_base_position_tolerance_m": (
                    INITIAL_BASE_POSITION_TOLERANCE_M
                ),
                "initial_base_yaw_tolerance_deg": (
                    INITIAL_BASE_YAW_TOLERANCE_DEG
                ),
                "maximum_initial_joint_speed": MAX_INITIAL_JOINT_SPEED,
                "maximum_initial_target_linear_speed": (
                    MAX_INITIAL_TARGET_LINEAR_SPEED
                ),
            },
        }
    except EvidenceUnavailable as exc:
        return {
            "status": "unverified",
            "passed": None,
            "reasons": [str(exc)],
            "checks": {},
            "metrics": {},
        }


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return _json(path, required=True)
    return _npz(path)


def _normalize_checker_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if "success" in value:
        passed = value.get("success")
    elif "ok" in value:
        passed = value.get("ok")
    else:
        passed = str(value.get("status", "")).lower() in {"pass", "passed"}
    if not isinstance(passed, (bool, np.bool_)):
        raise EvidenceUnavailable("checker returned no explicit boolean result")
    missing = list(value.get("missing_state_fields", ())) + list(
        value.get("missing_trace_fields", ())
    )
    if missing:
        raise EvidenceUnavailable(
            "current checker evidence is missing: " + ", ".join(map(str, missing))
        )
    all_gates = value.get("gates", value.get("acceptance", {}))
    gates = value.get("scored_gates", all_gates)
    if not isinstance(gates, Mapping):
        gates = {}
    if not isinstance(all_gates, Mapping):
        all_gates = gates
    reasons = value.get("reasons", value.get("failed_gates", ()))
    if isinstance(reasons, str):
        reasons = [reasons]
    # Preserve the actual failed condition, not only its umbrella gate name.
    reasons = list(reasons) + list(value.get("strict_place_reasons", ()))
    return {
        "status": "passed" if bool(passed) else "failed",
        "passed": bool(passed),
        "gates": {str(key): bool(item) for key, item in gates.items()},
        "diagnostic_gates": {
            str(key): bool(item)
            for key, item in all_gates.items()
            if key not in gates
        },
        "reasons": [str(item) for item in reasons],
        "margins": value.get("margins", value.get("metrics", {})),
        "raw_schema": value.get("schema"),
    }


def _family7_layout_transform(task_id, directory, result):
    """Resolve a replay's coordinate frame without moving its recorded state."""
    layout = result.get("house_layout_physics_canary")
    if layout is None:
        return None
    try:
        from . import house_layout, house_topology_visual, prepare_house_final_layouts
    except ImportError:
        import house_layout, house_topology_visual, prepare_house_final_layouts
    request = result.get("config", {}).get("house_layout", {})
    key = "benchmark_" + str(task_id).replace(".", "_")
    action = layout.get("action_application", {})
    scene = layout.get("scene_application", {})
    identity = layout.get("layout_id")
    if (layout.get("schema") != "real2sim.family7_house_layout_physics_canary.v1"
            or layout.get("live_simulator_rollout_completed") is not True
            or layout.get("live_collision_runtime_enabled") is not True
            or request.get("task_key") != key
            or request.get("task_frame") != "coffee_table"
            or layout.get("task_frame") != "coffee_table"
            or action.get("task_frame") != "coffee_table"
            or request.get("layout_id") != identity
            or action.get("layout_id") != identity
            or scene.get("topology_id") != identity
            or scene.get("physics_overlay_mode") is not True
            or scene.get("usd_author_save_reopen") != "passed"):
        raise EvidenceUnavailable("Family7 physical layout/action/scene binding disagrees")
    catalog = prepare_house_final_layouts.load_catalog()
    candidate, _ = house_topology_visual.candidate_for_id(identity, catalog)
    recipe = house_topology_visual.topology_layout(candidate, catalog["interaction_modules"])
    transform = house_layout.transform_for_frame(recipe, "coffee_table")
    expected = dict(pivot_xy_m=list(transform.pivot_xy_m),
                    translate_xy_m=list(transform.translate_xy_m), yaw_deg=transform.yaw_deg)
    for report in (layout.get("transform"), action.get("transform"),
                   result.get("scene_perturbation", {}).get("house_layout_transform")):
        if not isinstance(report, Mapping) or set(report) != set(expected):
            raise EvidenceUnavailable("Family7 physical layout transform is missing")
        for field, value in expected.items():
            observed = np.asarray(report[field], dtype=float)
            reference = np.asarray(value, dtype=float)
            if (observed.shape != reference.shape or not np.isfinite(observed).all()
                    or not np.allclose(observed, reference, rtol=0, atol=1e-9)):
                raise EvidenceUnavailable("Family7 recorded transform disagrees with current House recipe")
    # A layout receipt cannot substitute for physically captured state.
    state = _npz(directory / "full_state.npz")
    try:
        from ..teleop.physical_distractors import read_capture
    except ImportError:
        from physical_distractors import read_capture
    captured = read_capture(state)
    static = captured.get("static_collision_transforms", {})
    if not any(p.startswith("/World/HouseTopologyCandidate/Shell/Floors/") for p in static):
        raise EvidenceUnavailable("Family7 capture lacks the generated collision shell")
    source_path = result.get("config", {}).get("action_source_override")
    if not source_path or not Path(source_path).is_file():
        raise EvidenceUnavailable("Family7 physical layout requires the preserved source action")
    original = _npz(Path(source_path))
    mapped = house_layout.transform_action_arrays(original, recipe, frame="coffee_table")
    observed = _npz(directory / "action.npz")
    if (mapped.keys() != observed.keys() or any(
            not np.array_equal(mapped[k], observed[k]) for k in mapped)):
        raise EvidenceUnavailable("Family7 executed action is not the recorded recipe mapping of its source")
    return expected


def _run_checker(
    task_id: str,
    directory: Path,
    contract: Mapping[str, Any],
    episode: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("status") != "registered" or not contract.get("checker"):
        raise EvidenceUnavailable(str(
            contract.get("owner_requirement", "task checker is not registered")
        ))
    required = list(contract.get("required_artifacts", ()))
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise EvidenceUnavailable("missing checker evidence: " + ", ".join(missing))
    adapter = str(contract.get("adapter"))
    function = _entrypoint(str(contract["checker"]))
    parameters = dict(contract.get("parameters", {}))
    if adapter in {"livingroom_pick", "livingroom_place"}:
        runtime_task = parameters.get("runtime_task_id")
        if runtime_task is None:
            runtime_task = str(_scalar(_npz(directory / "full_state.npz")["task_id"]))
            if runtime_task not in parameters.get("runtime_task_ids", ()):
                return {
                    "status": "failed", "passed": False,
                    "gates": {"runtime_task_identity": False},
                    "reasons": ["recorded runtime is not a delivered implementation for this task"],
                    "margins": {},
                }
        raw = function(
            str(runtime_task),
            directory / "full_state.npz",
            directory / ("pick_contact_trace.npz" if adapter == "livingroom_pick" else "place_metrics.npz"),
            **({"semantic_only": True} if adapter == "livingroom_place" else {}),
        )
    elif adapter == "shelf_pick":
        raw = function(
            task_id,
            directory / "full_state.npz",
            directory / "pick_contact_trace.npz",
        )
    elif adapter == "family4_drawer_pick":
        raw = function(
            task_id, directory,
            expected_action_kind=parameters["expected_action_kind"],
        )
    elif adapter == "task6_place":
        try:
            from .task6_physical_layout import score_kwargs
        except ImportError:
            from task6_physical_layout import score_kwargs
        layout_kwargs = score_kwargs(
            task_id, directory / "full_state.npz",
            episode.get("queue_record", {}).get("task6_physical_layout"))
        raw = function(
            task_id,
            directory / "full_state.npz",
            directory / "place_metrics.npz",
            semantic_only=True,
            **layout_kwargs,
        )
    elif adapter == "family7_e2e":
        raw = function(
            str(parameters["mode"]),
            directory / "pick_contact_trace.npz",
            directory / "full_state.npz",
            directory / "action.npz",
            layout_transform=_family7_layout_transform(task_id, directory, result),
            semantic_only=True,
        )
    elif adapter == "task38_shelf":
        uid = result.get("source_episode_uid", episode.get("source_episode_uid"))
        if not uid:
            raise EvidenceUnavailable("task 3.29 has no source episode identity")
        raw = function(directory, str(uid))
    elif adapter == "tablecloth_clear":
        raw = function(task_id, directory)
    elif adapter == "tablecloth_spread":
        raw = function(directory)
    elif adapter == "metrics":
        alternatives = contract.get("required_artifact_alternatives", {})
        metrics_path = _first_existing(
            directory, list(alternatives.get("metrics", ()))
        )
        audit_path = _first_existing(
            directory, list(alternatives.get("randomization_audit", ()))
        )
        if metrics_path is None:
            raise EvidenceUnavailable("current task metrics are required")
        metrics = _load_mapping(metrics_path)
        audit = _load_mapping(audit_path) if audit_path is not None else None
        schema = contract.get("evidence_schema")
        if schema and str(_scalar(metrics.get("schema"))) != str(schema):
            raise EvidenceUnavailable("metrics schema does not match the catalog")
        if parameters.get("call_style") == "metrics_only":
            raw = function(
                metrics,
                randomization_audit=audit,
                semantic_only=True,
            )
        else:
            raw = function(
                task_id,
                metrics,
                randomization_audit=audit,
                semantic_only=True,
            )
    else:
        raise EvidenceUnavailable(f"unsupported checker adapter {adapter!r}")
    if not isinstance(raw, Mapping):
        raise EvidenceUnavailable("checker returned a non-object result")
    return _normalize_checker_result(raw)


def evaluate_task_success(
    benchmark_subtask: str,
    episode_directory: Path | str,
    *,
    write_result: bool = False,
) -> dict[str, Any]:
    """Evaluate one trajectory with task-semantic gates and quality diagnostics."""
    task_id = _task_id(benchmark_subtask)
    directory = Path(episode_directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"episode directory is missing: {directory}")
    contract = success_contract(task_id)
    try:
        episode = _json(directory / "episode.json")
        result = _json(directory / "result.json")
    except EvidenceUnavailable as exc:
        episode, result = {}, {}
        metadata_error = str(exc)
    else:
        metadata_error = ""
    identity = _identity_gate(task_id, episode, result, contract)
    environment = _environment_gate(episode, result)
    initial = _initial_state_gate(directory, result)
    action_state = _action_state_integrity_gate(directory)
    if metadata_error:
        identity = {
            "status": "unverified",
            "passed": None,
            "reasons": [metadata_error],
            "claims": [],
        }
    try:
        checker = _run_checker(task_id, directory, contract, episode, result)
    except (EvidenceUnavailable, OSError, ValueError, KeyError, TypeError) as exc:
        checker = {
            "status": "unverified",
            "passed": None,
            "gates": {},
            "reasons": [f"current checker could not decide: {exc}"],
            "margins": {},
        }
    components = {
        "task_identity": identity,
        "action_state_integrity": action_state,
        "default_environment": environment,
        "initial_state_reached_and_settled": initial,
        "task_outcome": checker,
    }
    required_components = {
        name: components[name]
        for name in (
            "task_identity", "action_state_integrity", "task_outcome"
        )
    }
    statuses = {value["status"] for value in required_components.values()}
    if "failed" in statuses:
        status, success = "failed", False
    elif "unverified" in statuses:
        status, success = "unverified", None
    else:
        status, success = "passed", True
    reasons = [
        f"{name}: {reason}"
        for name, value in required_components.items()
        for reason in value.get("reasons", ())
    ]
    diagnostics = [
        f"{name}: {reason}"
        for name in (
            "default_environment", "initial_state_reached_and_settled"
        )
        for reason in components[name].get("reasons", ())
    ]
    normalized = {
        "schema": SCHEMA,
        "benchmark_subtask": task_id,
        "task_key": _task_key(task_id),
        "episode_directory": str(directory),
        "status": status,
        "success": success,
        "environment_scope": DEFAULT_ENVIRONMENT,
        "checker": contract.get("checker"),
        "checker_registration_status": contract.get("status"),
        "evidence_schema": contract.get("evidence_schema"),
        "components": components,
        "required_components": list(required_components),
        "diagnostic_components": [
            "default_environment", "initial_state_reached_and_settled"
        ],
        "reasons": reasons,
        "diagnostics": diagnostics,
    }
    if write_result:
        output = directory / "task_success.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".task_success.", suffix=".json", dir=directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return normalized


def evaluate_rollout_stop(
    benchmark_subtask: str,
    episode_directory: Path | str,
) -> dict[str, Any]:
    """Use the same current checker to stop or abort a growing rollout.

    Owner checkers must make ``passed`` mean that their terminal condition has
    held through the end of the currently flushed prefix.  A normal semantic
    failure may keep running; missing evidence or a broken identity/action gate
    cannot become accepted merely by waiting longer and therefore aborts.
    """
    checked = evaluate_task_success(benchmark_subtask, episode_directory)
    components = checked.get("components", {})
    outcome = components.get("task_outcome", {})
    prerequisites = {
        name: components[name]
        for name in checked.get("required_components", ())
        if name != "task_outcome" and name in components
    }
    if checked.get("status") == "passed":
        decision = "stop_success"
        stop, abort = True, False
    elif (
        outcome.get("status") == "failed"
        and all(value.get("status") == "passed"
                for value in prerequisites.values())
    ):
        decision = "continue_task_not_finished"
        stop, abort = False, False
    else:
        decision = "abort_invalid_or_unscorable"
        stop, abort = False, True
    return {
        "schema": ROLLOUT_SCHEMA,
        "benchmark_subtask": checked["benchmark_subtask"],
        "decision": decision,
        "stop_rollout": stop,
        "abort_rollout": abort,
        "task_success": checked,
    }


def audit_root(root: Path | str) -> dict[str, Any]:
    directory = Path(root).resolve()
    catalog = _catalog()
    rows = []
    counts: Counter[str] = Counter()
    task_rows = []
    for task_id in sorted(
        catalog, key=lambda value: tuple(map(int, value.split(".")))
    ):
        catalog_task = catalog[task_id]["task"]
        task_directory = directory / _task_key(task_id)
        task_counts: Counter[str] = Counter()
        task_reasons: Counter[str] = Counter()
        task_diagnostics: Counter[str] = Counter()
        for episode in sorted((task_directory / "trajectories").glob("episode_*")):
            if not episode.is_dir():
                continue
            value = evaluate_task_success(task_id, episode)
            counts[value["status"]] += 1
            task_counts[value["status"]] += 1
            task_reasons.update(map(str, value.get("reasons", ())))
            task_diagnostics.update(map(str, value.get("diagnostics", ())))
            rows.append({
                "benchmark_subtask": task_id,
                "episode": episode.name,
                "status": value["status"],
                "success": value["success"],
                "reasons": value["reasons"],
                "diagnostics": value.get("diagnostics", []),
            })
        contract = success_contract(task_id)
        episode_count = sum(task_counts.values())
        task_rows.append({
            "benchmark_subtask": task_id,
            "task_key": _task_key(task_id),
            "owner": catalog_task.get("owner", ""),
            "checker_registration_status": contract.get("status"),
            "checker": contract.get("checker"),
            "data_directory": (
                str(task_directory) if task_directory.is_dir() else None
            ),
            "episode_count": episode_count,
            "counts": {
                status: task_counts.get(status, 0)
                for status in ("passed", "failed", "unverified")
            },
            "status": "no_data" if not episode_count else (
                "all_passed"
                if task_counts.get("passed", 0) == episode_count
                else "needs_attention"
            ),
            "reason_counts": dict(task_reasons.most_common()),
            "diagnostic_counts": dict(task_diagnostics.most_common()),
            "owner_requirement": contract.get("owner_requirement"),
        })
    return {
        "schema": AUDIT_SCHEMA,
        "root": str(directory),
        "episode_count": len(rows),
        "catalog_task_count": len(task_rows),
        "tasks_with_data": sum(row["episode_count"] > 0 for row in task_rows),
        "tasks_without_data": sum(row["episode_count"] == 0 for row in task_rows),
        "counts": {
            status: counts.get(status, 0)
            for status in ("passed", "failed", "unverified")
        },
        "tasks": task_rows,
        "episodes": rows,
    }


def audit_manifest(manifest: Path | str, *, workers: int = 4) -> dict[str, Any]:
    """Recheck explicitly bound episodes anywhere, without modifying inputs.

    Paths are relative to the manifest. This is an input list, not a task
    registry: all semantics and checkers still come from the Sheet catalog.
    A missing/corrupt episode remains visible and cannot abort other checks.
    """
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    source = Path(manifest).resolve()
    entries = _json(source, required=True).get("episodes")
    if not isinstance(entries, list):
        raise ValueError("manifest must contain an episodes list")
    bindings: dict[Path, str] = {}
    inputs = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each episode must be an object")
        task = _task_id(entry["benchmark_subtask"])
        success_contract(task)  # Reject unknown task IDs before starting workers.
        raw_path = entry.get("episode_directory")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("episode_directory must be a nonempty path")
        directory = (source.parent / raw_path).resolve()
        if directory in bindings:
            if bindings[directory] != task:
                raise ValueError(f"conflicting task identities for {directory}")
            continue
        bindings[directory] = task
        inputs.append((task, directory))

    def evaluate(item: tuple[str, Path]) -> dict[str, Any]:
        task, directory = item
        try:
            return evaluate_task_success(task, directory)
        except Exception as exc:
            return {
                "benchmark_subtask": task,
                "episode_directory": str(directory),
                "status": "unverified", "success": None,
                "reasons": [f"episode evaluation error: {type(exc).__name__}: {exc}"],
                "error_type": type(exc).__name__,
            }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, inputs))
    counts = Counter(row["status"] for row in results)
    tasks = []
    for task in sorted(set(bindings.values()), key=lambda t: tuple(map(int, t.split(".")))):
        rows = [row for row in results if row["benchmark_subtask"] == task]
        task_counts = Counter(row["status"] for row in rows)
        tasks.append({
            "benchmark_subtask": task, "episode_count": len(rows),
            "counts": {status: task_counts[status] for status in sorted(VALID_STATUSES)},
            "reason_counts": dict(Counter(
                reason for row in rows for reason in row.get("reasons", ())
            ).most_common()),
        })
    return {
        "schema": "real2sim.task_success_manifest_audit.v1",
        "manifest": str(source), "workers": workers,
        "evaluation_scope": "task_semantics_and_action_state_integrity",
        "render_eligibility_evaluated": False,
        "owner_release_approved": False,
        "input_count": len(entries), "episode_count": len(results),
        "duplicate_entries_skipped": len(entries) - len(results),
        "counts": {status: counts[status] for status in sorted(VALID_STATUSES)},
        "tasks": tasks, "episodes": results,
    }


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coverage = subparsers.add_parser("coverage")
    coverage.add_argument("--output", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("task")
    evaluate.add_argument("episode", type=Path)
    evaluate.add_argument("--write-result", action="store_true")
    evaluate.add_argument("--output", type=Path)
    rollout = subparsers.add_parser("rollout-stop")
    rollout.add_argument("task")
    rollout.add_argument("episode", type=Path)
    rollout.add_argument("--output", type=Path)
    audit = subparsers.add_parser("audit")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "coverage":
        value, output = coverage_report(), args.output
    elif args.command == "evaluate":
        value = evaluate_task_success(
            args.task, args.episode, write_result=args.write_result
        )
        output = args.output
    elif args.command == "rollout-stop":
        value = evaluate_rollout_stop(args.task, args.episode)
        output = args.output
    else:
        value, output = audit_root(args.root), args.output
    _emit(value, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

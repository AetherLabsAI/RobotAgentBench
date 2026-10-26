#!/usr/bin/env python3
"""Shared candidate producer for the two Family-4 drawer-pick tasks.

The benchmark catalog and ``recovered_task_contract`` own task semantics.  This
module only turns those definitions into one parameterized producer request,
then validates the action/state bundle returned by an Isaac runtime.  It does
not contain a second task registry and it does not promote a task before a
fresh-physics canary passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFLIGHT_SCHEMA = "real2sim.family4_drawer_pick_producer_preflight.v1"
REQUEST_SCHEMA = "real2sim.family4_drawer_pick_candidate_request.v1"
RESULT_SCHEMA = "real2sim.family4_drawer_pick_candidate_result.v1"
DEFAULT_OUTPUT = REPO / "dataset" / "sim_episodes_repro" / (
    "family4_drawer_pick_canary"
)


class DrawerPickRuntime(Protocol):
    """Simulator boundary used by the shared candidate producer."""

    def run_drawer_pick_candidate(
        self, *, request: Mapping[str, Any], output_dir: Path,
    ) -> Mapping[str, Any]: ...


def _task_definition(benchmark_subtask: str) -> dict[str, Any]:
    task = str(benchmark_subtask)
    try:
        from .recovered_scene_materialization import binding
        from .recovered_task_contract import ASSETS, TASKS
    except ImportError:
        from recovered_scene_materialization import binding
        from recovered_task_contract import ASSETS, TASKS

    try:
        contract = dict(TASKS[task])
    except KeyError as exc:
        raise ValueError(f"unknown recovered benchmark task {task}") from exc
    scene = binding(task)
    drawer = dict(contract.get("drawer_contract", {}))
    if contract.get("action") != "drawer_pick":
        raise ValueError(f"benchmark {task} is not a drawer-pick task")
    if (
        drawer.get("prim_path") != scene.get("support_prim")
        or drawer.get("initial_state") != "fully_closed"
        or drawer.get("final_state") != "open_allowed"
    ):
        raise ValueError(f"benchmark {task} drawer contract is inconsistent")
    target = str(contract["target"])
    references = list(dict.fromkeys(map(str, (
        *contract.get("real_data_reference_ids", ()),
        *contract.get("donor_real_data_reference_ids", ()),
    ))))
    return {
        "benchmark_subtask": task,
        "task_key": "benchmark_" + task.replace(".", "_"),
        "target": target,
        "target_asset": str(ASSETS[target]),
        "target_prim": str(scene["target_prim"]),
        "drawer": drawer,
        "robot_station": str(contract["robot_station"]),
        "controller_profile": str(contract["controller_profile"]),
        "optional_real_data_reference_ids": references,
    }


def _wrapper_path(task: str) -> Path:
    return HERE / "task_randomization" / (
        "benchmark_" + task.replace(".", "_")
    ) / "produce_cores.py"


def preflight(benchmark_subtask: str) -> dict[str, Any]:
    """Describe source/runtime readiness and reviewed canary acceptance."""
    task = _task_definition(benchmark_subtask)
    wrapper = _wrapper_path(task["benchmark_subtask"])
    missing_sources = [str(wrapper)] if not wrapper.is_file() else []
    target_asset = Path(task["target_asset"])
    missing_runtime_assets = (
        [] if target_asset.is_file() else [str(target_asset)]
    )
    request_ready = not missing_sources
    blockers = []
    if missing_sources:
        blockers.append(
            "missing producer source: " + ", ".join(missing_sources)
        )
    if missing_runtime_assets:
        blockers.append(
            "runtime-local target asset is not materialized: "
            + ", ".join(missing_runtime_assets)
        )
    task_root = wrapper.parent
    policy_path = task_root / "policy.json"
    canary_path = task_root / "canary.json"
    accepted_canary = False
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
        accepted_canary = bool(
            policy.get("schema")
            == "real2sim.task_initial_state_randomization_policy.v1"
            and policy.get("status") == "accepted"
            and policy.get("task_key") == task["task_key"]
            and canary.get("schema")
            == "real2sim.family4_drawer_pick_canary_acceptance.v1"
            and canary.get("status") == "accepted"
            and canary.get("task_key") == task["task_key"]
            and canary.get("strict_success") is True
            and canary.get("executed_action_replay", {}).get(
                "strict_success") is True
            and canary.get("visual_review", {}).get(
                "three_camera_full_decode") is True
            and canary.get("atomic_posture", {}).get("passed") is True
            and canary.get("atomic_posture", {}).get(
                "initial_arm_reference") == "compact_chest_front"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        accepted_canary = False
    if not accepted_canary:
        blockers.append(
            "fresh-physics closed-drawer open-and-pick canary has not been "
            "accepted for this benchmark task"
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        **task,
        "producer_entrypoint": str(wrapper.relative_to(REPO)),
        "candidate_request_ready": request_ready,
        "action_state_production_ready": bool(
            request_ready and not missing_runtime_assets and accepted_canary),
        "accepted_canary_available": accepted_canary,
        "execution_phase": "production" if accepted_canary else "canary",
        "blockers": blockers,
        "missing_paths": missing_sources,
        "runtime_assets_available": not missing_runtime_assets,
        "missing_runtime_assets": missing_runtime_assets,
        "action_source": "synthetic_code_control",
        "real_data_required": False,
        "required_sequence": [
            "restore_shared_house_baseline",
            "activate_target_inside_closed_drawer",
            "move_robot_to_catalog_station",
            "physically_pull_drawer_open",
            "grasp_and_lift_target",
            "hold_target_for_strict_checker",
        ],
        "success_semantics": {
            "drawer_must_open_by_contact": True,
            "target_must_be_lifted_and_held": True,
            "drawer_must_close_after_pick": False,
        },
    }


def _finite_vector(
    value: Sequence[float], *, size: int, name: str,
) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a {size}-value sequence")
    vector = [float(item) for item in value]
    if len(vector) != size or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain {size} finite values")
    if not any(abs(item) > 1e-12 for item in vector):
        raise ValueError(f"{name} must be physically nonzero")
    return vector


def build_candidate_request(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    arm_initial_delta_rad: Sequence[float],
    base_initial_delta_xyyaw: Sequence[float],
) -> dict[str, Any]:
    """Build one explicit, task-owned canary request.

    Numeric perturbations are mandatory inputs.  The shared component does not
    invent a global fallback scale for either task.
    """
    if (
        not isinstance(rollout_index, int)
        or isinstance(rollout_index, bool)
        or rollout_index < 0
    ):
        raise ValueError("rollout_index must be a non-negative integer")
    status = preflight(benchmark_subtask)
    if not status["candidate_request_ready"]:
        raise RuntimeError("drawer-pick candidate producer is incomplete")
    arm = _finite_vector(
        arm_initial_delta_rad, size=14,
        name="arm_initial_delta_rad",
    )
    base = _finite_vector(
        base_initial_delta_xyyaw, size=3,
        name="base_initial_delta_xyyaw",
    )
    task = status["benchmark_subtask"]
    family, subtask = (int(part) for part in task.split("."))
    seed = (family * 1_000 + subtask) * 1_000_000 + rollout_index + 1
    return {
        "schema": REQUEST_SCHEMA,
        "benchmark_subtask": task,
        "task_key": status["task_key"],
        "rollout_index": rollout_index,
        "seed": seed,
        "execution_phase": "canary",
        "production_acceptance_claimed": False,
        "target": {
            "name": status["target"],
            "prim_path": status["target_prim"],
            "asset_path": status["target_asset"],
            "optional_real_data_reference_ids": status[
                "optional_real_data_reference_ids"
            ],
        },
        "drawer": status["drawer"],
        "robot_station": status["robot_station"],
        "controller_profile": status["controller_profile"],
        "action_source": "synthetic_code_control",
        "robot_arm_initial_delta_rad": arm,
        "robot_base_initial_delta_xyyaw": base,
        "retarget_robot_relative_action_with_base": True,
        "furniture_transforms_changed": False,
        "camera_randomized": False,
        "required_sequence": status["required_sequence"],
        "success_semantics": status["success_semantics"],
    }


def plan_candidate(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    arm_initial_delta_rad: Sequence[float],
    base_initial_delta_xyyaw: Sequence[float],
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Create the shared-scene initializer plan consumed by an Isaac runtime."""
    request = build_candidate_request(
        benchmark_subtask,
        rollout_index=rollout_index,
        arm_initial_delta_rad=arm_initial_delta_rad,
        base_initial_delta_xyyaw=base_initial_delta_xyyaw,
    )
    try:
        from .recovered_task_contract import prepare_rollout
    except ImportError:
        from recovered_task_contract import prepare_rollout
    planned = prepare_rollout(
        request["benchmark_subtask"],
        rollout_index=rollout_index,
        output_root=Path(output_root),
    )
    if planned.get("decision") != "requires_runtime_materialization":
        raise RuntimeError(
            "drawer-pick initializer plan is not ready for materialization: "
            + str(planned.get("blockers", planned.get("decision")))
        )
    return {
        **request,
        "initializer_plan": str(planned["initializer_plan"]),
        "randomization_audit": str(planned["randomization_audit"]),
        "runtime_materialized": False,
    }


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _validate_action_state(episode: Path) -> tuple[Path, Path]:
    action_path = episode / "action.npz"
    state_path = episode / "full_state.npz"
    if not action_path.is_file() or not state_path.is_file():
        raise ValueError("runtime did not produce action.npz and full_state.npz")
    with np.load(action_path, allow_pickle=False) as action:
        required = {
            "fps", "arm", "waist", "head", "base_xyyaw", "lgrip", "rgrip",
        }
        missing = sorted(required - set(action.files))
        if missing:
            raise ValueError("action.npz is missing: " + ", ".join(missing))
        frames = int(np.asarray(action["arm"]).shape[0])
        shapes = {
            "arm": (frames, 14),
            "waist": (frames, 2),
            "head": (frames, 2),
            "base_xyyaw": (frames, 3),
            "lgrip": (frames,),
            "rgrip": (frames,),
        }
        if frames < 2 or any(
            np.asarray(action[name]).shape != shape
            for name, shape in shapes.items()
        ):
            raise ValueError("action.npz channel shapes are inconsistent")
        if not all(
            np.isfinite(np.asarray(action[name], dtype=float)).all()
            for name in shapes
        ):
            raise ValueError("action.npz contains non-finite commands")
        fps = np.asarray(action["fps"])
        if fps.shape != () or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("action.npz fps must be one positive scalar")
    with np.load(state_path, allow_pickle=False) as state:
        if not state.files:
            raise ValueError("full_state.npz has no arrays")
        for name in state.files:
            np.asarray(state[name])
    return action_path, state_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def produce_candidate(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    arm_initial_delta_rad: Sequence[float],
    base_initial_delta_xyyaw: Sequence[float],
    runtime: DrawerPickRuntime,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Run one candidate and apply the current benchmark checker.

    A passing result is canary evidence only.  Production promotion remains a
    separate reviewed Git change after the physical result is inspected.
    """
    request = plan_candidate(
        benchmark_subtask,
        rollout_index=rollout_index,
        arm_initial_delta_rad=arm_initial_delta_rad,
        base_initial_delta_xyyaw=base_initial_delta_xyyaw,
        output_root=output_root,
    )
    episode = Path(output_root) / request["task_key"] / (
        f"candidate_{rollout_index:06d}"
    )
    episode.mkdir(parents=True, exist_ok=True)
    runtime_result = dict(runtime.run_drawer_pick_candidate(
        request=request, output_dir=episode,
    ))
    if runtime_result.get("status") not in {"pass", "fail"}:
        raise ValueError("runtime result status must be pass or fail")
    action_path, state_path = _validate_action_state(episode)
    metrics = _json_object(runtime_result.get("metrics"), name="metrics")
    audit = _json_object(
        runtime_result.get("randomization_audit"),
        name="randomization_audit",
    )
    try:
        from .recovered_task_check_success import check_task
    except ImportError:
        from recovered_task_check_success import check_task
    checked = check_task(
        request["benchmark_subtask"], metrics,
        randomization_audit=audit,
    )
    strict_success = checked.get("success") is True
    result = {
        "schema": RESULT_SCHEMA,
        "benchmark_subtask": request["benchmark_subtask"],
        "task_key": request["task_key"],
        "rollout_index": rollout_index,
        "status": "pass" if strict_success else "fail",
        "strict_success": strict_success,
        "production_acceptance_claimed": False,
        "action_path": str(action_path),
        "state_path": str(state_path),
        "checker": checked,
        "request": request,
    }
    _write_json(episode / "validation.json", {
        "validation_ok": strict_success,
        **result,
    })
    return result


def _numbers(raw: str, *, size: int, name: str) -> list[float]:
    try:
        values = [float(item) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers") from exc
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    return values


def cli(benchmark_subtask: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="preflight",
        choices=("preflight", "plan-canary"),
    )
    parser.add_argument("--rollout-index", type=int)
    parser.add_argument("--arm-delta")
    parser.add_argument("--base-delta")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(benchmark_subtask)
        else:
            if (
                args.rollout_index is None
                or args.arm_delta is None
                or args.base_delta is None
            ):
                parser.error(
                    "plan-canary requires --rollout-index, --arm-delta, and "
                    "--base-delta"
                )
            result = plan_candidate(
                benchmark_subtask,
                rollout_index=args.rollout_index,
                arm_initial_delta_rad=_numbers(
                    args.arm_delta, size=14, name="arm-delta"),
                base_initial_delta_xyyaw=_numbers(
                    args.base_delta, size=3, name="base-delta"),
                output_root=args.output_root,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


__all__ = [
    "DrawerPickRuntime",
    "build_candidate_request",
    "plan_candidate",
    "preflight",
    "produce_candidate",
]

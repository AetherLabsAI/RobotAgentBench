#!/usr/bin/env python3
"""Recovered exact-L2 contracts for benchmark family 2 air-fryer tasks."""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from .house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_DEMO = SOURCE_ROOT
SCENE = WORKING_SCENE
BASELINE = SOURCE_BASELINE
DRAWER_ASSET = (
    SOURCE_ROOT / "l2_imports"
    / "air_fryer_v19" / "air_fryer_articulated_handle_v19_task66.usda"
)
LOAD_ASSET = (
    HERE / "assets" / "kitchen_task_appliances"
    / "air_fryer_chefman_task63" / "usd"
    / "air_fryer_chefman_photo_physics_v8_repro_v1.usda"
)
OUTPUT = REPO / "real2sim" / "output" / "air_fryer_rollouts"
RANDOMIZATION_SCHEMA = "real2sim.air_fryer_randomization.applied.v1"

TASKS = {
    "2.1": {
        "action": "load_close_start", "sugar": False,
        "real_data_reference_ids": ("65",),
        "runtime_ready": False,
        "blocker": (
            "component reference recordings exist, but no accepted end-to-end "
            "load-close-start initializer/trajectory is bound"
        ),
    },
    "2.2": {
        "action": "load_close_start", "sugar": True,
        "real_data_reference_ids": ("63",),
        "runtime_ready": False,
        "blocker": (
            "a sugar-bread load/start reference exists, but no accepted "
            "complete load-close-start initializer/trajectory is bound"
        ),
    },
    "2.3": {
        "action": "drawer_remove", "sugar": True,
        "real_data_reference_ids": ("66",), "runtime_ready": False,
        "blocker": (
            "the obsolete DiningTableAirFryer v19 scene fixture was removed; "
            "a replacement drawer asset and trajectory calibration are required"
        ),
    },
    "2.4": {
        "action": "drawer_remove", "sugar": False,
        "real_data_reference_ids": ("67",), "runtime_ready": False,
        "blocker": (
            "the obsolete DiningTableAirFryer v19 scene fixture was removed; "
            "a replacement drawer asset and trajectory calibration are required"
        ),
    },
}


def preflight(benchmark_subtask: str) -> dict:
    task = str(benchmark_subtask)
    contract = dict(TASKS[task])
    asset = DRAWER_ASSET if contract["action"] == "drawer_remove" else LOAD_ASSET
    missing = [
        str(path) for path in (BASELINE, asset) if not path.is_file()
    ]
    blockers = [contract["blocker"]] if contract.get("blocker") else []
    if missing:
        blockers.append("missing exact L2 dependency: " + ", ".join(missing))
    return {
        "benchmark_subtask": task,
        "registered": True,
        "strict_checker_ready": True,
        "runtime_ready": bool(contract["runtime_ready"] and not missing),
        "contract": contract,
        "canonical_scene": str(SCENE),
        "restore_source": str(BASELINE),
        "asset": str(asset),
        "missing_paths": missing,
        "blockers": blockers,
    }


def prepare_rollout(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    output_root: Path = OUTPUT,
) -> dict:
    task = str(benchmark_subtask)
    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    decision = preflight(task)
    if not decision["runtime_ready"]:
        return {
            **decision,
            "schema": RANDOMIZATION_SCHEMA,
            "rollout_index": int(rollout_index),
            "schedulable": False,
            "decision": "blocked",
        }
    seed = f"{RANDOMIZATION_SCHEMA}:{task}:{rollout_index}"
    rollout_dir = (
        Path(output_root) / f"benchmark_{task.replace('.', '_')}"
        / f"rollout_{int(rollout_index):06d}"
    )
    plan_path = rollout_dir / "initializer_plan.json"
    asset = (
        DRAWER_ASSET
        if decision["contract"]["action"] == "drawer_remove"
        else LOAD_ASSET
    )
    plan = {
        "schema": "real2sim.air_fryer_initializer_plan.v1",
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "seed": seed,
        "restore_source": str(BASELINE),
        "working_scene": str(SCENE),
        "asset": str(asset),
        "object_randomization": (
            "none for drawer-remove: exact accepted L2 episode pose"
        ),
        "robot_adaptation": (
            "robot station and robot-relative trajectory move together; "
            "air fryer/furniture never move"
        ),
    }
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((plan_path, plan),):
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return {
        "schema": "real2sim.air_fryer_initializer_plan.pending.v1",
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "registered": True,
        "pipeline_ready_for_materialization": True,
        "runtime_materialized": False,
        "schedulable": False,
        "decision": "requires_runtime_materialization",
        "initializer_plan": str(plan_path),
        "required_runtime_evidence": [
            "canonical session restore",
            "active-body overlap audit",
            "drawer-joint swept-volume audit",
            "measured collision-free frame zero",
        ],
    }

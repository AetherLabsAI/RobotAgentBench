#!/usr/bin/env python3
"""Self-contained runtime contracts for the dining benchmark tasks.

Legacy recording labels are optional data references only.
``strict_checker_ready`` describes the scorer.  ``runtime_ready`` additionally
requires an accepted initializer/asset closure and is the scheduling boundary.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from .house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEMO = WORKING_SCENE.parent
CANONICAL_BASELINE = SOURCE_BASELINE
CANONICAL_SCENE = WORKING_SCENE
ASSET_ROOT = SOURCE_ROOT / "kitchen_task_assets" / "dining"
WORK = HERE / "kitchen_task_pipelines"
TAKE = WORK / "microwave_take"
SERVE = WORK / "serve_meal"
PUT = WORK / "microwave_put"

ASSETS = {
    "beef_box": (
        ASSET_ROOT / "beef_box" / "beefbox.usda"
    ),
    "meal_box": (
        ASSET_ROOT / "foodbox_burger_v1"
        / "foodbox_task108_physics_v2.usda"
    ),
    "salad_box": (
        ASSET_ROOT / "salad_box" / "saladbox.usda"
    ),
    "plate": (
        ASSET_ROOT / "black_plate_325mm"
        / "black_plate_325mm_physics.usda"
    ),
    "microwave": (
        SOURCE_ROOT / "kitchen_task_assets"
        / "microwave_task106" / "microwave.usda"
    ),
}


def _task(
    *,
    action: str,
    payload: str,
    real_data_reference_ids: tuple[str, ...] = (),
    checker: str,
    pipeline: str,
    runtime_ready: bool,
    blocker: str = "",
) -> dict[str, Any]:
    return {
        "action_family": action,
        "payload_type": payload,
        "canonical_real_data_reference_ids": real_data_reference_ids,
        "checker": checker,
        "pipeline": pipeline,
        "strict_checker_ready": True,
        "runtime_ready": runtime_ready,
        "blockers": [blocker] if blocker else [],
        "coordinate_contract": {
            "world": "USD metres, Z-up",
            "microwave_containment": "live articulated microwave-body local",
            "destination_containment": "live destination-body local footprint",
        },
    }


DINING_TASKS = {
    "9.1": _task(
        action="take_from_refrigerator", payload="beef_box",
        real_data_reference_ids=("95",), checker="check_refrigerator_take",
        pipeline="refrigerator_take",
        runtime_ready=False,
        blocker=(
            "reference recordings exist, but L2 only has refrigerator "
            "room-shell visuals, not an accepted "
            "articulated refrigerator initializer/physics closure"
        ),
    ),
    "9.2": _task(
        action="put_into_microwave", payload="beef_box",
        real_data_reference_ids=("94",), checker="check_microwave_put",
        pipeline="microwave_put_handoff", runtime_ready=True,
    ),
    "9.3": _task(
        action="take_from_refrigerator", payload="meal_box",
        real_data_reference_ids=("96",), checker="check_refrigerator_take",
        pipeline="refrigerator_take",
        runtime_ready=False,
        blocker=(
            "reference recordings exist, but the articulated refrigerator "
            "runtime is absent"
        ),
    ),
    "9.4": _task(
        action="put_into_microwave", payload="meal_box",
        real_data_reference_ids=("93",), checker="check_microwave_put",
        pipeline="microwave_put_handoff", runtime_ready=True,
    ),
    "9.5": _task(
        action="take_from_refrigerator", payload="salad_box",
        real_data_reference_ids=("97",), checker="check_refrigerator_take",
        pipeline="refrigerator_take",
        runtime_ready=False,
        blocker=(
            "reference recordings exist, but the articulated refrigerator "
            "runtime is absent"
        ),
    ),
    "10.1": _task(
        action="take_from_microwave", payload="beef_box",
        real_data_reference_ids=("106",), checker="check_microwave_take",
        pipeline="task106_take", runtime_ready=True,
    ),
    "10.2": _task(
        action="serve_to_dining_table", payload="beef_box",
        checker="check_serve_plate", pipeline="task108_serve",
        runtime_ready=False,
        blocker=(
            "checker and exact L2 payload/plate assets are ready, but no "
            "accepted dining-table robot station/initializer has passed"
        ),
    ),
    "10.3": _task(
        action="serve_to_coffee_table", payload="beef_box",
        real_data_reference_ids=("108", "112"), checker="check_serve_plate",
        pipeline="task108_serve", runtime_ready=True,
    ),
    "10.4": _task(
        action="take_from_microwave", payload="meal_box",
        real_data_reference_ids=("106",), checker="check_microwave_take",
        pipeline="task106_take", runtime_ready=True,
    ),
    "10.5": _task(
        action="serve_to_dining_table", payload="meal_box",
        checker="check_serve_plate", pipeline="task108_serve",
        runtime_ready=False,
        blocker=(
            "checker and exact L2 payload/plate assets are ready, but no "
            "accepted dining-table robot station/initializer has passed"
        ),
    ),
    "10.6": _task(
        action="serve_to_coffee_table", payload="meal_box",
        real_data_reference_ids=("108",), checker="check_serve_plate",
        pipeline="task108_serve", runtime_ready=True,
    ),
    "10.7": _task(
        action="take_from_microwave", payload="salad_box",
        real_data_reference_ids=("106",),
        checker="check_microwave_take", pipeline="task106_take",
        runtime_ready=False,
        blocker=(
            "Task106 production accepted beef and meal-box payloads only; "
            "the salad take initializer has no accepted strict rollout"
        ),
    ),
    "10.8": _task(
        action="serve_to_dining_table", payload="salad_box",
        checker="check_serve_plate", pipeline="task108_serve",
        runtime_ready=False,
        blocker=(
            "checker and exact L2 salad/plate assets are ready, but no "
            "accepted dining-table robot station/initializer has passed"
        ),
    ),
    "10.9": _task(
        action="serve_to_coffee_table", payload="salad_box",
        real_data_reference_ids=("108",), checker="check_serve_plate",
        pipeline="task108_serve", runtime_ready=True,
    ),
}


PIPELINE_REQUIREMENTS = {
    "microwave_put_handoff": (
        PUT / "README.md",
        PUT / "mw_guided.py",
        ASSETS["microwave"],
    ),
    "task106_take": (
        TAKE / "README.md",
        TAKE / "validate_take_multibaseline_v8.py",
        ASSETS["microwave"],
        ASSETS["plate"],
    ),
    "task108_serve": (
        SERVE / "README.md",
        SERVE / "task108_scaleup.py",
        ASSETS["plate"],
    ),
}


def contract(benchmark_subtask: str) -> dict[str, Any]:
    task = str(benchmark_subtask)
    if task not in DINING_TASKS:
        raise KeyError(f"{task} is not a dining benchmark task")
    return dict(DINING_TASKS[task])


def preflight(benchmark_subtask: str) -> dict[str, Any]:
    task = str(benchmark_subtask)
    value = contract(task)
    required = PIPELINE_REQUIREMENTS.get(value["pipeline"], ())
    required = (
        CANONICAL_BASELINE,
        *required,
        ASSETS[value["payload_type"]],
    )
    missing = [str(path) for path in required if not path.is_file()]
    blockers = list(value["blockers"])
    if missing:
        blockers.append(
            "exact L2 dependency closure is incomplete: " + ", ".join(missing)
        )
    ready = bool(value["runtime_ready"] and not missing)
    return {
        "benchmark_subtask": task,
        "registered": True,
        "strict_checker_ready": True,
        "runtime_ready": ready,
        "required_paths": [str(path) for path in required],
        "missing_paths": missing,
        "blockers": blockers,
        "contract": value,
    }

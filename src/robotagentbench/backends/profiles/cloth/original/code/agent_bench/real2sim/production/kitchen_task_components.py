#!/usr/bin/env python3
"""Optional Kitchen runtime components for the Sheet-backed task catalog.

This module configures the four Kitchen pipeline groups on the canonical House
scene.  It does not define another task identity layer, and configuration alone
is not a claim that a rollout passed runtime physics checks.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

try:
    from .git_provenance import source_revision
except ImportError:
    from git_provenance import source_revision
try:
    from .house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_BASELINE, SOURCE_ROOT, WORKING_SCENE


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMPONENTS_PATH = HERE / "kitchen_task_components_v1.json"
SOURCE_DEMO = SOURCE_ROOT
CANONICAL_BASELINE = SOURCE_BASELINE
RUNTIME_DEMO = WORKING_SCENE.parent
CANONICAL_SCENE = WORKING_SCENE

ASSET_REQUIREMENTS = {
    "coffee_bean": (
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/task75/coffee_bag/coffee_bag_task75.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/task75/coffee_beans/coffee_bean_a_task75.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/task75/grinder/grinder_task75.usda",
    ),
    "air_fryer": (
        "arena_scene/v1_2/demo_sim/l2_imports/air_fryer_v19/air_fryer_articulated_handle_v19_task66.usda",
    ),
    "microwave_fridge": (
        "arena_scene/v1_2/demo_sim/fridge_articulated_v1/fridge.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/microwave_task106/microwave.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/beef_box/beefbox.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/foodbox_burger_v1/foodbox_task108_physics_v2.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/salad_box/saladbox.usda",
        "real2sim/production/kitchen_task_pipelines/microwave_put/mw_guided.py",
        "real2sim/production/kitchen_task_pipelines/microwave_take/validate_take_multibaseline_v8.py",
    ),
    "serve_meal": (
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/black_plate_325mm/black_plate_325mm_flat_collision_v6.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/beef_box/beefbox.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/foodbox_burger_v1/foodbox_smooth_collision_v3.usda",
        "arena_scene/v1_2/demo_sim/kitchen_task_assets/dining/salad_box/saladbox.usda",
        "real2sim/production/kitchen_task_pipelines/serve_meal/task108_scaleup.py",
    ),
}

PIPELINE_ENTRYPOINTS = {
    "task75_pour": (
        "real2sim/production/kitchen_randomization.py:prepare_rollout"
    ),
    "air_fryer_load_close_start": (
        "real2sim/production/air_fryer_task_contract.py:prepare_rollout"
    ),
    "air_fryer_drawer_remove": (
        "real2sim/production/air_fryer_task_contract.py:prepare_rollout"
    ),
    "refrigerator_take": (
        "real2sim/production/dining_randomization.py:prepare_rollout"
    ),
    "microwave_put_handoff": (
        "real2sim/production/kitchen_task_pipelines/microwave_put/"
        "canonical_pipeline.py:prepare_rollout"
    ),
    "task106_take": (
        "real2sim/production/kitchen_task_pipelines/microwave_take/"
        "validate_take_multibaseline_v8.py:prepare_rollout"
    ),
    "task108_serve": (
        "real2sim/production/kitchen_task_pipelines/serve_meal/"
        "task108_scaleup.py:prepare_rollout"
    ),
}

PIPELINE_SOURCE_REQUIREMENTS = {
    "microwave_put_handoff": (
        "real2sim/production/kitchen_task_pipelines/microwave_put/"
        "mw_guided.py",
    ),
}


def _entrypoint_issue(entrypoint: str) -> str:
    """Verify a configured ``path.py:function`` without importing runtime code."""
    try:
        relative, function = str(entrypoint).rsplit(":", 1)
    except ValueError:
        return f"invalid entrypoint syntax: {entrypoint}"
    path = REPO / relative
    if not path.is_file():
        return f"entrypoint source is missing: {path}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return f"entrypoint source cannot be inspected: {path}: {exc}"
    functions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function not in functions:
        return f"entrypoint callable is missing: {entrypoint}"
    return ""


def _load() -> dict[str, Any]:
    payload = json.loads(COMPONENTS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "real2sim.kitchen_task_components.v1":
        raise ValueError("unsupported Kitchen component schema")
    if payload.get("runtime_scene_basename") != "scene_room.usda":
        raise ValueError("Kitchen components may select only scene_room.usda")
    components = payload.get("components")
    groups = payload.get("groups")
    if not isinstance(components, dict) or not isinstance(groups, dict):
        raise ValueError(
            "Kitchen component document must contain component/group objects"
        )
    for task_id, component in components.items():
        if not isinstance(component, dict):
            raise ValueError(f"Kitchen component {task_id} must be an object")
        group = component.get("group")
        pipeline = component.get("pipeline")
        if group not in groups or group not in ASSET_REQUIREMENTS:
            raise ValueError(
                f"Kitchen component {task_id} has unknown group {group!r}"
            )
        if pipeline not in PIPELINE_ENTRYPOINTS:
            raise ValueError(
                f"Kitchen component {task_id} has unknown pipeline {pipeline!r}"
            )
        if not isinstance(component.get("checker"), str):
            raise ValueError(
                f"Kitchen component {task_id} must name a checker"
            )
        references = component.get("real_data_reference_ids", ())
        if not isinstance(references, list):
            raise ValueError(
                f"Kitchen component {task_id} real-data references must be a list"
            )
    return payload


DOCUMENT = _load()
COMPONENTS = DOCUMENT["components"]
GROUPS = DOCUMENT["groups"]


def resolve_component(task_id: str) -> dict[str, Any]:
    """Return one optional Kitchen implementation descriptor."""
    task = str(task_id)
    if task not in COMPONENTS:
        raise KeyError(f"no shared Kitchen component for task {task}")
    entry = copy.deepcopy(COMPONENTS[task])
    entry.update({
        "task_id": task,
        "canonical_scene": str(CANONICAL_SCENE),
        "restore_source": str(CANONICAL_BASELINE),
        "runtime_scene_basename": "scene_room.usda",
        "appearance_entrypoint": DOCUMENT["appearance_entrypoint"],
        "tableware_entrypoint": DOCUMENT["tableware_entrypoint"],
        "initializer_entrypoint": PIPELINE_ENTRYPOINTS[entry["pipeline"]],
    })
    return entry


def preflight_component(task_id: str) -> dict[str, Any]:
    """Check the migrated source closure without self-certifying a rollout."""
    entry = resolve_component(task_id)
    initializer_entrypoint = PIPELINE_ENTRYPOINTS[entry["pipeline"]]
    checker_entrypoint = str(entry["checker"])
    def required_path(value: str) -> Path:
        prefix = "arena_scene/v1_2/demo_sim/"
        return (
            SOURCE_ROOT / value[len(prefix):]
            if value.startswith(prefix)
            else REPO / value
        )

    required = (
        CANONICAL_BASELINE,
        REPO / DOCUMENT["appearance_entrypoint"],
        REPO / DOCUMENT["tableware_entrypoint"],
        *(required_path(path) for path in ASSET_REQUIREMENTS[entry["group"]]),
        *(REPO / path for path in PIPELINE_SOURCE_REQUIREMENTS.get(
            entry["pipeline"], ())),
        REPO / initializer_entrypoint.split(":", 1)[0],
        REPO / checker_entrypoint.split(":", 1)[0],
    )
    missing = [str(path) for path in required if not path.is_file()]
    entrypoint_issues = list(filter(None, (
        _entrypoint_issue(initializer_entrypoint),
        _entrypoint_issue(checker_entrypoint),
    )))
    pipeline_ready = bool(
        entry["accepted_pipeline_ready"]
        and not missing
        and not entrypoint_issues
    )
    blockers = []
    if not entry["accepted_pipeline_ready"]:
        blockers.append(
            "this benchmark task does not yet have a complete current pipeline"
        )
    if missing:
        blockers.append("migrated Kitchen source closure is incomplete: " + ", ".join(missing))
    if entrypoint_issues:
        blockers.append(
            "Kitchen initializer/checker configuration is invalid: "
            + "; ".join(entrypoint_issues)
        )
    return {
        "schema": "real2sim.kitchen_component_preflight.v1",
        "task_id": str(task_id),
        "component_available": True,
        "group": entry["group"],
        "canonical_scene": str(CANONICAL_SCENE),
        "pipeline_ready_for_materialization": pipeline_ready,
        "runtime_materialized": False,
        "schedulable": False,
        "decision": "requires_runtime_materialization" if pipeline_ready else "blocked",
        "required_paths": [str(path) for path in required],
        "missing_paths": missing,
        "source_revision": (
            source_revision((
                COMPONENTS_PATH,
                REPO / initializer_entrypoint.split(":", 1)[0],
                REPO / checker_entrypoint.split(":", 1)[0],
            )) if not missing and not entrypoint_issues else None
        ),
        "entrypoint_issues": entrypoint_issues,
        "blockers": blockers,
        "component": entry,
    }


def component_summary() -> dict[str, Any]:
    decisions = {
        task: preflight_component(task) for task in COMPONENTS
    }
    return {
        "schema": "real2sim.kitchen_component_summary.v1",
        "canonical_scene": str(CANONICAL_SCENE),
        "group_count": len(GROUPS),
        "component_count": len(COMPONENTS),
        "pipeline_ready_for_materialization": sorted(
            task for task, row in decisions.items()
            if row["pipeline_ready_for_materialization"]
        ),
        "blocked_contracts": sorted(
            task for task, row in decisions.items()
            if not row["pipeline_ready_for_materialization"]
        ),
    }

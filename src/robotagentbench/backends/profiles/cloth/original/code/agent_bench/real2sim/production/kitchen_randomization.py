#!/usr/bin/env python3
"""Plan one unified House appearance/layout initialization for Kitchen tasks.

The only selectable stage is the canonical ``scene_room.usda``.  This module
may build a deterministic plan before a session starts, but it does not label
that plan as a restored scene or measured collision audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from .house_paths import SOURCE_BASELINE, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_BASELINE, WORKING_SCENE


POLICY = "real2sim.kitchen_rollout_randomization.v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CANONICAL_SCENE = WORKING_SCENE
CANONICAL_BASELINE = SOURCE_BASELINE
OUTPUT = REPO / "real2sim" / "output" / "kitchen_rollouts"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rollout_seed(task_id: str, rollout_index: int, channel: str) -> str:
    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    return f"task-{task_id}.rollout-{rollout_index}.{channel}"


def _xy_half_extents(asset: Any, yaw_deg: float) -> tuple[float, float]:
    if float(asset.length) <= 0:
        radius = float(asset.diameter) / 2
        return radius, radius
    theta = math.radians(yaw_deg)
    half_long = float(asset.length) / 2
    half_wide = float(asset.diameter) / 2
    # Source cutlery is long along local Y.
    return (
        abs(math.sin(theta)) * half_long + abs(math.cos(theta)) * half_wide,
        abs(math.cos(theta)) * half_long + abs(math.sin(theta)) * half_wide,
    )


def _bounds(item: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    asset = assets[item["asset"]]
    x, y, z = map(float, item["translate"])
    yaw = float(item["rotate_xyz"][2])
    hx, hy = _xy_half_extents(asset, yaw)
    height = float(asset.height)
    if float(asset.length) > 0:
        z0, z1 = z - height / 2, z + height / 2
    else:
        z0, z1 = z, z + height
    out = {
        "name": item["name"],
        "asset": item["asset"],
        "minimum": [x - hx, y - hy, z0],
        "maximum": [x + hx, y + hy, z1],
        "center_xy": [x, y],
    }
    if float(asset.length) > 0:
        out["shape"] = "oriented_rectangle"
        out["half_length_m"] = float(asset.length) / 2
        out["half_width_m"] = float(asset.diameter) / 2
        out["yaw_deg"] = yaw
    else:
        out["shape"] = "circle"
        out["radius_m"] = float(asset.diameter) / 2
    return out


def _overlap(a: dict[str, Any], b: dict[str, Any], clearance: float) -> bool:
    alo, ahi = a["minimum"], a["maximum"]
    blo, bhi = b["minimum"], b["maximum"]
    if not (alo[2] <= bhi[2] and ahi[2] >= blo[2]):
        return False

    ax, ay = a["center_xy"]
    bx, by = b["center_xy"]
    if a["shape"] == b["shape"] == "circle":
        return math.hypot(ax - bx, ay - by) <= (
            float(a["radius_m"]) + float(b["radius_m"]) + clearance)

    def axes(rect: dict[str, Any]) -> tuple[tuple[float, float], ...]:
        # Local long axis is Y at yaw=0; local width axis is X.
        theta = math.radians(float(rect["yaw_deg"]))
        return ((-math.sin(theta), math.cos(theta)),
                (math.cos(theta), math.sin(theta)))

    def circle_rect(circle: dict[str, Any], rect: dict[str, Any]) -> bool:
        cx, cy = circle["center_xy"]
        rx, ry = rect["center_xy"]
        long_axis, width_axis = axes(rect)
        dx, dy = cx - rx, cy - ry
        local_long = dx * long_axis[0] + dy * long_axis[1]
        local_width = dx * width_axis[0] + dy * width_axis[1]
        q_long = max(
            -float(rect["half_length_m"]),
            min(float(rect["half_length_m"]), local_long))
        q_width = max(
            -float(rect["half_width_m"]),
            min(float(rect["half_width_m"]), local_width))
        dd_long = local_long - q_long
        dd_width = local_width - q_width
        return math.hypot(dd_long, dd_width) <= (
            float(circle["radius_m"]) + clearance)

    if a["shape"] == "circle":
        return circle_rect(a, b)
    if b["shape"] == "circle":
        return circle_rect(b, a)

    # Separating-axis test for two oriented rectangles.
    a_axes, b_axes = axes(a), axes(b)
    delta = (bx - ax, by - ay)
    for axis in (*a_axes, *b_axes):
        center_distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        a_radius = (
            float(a["half_length_m"])
            * abs(a_axes[0][0] * axis[0] + a_axes[0][1] * axis[1])
            + float(a["half_width_m"])
            * abs(a_axes[1][0] * axis[0] + a_axes[1][1] * axis[1])
        )
        b_radius = (
            float(b["half_length_m"])
            * abs(b_axes[0][0] * axis[0] + b_axes[0][1] * axis[1])
            + float(b["half_width_m"])
            * abs(b_axes[1][0] * axis[0] + b_axes[1][1] * axis[1])
        )
        if center_distance > a_radius + b_radius + clearance:
            return False
    return True


def validate_tableware_plan(
    plan: dict[str, Any],
    assets: dict[str, Any],
    *,
    clearance_m: float = 0.003,
) -> dict[str, Any]:
    """Validate complete object bounds, not only authored origins."""
    safe = plan["safe_region"]
    all_bounds: list[dict[str, Any]] = []
    prim_keys: set[tuple[int, str]] = set()
    for cluster in plan["clusters"]:
        cluster_index = int(cluster["index"])
        for item in cluster["items"]:
            key = (cluster_index, str(item["name"]))
            if key in prim_keys:
                raise ValueError(f"duplicate tableware prim: {key}")
            prim_keys.add(key)
            bounds = _bounds(item, assets)
            bounds["cluster"] = cluster_index
            lo, hi = bounds["minimum"], bounds["maximum"]
            if not (
                lo[0] >= float(safe["x_min"])
                and hi[0] <= float(safe["x_max"])
                and lo[1] >= float(safe["y_min"])
                and hi[1] <= float(safe["y_max"])
                and lo[2] >= float(plan["table_top_z_m"])
            ):
                raise ValueError(
                    f"full footprint outside safe tabletop region: {key}")
            for other in all_bounds:
                if _overlap(bounds, other, clearance_m):
                    raise ValueError(
                        "initial tableware collision: "
                        f"cluster {other['cluster']}/{other['name']} vs "
                        f"cluster {cluster_index}/{bounds['name']}")
            all_bounds.append(bounds)
    return {
        "schema": "real2sim.kitchen_tableware_collision_audit.v1",
        "collision_free": True,
        "clearance_m": clearance_m,
        "object_count": len(all_bounds),
        "objects": all_bounds,
    }


def sample_collision_free_tableware_plan(
    tableware: ModuleType,
    *,
    seed: str,
    base_layer: str,
    max_attempts: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rejection-sample the l2 layout policy without changing tuned physics."""
    failures: list[str] = []
    for attempt in range(max_attempts):
        candidate_seed = f"{seed}.candidate-{attempt}"
        plan = tableware.build_plan(candidate_seed, base_layer, None)
        tableware.validate_plan(plan)
        try:
            audit = validate_tableware_plan(plan, tableware.ASSETS)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        audit.update({
            "requested_seed": seed,
            "selected_candidate_seed": candidate_seed,
            "selection_attempt": attempt,
            "rejected_candidate_count": len(failures),
            "rejection_reasons": failures,
        })
        return plan, audit
    raise RuntimeError(
        f"no collision-free Kitchen tableware layout after {max_attempts} "
        f"candidates; last failures: {failures[-3:]}")


def preflight_runtime(runtime_dir: Path | None = None) -> dict[str, Any]:
    """Validate current modules; ``runtime_dir`` is ignored for compatibility."""
    required = [
        CANONICAL_SCENE,
        CANONICAL_BASELINE,
        HERE / "appearance.py",
        HERE / "kitchen_material_randomization.py",
        HERE / "kitchen_tableware_randomization.py",
        HERE / "livingroom_environment_randomization.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Kitchen runtime dependency closure is incomplete:\n- "
            + "\n- ".join(missing))
    return {
        "schema": "real2sim.kitchen_runtime_preflight.v1",
        "canonical_scene": str(CANONICAL_SCENE),
        "restore_source": str(CANONICAL_BASELINE),
        "legacy_runtime_dir_ignored": str(runtime_dir) if runtime_dir else "",
        "required_files": [str(path.resolve()) for path in required],
        "complete": True,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def prepare_rollout(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    output_root: Path = OUTPUT,
) -> dict[str, Any]:
    task = str(benchmark_subtask)
    if task != "1.1":
        raise KeyError(f"{task} is not the coffee-bean Kitchen pipeline")
    preflight = preflight_runtime()
    try:
        from . import appearance
        from . import kitchen_tableware_randomization as tableware
    except ImportError:
        import appearance
        import kitchen_tableware_randomization as tableware

    seed_hex = rollout_seed(task, rollout_index, "appearance")
    gen_seed = 1_000_000 + sum(
        (index + 1) * ord(character)
        for index, character in enumerate(seed_hex)
    ) % (2**63)
    appearance_plan = appearance.build_plan(
        task_id=f"benchmark-{task}", gen_seed=gen_seed
    )
    tableware_plan, footprint_check = tableware.build_collision_free_plan(
        rollout_seed(task, rollout_index, "tableware"),
        "../scene_room.usda",
    )
    plan = {
        "schema": "real2sim.kitchen_initializer_plan.v2",
        "policy": POLICY,
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "restore_source": str(CANONICAL_BASELINE),
        "working_scene": str(CANONICAL_SCENE),
        "runtime_preflight": preflight,
        "appearance": appearance_plan,
        "tableware": tableware_plan,
        "planned_tableware_footprint_check": footprint_check,
        "runtime_evidence_required": [
            "canonical session restore",
            "Task75 activation and accepted episode pose",
            "all-active-body overlap audit",
            "measured collision-free frame zero",
        ],
    }
    rollout_dir = (
        Path(output_root) / "benchmark_1_1"
        / f"rollout_{int(rollout_index):06d}"
    )
    plan_path = rollout_dir / "initializer_plan.json"
    _atomic_write(plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {
        "schema": "real2sim.kitchen_initializer_plan.pending.v2",
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "registered": True,
        "pipeline_ready_for_materialization": True,
        "runtime_materialized": False,
        "schedulable": False,
        "decision": "requires_runtime_materialization",
        "initializer_plan": str(plan_path),
        "planned_tableware_footprint_check": footprint_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="1.1")
    parser.add_argument("--rollout-index", type=int, required=True)
    args = parser.parse_args()
    result = prepare_rollout(
        args.task,
        rollout_index=args.rollout_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

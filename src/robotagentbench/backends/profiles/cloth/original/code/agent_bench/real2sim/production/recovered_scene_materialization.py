#!/usr/bin/env python3
"""Materialize and measure recovered living-room rollout plans.

This module is deliberately split into a pure validator and a USD adapter.  The
pure part is exercised in the normal test environment; the adapter is run with
the repository's ``mesh311`` environment after an initializer has staged the
target and before simulator playback starts.

The audit is not a planner's estimate.  It reads the exclusively claimed
canonical working scene, measures the target and every active collision body,
and compares structure transforms with the immutable baseline.  A caller may
start physics only after this program has written ``measured_pass``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .house_paths import (
        RUNTIME_BASELINE, RUNTIME_ROOT, SOURCE_BASELINE, SOURCE_ROOT,
        WORKING_SCENE,
    )
except ImportError:
    from house_paths import (
        RUNTIME_BASELINE, RUNTIME_ROOT, SOURCE_BASELINE, SOURCE_ROOT,
        WORKING_SCENE,
    )

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_DEMO = SOURCE_ROOT
DEMO = RUNTIME_ROOT
CANONICAL_SCENE = WORKING_SCENE
CANONICAL_BASELINE = RUNTIME_BASELINE
SESSION_SIDECAR = DEMO / "scene_room.usda.session.json"
PLAN_SCHEMA = "real2sim.recovered_initializer_plan.v2"
AUDIT_SCHEMA = "real2sim.recovered_rollout_randomization.applied.v1"
COLLISION_SCHEMA = "real2sim.recovered_initial_collision_audit.v1"
RECEIPT_SCHEMA = "real2sim.recovered_initializer_receipt.v1"

# Exact canonical identities.  A benchmark label is never used as a robot
# small-task ID and an initializer is not permitted to substitute another
# object's prim merely because the geometry looks similar.
RUNTIME_BINDINGS: dict[str, dict[str, Any]] = {
    "3.1": {
        "target_prim": "/World/Assets/ShelfItems/cayman_jacked",
        "support_prim": "/World/Assets/Tier3Overlay",
        "intentional_collision_roots": ["/World/Assets/ShelvingUnit"],
        "initial_relation": "on_support",
        "support_surface_z_m": 1.1825,
    },
    "3.7": {
        "target_prim": "/World/Assets/ShelfItems/starbucks",
        "support_prim": "/World/Assets/Tier3Overlay",
        "intentional_collision_roots": ["/World/Assets/ShelvingUnit"],
        "initial_relation": "on_support",
        "support_surface_z_m": 1.1825,
    },
    "3.11": {
        "target_prim": "/World/Assets/ShelfItems/grapefruit_can",
        "support_prim": "/World/Assets/Tier3Overlay",
        "intentional_collision_roots": ["/World/Assets/ShelvingUnit"],
        "initial_relation": "on_support",
        "support_surface_z_m": 1.1825,
    },
    "3.24": {
        "target_prim": "/World/Assets/Tier2Items/oreo_gf",
        "support_prim": "/World/Assets/CoffeeTable",
    },
    "4.4": {
        "target_prim": "/World/Assets/GlassesCase",
        "support_prim": "/World/Assets/CoffeeTable/black_box_1",
        "intentional_collision_roots": ["/World/Assets/CoffeeTable"],
        "initial_relation": "inside_support",
    },
    "4.10": {
        "target_prim": "/World/Assets/Wallet",
        "support_prim": "/World/Assets/CoffeeTable/black_box_1",
        "intentional_collision_roots": ["/World/Assets/CoffeeTable"],
        "initial_relation": "inside_support",
    },
    "4.6": {
        "target_prim": "/World/Assets/GlassesCase",
        "episode_destination_required": True,
        "allowed_destination_modes": [
            "specified_surface", "user_handoff",
        ],
    },
    "4.12": {
        "target_prim": "/World/Assets/Wallet",
        "episode_destination_required": True,
        "allowed_destination_modes": [
            "specified_surface", "user_handoff",
        ],
    },
    "4.16": {
        "target_prim": "/World/Assets/Remote",
        "support_prim": "/World/Assets/Tier2Overlay",
        "intentional_collision_roots": ["/World/Assets/ShelvingUnit"],
        "initial_relation": "on_support",
        "support_surface_z_m": 0.751,
    },
    "4.17": {
        "target_prim": "/World/Assets/Remote",
        "support_prim": "/World/Assets/TvTable",
    },
    "5.3": {
        "target_prim": "/World/Assets/Task5Foodbox",
        "payload_type": "salad_box",
        "asset_selection": "user-selected foodbox_burger_v1/foodbox_task108_physics_v2.usda",
        "support_prim": "/World/Background/Shell/CounterTop_17",
        "initial_relation": "on_support",
        "support_surface_z_m": 0.92,
    },
    "5.4": {
        "target_prim": "/World/Assets/Task5Foodbox",
        "payload_type": "salad_box",
        "asset_selection": "user-selected foodbox_burger_v1/foodbox_task108_physics_v2.usda",
        "support_prim": "/World/Task73/Tote",
        "activation_prims": [
            "/World/Assets/Task5Foodbox", "/World/Task73",
        ],
    },
    "5.5": {
        "target_prim": "/World/Assets/PowerBank",
        "support_prim": "/World/Assets/Sofa",
        "initial_relation": "on_support",
        "support_surface_z_m": 0.452,
    },
    "5.6": {
        "target_prim": "/World/Assets/PowerBank",
        "support_prim": "/World/Task73/Tote",
        "activation_prims": [
            "/World/Assets/PowerBank", "/World/Task73",
        ],
    },
    "5.11": {
        # small Task 73 is one long sofa-pick -> tote-place episode, so its
        # tote and existing contents are inherited context, not distractors.
        "target_prim": "/World/Task73/Bottle",
        "support_prim": "/World/Assets/Sofa",
        "activation_prims": ["/World/Task73"],
        "initial_relation": "on_support",
        "support_surface_z_m": 0.452,
    },
    "8.8": {
        "target_prim": "/World/Assets/OldTablecloth",
        "support_prim": "/World/Assets/TableclothBasket",
        "canonical_target_available": False,
    },
    "8.9": {
        "target_prim": "/World/Assets/CleanTablecloth",
        "support_prim": "/World/Assets/TableclothBasket",
        "canonical_target_available": False,
    },
}

IN_HAND_ACTIONS = frozenset({"surface_place", "tote_place"})


def binding(
    benchmark_subtask: str,
    episode_destination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve scene identities, including an episode-selected destination.

    Tasks 4.6 and 4.12 intentionally have no default destination: the Sheet
    says ``user / specified location``.  A specified surface is executable by
    the current physical-place adapter once the episode supplies its live prim,
    layout frame and robot station.  User handoff remains a valid task branch,
    but is reported as unsupported until a user-handoff checker is delivered.
    """
    task = str(benchmark_subtask)
    if task not in RUNTIME_BINDINGS:
        raise KeyError(f"no recovered runtime binding for benchmark {task}")
    result = dict(RUNTIME_BINDINGS[task])
    result.setdefault("canonical_target_available", True)
    result.setdefault("episode_destination_required", False)
    if result["episode_destination_required"]:
        result.update({
            "episode_destination": None,
            "destination_selected": False,
            "runtime_supported": False,
            "support_prim": "",
            "intentional_collision_roots": [],
            "initial_relation": "staged_for_inhand",
            "activation_prims": [result["target_prim"]],
        })
        if episode_destination is None:
            return result
        if not isinstance(episode_destination, Mapping):
            raise ValueError("episode_destination must be an object")
        mode = str(episode_destination.get("mode", ""))
        allowed = tuple(map(str, result["allowed_destination_modes"]))
        if mode not in allowed:
            raise ValueError(
                "episode_destination.mode must be one of "
                + ", ".join(allowed)
            )
        if mode == "user_handoff":
            result.update({
                "episode_destination": {"mode": mode},
                "destination_selected": True,
                "unsupported_reason": (
                    "user_handoff runtime/checker component is not delivered"
                ),
            })
            return result
        selected = {
            key: str(episode_destination.get(key, "")).strip()
            for key in ("prim_path", "layout_frame", "robot_station")
        }
        missing = [key for key, value in selected.items() if not value]
        if missing:
            raise ValueError(
                "specified_surface destination is missing: "
                + ", ".join(missing)
            )
        if not selected["prim_path"].startswith("/World/"):
            raise ValueError(
                "specified_surface prim_path must be below /World"
            )
        destination = {"mode": mode, **selected}
        result.update({
            "episode_destination": destination,
            "destination_selected": True,
            "runtime_supported": True,
            "support_prim": selected["prim_path"],
            "intentional_collision_roots": [selected["prim_path"]],
            "robot_station": selected["robot_station"],
            "layout_frame": selected["layout_frame"],
        })
        return result
    if episode_destination is not None:
        raise ValueError(
            f"benchmark {task} does not accept an episode destination"
        )
    result["runtime_supported"] = True
    result["intentional_collision_roots"] = list(result.get(
        "intentional_collision_roots", [result["support_prim"]]
    ))
    result.setdefault("initial_relation", "staged_for_inhand")
    result["activation_prims"] = list(result.get(
        "activation_prims", [result["target_prim"]]
    ))
    return result


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported recovered initializer plan schema")
    task = str(plan.get("benchmark_subtask", ""))
    expected = binding(task, plan.get("episode_destination"))
    if expected.get("runtime_supported") is not True:
        raise ValueError("plan has no supported episode destination")
    if plan.get("episode_destination") != expected.get(
        "episode_destination"
    ):
        raise ValueError("plan episode destination is not normalized")
    for key in ("target_prim", "support_prim"):
        if str(plan.get(key, "")) != expected[key]:
            raise ValueError(f"{key} disagrees with benchmark {task} binding")
    if list(map(str, plan.get("intentional_collision_roots", ()))) != expected[
        "intentional_collision_roots"
    ]:
        raise ValueError(
            "intentional_collision_roots disagree with the canonical binding"
        )
    if list(map(str, plan.get("activation_prims", ()))) != expected[
        "activation_prims"
    ]:
        raise ValueError("activation_prims disagree with the canonical binding")
    if Path(str(plan.get("restore_source", ""))).resolve() != (
        CANONICAL_BASELINE.resolve()
    ):
        raise ValueError("plan does not restore the canonical House baseline")
    if Path(str(plan.get("working_scene", ""))).resolve() != (
        CANONICAL_SCENE.resolve()
    ):
        raise ValueError("plan does not use the canonical House working scene")
    if Path(str(plan.get("source_baseline", ""))).resolve() != (
        SOURCE_BASELINE.resolve()
    ):
        raise ValueError("plan source baseline does not match House authority")
    if plan.get("furniture_transforms_changed") is not False:
        raise ValueError("plan may not move task-local furniture")
    station = plan.get("robot_station_pose")
    if not isinstance(station, Mapping):
        raise ValueError("plan has no robot_station_pose")
    xy = station.get("xy_m")
    yaw = station.get("yaw_deg")
    if (
        not isinstance(xy, Sequence) or len(xy) != 2
        or not all(math.isfinite(float(value)) for value in xy)
        or not math.isfinite(float(yaw))
        or str(station.get("key", "")) != str(plan.get("robot_station", ""))
    ):
        raise ValueError("plan robot_station_pose is malformed")
    return expected


def _bounds(value: Any, *, name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} bounds are missing")
    minimum = value.get("min_m")
    maximum = value.get("max_m")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
        raise ValueError(f"{name} bounds are malformed")
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError(f"{name} bounds must have three axes")
    lo = tuple(float(number) for number in minimum)
    hi = tuple(float(number) for number in maximum)
    if not all(math.isfinite(number) for number in (*lo, *hi)):
        raise ValueError(f"{name} bounds must be finite")
    if any(high <= low for low, high in zip(lo, hi)):
        raise ValueError(f"{name} bounds must have positive extent")
    return lo, hi


def _penetration_m(
    first: tuple[tuple[float, ...], tuple[float, ...]],
    second: tuple[tuple[float, ...], tuple[float, ...]],
) -> tuple[float, float, float]:
    return tuple(
        min(first[1][axis], second[1][axis])
        - max(first[0][axis], second[0][axis])
        for axis in range(3)
    )


def assess_observation(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    penetration_tolerance_m: float = 0.001,
) -> dict[str, Any]:
    """Turn measured scene facts into a fail-closed rollout audit."""
    expected = validate_plan(plan)
    task = str(plan["benchmark_subtask"])
    errors: list[str] = []
    receipt = observation.get("initializer_receipt")
    if not isinstance(receipt, Mapping):
        errors.append("initializer completion receipt is missing")
        receipt = {}
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("initializer receipt schema mismatch")
    if str(receipt.get("benchmark_subtask", "")) != task:
        errors.append("initializer receipt benchmark identity mismatch")
    if int(receipt.get("rollout_index", -1)) != int(plan["rollout_index"]):
        errors.append("initializer receipt rollout identity mismatch")
    if str(receipt.get("initializer_plan_id", "")) != str(
        observation.get("initializer_plan_id", "")
    ):
        errors.append("initializer receipt plan identity mismatch")
    if str(observation.get("initializer_plan_id", "")) != str(
        plan.get("plan_id", "")
    ):
        errors.append("observation plan identity mismatch")
    initializer = receipt.get("initializer_result")
    if not isinstance(initializer, Mapping):
        errors.append("initializer receipt has no result")
        initializer = {}
    if initializer.get("ok") is not True:
        errors.append("initializer did not report completion")
    if str(initializer.get("track_prim", "")) != expected["target_prim"]:
        errors.append("initializer staged the wrong target prim")
    # A legacy recording label may describe the initializer's input, but the
    # benchmark task and its measured scene outcome are the authority here.
    # Robot base/yaw are deliberately excluded from scene-materialization
    # acceptance.  Task-owned producers perturb them after session restore;
    # their captured state/action and task physics own that evidence.  This
    # audit remains responsible for scene identity, activation, furniture and
    # measured body conflicts.
    session = observation.get("session")
    if not isinstance(session, Mapping):
        errors.append("canonical scene has no readable open-session sidecar")
        session = {}
    if session.get("schema") != "real2sim.scene_session.v1":
        errors.append("session schema mismatch")
    if session.get("status") != "open":
        errors.append("canonical scene session is not open")
    if Path(str(session.get("scene", ""))).resolve() != CANONICAL_SCENE.resolve():
        errors.append("session does not own the canonical working scene")
    if session.get("restored_from_baseline") is not True:
        errors.append("session does not prove a baseline restore")
    if Path(str(session.get("baseline", ""))).resolve() != CANONICAL_BASELINE.resolve():
        errors.append("session does not identify the canonical baseline")
    if Path(str(observation.get("baseline", ""))).resolve() != CANONICAL_BASELINE.resolve():
        errors.append("observation does not identify the canonical baseline")
    if Path(str(plan.get("restore_source", ""))).resolve() != CANONICAL_BASELINE.resolve():
        errors.append("plan does not restore the canonical baseline")
    activated = set(map(str, session.get("activated", ())))
    missing_activation = [
        path for path in expected["activation_prims"] if path not in activated
    ]
    if missing_activation:
        errors.append("session omitted activation prims: " + ", ".join(missing_activation))
    if observation.get("target_active") is not True:
        errors.append("target prim is not active after initialization")
    if observation.get("support_active") is not True:
        errors.append("support prim is not active after initialization")
    if observation.get("furniture_transforms_unchanged") is not True:
        errors.append("canonical furniture transforms changed during initialization")

    try:
        target_bounds = _bounds(observation.get("target_bounds"), name="target")
        _bounds(observation.get("support_bounds"), name="support")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        target_bounds = None

    bodies = observation.get("active_collision_bodies")
    conflicts: list[dict[str, Any]] = []
    measured_body_count = 0
    if not isinstance(bodies, list):
        errors.append("active collision-body measurement is missing")
        bodies = []
    target_path = expected["target_prim"]
    support_path = expected["support_prim"]
    intentional_roots = tuple(expected["intentional_collision_roots"])
    relation = expected["initial_relation"]
    support_relation_passed = False
    if target_bounds is not None:
        try:
            support_bounds = _bounds(
                observation.get("support_bounds"), name="support"
            )
            if relation == "on_support":
                margin_m = 0.04
                footprint_inside = all(
                    target_bounds[0][axis] >= support_bounds[0][axis] - margin_m
                    and target_bounds[1][axis] <= support_bounds[1][axis] + margin_m
                    for axis in (0, 1)
                )
                height_error = abs(
                    target_bounds[0][2]
                    - float(expected["support_surface_z_m"])
                )
                support_relation_passed = footprint_inside and height_error <= 0.03
            elif relation == "inside_support":
                margin_m = 0.03
                support_relation_passed = all(
                    target_bounds[0][axis] >= support_bounds[0][axis] - margin_m
                    and target_bounds[1][axis] <= support_bounds[1][axis] + margin_m
                    for axis in range(3)
                )
            else:
                support_relation_passed = True
        except (TypeError, ValueError, KeyError):
            support_relation_passed = False
    if not support_relation_passed:
        errors.append(
            f"target does not satisfy measured initial relation {relation}"
        )
    for body in bodies:
        if not isinstance(body, Mapping):
            errors.append("collision-body row is malformed")
            continue
        path = str(body.get("prim_path", ""))
        if not path:
            errors.append("collision-body row has no prim_path")
            continue
        if (
            path == target_path or path.startswith(target_path + "/")
            or any(
                path == root or path.startswith(root + "/")
                for root in intentional_roots
            )
        ):
            continue
        try:
            other_bounds = _bounds(body.get("bounds"), name=path)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        measured_body_count += 1
        if target_bounds is None:
            continue
        penetration = _penetration_m(target_bounds, other_bounds)
        if all(value > penetration_tolerance_m for value in penetration):
            conflicts.append({
                "target_prim": target_path,
                "other_prim": path,
                "axis_penetration_m": list(penetration),
            })
    if measured_body_count == 0:
        errors.append("no non-target active collision bodies were measured")
    if conflicts:
        errors.append("target overlaps active collision bodies")

    collision_free = not errors and not conflicts
    return {
        "schema": AUDIT_SCHEMA,
        "benchmark_subtask": task,
        "rollout_index": int(plan["rollout_index"]),
        "episode_destination": expected.get("episode_destination"),
        "scene_materialized": True,
        "fresh_scene_per_rollout": bool(
            session.get("restored_from_baseline") is True
        ),
        "collision_free": collision_free,
        "collision_audit": {
            "schema": COLLISION_SCHEMA,
            "status": "measured_pass" if collision_free else "measured_fail",
            "method": "USD world AABB target versus every active CollisionAPI body",
            "penetration_tolerance_m": penetration_tolerance_m,
            "measured_non_target_body_count": measured_body_count,
            "intentional_support_prim": support_path,
            "intentional_collision_roots": list(intentional_roots),
            "initial_relation": relation,
            "initial_relation_passed": support_relation_passed,
            "conflicts": conflicts,
            "errors": errors,
        },
        "furniture_transforms_changed": observation.get(
            "furniture_transforms_unchanged"
        ) is not True,
        "camera_randomized": False,
        "object_pose_writes_during_rollout": False,
        "initializer_plan": str(plan.get("plan_path", "")),
        "canonical_scene": str(CANONICAL_SCENE),
        "canonical_baseline": str(observation.get("baseline", "")),
    }


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _usd_observation(
    plan: Mapping[str, Any], receipt_path: Path,
) -> dict[str, Any]:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("USD audit requires the mesh311 environment") from exc

    expected = validate_plan(plan)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if not SESSION_SIDECAR.is_file():
        session: dict[str, Any] = {}
    else:
        session = json.loads(SESSION_SIDECAR.read_text(encoding="utf-8"))
    working = Usd.Stage.Open(str(CANONICAL_SCENE))
    baseline = Usd.Stage.Open(str(CANONICAL_BASELINE))
    if working is None or baseline is None:
        raise RuntimeError("could not open canonical working scene and baseline")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )

    def bounds_for(prim: Any) -> dict[str, list[float]]:
        cache.Clear()
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return {
            "min_m": [float(number) for number in value.GetMin()],
            "max_m": [float(number) for number in value.GetMax()],
        }

    target = working.GetPrimAtPath(expected["target_prim"])
    support = working.GetPrimAtPath(expected["support_prim"])
    target_active = bool(target and target.IsValid() and target.IsActive())
    support_active = bool(support and support.IsValid() and support.IsActive())
    target_bounds = bounds_for(target) if target_active else None
    support_bounds = bounds_for(support) if support_active else None
    robot = working.GetPrimAtPath("/World/Robot")
    robot_matrix = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(robot)
    robot_translation = robot_matrix.ExtractTranslation()
    robot_quaternion = Gf.Transform(robot_matrix).GetRotation().GetQuat()
    robot_imaginary = robot_quaternion.GetImaginary()
    robot_yaw_deg = math.degrees(math.atan2(
        2.0 * (
            float(robot_quaternion.GetReal()) * float(robot_imaginary[2])
            + float(robot_imaginary[0]) * float(robot_imaginary[1])
        ),
        1.0 - 2.0 * (
            float(robot_imaginary[1]) ** 2
            + float(robot_imaginary[2]) ** 2
        ),
    ))

    bodies = []
    seen = set()
    for prim in working.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Boundable):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if path in seen:
            continue
        seen.add(path)
        try:
            measured = bounds_for(prim)
            _bounds(measured, name=path)
        except (RuntimeError, TypeError, ValueError):
            continue
        bodies.append({"prim_path": path, "bounds": measured})

    # Compare every active structure transform.  Session-object activation and
    # target placement are expected differences; their top-level roots are the
    # only exclusions.  This is a transform audit, not a byte comparison, so
    # harmless active opinions do not masquerade as furniture movement.
    manifest_path = HERE.parent / "teleop" / "scene_baseline_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform_exclusions = set(map(
        str, manifest.get("session_objects_inactive", ())
    ))
    transform_exclusions.update(expected["activation_prims"])
    transform_exclusions.add("/World/Robot")
    excluded = tuple(path + "/" for path in transform_exclusions)
    furniture_changes = []
    xform_cache_working = UsdGeom.XformCache(Usd.TimeCode.Default())
    xform_cache_baseline = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in baseline.Traverse():
        path = str(prim.GetPath())
        if path in transform_exclusions or path.startswith(excluded):
            continue
        current = working.GetPrimAtPath(path)
        if not current or not current.IsValid() or not prim.IsActive() or not current.IsActive():
            continue
        if not prim.IsA(UsdGeom.Xformable) or not current.IsA(UsdGeom.Xformable):
            continue
        left = _matrix_values(xform_cache_baseline.GetLocalToWorldTransform(prim))
        right = _matrix_values(xform_cache_working.GetLocalToWorldTransform(current))
        if any(abs(a - b) > 1e-9 for a, b in zip(left, right)):
            furniture_changes.append(path)

    return {
        "initializer_receipt": receipt,
        "initializer_plan_id": str(plan.get("plan_id", "")),
        "session": session,
        "baseline": str(CANONICAL_BASELINE),
        "target_active": target_active,
        "support_active": support_active,
        "target_bounds": target_bounds,
        "support_bounds": support_bounds,
        "robot_base_xy_m": [
            float(robot_translation[0]), float(robot_translation[1]),
        ],
        "robot_yaw_deg": robot_yaw_deg,
        "active_collision_bodies": bodies,
        "furniture_transforms_unchanged": not furniture_changes,
        "furniture_transform_changes": furniture_changes,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize_and_audit(
    plan_path: Path, receipt_path: Path, audit_path: Path,
) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan["plan_path"] = str(Path(plan_path).resolve())
    observation = _usd_observation(plan, Path(receipt_path))
    audit = assess_observation(plan, observation)
    audit["observation"] = observation
    _atomic_json(Path(audit_path), audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit = materialize_and_audit(args.plan, args.receipt, args.audit)
    except Exception as exc:  # fail closed but keep a concise CLI error
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": bool(audit["collision_free"]),
        "audit": str(args.audit.resolve()),
        "status": audit["collision_audit"]["status"],
        "conflicts": len(audit["collision_audit"]["conflicts"]),
    }, ensure_ascii=False))
    return 0 if audit["collision_free"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

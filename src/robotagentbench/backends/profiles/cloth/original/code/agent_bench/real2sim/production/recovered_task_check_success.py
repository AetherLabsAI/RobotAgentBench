#!/usr/bin/env python3
"""Strict standardized checkers for L2-recovered personal-item tasks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SCHEMA = "real2sim.recovered_task_metrics.v1"
RANDOMIZATION_SCHEMA = "real2sim.recovered_rollout_randomization.applied.v1"
TASKS = {
    "3.1": ("pick", "alcoholic_drink", ("85",)),
    "3.7": ("pick", "caffeinated_drink", ("48",)),
    "3.11": ("pick", "sparkling_water", ("50",)),
    "3.24": ("surface_place", "oreo_pack", ("45",)),
    "4.4": ("drawer_pick", "glasses_case", ()),
    "4.6": ("surface_place", "glasses_case", ()),
    "4.10": ("drawer_pick", "wallet", ("92",)),
    "4.12": ("surface_place", "wallet", ()),
    "4.16": ("pick", "remote", ()),
    "4.17": ("surface_place", "remote", ("102",)),
    "5.3": ("pick", "salad_box", ("107",)),
    "5.4": ("tote_place", "salad_box", ("99",)),
    "5.5": ("pick", "power_bank", ("111",)),
    "5.6": ("tote_place", "power_bank", ("103",)),
    "5.11": ("pick", "bottle", ("73",)),
    "8.8": ("cloth_place", "old_tablecloth", ()),
    "8.9": ("cloth_pick", "clean_tablecloth", ()),
}
TARGET_IDENTITY_ALIASES = {
    "salad_box": frozenset({"salad_box", "lunchbox"}),
}
EPISODE_DESTINATION_TASKS = frozenset({"4.6", "4.12"})


def _scalar(metrics: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(metrics.get(key))
    if value.shape != ():
        raise ValueError(f"{key} must be scalar")
    return value.item()


def _float(metrics: Mapping[str, Any], key: str) -> float:
    value = float(_scalar(metrics, key))
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _bool(metrics: Mapping[str, Any], key: str) -> bool:
    value = _scalar(metrics, key)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be an explicit boolean")
    return bool(value)


def _common(
    benchmark_subtask: str,
    metrics: Mapping[str, Any],
    randomization_audit: Mapping[str, Any] | None,
    semantic_only: bool = False,
) -> tuple[list[str], tuple[str, str, tuple[str, ...]] | None]:
    task = str(benchmark_subtask)
    contract = TASKS.get(task)
    if contract is None:
        return [f"unknown recovered task {task}"], None
    _, target, small_ids = contract
    errors = []
    for key, expected in (
        ("schema", SCHEMA),
        ("benchmark_subtask", task),
        ("coordinate_frame", "USD world metres Z-up"),
    ):
        try:
            if str(_scalar(metrics, key)) != expected:
                errors.append(f"{key} identity mismatch")
        except ValueError as exc:
            errors.append(str(exc))
    try:
        identity = str(_scalar(metrics, "target_identity"))
        accepted = TARGET_IDENTITY_ALIASES.get(target, frozenset({target}))
        if identity not in accepted:
            errors.append("target_identity identity mismatch")
    except ValueError as exc:
        errors.append(str(exc))
    # Real-recording labels are optional provenance.  Validate one when a
    # runtime reports it, but never require synthetic code-control to pretend
    # it came from a donor recording.
    if small_ids and "small_task_id" in metrics:
        try:
            if str(_scalar(metrics, "small_task_id")) not in small_ids:
                errors.append("small_task_id identity mismatch")
        except ValueError as exc:
            errors.append(str(exc))
    if not semantic_only:
        for key in ("runtime_source_revision", "scene_path", "target_asset_path"):
            try:
                if not str(_scalar(metrics, key)).strip():
                    errors.append(f"{key} is empty")
            except ValueError as exc:
                errors.append(str(exc))
    boolean_true_fields = ["contact_available"]
    if not semantic_only:
        boolean_true_fields.append("trusted_identity_bound")
    for key in boolean_true_fields:
        try:
            if not _bool(metrics, key):
                errors.append(f"{key} is not explicit true")
        except ValueError as exc:
            errors.append(str(exc))
    for key in (
        "attachment_used", "kinematic_target", "trajectory_pose_writes",
    ):
        try:
            if _bool(metrics, key):
                errors.append(f"forbidden physical shortcut: {key}")
        except ValueError as exc:
            errors.append(str(exc))
    if semantic_only:
        pass
    elif not isinstance(randomization_audit, Mapping):
        errors.append("missing rollout randomization audit")
    else:
        if randomization_audit.get("schema") != RANDOMIZATION_SCHEMA:
            errors.append("randomization schema mismatch")
        if str(randomization_audit.get("benchmark_subtask")) != task:
            errors.append("randomization task identity mismatch")
        if randomization_audit.get("scene_materialized") is not True:
            errors.append("initializer plan was not materialized")
        if randomization_audit.get("fresh_scene_per_rollout") is not True:
            errors.append("rollout did not restore a fresh scene")
        if randomization_audit.get("collision_free") is not True:
            errors.append("initial layout is not collision-free")
        collision = randomization_audit.get("collision_audit")
        if not isinstance(collision, Mapping):
            errors.append("missing measured collision audit")
        else:
            if collision.get("status") != "measured_pass":
                errors.append("collision audit did not measure a pass")
            conflicts = collision.get("conflicts")
            if not isinstance(conflicts, list) or conflicts:
                errors.append("collision audit has conflicts or no conflict list")
            measured = collision.get("measured_non_target_body_count")
            if isinstance(measured, bool) or not isinstance(measured, int) or measured < 1:
                errors.append("collision audit measured no active scene bodies")
        if randomization_audit.get("furniture_transforms_changed") is not False:
            errors.append("task-local furniture movement is forbidden")
        if randomization_audit.get("object_pose_writes_during_rollout") is not False:
            errors.append("rollout object-pose writes are forbidden")
    return errors, contract


def _finish(
    task: str,
    errors: list[str],
    gates: Mapping[str, bool],
    margins: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = list(errors)
    reasons.extend(
        f"failed strict gate: {key}"
        for key, passed in gates.items() if not passed
    )
    return {
        "success": not reasons,
        "reasons": reasons,
        "gates": dict(gates),
        "margins": {
            "benchmark_subtask": task,
            "coordinate_frame": "USD world metres Z-up",
            **dict(margins),
        },
    }


def check_task(
    benchmark_subtask: str,
    metrics: Mapping[str, Any],
    *,
    randomization_audit: Mapping[str, Any] | None,
    semantic_only: bool = False,
) -> dict[str, Any]:
    task = str(benchmark_subtask)
    errors, contract = _common(
        task, metrics, randomization_audit, semantic_only=semantic_only
    )
    if contract is None:
        return _finish(task, errors, {}, {})
    action, _, _ = contract
    numbers: dict[str, float] = {}
    for key in (
        "bilateral_contact_fraction",
        "minimum_pad_force_n",
        "target_displacement_m",
        "maximum_one_frame_displacement_m",
        "final_target_speed_mps",
    ):
        try:
            numbers[key] = _float(metrics, key)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    gates = {
        "sustained_bilateral_contact":
            numbers.get("bilateral_contact_fraction", -1) >= 0.90,
        "physical_pad_force":
            numbers.get("minimum_pad_force_n", -1) >= 3.0,
        "target_moved":
            numbers.get("target_displacement_m", -1) >= 0.05,
        "no_single_frame_teleport":
            numbers.get("maximum_one_frame_displacement_m", 1e9) <= 0.12,
        "target_stable":
            numbers.get("final_target_speed_mps", 1e9) <= 0.20,
    }
    if action in {"pick", "drawer_pick", "drawer_pick_close", "cloth_pick"}:
        for key in ("target_lift_m",):
            try:
                numbers[key] = _float(metrics, key)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        try:
            held = _bool(metrics, "target_held_at_tail")
        except ValueError as exc:
            errors.append(str(exc))
            held = False
        gates.update({
            "target_lifted": numbers.get("target_lift_m", -1) >= 0.05,
            "target_held_at_tail": held,
        })
    if action in {"drawer_pick", "drawer_pick_close"}:
        try:
            drawer_joint = _float(metrics, "drawer_joint_displacement_m")
            initial_gap = _float(metrics, "drawer_initial_gap_m")
            maximum_gap = _float(metrics, "drawer_maximum_gap_m")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            drawer_joint, initial_gap, maximum_gap = -1, 1e9, -1
        try:
            drawer_physical = _bool(metrics, "drawer_moved_by_contact")
        except ValueError as exc:
            errors.append(str(exc))
            drawer_physical = False
        numbers.update({
            "drawer_joint_displacement_m": drawer_joint,
            "drawer_initial_gap_m": initial_gap,
            "drawer_maximum_gap_m": maximum_gap,
        })
        gates.update({
            "drawer_initially_closed": initial_gap <= 0.010,
            "drawer_opened": drawer_joint >= 0.10 and maximum_gap >= 0.10,
            "drawer_moved_physically": drawer_physical,
        })
    if action == "drawer_pick_close":
        try:
            final_gap = _float(metrics, "drawer_final_gap_m")
            closed_after_pick = _bool(
                metrics, "drawer_closed_after_target_grasp")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            final_gap = 1e9
            closed_after_pick = False
        numbers["drawer_final_gap_m"] = final_gap
        gates.update({
            "drawer_closed_after_pick": (
                closed_after_pick and final_gap <= 0.035
            ),
        })
    if action == "tote_place":
        try:
            containment = str(_scalar(metrics, "containment_method"))
            inside = _bool(metrics, "target_inside_live_tote_volume")
            released = _bool(metrics, "gripper_released")
            tote_physical = _bool(metrics, "tote_remained_deformable")
        except ValueError as exc:
            errors.append(str(exc))
            containment, inside, released, tote_physical = "", False, False, False
        if containment != "live_tote_body_local_interior":
            errors.append("tote containment is not live body-relative")
        gates.update({
            "released": released,
            "inside_live_tote": inside,
            "tote_remained_physical": tote_physical,
        })
    if action == "surface_place":
        try:
            containment = str(_scalar(metrics, "containment_method"))
            inside = _bool(metrics, "target_inside_live_destination_footprint")
            supported = _bool(metrics, "target_supported_by_destination")
            released = _bool(metrics, "gripper_released")
        except ValueError as exc:
            errors.append(str(exc))
            containment, inside, supported, released = "", False, False, False
        if containment != "live_destination_body_local_footprint":
            errors.append("destination containment is not live body-relative")
        if task in EPISODE_DESTINATION_TASKS:
            destination = (
                randomization_audit.get("episode_destination")
                if isinstance(randomization_audit, Mapping) else None
            )
            if (
                not isinstance(destination, Mapping)
                or destination.get("mode") != "specified_surface"
                or not str(destination.get("prim_path", ""))
            ):
                errors.append(
                    "measured audit has no selected specified-surface "
                    "destination"
                )
            else:
                try:
                    measured_destination = str(_scalar(
                        metrics, "destination_body_path"
                    ))
                    if measured_destination != str(destination["prim_path"]):
                        errors.append("destination_body_path identity mismatch")
                except ValueError as exc:
                    errors.append(str(exc))
        gates.update({
            "released": released,
            "inside_live_destination": inside,
            "destination_support": supported,
        })
    if action in {"cloth_pick", "cloth_place"}:
        try:
            deformable = _bool(metrics, "deformable_state_available")
            particle_contact = _float(
                metrics, "contacted_particle_fraction")
            particle_motion = _float(
                metrics, "particle_displacement_fraction")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            deformable, particle_contact, particle_motion = False, -1, -1
        gates.update({
            "deformable_state_available": deformable,
            "cloth_particles_contacted": particle_contact >= 0.05,
            "cloth_particles_moved": particle_motion >= 0.50,
        })
        if action == "cloth_place":
            try:
                in_zone = _float(
                    metrics, "final_particle_fraction_in_live_destination")
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
                in_zone = -1
            gates["cloth_in_destination"] = in_zone >= 0.90
    return _finish(task, errors, gates, numbers)

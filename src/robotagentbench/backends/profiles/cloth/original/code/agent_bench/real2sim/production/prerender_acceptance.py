"""Fail-closed evidence checks before spending GPU time on task videos.

Success is evaluated by the sole task catalog. Perturbation and visual evidence
are independent quality gates, never alternate task semantics.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _vector(value, sizes):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return a if a.ndim == 1 and a.size in sizes and np.isfinite(a).all() else None


def _same(a, b, atol=2e-6):
    return a is not None and b is not None and a.shape == b.shape and bool(
        np.allclose(a, b, rtol=0, atol=atol))


def evaluate_initial_perturbation(core: Path, task_key: str) -> dict:
    """Verify recorded injection against arrays, including preserved trim input.

    Legacy metadata alone cannot prove that an injection actually ran. Both
    recorded initializer deltas and the existing action-noise producer are
    recognized. No universal perturbation amplitude is imposed.
    """
    core = Path(core)
    if task_key == "benchmark_8_10":
        from real2sim.production.task_randomization.benchmark_8_10.inject_robot_initial_state import check_episode_perturbation
        return check_episode_perturbation(core)
    gates, metrics, reasons = {}, {}, []
    try:
        result = _read(core / "result.json")
        evidence = result.get("task_initial_state_randomization", {})
        arm_delta = _vector(evidence.get("robot_arm_initial_state_delta"), (14,))
        base_delta = _vector(evidence.get("robot_base_initial_state_delta"), (2, 3))
        gates["direct_task_evidence"] = evidence.get("task_key") == task_key
        gates["fresh_physics_declared"] = evidence.get("fresh_physics_rollout") is True
        gates["nonzero_arm_delta"] = arm_delta is not None and bool(np.any(arm_delta != 0))
        gates["nonzero_base_delta"] = base_delta is not None and bool(np.any(base_delta != 0))
        if arm_delta is not None:
            metrics["arm_max_abs_rad"] = float(np.max(np.abs(arm_delta)))
        if base_delta is not None:
            metrics["base_xy_norm_m"] = float(np.linalg.norm(base_delta[:2]))

        observed_core = core
        trim = _read(core / "trim.json") if (core / "trim.json").is_file() else None
        if trim:
            observed_core = Path(trim["source_episode_directory"])
            start, end = int(trim["source_start_frame_inclusive"]), int(trim["source_end_frame_exclusive"])
            with np.load(core / "action.npz", allow_pickle=False) as current, np.load(observed_core / "action.npz", allow_pickle=False) as parent:
                n = len(parent["arm"])
                gates["trim_action_matches_parent"] = set(current.files) == set(parent.files) and all(
                    np.array_equal(current[k], parent[k][start:end] if parent[k].ndim and len(parent[k]) == n else parent[k])
                    for k in current.files)
            with np.load(core / "full_state.npz", allow_pickle=False) as current, np.load(observed_core / "full_state.npz", allow_pickle=False) as parent:
                fields = ("robot_joint_position_rad_or_m", "command_arm_rad", "prop_xyz_m", "prop_wxyz")
                gates["trim_state_matches_parent"] = all(
                    k in current and k in parent and np.array_equal(current[k], parent[k][start:end]) for k in fields)
            metrics["observed_untrimmed_source"] = str(observed_core)
            metrics["trim_source_range"] = [start, end]

        with np.load(observed_core / "full_state.npz", allow_pickle=False) as archive:
            state = {k: archive[k] for k in archive.files}
        with np.load(observed_core / "action.npz", allow_pickle=False) as archive:
            action = {k: archive[k] for k in archive.files}
        gates["command_arm_corresponds_to_action"] = _same(
            state.get("command_arm_rad"), action.get("arm"))
        gates["finite_action_and_motion_state"] = all(
            np.isfinite(value).all() for value in action.values()
            if np.issubdtype(value.dtype, np.number)) and all(
                key in state and np.isfinite(state[key]).all()
                for key in ("prop_xyz_m", "prop_wxyz", "robot_joint_position_rad_or_m"))
        names = np.asarray(state["robot_dof_names"]).astype(str).tolist()
        indices = [i for i, name in enumerate(names) if "_arm_l_joint" in name or "_arm_r_joint" in name]
        declared_arm = _vector(evidence.get("applied_robot_arm_initial_state"), (14, len(names)))
        report = result.get("action_randomization", {})
        derivation_path = observed_core / "action_derivation.json"
        if not report and derivation_path.is_file():
            derivation = _read(derivation_path)
            if derivation.get("schema") == "real2sim.family4_drawer_pick_action_derivation.v1":
                report = derivation.get("controller_input", {})
        noise = report.get("initial_arm_noise", {})
        # The family7 producer explicitly records the first captured state;
        # initializer-based producers record the pre-step initialization.
        action_noise = report.get("schema") == "real2sim.family7_action_randomization.v1"
        measured = np.asarray(state["robot_joint_position_rad_or_m"])[0] if action_noise else state.get("robot_initial_joint_position_rad_or_m")
        if measured is not None and declared_arm is not None and len(indices) == 14:
            measured = np.asarray(measured)
            if declared_arm.size == 14:
                measured = measured[indices]
            gates["declared_initial_arm_observed"] = _same(measured, declared_arm)
            metrics["initial_arm_max_error_rad"] = float(np.max(np.abs(measured-declared_arm)))
        else:
            gates["declared_initial_arm_observed"] = False
        declared_base = _vector(evidence.get("applied_robot_base_initial_state"), (2, 3, 7))
        actual_base = _vector(state.get("robot_root_xyz_m"), (3,))
        gates["declared_initial_base_xy_observed"] = declared_base is not None and actual_base is not None and _same(declared_base[:2], actual_base[:2])
        if task_key in {"benchmark_4_3", "benchmark_4_8", "benchmark_4_9"}:
            from real2sim.teleop.robot_positions import robot_position
            from real2sim.production.benchmark_eval_registry import CATALOG
            task_id = task_key.removeprefix("benchmark_").replace("_", ".")
            station = robot_position(CATALOG[task_id]["task"]["runtime"]["environment"]["robot_station"]["key"])
            quat = _vector(state.get("current_robot_orient_wxyz"), (4,))
            requested_yaw = _vector([report.get("base_delta_yaw_rad")], (1,))
            gates["base_xy_and_yaw_observed_at_task_station"] = False
            if quat is not None and requested_yaw is not None and base_delta is not None and base_delta.size == 3 and actual_base is not None:
                w, x, y, z = quat
                yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
                error = (yaw - np.deg2rad(station.yaw_deg) - requested_yaw[0] + np.pi) % (2*np.pi) - np.pi
                gates["base_xy_and_yaw_observed_at_task_station"] = bool(
                    _same(actual_base[:2] - np.asarray(station.xy_m), base_delta[:2])
                    and abs(error) < 2e-6
                    and abs(requested_yaw[0] - base_delta[2]) < 1e-10)
                metrics["base_yaw_readback_error_rad"] = float(abs(error))
        applied = _vector(state.get("task_owned_robot_arm_initial_delta_applied_rad"), (14,))
        requested = _vector(state.get("task_owned_robot_arm_initial_delta_requested_rad"), (14,))
        initializer = bool(np.asarray(state.get("task_owned_robot_arm_initial_delta_enabled", False)).item())
        if initializer:
            applied_matches_request = _same(applied, arm_delta)
            if (task_key, str(np.asarray(state.get("task_id", "")).item())) in {
                ("benchmark_3_29", "38"), ("benchmark_4_3", "405"),
                ("benchmark_4_8", "55"), ("benchmark_4_9", "60"),
            }:
                # These task runtimes settle to the already perturbed arm[0].
                # Its applied delta includes the measured servo residual;
                # unlike a direct state write, it need not equal the request
                # to two microradians. Reconstruct that residual independently
                # in command order and retain the runtime's 2e-4 rad limit.
                command_names = [
                    f"idx{start + joint}_arm_{hand}_joint{joint}"
                    for hand, start in (("l", 20), ("r", 60))
                    for joint in range(1, 8)
                ]
                if all(name in names for name in command_names):
                    initial = np.asarray(state[
                        "robot_initial_joint_position_rad_or_m"])
                    residual = initial[[names.index(name) for name in command_names]] - action["arm"][0]
                    metrics["action_frame_zero_servo_residual_rad"] = float(np.max(np.abs(residual)))
                    applied_matches_request = (
                        applied is not None and requested is not None
                        and bool(np.isfinite(residual).all())
                        and float(np.max(np.abs(residual))) <= 2e-4
                        and _same(applied - requested, residual)
                    )
                else:
                    applied_matches_request = False
            gates["injection_observed"] = (applied is not None and requested is not None
                and arm_delta is not None and bool(np.any(applied != 0))
                and _same(requested, arm_delta, atol=1e-10)
                and applied_matches_request
                and np.array_equal(np.sign(applied), np.sign(requested)))
            metrics["injection_method"] = "recorded_initializer_delta"
        elif action_noise:
            gates["injection_observed"] = noise.get("enabled") is True and _same(_vector(noise.get("delta_rad"), (14,)), arm_delta)
            metrics["injection_method"] = "perturbed_action_with_recorded_initial_state"
        else:
            gates["injection_observed"] = False
            reasons.append("legacy initializer/action injection needs measured evidence; metadata delta alone is insufficient")
        if report.get("base_delta_xy_m") is not None:
            gates["base_request_matches_evidence"] = base_delta is not None and _same(_vector(report["base_delta_xy_m"], (2,)), base_delta[:2])
        else:
            gates["base_request_matches_evidence"] = False
        if action_noise:
            baseline = _vector(report.get("canonical_base_xy_m"), (2,))
            gates["base_shift_observed"] = baseline is not None and actual_base is not None and base_delta is not None and _same(actual_base[:2]-baseline, base_delta[:2])
        elif result.get("observed_robot_base_delta_xy_m") is not None:
            gates["base_shift_observed"] = _same(_vector(result["observed_robot_base_delta_xy_m"], (2,)), base_delta[:2] if base_delta is not None else None)
        else:
            audit_path = observed_core / "randomization_audit.json"
            setup_path = observed_core / "scene_setup.json"
            baseline = planned = None
            if audit_path.is_file() and setup_path.is_file():
                audit, setup = _read(audit_path), _read(setup_path)
                if audit.get("benchmark_subtask") == task_key.removeprefix("benchmark_").replace("_", "."):
                    baseline = _vector(audit.get("initializer", {}).get("robot_station", {}).get("xy_m"), (2,))
                    planned = _vector(setup.get("plan", {}).get("robot", {}).get("base_xy_yaw"), (3,))
            gates["base_shift_observed"] = (baseline is not None and planned is not None
                and actual_base is not None and base_delta is not None
                and _same(planned[:2], actual_base[:2])
                and _same(actual_base[:2]-baseline, base_delta[:2]))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        reasons.append(f"evidence unreadable: {type(exc).__name__}: {exc}")
        gates["evidence_readable"] = False
    reasons += [name for name, passed in gates.items() if not passed]
    return {"status": "passed" if gates and all(gates.values()) else "unverified",
            "gates": gates, "metrics": metrics, "reasons": reasons}


def evaluate_action_state_alignment(core: Path) -> dict:
    """Check recorded motion independently of optional initializer evidence."""
    gates, reasons = {}, []
    try:
        with np.load(Path(core) / "action.npz", allow_pickle=False) as archive:
            action = {key: archive[key] for key in archive.files}
        with np.load(Path(core) / "full_state.npz", allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        arm = action.get("arm")
        gates["nonempty_arm_action"] = (
            arm is not None and arm.ndim == 2 and len(arm) > 0
            and arm.shape[1] == 14)
        gates["command_arm_corresponds_to_action"] = _same(
            state.get("command_arm_rad"), arm)
        gates["finite_action_and_motion_state"] = all(
            np.isfinite(value).all() for value in action.values()
            if np.issubdtype(value.dtype, np.number)) and all(
                key in state and np.isfinite(state[key]).all()
                for key in ("prop_xyz_m", "prop_wxyz", "robot_joint_position_rad_or_m"))
        gates["motion_state_clock_matches_action"] = bool(
            gates["nonempty_arm_action"] and all(
                key in state and state[key].ndim > 0
                and len(state[key]) == len(arm)
                for key in ("prop_xyz_m", "prop_wxyz", "robot_joint_position_rad_or_m")))
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
    reasons.extend(key for key, passed in gates.items() if not passed)
    return {"passed": bool(gates) and all(gates.values()) and not reasons,
            "gates": gates, "reasons": reasons}


def evaluate_core(task_key: str, core: Path, *,
                  require_initial_perturbation: bool = False,
                  require_physical_distractors: bool = True) -> dict:
    try:
        from .benchmark_eval_registry import resolve
        from .task_success import evaluate_task_success
    except ImportError:
        from benchmark_eval_registry import resolve
        from task_success import evaluate_task_success
    task_id = task_key.removeprefix("benchmark_").replace("_", ".")
    if resolve(task_id)["task"]["key"] != task_key:
        raise ValueError("noncanonical task identity")
    semantic = evaluate_task_success(task_id, core)
    alignment = evaluate_action_state_alignment(core)
    perturbation = evaluate_initial_perturbation(core, task_key)
    physical = {"passed": False}
    try:
        from real2sim.teleop.physical_distractors import read_capture
        with np.load(Path(core) / "full_state.npz", allow_pickle=False) as state:
            captured = read_capture(state)
        physical = {"passed": True, "schema": captured["schema"],
                    "objects": len(captured["objects"]),
                    "robot_contact_observed": captured.get("robot_contact_observed", False)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        physical["reason"] = str(exc)

    passed = (semantic.get("status") == "passed" and alignment["passed"]
              and (not require_physical_distractors or physical["passed"])
              and (not require_initial_perturbation
                   or perturbation["status"] == "passed"))
    return {"schema": "real2sim.prerender_core_acceptance.v1", "task_key": task_key,
            "core": str(core), "passed": passed, "semantic": semantic,
            "action_state_alignment": alignment,
            "physical_distractors": physical,
            "physical_distractors_required": require_physical_distractors,
            "initial_perturbation": perturbation,
            "initial_perturbation_required": require_initial_perturbation,
            "owner_release_approved": False}


def evaluate_scene_randomization(*, task_key, seed, palette, scene, appearance,
                                 layout=None) -> dict:
    """Validate evidence authored by the existing randomizers before rendering."""
    app = appearance.get("appearance", {})
    objects = scene.get("objects", [])
    env = app.get("environment", {})
    registry = app.get("randomization_registry", {}).get("components", {})
    from visual_scene_randomization import shelf_count_report
    gates = {
        "scene_applied": scene.get("ok") is True,
        "scene_identity_seed": scene.get("task_key") == task_key and scene.get("seed") == seed,
        "multiple_actual_distractors": isinstance(objects, list) and len(objects) >= 2 and scene.get("applied_count") == len(objects),
        "five_distractors_per_shelf_layer": shelf_count_report(objects)['passed'],
        "captured_bodies_unchanged": scene.get("captured_bodies_unchanged") is True,
        "captured_sweeps_clear": scene.get("captured_body_sweeps_clear") is True,
        "robot_full_motion_sweeps_clear": scene.get("robot_full_motion_sweeps_clear") is True,
        "reasonable_distractor_density": scene.get("density", {}).get("passed") is True,
        "original_action_state_unchanged": scene.get("source_action_state_modified") is False,
        "appearance_applied": appearance.get("ok") is True,
        "palette_and_seed": app.get("style") == palette and app.get("gen_seed") == seed,
        "semantic_objects_untouched": app.get("semantic_prims_untouched") is True,
        "lights_applied": bool(app.get("lights")) and bool(app.get("lighting_scheme")),
        "background_tone_authored": bool(env.get("authored_attribute_paths")) and bool(env.get("floor_texture_asset")),
        "all_registered_components": set(registry) == {"kitchen", "central_table", "livingroom_lamps", "livingroom_plants"},
        "materials_verified": app.get("kitchen_material_application", {}).get("material_bindings_verified") is True,
        "decor_selected_and_authored": bool(env.get("decor_selection", {}).get("plants")) and bool(env.get("decor_selection", {}).get("lamp_replacement")) and bool(env.get("authored_variant_selections")),
    }
    physical = scene.get("additional_props_mode") == "captured_dynamic_physics"
    gates["physical_clutter_capture"] = (
        physical and scene.get("physics_rollout_performed") is True
        and scene.get("resampled_after_physics") is False
        and scene.get("physical_capture", {}).get("schema") == "real2sim.physical_distractors.v1")
    if physical:
        # Real contacts may intersect the old swept volume and move a prop.
        # The owner task checker and recorded physics judge that outcome.
        for name in ("captured_sweeps_clear", "robot_full_motion_sweeps_clear", "reasonable_distractor_density"):
            gates.pop(name)
    for component in registry.values():
        value = app
        for field in str(component.get("plan_path", "")).split("."):
            value = value.get(field) if isinstance(value, dict) else None
        gates["component_plan:" + str(component.get("plan_path"))] = bool(value)
    if layout is not None:
        gates["layout_authored"] = layout.get("ok") is True
    return {"schema": "real2sim.prerender_visual_acceptance.v1",
            "passed": all(gates.values()), "gates": gates,
            "reasons": [key for key, value in gates.items() if not value],
            "physics_validation_claimed": False}

#!/usr/bin/env python3
"""Fail-closed authored and cumulative robot-initial randomization validator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/lingjun/ziche")
LIMIT_XY_M = 0.05
LIMIT_YAW_DEG = 5.0
LIMIT_JOINT_DEG = 4.0
SPREAD_XY_M = 0.02
SPREAD_YAW_DEG = 2.0
SPREAD_ARM_JOINT_DEG = 2.0
PAIR_XY_M = 0.01
PAIR_YAW_DEG = 1.0
PAIR_ARM_RMS_DEG = 1.0
MIN_DIVERSE_SUBSET = 12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def state_path(path: Path) -> Path | None:
    path = path.resolve()
    for candidate in (path / "full_state.npz", path / "physics/full_state.npz"):
        if candidate.is_file():
            return candidate
    return None


def metadata_path(path: Path) -> Path | None:
    for candidate in (path / "inject_meta.json", path / "physics/inject_meta.json"):
        if candidate.is_file():
            return candidate
    return None


def parent_reference(path: Path) -> tuple[Path | None, str | None, Path | None]:
    """Return one explicit parent; never infer a nominal from numeric proximity."""
    meta_path = metadata_path(path)
    candidates = []
    if meta_path:
        meta = read_json(meta_path)
        for key in ("source_reference", "source_episode", "mother_physics_state"):
            value = meta.get(key)
            if value:
                parent = Path(value)
                if parent.name == "full_state.npz":
                    parent = parent.parent
                candidates.append((parent.resolve(), f"{meta_path}:{key}"))
    acceptance_roots = [path]
    if path.name == "physics":
        acceptance_roots.append(path.parent)
    for root in acceptance_roots:
        for candidate in (root / "trajectory_acceptance.json", root / "repair_acceptance.json"):
            if candidate.is_file():
                value = read_json(candidate).get("source_episode")
                if value:
                    candidates.append((Path(value).resolve(), f"{candidate}:source_episode"))
    unique = {parent for parent, _ in candidates}
    if len(unique) > 1:
        detail = "|".join(f"{edge}->{parent}" for parent, edge in candidates)
        return None, "ERROR:ambiguous_parent_references:" + detail, meta_path
    if candidates:
        return candidates[0][0], "|".join(edge for _, edge in candidates), meta_path
    # Planning repairs can retain the original inject_meta verbatim while
    # recording their source only in episode.json. Such a repaired initial
    # state is not the terminal randomized observation of the nominal.
    episode_path = path / "episode.json"
    if episode_path.is_file():
        value = read_json(episode_path).get("source_reference")
        if value:
            parent = Path(value)
            if parent.name == "full_state.npz":
                parent = parent.parent
            return parent.resolve(), f"{episode_path}:source_reference", meta_path
    for candidate in (path / "trajectory_acceptance.json", path / "repair_acceptance.json"):
        if candidate.is_file():
            value = read_json(candidate).get("source_episode")
            if value:
                return Path(value).resolve(), f"{candidate}:source_episode", candidate
    if path.name == "physics":
        return parent_reference(path.parent)
    return None, None, meta_path


def trace_nominal(start: Path) -> dict:
    current = start.resolve()
    chain = []
    seen = set()
    for _ in range(16):
        if current in seen:
            return {"passed": False, "error": "provenance_cycle", "chain": chain}
        seen.add(current)
        state = state_path(current)
        if state is None:
            return {"passed": False, "error": "missing_state:" + str(current), "chain": chain}
        parent, edge, meta = parent_reference(current)
        chain.append({"episode": str(current), "state": str(state), "state_sha256": digest(state),
                      "metadata": str(meta) if meta else None,
                      "metadata_sha256": digest(meta) if meta and meta.is_file() else None,
                      "parent_edge": edge})
        if edge and edge.startswith("ERROR:"):
            return {"passed": False, "error": edge, "chain": chain}
        if parent is None:
            return {"passed": True, "nominal_episode": str(current), "nominal_state": str(state), "chain": chain}
        current = parent
    return {"passed": False, "error": "provenance_depth_exceeded", "chain": chain}


def initial(state: Path) -> dict:
    with np.load(state, allow_pickle=False) as data:
        names = list(map(str, data["robot_dof_names"]))
        joints = np.asarray(data["robot_initial_joint_position_rad_or_m"] if "robot_initial_joint_position_rad_or_m" in data else data["robot_joint_position_rad_or_m"][0], dtype=float)
        xyz = np.asarray(data["robot_root_xyz_m"], dtype=float)
        quat = np.asarray(data["robot_root_wxyz"], dtype=float)
    if joints.shape != (len(names),) or xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[joints, xyz, quat]).all():
        raise ValueError("invalid initial robot state: " + str(state))
    return {"names": names, "joints": joints, "xyz": xyz, "quat": quat}


def arm_indices(names: list[str], side: str) -> list[int]:
    result = [i for i, name in enumerate(names) if f"arm_{side}_joint" in name]
    if len(result) != 7:
        raise ValueError(f"expected seven {side} arm joints, found {result}")
    return result


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def rotation_delta(actual_wxyz, nominal_wxyz) -> dict:
    actual = Rotation.from_quat(np.asarray(actual_wxyz)[[1, 2, 3, 0]])
    nominal = Rotation.from_quat(np.asarray(nominal_wxyz)[[1, 2, 3, 0]])
    a = actual.as_matrix()
    n = nominal.as_matrix()
    yaw = wrap_degrees(math.degrees(math.atan2(a[1, 0], a[0, 0]) - math.atan2(n[1, 0], n[0, 0])))
    relative_xyz = (nominal.inv() * actual).as_euler("xyz", degrees=True)
    return {"world_z_yaw_deg": yaw, "relative_roll_pitch_deg": relative_xyz[:2].tolist(),
            "relative_rotation_angle_deg": float((nominal.inv() * actual).magnitude() * 180 / math.pi)}


def declared(acceptance: dict, source: Path, task: str) -> dict:
    values = acceptance.get("perturbation", {})
    meta = metadata_path(source)
    raw = read_json(meta) if meta else {}
    arm = values.get("arm_delta", values.get("inherited_arm_delta_rad", raw.get("arm_delta", raw.get("arm_delta_rad"))))
    base = values.get("base_delta_xy_m", values.get("inherited_base_delta_xy_m", raw.get("base_delta_xy_m")))
    yaw = values.get("base_yaw_delta_deg", raw.get("base_yaw_delta_deg"))
    yaw_source = "metadata"
    policy_path = ROOT / f"code/family3_repair/real2sim/production/task_randomization/{task}/policy.json"
    if yaw is None and policy_path.is_file():
        policy = read_json(policy_path).get("robot_base_initial_state", {})
        scale = policy.get("experimentally_accepted_scale", {})
        if float(scale.get("base_yaw_deg", float("nan"))) == 0.0 or "yaw deferred (0)" in policy.get("method", "").lower():
            yaw, yaw_source = 0.0, str(policy_path) + ":legacy_yaw_deferred"
    errors = []
    try:
        arm = np.asarray(arm, dtype=float)
        if arm.shape != (14,) or not np.isfinite(arm).all():
            raise ValueError
    except Exception:
        errors.append("missing_or_invalid_declared_14_arm_delta")
        arm = np.full(14, np.nan)
    try:
        base = np.asarray(base, dtype=float)
        if base.shape != (2,) or not np.isfinite(base).all():
            raise ValueError
    except Exception:
        errors.append("missing_or_invalid_declared_base_xy_delta")
        base = np.full(2, np.nan)
    if yaw is None or not math.isfinite(float(yaw)):
        errors.append("missing_or_invalid_declared_base_yaw_delta")
        yaw = float("nan")
    within = bool(not errors and np.max(np.abs(base)) <= LIMIT_XY_M + 1e-9
                  and abs(float(yaw)) <= LIMIT_YAW_DEG + 1e-9
                  and np.max(np.abs(np.degrees(arm))) <= LIMIT_JOINT_DEG + 1e-7)
    return {"passed": within, "errors": errors, "base_xy_m": base.tolist(), "base_yaw_deg": float(yaw),
            "arm_delta_deg": np.degrees(arm).tolist(), "maximum_absolute_arm_joint_deg": float(np.nanmax(np.abs(np.degrees(arm)))) if np.isfinite(arm).any() else None,
            "metadata": str(meta) if meta else None, "yaw_source": yaw_source,
            "metadata_sha256": digest(meta) if meta and meta.is_file() else None,
            "raw_base_change_from_mother_m": raw.get("base_change_from_mother_m"),
            "raw_parent_design_seed": raw.get("parent_design_seed")}


def recover_recorded_nominal(task: str, trace: dict) -> dict:
    """Recover the pre-randomization nominal behind the oldest traced episode.

    The legacy producers store a randomized physical episode as the provenance
    root.  Its inject_meta draw is relative to the task's canonical recorded
    station/arm state, so treating that episode itself as nominal compounds
    later randomization.  Subtracting that oldest declared draw from its actual
    initial state reconstructs the authored recorded nominal without importing
    task implementation modules or guessing from a nearby episode.
    """
    terminal_episode = Path(trace["nominal_episode"])
    terminal_state_path = Path(trace["nominal_state"])
    terminal = initial(terminal_state_path)
    root_draw = declared({}, terminal_episode, task)
    errors = list(root_draw["errors"])
    if not root_draw["passed"]:
        errors.append("oldest_source_draw_does_not_pass_authored_bounds")
    try:
        left = arm_indices(terminal["names"], "l")
        right = arm_indices(terminal["names"], "r")
        indices = left + right
        draw_arm = np.radians(np.asarray(root_draw["arm_delta_deg"], dtype=float))
        nominal_arm = terminal["joints"][indices] - draw_arm
        actual_yaw = math.degrees(math.atan2(
            Rotation.from_quat(terminal["quat"][[1, 2, 3, 0]]).as_matrix()[1, 0],
            Rotation.from_quat(terminal["quat"][[1, 2, 3, 0]]).as_matrix()[0, 0]))
        result = {
            "passed": not errors,
            "errors": errors,
            "method": "oldest traced physical initial state minus its inject_meta declared draw",
            "terminal_randomized_episode": str(terminal_episode),
            "terminal_randomized_state": str(terminal_state_path),
            "terminal_randomized_state_sha256": digest(terminal_state_path),
            "terminal_declared_draw": root_draw,
            "base_xyz_m": [
                float(terminal["xyz"][0] - root_draw["base_xy_m"][0]),
                float(terminal["xyz"][1] - root_draw["base_xy_m"][1]),
                float(terminal["xyz"][2]),
            ],
            "base_yaw_deg": wrap_degrees(actual_yaw - root_draw["base_yaw_deg"]),
            "base_roll_pitch_reference_wxyz": terminal["quat"].tolist(),
            "arm_dof_names": [terminal["names"][i] for i in indices],
            "arm_joint_rad": nominal_arm.tolist(),
        }
        return result
    except Exception as error:
        return {"passed": False, "errors": errors + [f"{type(error).__name__}: {error}"],
                "terminal_randomized_episode": str(terminal_episode),
                "terminal_randomized_state": str(terminal_state_path)}


def assess(task: str, episode: str, actual_state: Path, source: Path, acceptance_path: Path, cohort: str) -> dict:
    acceptance = read_json(acceptance_path)
    trace = trace_nominal(source)
    result = {"task": task, "episode": episode, "cohort": cohort, "acceptance": str(acceptance_path),
              "acceptance_sha256": digest(acceptance_path),
              "actual_state": str(actual_state), "source_episode": str(source), "provenance": trace}
    draw = declared(acceptance, source, task)
    result["declared_draw"] = draw
    if not trace["passed"]:
        result.update(within_bounds=False, classification="unprovable_nominal")
        return result
    try:
        actual = initial(actual_state)
        nominal = recover_recorded_nominal(task, trace)
        result["recorded_nominal"] = nominal
        if not nominal["passed"]:
            raise ValueError("recorded nominal recovery failed: " + ";".join(nominal["errors"]))
        left = arm_indices(actual["names"], "l")
        right = arm_indices(actual["names"], "r")
        names = [actual["names"][i] for i in left + right]
        if names != nominal["arm_dof_names"]:
            raise ValueError("robot arm DOF names mismatch")
        delta_deg = np.degrees(actual["joints"][left + right] - np.asarray(nominal["arm_joint_rad"]))
        actual_rotation = Rotation.from_quat(actual["quat"][[1, 2, 3, 0]])
        actual_yaw = math.degrees(math.atan2(actual_rotation.as_matrix()[1, 0], actual_rotation.as_matrix()[0, 0]))
        terminal = initial(Path(trace["nominal_state"]))
        rotation = rotation_delta(actual["quat"], terminal["quat"])
        cumulative = {"base_xy_m": (actual["xyz"][:2] - np.asarray(nominal["base_xyz_m"][:2])).tolist(),
                      "base_yaw_deg": wrap_degrees(actual_yaw - nominal["base_yaw_deg"]),
                      "base_relative_roll_pitch_deg": rotation["relative_roll_pitch_deg"],
                      "left_arm_delta_deg": delta_deg[:7].tolist(), "right_arm_delta_deg": delta_deg[7:].tolist(),
                      "maximum_absolute_arm_joint_deg": float(np.max(np.abs(delta_deg))),
                      "actual_state_sha256": digest(actual_state),
                      "recorded_nominal_recovery_source_sha256": nominal["terminal_randomized_state_sha256"]}
        declared_arm = np.asarray(draw["arm_delta_deg"], dtype=float)
        cumulative_arm = np.asarray(cumulative["left_arm_delta_deg"] + cumulative["right_arm_delta_deg"], dtype=float)
        cumulative["realization_residual_vs_declared"] = {
            "base_xy_mm": (1000.0 * (np.asarray(cumulative["base_xy_m"]) - np.asarray(draw["base_xy_m"]))).tolist(),
            "base_yaw_deg": wrap_degrees(cumulative["base_yaw_deg"] - draw["base_yaw_deg"]),
            "arm_joint_deg": (cumulative_arm - declared_arm).tolist(),
            "maximum_absolute_arm_joint_deg": float(np.max(np.abs(cumulative_arm - declared_arm))),
        }
        gates = {
            "declared_base_each_world_xy_within_0p05m": bool(np.max(np.abs(draw["base_xy_m"])) <= LIMIT_XY_M + 1e-9),
            "declared_base_world_z_yaw_within_5deg": bool(abs(draw["base_yaw_deg"]) <= LIMIT_YAW_DEG + 1e-9),
            "declared_each_arm_joint_within_4deg": bool(draw["maximum_absolute_arm_joint_deg"] is not None and draw["maximum_absolute_arm_joint_deg"] <= LIMIT_JOINT_DEG + 1e-7),
            "cumulative_base_each_world_xy_within_0p05m": bool(np.max(np.abs(cumulative["base_xy_m"])) <= LIMIT_XY_M + 1e-6),
            "cumulative_base_world_z_yaw_within_5deg": bool(abs(cumulative["base_yaw_deg"]) <= LIMIT_YAW_DEG + 1e-5),
            "base_roll_pitch_unchanged_within_0p1deg": bool(np.max(np.abs(cumulative["base_relative_roll_pitch_deg"])) <= 0.1),
            "cumulative_each_arm_joint_within_4deg": bool(cumulative["maximum_absolute_arm_joint_deg"] <= LIMIT_JOINT_DEG + 1e-4),
        }
        cumulative_pass = all(gates[key] for key in gates if key.startswith("cumulative_") or key.startswith("base_roll"))
        result["cumulative_from_recorded_nominal"] = cumulative
        result["per_episode_gates"] = gates
        result["within_bounds"] = bool(all(gates.values()) and draw["passed"] and cumulative_pass)
        result["failed_gates"] = [key for key, passed in gates.items() if not passed]
        result["classification"] = "within_bounds" if result["within_bounds"] else "bounds_violation"
    except Exception as error:
        result.update(within_bounds=False, classification="unprovable_nominal", error=f"{type(error).__name__}: {error}")
    return result


def separated(a: dict, b: dict) -> bool:
    ac, bc = a["cumulative_from_recorded_nominal"], b["cumulative_from_recorded_nominal"]
    xy = np.linalg.norm(np.asarray(ac["base_xy_m"]) - bc["base_xy_m"])
    yaw = abs(wrap_degrees(ac["base_yaw_deg"] - bc["base_yaw_deg"]))
    aa = np.asarray(ac["left_arm_delta_deg"] + ac["right_arm_delta_deg"])
    ba = np.asarray(bc["left_arm_delta_deg"] + bc["right_arm_delta_deg"])
    arm = float(np.sqrt(np.mean((aa - ba) ** 2)))
    return bool(xy >= PAIR_XY_M or yaw >= PAIR_YAW_DEG or arm >= PAIR_ARM_RMS_DEG)


def clique_at_least(rows: list[dict], target: int) -> list[int]:
    adjacency = [{j for j in range(len(rows)) if j != i and separated(rows[i], rows[j])} for i in range(len(rows))]
    def search(chosen, candidates):
        if len(chosen) >= target:
            return chosen
        if len(chosen) + len(candidates) < target:
            return None
        while candidates:
            vertex = min(candidates)
            candidates.remove(vertex)
            result = search(chosen + [vertex], candidates & adjacency[vertex])
            if result is not None:
                return result
        return None
    return search([], set(range(len(rows)))) or []


def batch_summary(task: str, rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("within_bounds") and "cumulative_from_recorded_nominal" in row]
    if valid:
        xy = np.asarray([row["cumulative_from_recorded_nominal"]["base_xy_m"] for row in valid])
        yaw = np.asarray([row["cumulative_from_recorded_nominal"]["base_yaw_deg"] for row in valid])
        left = np.asarray([row["cumulative_from_recorded_nominal"]["left_arm_delta_deg"] for row in valid])
        right = np.asarray([row["cumulative_from_recorded_nominal"]["right_arm_delta_deg"] for row in valid])
        spans = {"base_x_m": float(np.ptp(xy[:, 0])), "base_y_m": float(np.ptp(xy[:, 1])),
                 "base_yaw_deg": float(np.ptp(yaw)), "left_arm_max_joint_span_deg": float(np.max(np.ptp(left, axis=0))),
                 "right_arm_max_joint_span_deg": float(np.max(np.ptp(right, axis=0)))}
        target = MIN_DIVERSE_SUBSET
        clique = clique_at_least(valid, target) if len(valid) >= target else []
        near_zero = [row for row in valid if (np.linalg.norm(row["cumulative_from_recorded_nominal"]["base_xy_m"]) < .01
                    and abs(row["cumulative_from_recorded_nominal"]["base_yaw_deg"]) < 1.0
                    and np.sqrt(np.mean(np.square(row["cumulative_from_recorded_nominal"]["left_arm_delta_deg"] + row["cumulative_from_recorded_nominal"]["right_arm_delta_deg"]))) < 1.0)]
    else:
        spans, clique, near_zero = {}, [], []
    gates = {"at_least_12_within_bounds": len(valid) >= 12,
             "base_x_span_at_least_20mm": spans.get("base_x_m", 0) >= SPREAD_XY_M,
             "base_y_span_at_least_20mm": spans.get("base_y_m", 0) >= SPREAD_XY_M,
             "base_yaw_span_at_least_2deg": spans.get("base_yaw_deg", 0) >= SPREAD_YAW_DEG,
             "left_arm_joint_span_at_least_2deg": spans.get("left_arm_max_joint_span_deg", 0) >= SPREAD_ARM_JOINT_DEG,
             "right_arm_joint_span_at_least_2deg": spans.get("right_arm_max_joint_span_deg", 0) >= SPREAD_ARM_JOINT_DEG,
             "pairwise_separated_12_case_subset": len(clique) >= 12,
             "near_zero_fraction_at_most_20_percent": len(near_zero) <= .2 * len(valid) if valid else False}
    return {"task": task, "episode_count": len(rows), "within_bounds_count": len(valid),
            "bounds_violation_count": sum(row.get("classification") == "bounds_violation" for row in rows),
            "unprovable_count": sum(row.get("classification") == "unprovable_nominal" for row in rows),
            "spans": spans, "near_zero_count": len(near_zero),
            "pairwise_separated_subset": [valid[i]["episode"] for i in clique], "spread_gates": gates,
            "batch_spread_passed": all(gates.values())}


def published_rows(root: Path):
    for episode in sorted(root.glob("benchmark_*/trajectories/episode_*")):
        acceptance_path = episode / "repair_acceptance.json"
        if not acceptance_path.is_file():
            yield {"task": episode.parents[1].name, "episode": episode.name, "classification": "unprovable_nominal", "error": "missing_repair_acceptance", "cohort": "published"}
            continue
        acceptance = read_json(acceptance_path)
        source = Path(acceptance.get("source_episode", ""))
        actual_state = episode / "full_state.npz"
        yield assess(episode.parents[1].name, episode.name, actual_state, source, acceptance_path, "published")


def accepted_rows(inventory: dict):
    for task in inventory["tasks"]:
        for episode, versions in sorted(task.get("physical_versions", {}).items()):
            acceptance_path = Path(versions[0])
            acceptance = read_json(acceptance_path)
            source = Path(acceptance.get("source_episode", ""))
            case = Path(acceptance.get("case", acceptance_path.parent))
            actual_state = state_path(case)
            if actual_state is None:
                yield {"task": task["task"], "episode": episode, "classification": "unprovable_nominal", "error": "missing_actual_state", "cohort": "accepted_inventory", "acceptance": str(acceptance_path)}
                continue
            row = assess(task["task"], episode, actual_state, source, acceptance_path, "accepted_inventory")
            row["accepted_version_count"] = len(versions)
            row["accepted_versions"] = versions
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, default=ROOT / "outputs/family3_repaired")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inventory", type=Path)
    source.add_argument("--candidate-spec", type=Path,
                        help="JSON with task, episode, actual_state, source_episode, acceptance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true",
                        help="exit nonzero unless every episode and every task spread gate passes")
    args = parser.parse_args()
    if args.candidate_spec:
        spec = read_json(args.candidate_spec)
        required = ("task", "episode", "actual_state", "source_episode", "acceptance")
        missing = [key for key in required if not spec.get(key)]
        if missing:
            report = {"schema": "family3.robot_initial_randomization_candidate_gate.v1",
                      "passed": False, "error": "missing candidate spec fields: " + ",".join(missing),
                      "candidate_spec": str(args.candidate_spec.resolve()),
                      "candidate_spec_sha256": digest(args.candidate_spec)}
        else:
            try:
                row = assess(spec["task"], spec["episode"], Path(spec["actual_state"]),
                             Path(spec["source_episode"]), Path(spec["acceptance"]),
                             spec.get("cohort", "candidate"))
                report = {"schema": "family3.robot_initial_randomization_candidate_gate.v1",
                          "passed": bool(row.get("within_bounds")),
                          "candidate_spec": str(args.candidate_spec.resolve()),
                          "candidate_spec_sha256": digest(args.candidate_spec), "episode": clean(row)}
            except Exception as error:
                report = {"schema": "family3.robot_initial_randomization_candidate_gate.v1",
                          "passed": False, "error": f"{type(error).__name__}: {error}",
                          "candidate_spec": str(args.candidate_spec.resolve()),
                          "candidate_spec_sha256": digest(args.candidate_spec)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "passed": report["passed"]}, indent=2))
        if not report["passed"]:
            sys.exit(2)
        return
    inventory = read_json(args.inventory)
    cohorts = {"published": list(published_rows(args.published_root)),
               "accepted_inventory": list(accepted_rows(inventory))}
    report = {"schema": "family3.robot_initial_randomization_audit.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "bounds": {"base_each_world_xy_m": LIMIT_XY_M, "base_world_z_yaw_deg": LIMIT_YAW_DEG, "each_left_and_right_arm_joint_deg": LIMIT_JOINT_DEG},
              "spread_thresholds": {"minimum_episode_count": 12, "each_world_xy_span_m": SPREAD_XY_M, "world_z_yaw_span_deg": SPREAD_YAW_DEG,
                                    "each_arm_any_joint_span_deg": SPREAD_ARM_JOINT_DEG, "pairwise_separation_any_of": {"base_xy_m": PAIR_XY_M, "yaw_deg": PAIR_YAW_DEG, "arm_rms_deg": PAIR_ARM_RMS_DEG}, "near_zero_fraction_max": .2},
              "semantics": {"translation": "candidate actual initial robot root minus reconstructed pre-randomization recorded nominal in world X/Y", "rotation": "shortest base heading difference about world Z, the table/world up axis", "joints": "candidate actual initial joint positions minus reconstructed pre-randomization recorded nominal, seven joints on each arm", "nominal_recovery": "recursively trace explicit parents; the oldest episode is normally already randomized, so subtract its own inject_meta declared draw from its physical initial state", "declared_vs_cumulative": "both must be independently within bounds; missing provenance or oldest-source draw fails closed", "batch_spread": "computed from actual cumulative deltas and reported separately from per-episode bounds"},
              "inventory": str(args.inventory.resolve()), "inventory_sha256": digest(args.inventory), "cohorts": {}}
    for name, rows in cohorts.items():
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row)
        report["cohorts"][name] = {"episode_count": len(rows),
            "within_bounds_count": sum(row.get("within_bounds", False) for row in rows),
            "bounds_violation_count": sum(row.get("classification") == "bounds_violation" for row in rows),
            "unprovable_count": sum(row.get("classification") == "unprovable_nominal" for row in rows),
            "task_summaries": [batch_summary(task, task_rows) for task, task_rows in sorted(by_task.items())], "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = clean(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {name: {key: value[key] for key in ("episode_count", "within_bounds_count", "bounds_violation_count", "unprovable_count")} for name, value in report["cohorts"].items()}
    print(json.dumps({"output": str(args.output), "cohorts": summary}, indent=2))
    if args.enforce:
        passed = all(cohort["episode_count"] == cohort["within_bounds_count"]
                     and all(task["batch_spread_passed"] for task in cohort["task_summaries"])
                     for cohort in report["cohorts"].values())
        if not passed:
            sys.exit(2)


if __name__ == "__main__":
    main()

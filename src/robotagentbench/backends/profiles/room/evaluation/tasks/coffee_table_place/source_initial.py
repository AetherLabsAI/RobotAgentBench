"""Extract measured L1 frame-zero state, never import the old controller.

These are source candidates, NOT physically accepted resets or runnable API
recordings. Source material/drive overrides and release helpers are reported,
not applied. Missing definitions deliberately have no numerical defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

TASK = Path(__file__).resolve().parent
REFERENCE = TASK / "references/l1_place_20260914/l1"
COFFEE_TABLE = "/World/Assets/CoffeeTable"
SCHEMA = "arena.coffee_table_place.source_initial_candidate.v1"
MEALS = {
    "30-1": ("10.3", "beef_box", "将盒装牛排放到茶几"),
    "30-2": ("10.6", "meal_box", "将盒装剩饭放到茶几"),
    "30-3": ("10.9", "salad_box", "将沙拉饭盒放到茶几"),
}


def catalog():
    index = json.loads((TASK / "apis/api_index.json").read_text())
    if set(index) != {f"5-{i}" for i in range(1, 15)}:
        raise ValueError("Expected exactly the 14 approved API identities")
    rows = {}
    for api, row in index.items():
        rows[api] = dict(
            api_id=api, benchmark=row["bm"], source_runtime_id=row["small"],
            title=row["title"], held_body="object",
            # 5-7：用户 2026-09-23 裁定目标身份为 `coke_c3`（与定义表的实现绑定行、
            # 2026-09-18 新数据以及 4-7 一致）；`prim_newdata` 就是这条裁定的记录。
            source_target=f'/World/Assets/{row["root"]}/{row.get("prim_newdata", row["prim"])}',
            source_destination=row["dest"], destination_family=COFFEE_TABLE,
            definition_status="pending_confirmation", runtime_ready=False,
        )
    for api, (benchmark, payload, title) in MEALS.items():
        rows[api] = dict(
            api_id=api, benchmark=benchmark, source_runtime_id="108",
            title=title, held_body="plate", payload=payload,
            source_destination="table", destination_family=COFFEE_TABLE,
            definition_status="pending_confirmation", runtime_ready=False,
            source_state_status="no_episode_in_l1_task_owned_50x4",
        )
    return rows


def source_directory(api_id, episode="episode_000"):
    row = catalog()[api_id]
    if api_id in MEALS:
        raise ValueError("L1 meal wrapper is a planner, not a measured initial-state episode")
    if not episode.startswith("episode_") or not episode[8:].isdigit():
        raise ValueError("Expected episode_<digits>, not a filesystem path")
    path = REFERENCE / ("benchmark_" + row["benchmark"].replace(".", "_")) / "trajectories" / episode
    if not path.resolve().is_relative_to(REFERENCE.resolve()):
        raise ValueError("Source escaped task-owned references")
    return path


def pose(position, quaternion):
    p, q = list(map(float, position)), list(map(float, quaternion))
    if len(p) != 3 or len(q) != 4 or not all(math.isfinite(v) for v in p + q):
        raise ValueError("Invalid measured pose")
    if abs(sum(v*v for v in q) - 1) > 1e-5:
        raise ValueError("Source quaternion is not unit length")
    return dict(position_m=p, quaternion_wxyz=q)


def extract(api_id, state, action, episode, injection, metrics):
    """Pure conversion, no scene writes, joint clamping, IK, or invented states."""
    import numpy as np
    row = catalog()[api_id]
    target = row.get("source_target")
    if target is None:
        raise ValueError("Meal source reset has not been recovered")
    runtime = row["source_runtime_id"]
    identities = [str(state["task_id"]), str(episode["task_id"]), str(injection["small_task"]), str(metrics["task_id"])]
    if any(value != runtime for value in identities):
        raise ValueError("Source runtime identity mismatch")
    if injection["task_key"] != "benchmark_" + row["benchmark"].replace(".", "_"):
        raise ValueError("Source benchmark identity mismatch")
    paths = state["rigid_body_paths"].tolist()
    if (str(state["target_prim"]) != target or metrics["target_body_path"] != target
            or paths.count(target) != 1 or len(paths) != len(set(paths))):
        raise ValueError("Target body mismatch or duplicate body")
    fps = float(state["fps"])
    if not math.isfinite(fps) or fps <= 0 or fps != float(action["fps"]):
        raise ValueError("Action/state clocks differ")
    frame_count = len(state["frame_index"])
    if frame_count < 1 or not np.array_equal(state["frame_index"], np.arange(frame_count)):
        raise ValueError("Non-contiguous source frame sequence")
    names = state["robot_dof_names"].tolist()
    if len(names) != len(set(names)) or not all(isinstance(n, str) for n in names):
        raise ValueError("Duplicate or invalid joint names")
    shapes = {
        "prop_xyz_m": (frame_count, len(paths), 3),
        "prop_wxyz": (frame_count, len(paths), 4),
        "robot_joint_position_rad_or_m": (frame_count, len(names)),
    }
    for key, shape in shapes.items():
        if state[key].shape != shape or not np.isfinite(state[key]).all():
            raise ValueError("Invalid state array: " + key)
    for key, width in (("arm", 14), ("waist", 2), ("head", 2), ("base_xyyaw", 3), ("lgrip", None), ("rgrip", None)):
        shape = (frame_count,) if width is None else (frame_count, width)
        if action[key].shape != shape or not np.isfinite(action[key]).all():
            raise ValueError("Invalid action array: " + key)
    dest = "/World/Assets/Tray" if row["source_destination"] == "tray" else COFFEE_TABLE
    if metrics["destination_body_path"] != dest:
        raise ValueError("Source destination differs from indexed task")
    if injection["hand"] != metrics["hand"] or metrics["hand"] not in ("l", "r"):
        raise ValueError("Inconsistent holding-hand provenance")
    queue = episode.get("queue_record", {})
    warnings = ["Historical success labels do not validate current native robot/asset physics"]
    source_overrides = queue.get("env", {})
    if source_overrides:
        warnings.append("Source used runtime overrides; none are applied by this extractor")
    if dest.endswith("/Tray") and dest not in paths:
        warnings.append("Source full_state does not include the tray pose; destination reset is incomplete")
    furniture = str(state.get("current_furniture_body_path", ""))
    if furniture != COFFEE_TABLE:
        warnings.append("No measured CoffeeTable frame in this source; do not assume a layout transform")
    idx = paths.index(target)
    return dict(
        schema=SCHEMA, api_id=api_id, benchmark=row["benchmark"],
        source_runtime_id=runtime, source_frame_index=0, source_fps=fps,
        source_frames=frame_count, source_duration_s=frame_count/fps,
        source_episode_uid=str(state["episode_uid"]), target_path=target,
        robot=pose(state["robot_root_xyz_m"], state["robot_root_wxyz"]),
        target=pose(state["prop_xyz_m"][0, idx], state["prop_wxyz"][0, idx]),
        joint_positions=dict(zip(names, map(float, state["robot_joint_position_rad_or_m"][0]))),
        source_hand=metrics["hand"], source_destination=dict(
            body_path=dest, zone=metrics["zone"],
            furniture_body_path=furniture,
            furniture_aabb_min_xyz_m=state["current_furniture_aabb_min_xyz_m"].tolist(),
            furniture_aabb_max_xyz_m=state["current_furniture_aabb_max_xyz_m"].tolist()),
        source_physics_overrides_not_applied=source_overrides,
        source_release_helpers_not_applied={k: metrics.get(k) for k in (
            "release_retreat_m", "release_peel_angle_deg", "release_lower_m", "tail_steps")},
        historical_bilateral_contact=metrics.get("clamp_bilateral_contact"),
        source_warnings=warnings, physical_acceptance=False, execution_ready=False,
        approved_scoring=None, approved_timeout_s=None,
    )


def load_candidate(api_id, episode="episode_000"):
    import numpy as np
    path = source_directory(api_id, episode)
    names = ("full_state.npz", "action.npz", "episode.json", "inject_meta.json", "place_metrics.json")
    manifest = json.loads((REFERENCE.parent / "manifest.json").read_text())
    expected = {row["local_path"]: row["sha256"] for row in manifest["files"]}
    provenance = {}
    for name in names:
        f = (path / name).resolve(strict=True)
        if not f.is_relative_to(REFERENCE.resolve()):
            raise ValueError("Source file escaped reference directory")
        relative = str(f.relative_to(TASK))
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if expected.get(relative) != sha:
            raise ValueError("Source snapshot differs from captured manifest: " + relative)
        provenance[relative] = sha
    with np.load(path / "full_state.npz", allow_pickle=False) as state, np.load(path / "action.npz", allow_pickle=False) as action:
        result = extract(api_id, state, action, *[
            json.loads((path/name).read_text()) for name in names[2:]])
    result["source_files_sha256"] = provenance
    return result


def recorded_hold_targets(state):
    """Read native targets, not binary actions or measured joint coordinates.

    L1 full_state stores waist as [angular body_joint2, linear body_joint1],
    arms left seven then right seven, and grippers left then right.
    """
    import numpy as np
    groups = {
        'command_arm_rad': tuple(f'idx{base+i}_arm_{side}_joint{i}'
                                for base, side in ((20, 'l'), (60, 'r')) for i in range(1, 8)),
        'command_waist_rad_m': ('idx02_body_joint2', 'idx01_body_joint1'),
        'command_head_rad': ('idx11_head_joint1', 'idx12_head_joint2'),
        'command_grip_rad': ('idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1'),
    }
    result = {}
    for key, names in groups.items():
        values = np.asarray(state[key])
        if values.shape != (len(state['frame_index']), len(names)) or not np.isfinite(values).all():
            raise ValueError('Invalid recorded native targets: '+key)
        result.update(zip(names, map(float, values[0])))
    return result


def load_recorded_hold_targets(api_id, episode="episode_000"):
    import numpy as np
    candidate = load_candidate(api_id, episode)  # Includes source hash/identity validation.
    with np.load(source_directory(api_id, episode)/'full_state.npz', allow_pickle=False) as state:
        return candidate, recorded_hold_targets(state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((TASK / "artifacts").resolve()):
        raise ValueError("Use a new directory within coffee_table_place/artifacts")
    results = {}
    for api in catalog():
        if api in MEALS:
            results[api] = dict(status="missing_measured_source", **catalog()[api])
        else:
            results[api] = dict(status="source_candidate_only", candidate=load_candidate(api))
    # Validate every input before writing any generated files.
    out.mkdir(parents=True)
    for api, row in results.items():
        (out / (api + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({api: row["status"] for api, row in results.items()}))


if __name__ == "__main__":
    main()

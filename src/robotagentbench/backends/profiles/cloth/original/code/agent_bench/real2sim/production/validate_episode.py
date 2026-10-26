#!/usr/bin/env python3
"""Structural validation for one aligned RGB/state/action episode.

This validator deliberately treats a render made after the physics rollout as
an appearance DLC, not as evidence that the base training tuple was captured
on one clock.  A publishable base episode needs three H.264 videos with exactly
the same frame count and rate as action.npz and full_state.npz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import av


SCHEMA = "real2sim.livingroom_episode_validation.v1"
ACTION_CHANNELS = {
    "arm": 14,
    "waist": 2,
    "head": 2,
    "base_xyyaw": 3,
}
STATE_FRAME_FIELDS = {
    "prop_xyz_m",
    "prop_wxyz",
    "prop_velocity_mps_radps",
    "robot_joint_position_rad_or_m",
    "robot_joint_velocity_radps_or_mps",
    "command_arm_rad",
    "command_waist_rad_m",
    "command_head_rad",
    "command_grip_rad",
}
VIDEO_NAMES = ("head", "hand_left", "hand_right")
REVIEW_CHECKS = frozenset({
    "target_identity_and_task_are_correct",
    "grasp_and_lift_are_physically_plausible",
    "no_visible_teleport_penetration_or_furniture_motion",
    "all_three_cameras_are_clear_and_temporally_complete",
    "video_motion_agrees_with_action_and_state_metrics",
})


def _scalar(array: np.ndarray, name: str) -> Any:
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got {array.shape}")
    return array.item()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _all_finite(name: str, value: np.ndarray) -> None:
    if value.dtype.kind not in "biufc":
        return
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _same(label: str, action: np.ndarray, command: np.ndarray) -> float:
    if action.shape != command.shape:
        raise ValueError(
            f"{label} action/command shape mismatch: "
            f"{action.shape} != {command.shape}"
        )
    delta = float(np.max(np.abs(
        action.astype(np.float64) - command.astype(np.float64)
    ))) if action.size else 0.0
    if delta > 5e-6:
        raise ValueError(
            f"{label} action differs from captured command by {delta:.3g}"
        )
    return delta


def _grip_mapping(action: np.ndarray, command: np.ndarray, hand: str) -> dict:
    """Validate the documented normalized-action to joint-radian mapping.

    Existing Family 7 actions store 0=open and 1=closed, while the captured
    command stores joint radians.  Fitting the two endpoints avoids baking a
    robot-specific gripper calibration into the general episode schema.
    """
    x = action.astype(np.float64).reshape(-1)
    y = command.astype(np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"{hand} grip action/command length mismatch")
    if not len(x):
        raise ValueError(f"{hand} grip channel is empty")
    design = np.stack((np.ones_like(x), x), axis=1)
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    residual = float(np.max(np.abs(prediction - y)))
    if residual > 5e-5:
        raise ValueError(
            f"{hand} grip is not one affine action-to-command mapping; "
            f"max residual {residual:.3g} rad"
        )
    if np.ptp(x) > 1e-6 and slope >= 0:
        raise ValueError(
            f"{hand} normalized grip convention drifted: closing must reduce "
            "the captured joint target"
        )
    return {
        "open_command_rad": round(float(intercept), 8),
        "closed_command_rad": round(float(intercept + slope), 8),
        "slope_rad_per_action": round(float(slope), 8),
        "max_residual_rad": residual,
    }


def _top_level_atom_order(path: Path) -> dict[str, int]:
    positions: dict[str, int] = {}
    size = path.stat().st_size
    offset = 0
    with path.open("rb") as stream:
        while offset + 8 <= size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) < 8:
                break
            atom_size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8].decode("latin-1")
            header_size = 8
            if atom_size == 1:
                if len(header) < 16:
                    break
                atom_size = int.from_bytes(header[8:16], "big")
                header_size = 16
            elif atom_size == 0:
                atom_size = size - offset
            if atom_size < header_size or offset + atom_size > size:
                raise ValueError(f"{path.name} has an invalid MP4 atom table")
            positions.setdefault(atom_type, offset)
            offset += atom_size
    return positions


def _video_probe(path: Path, expected_frames: int, expected_fps: float) -> dict:
    frames = 0
    with av.open(str(path), "r") as container:
        streams = [
            stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one video stream")
        stream = streams[0]
        codec = stream.codec_context.name
        pixel = stream.codec_context.format.name
        fps = float(stream.average_rate)
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        for frame in container.decode(stream):
            frame.to_ndarray(format="yuv420p")
            frames += 1
    if codec != "h264" or pixel != "yuv420p":
        raise ValueError(
            f"{path.name} is {codec}/{pixel}, expected h264/yuv420p")
    if frames != expected_frames:
        raise ValueError(
            f"{path.name} has {frames} decoded frames, expected "
            f"{expected_frames}"
        )
    if not math.isclose(fps, expected_fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f"{path.name} is {fps:g} fps, expected {expected_fps:g}")
    atoms = _top_level_atom_order(path)
    if "moov" not in atoms or "mdat" not in atoms or atoms["moov"] > atoms["mdat"]:
        raise ValueError(f"{path.name} is not MP4 fast-start (moov before mdat)")
    return {
        "codec": codec,
        "pixel_format": pixel,
        "fps": fps,
        "decoded_frames": frames,
        "width": width,
        "height": height,
        "faststart": True,
        "complete_decode": True,
        "size_bytes": path.stat().st_size,
    }


def validate_arrays(directory: Path) -> dict[str, Any]:
    action_path = directory / "action.npz"
    state_path = directory / "full_state.npz"
    trace_path = directory / "pick_contact_trace.npz"
    for required in (action_path, state_path, trace_path):
        if not required.is_file():
            raise ValueError(f"missing required array file: {required.name}")

    with np.load(action_path, allow_pickle=False) as source:
        action = {key: np.asarray(source[key]) for key in source.files}
    required_action = {"fps", "arm", "lgrip", "rgrip", *ACTION_CHANNELS}
    missing = sorted(required_action - set(action))
    if missing:
        raise ValueError(f"action.npz lacks {', '.join(missing)}")
    fps = float(_scalar(action["fps"], "action.fps"))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("action fps must be finite and positive")
    frames = int(action["arm"].shape[0])
    if frames < 2:
        raise ValueError("episode must have at least two action frames")
    for name, width in ACTION_CHANNELS.items():
        expected = (frames, width)
        if action[name].shape != expected:
            raise ValueError(
                f"action.{name} is {action[name].shape}, expected {expected}")
        _all_finite(f"action.{name}", action[name])
    for name in ("lgrip", "rgrip"):
        if action[name].shape != (frames,):
            raise ValueError(f"action.{name} must have shape ({frames},)")
        _all_finite(f"action.{name}", action[name])

    with np.load(state_path, allow_pickle=False) as source:
        state = {key: np.asarray(source[key]) for key in source.files}
    required_state = {
        "schema", "fps", "frame_index",
        "rigid_body_paths", "robot_dof_names", "target_prim",
        *STATE_FRAME_FIELDS,
    }
    missing = sorted(required_state - set(state))
    if missing:
        raise ValueError(f"full_state.npz lacks {', '.join(missing)}")
    schema = str(_scalar(state["schema"], "state.schema"))
    if schema not in {
        "real2sim.family7_full_state.v1",
        "real2sim.livingroom_full_state.v1",
    }:
        raise ValueError(f"unsupported full-state schema {schema!r}")
    capture_scope = (
        str(_scalar(state["capture_scope"], "state.capture_scope"))
        if "capture_scope" in state else "unspecified"
    )
    state_fps = float(_scalar(state["fps"], "state.fps"))
    if not math.isclose(state_fps, fps, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"state/action fps mismatch: {state_fps} != {fps}")
    if not np.array_equal(state["frame_index"], np.arange(frames)):
        raise ValueError("state frame_index is not exactly 0..T-1")
    for name in STATE_FRAME_FIELDS:
        value = state[name]
        if value.ndim < 1 or value.shape[0] != frames:
            raise ValueError(
                f"state.{name} has no one-to-one action frame axis")
        _all_finite(f"state.{name}", value)

    rigid_paths = [str(value) for value in state["rigid_body_paths"].tolist()]
    if len(rigid_paths) != len(set(rigid_paths)):
        raise ValueError("rigid_body_paths contains duplicates")
    bodies = len(rigid_paths)
    for name, tail in (
        ("prop_xyz_m", (bodies, 3)),
        ("prop_wxyz", (bodies, 4)),
        ("prop_velocity_mps_radps", (bodies, 6)),
    ):
        if state[name].shape != (frames, *tail):
            raise ValueError(
                f"state.{name} shape does not match rigid_body_paths")
    target = str(_scalar(state["target_prim"], "state.target_prim"))
    if rigid_paths.count(target) != 1:
        raise ValueError(
            "target_prim must occur exactly once in rigid_body_paths")

    command_delta = {
        "arm": _same("arm", action["arm"], state["command_arm_rad"]),
        "waist": _same(
            "waist", action["waist"], state["command_waist_rad_m"]),
        "head": _same("head", action["head"], state["command_head_rad"]),
    }
    grip = {
        "left": _grip_mapping(
            action["lgrip"], state["command_grip_rad"][:, 0], "left"),
        "right": _grip_mapping(
            action["rgrip"], state["command_grip_rad"][:, 1], "right"),
    }
    base_span = np.ptp(action["base_xyyaw"].astype(np.float64), axis=0)
    if np.max(np.abs(base_span)) > 5e-8:
        if "robot_root_xyz_m_per_frame" not in state:
            raise ValueError(
                "time-varying base action lacks per-frame robot-root state")

    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: np.asarray(source[key]) for key in source.files}
    for name in ("target_body_path", "frame_count", "fabric_obj_xyz"):
        if name not in trace:
            raise ValueError(f"pick_contact_trace.npz lacks {name}")
    trace_target = str(_scalar(
        trace["target_body_path"], "trace.target_body_path"))
    if trace_target != target:
        raise ValueError(f"trace/state target mismatch: {trace_target} != {target}")
    if int(_scalar(trace["frame_count"], "trace.frame_count")) != frames:
        raise ValueError("trace/state/action frame counts disagree")
    target_index = rigid_paths.index(target)
    trace_xyz = trace["fabric_obj_xyz"]
    if trace_xyz.shape != (frames, 3):
        raise ValueError("trace.fabric_obj_xyz has the wrong shape")
    trace_delta = float(np.max(np.abs(
        trace_xyz.astype(np.float64)
        - state["prop_xyz_m"][:, target_index, :].astype(np.float64)
    )))
    if trace_delta > 5e-6:
        raise ValueError(
            f"target state differs from scoring trace by {trace_delta:.3g} m")

    array_files = {
        "action.npz": {"size_bytes": action_path.stat().st_size},
        "full_state.npz": {"size_bytes": state_path.stat().st_size},
        "pick_contact_trace.npz": {"size_bytes": trace_path.stat().st_size},
    }
    for optional in ("place_metrics.npz", "place_metrics.json"):
        optional_path = directory / optional
        if optional_path.is_file():
            array_files[optional] = {
                "size_bytes": optional_path.stat().st_size}
    return {
        "frames": frames,
        "fps": fps,
        "duration_s": (frames - 1) / fps,
        "action_schema": "real2sim.action_npz.v1",
        "state_schema": schema,
        "state_capture_scope": capture_scope,
        "task_id": (
            str(_scalar(state["task_id"], "state.task_id"))
            if "task_id" in state else ""
        ),
        "episode_uid": (
            str(_scalar(state["episode_uid"], "state.episode_uid"))
            if "episode_uid" in state else ""
        ),
        "active_rigid_body_count": bodies,
        "target_prim": target,
        "max_command_delta": command_delta,
        "gripper_mappings": grip,
        "target_state_trace_max_delta_m": trace_delta,
        "array_files": array_files,
    }


def validate_episode(
    episode_directory: Path | str, *, allow_missing_video: bool = False
) -> dict[str, Any]:
    directory = Path(episode_directory).resolve()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "episode_directory": str(directory),
        "accepted": False,
        "training_accepted": False,
        "status": "invalid",
        "errors": [],
        "blocking_reasons": [],
    }
    try:
        arrays = validate_arrays(directory)
        result["arrays"] = arrays
        episode_path = directory / "episode.json"
        score_path = directory / "score.json"
        if not episode_path.is_file():
            raise ValueError("episode.json is required")
        episode = _json(episode_path)
        # score.json is retained as optional historical/owner diagnostics.
        # Acceptance is decided by the current Sheet-bound checker below, so
        # a copied old score file can never certify a changed trajectory.
        score = _json(score_path) if score_path.is_file() else {}
        state_task = arrays.get("task_id", "")
        state_episode = arrays.get("episode_uid", "")
        episode_task = str(episode.get(
            "task_id", score.get("runtime_task_id", "")))
        episode_uid = str(episode.get(
            "source_episode_uid", episode.get("episode_uid", "")))
        if episode_task and state_task and episode_task != state_task:
            raise ValueError(
                f"episode/state task mismatch: {episode_task} != {state_task}")
        if episode_uid and state_episode and episode_uid != state_episode:
            raise ValueError(
                f"episode/state UUID mismatch: {episode_uid} != {state_episode}")
        if episode.get("furniture_moved") is True:
            raise ValueError("episode moved furniture, violating global scene state")
        if episode.get("object_pose_or_velocity_writes") is True:
            raise ValueError("episode used non-physical object pose/velocity writes")
        owner_result = (
            _json(directory / "result.json")
            if (directory / "result.json").is_file() else {}
        )
        benchmark_task = str(episode.get(
            "benchmark_subtask", episode.get("benchmark_task_id", "")
        )).strip()
        if not benchmark_task:
            benchmark_task = str(owner_result.get(
                "benchmark_subtask", owner_result.get("benchmark_task_id", "")
            )).strip()
        if not benchmark_task:
            task_key = str(owner_result.get("task_key", ""))
            if task_key.startswith("benchmark_"):
                pieces = task_key[len("benchmark_"):].split("_")
                if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
                    benchmark_task = f"{int(pieces[0])}.{int(pieces[1])}"
        if benchmark_task:
            try:
                try:
                    from .task_success import evaluate_task_success
                except ImportError:
                    from task_success import evaluate_task_success
                task_success = evaluate_task_success(benchmark_task, directory)
            except (KeyError, TypeError, ValueError) as exc:
                task_success = {
                    "schema": "real2sim.task_success_result.v1",
                    "benchmark_subtask": benchmark_task,
                    "status": "unverified",
                    "success": None,
                    "reasons": [f"current task checker could not run: {exc}"],
                }
        else:
            task_success = {
                "schema": "real2sim.task_success_result.v1",
                "benchmark_subtask": None,
                "status": "unverified",
                "success": None,
                "reasons": [
                    "episode.json has no Sheet benchmark_subtask; a robot "
                    "recording label cannot select the task checker"
                ],
            }
        result["task_success"] = task_success
        if task_success.get("status") != "passed":
            result["blocking_reasons"].append(
                "current task success is "
                + str(task_success.get("status", "unverified"))
                + ": "
                + "; ".join(map(str, task_success.get("reasons", ())))
            )
        if benchmark_task:
            try:
                try:
                    from .task_data_compatibility import (
                        evaluate_episode_compatibility,
                    )
                except ImportError:
                    from task_data_compatibility import (
                        evaluate_episode_compatibility,
                    )
                data_compatibility = evaluate_episode_compatibility(
                    benchmark_task, directory)
            except (
                KeyError, OSError, RuntimeError, TypeError, ValueError,
            ) as exc:
                data_compatibility = {
                    "schema": "real2sim.task_data_compatibility.v1",
                    "benchmark_subtask": benchmark_task,
                    "disposition": "unverified",
                    "reason": (
                        "current data compatibility could not run: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        else:
            data_compatibility = {
                "schema": "real2sim.task_data_compatibility.v1",
                "benchmark_subtask": None,
                "disposition": "unverified",
                "reason": "Sheet benchmark task identity is missing",
            }
        result["data_compatibility"] = data_compatibility
        if data_compatibility.get("disposition") != "current":
            result["blocking_reasons"].append(
                "current default-House data compatibility is "
                + str(data_compatibility.get("disposition", "unverified"))
            )
        result["strict_success"] = {
            "score_schema": score.get("schema"),
            "score_ok": task_success.get("status") == "passed",
            "failed_gates": (
                [] if task_success.get("status") == "passed"
                else list(task_success.get("reasons", ()))
            ),
            "current_task_success_status": task_success.get("status"),
            "current_task_checker": task_success.get("checker"),
            "current_data_compatibility_ok": (
                data_compatibility.get("disposition") == "current"
            ),
            "current_data_compatibility_disposition": (
                data_compatibility.get("disposition", "unverified")
            ),
            "array_files_opened_and_structurally_verified": sorted(
                arrays["array_files"]),
            "furniture_moved": episode.get("furniture_moved", False),
            "object_pose_or_velocity_writes": episode.get(
                "object_pose_or_velocity_writes", False),
        }

        videos: dict[str, Any] = {}
        missing_videos: list[str] = []
        for name in VIDEO_NAMES:
            path = directory / "videos" / f"{name}.mp4"
            if not path.is_file():
                missing_videos.append(str(path.relative_to(directory)))
                continue
            videos[name] = _video_probe(
                path, arrays["frames"], arrays["fps"])
        result["videos"] = videos
        if missing_videos:
            result["blocking_reasons"].append(
                "missing same-rollout videos: " + ", ".join(missing_videos))
            if not allow_missing_video:
                raise ValueError(result["blocking_reasons"][-1])
            result["status"] = "diagnostic_action_state_only"
        else:
            capture = episode.get("capture", {})
            capture_mode = capture.get("mode") if isinstance(capture, dict) else None
            expected_mode = "physics_rgb_state_action_same_rollout_same_tick_clock"
            if capture_mode != expected_mode:
                raise ValueError(
                    "episode.json lacks authoritative same-tick capture mode")
            review_path = directory / "review.json"
            if task_success.get("status") != "passed":
                result["status"] = "task_success_" + str(
                    task_success.get("status", "unverified")
                )
            elif data_compatibility.get("disposition") != "current":
                result["status"] = "data_compatibility_" + str(
                    data_compatibility.get("disposition", "unverified")
                )
            elif review_path.is_file():
                review = _json(review_path)
                expected_episode_id = str(episode.get(
                    "episode_id", episode.get("label", directory.name)))
                if (
                    review.get("schema")
                    != "real2sim.livingroom_video_human_review.v1"
                    or review.get("approved") is not True
                    or review.get("episode_id") != expected_episode_id
                    or review.get("reviewed_videos") != {
                        name: f"videos/{name}.mp4" for name in VIDEO_NAMES}
                    or not review.get("reviewer")
                    or set(review.get("checklist", {})) != REVIEW_CHECKS
                    or not all(review["checklist"].values())
                ):
                    raise ValueError("review.json is invalid or stale")
                result["human_review"] = review
                result["accepted"] = True
                result["training_accepted"] = True
                result["status"] = "publishable"
            else:
                result["blocking_reasons"].append(
                    "representative same-tick video human review is pending")
                result["status"] = "review_sample"
        result["validation_ok"] = True
    except Exception as exc:
        result["validation_ok"] = False
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_directory", type=Path)
    parser.add_argument(
        "--allow-missing-video", action="store_true",
        help="audit legacy action/state data without accepting it for training",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate_episode(
        args.episode_directory, allow_missing_video=args.allow_missing_video)
    if args.output:
        _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["validation_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

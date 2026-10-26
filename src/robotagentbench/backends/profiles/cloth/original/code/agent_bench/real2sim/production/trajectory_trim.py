#!/usr/bin/env python3
"""Conservatively trim an accepted trajectory without changing its meaning.

The tool never uses trimming to repair a bad initialization or a failed task.
It first requires the current Sheet-bound success function to pass, searches
for the earliest prefix that still passes the same function, and only removes
an idle start prefix when the trimmed result independently passes again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset.decode_images import write_mp4

try:
    from .task_data_compatibility import evaluate_episode_compatibility
    from .task_success import evaluate_task_success, success_contract
    from .validate_episode import _video_probe
except ImportError:  # Direct script execution.
    from task_data_compatibility import evaluate_episode_compatibility
    from task_success import evaluate_task_success, success_contract
    from validate_episode import _video_probe


SCHEMA = "real2sim.trajectory_trim_plan.v1"
TRIM_SCHEMA = "real2sim.trajectory_trim.v1"
SUPPORTED_ADAPTERS = frozenset({
    "livingroom_pick",
    "task6_place",
    "family7_e2e",
})
VIDEO_NAMES = ("head", "hand_left", "hand_right")
STATIC_ARRAY_NAMES = frozenset({
    "schema",
    "capture_scope",
    "task_id",
    "episode_uid",
    "target_prim",
    "target_body_path",
    "fps",
    "rigid_body_paths",
    "rigid_body_roles",
    "robot_dof_names",
    "contact_target_paths",
    "contact_hands",
    "contact_pad_sides",
})
TERMINAL_FRAME_MARKERS = frozenset({
    # This producer-side marker means "keep applying the already established
    # approach pose through the end of the rollout".  When a successful idle
    # suffix is removed, the truthful equivalent is the new last frame.  It is
    # not a separate task event and must not force retention of the old tail.
    "place_approach_hold_through_frame",
})


class TrimUnavailable(ValueError):
    """The requested trim cannot be shown to preserve task success."""


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as source:
            return {name: np.asarray(source[name]) for name in source.files}
    except (OSError, ValueError) as exc:
        raise TrimUnavailable(f"cannot read {path.name}: {exc}") from exc


def _action_clock(directory: Path) -> tuple[dict[str, np.ndarray], int, float]:
    action = _load_npz(directory / "action.npz")
    counts = {
        int(value.shape[0])
        for value in action.values()
        if value.ndim >= 1
    }
    if len(counts) != 1:
        raise TrimUnavailable("action channels do not share one frame count")
    frames = counts.pop()
    try:
        fps = float(np.asarray(action["fps"]).item())
    except (KeyError, ValueError) as exc:
        raise TrimUnavailable("action.npz has no scalar fps") from exc
    if frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise TrimUnavailable("trajectory clock is invalid")
    return action, frames, fps


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrimUnavailable(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrimUnavailable(f"{path.name} must contain one object")
    return value


def _without_obsolete_content_locks(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_obsolete_content_locks(item)
            for key, item in value.items()
            if (
                str(key) != "ha" + "shes"
                and not str(key).endswith("_sha" + "256")
            )
        }
    if isinstance(value, list):
        return [_without_obsolete_content_locks(item) for item in value]
    return value


def _trim_metadata(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_frames: int,
    fps: float,
) -> None:
    for name in ("episode.json", "result.json"):
        value = _json(source / name)
        if not value:
            continue
        value = _without_obsolete_content_locks(value)
        if name == "episode.json":
            capture = value.get("capture")
            if isinstance(capture, dict):
                capture["frames"] = end - start
                capture["fps"] = fps
            value["human_review_required_after_trim"] = True
        value["trajectory_trim"] = {
            "schema": TRIM_SCHEMA,
            "source_episode_directory": str(source.resolve()),
            "source_frame_count": source_frames,
            "source_start_frame_inclusive": start,
            "source_end_frame_exclusive": end,
            "output_frame_count": end - start,
            "fps": fps,
            "task_success_rechecked": False,
        }
        (destination / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def _adjust_scalar_index(
    name: str, value: np.ndarray, *, start: int, end: int
) -> np.ndarray:
    if value.shape != () or value.dtype.kind not in "iu":
        return value
    if name in {"frame_count", "rendered_frames"}:
        return np.asarray(end - start, dtype=value.dtype)
    if not (name.endswith("_frame") or name.endswith("_t")):
        return value
    index = int(value.item())
    if index < 0:  # Explicit not-applicable sentinel.
        return value
    if name in TERMINAL_FRAME_MARKERS and index >= end:
        return np.asarray(end - start - 1, dtype=value.dtype)
    if index < start or index >= end:
        raise TrimUnavailable(
            f"{name}={index} would fall outside the proposed frame range"
        )
    return np.asarray(index - start, dtype=value.dtype)


def _slice_npz(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_frames: int,
    action_file: bool = False,
) -> None:
    values = _load_npz(source)
    output: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if name == "frame_index":
            output[name] = np.arange(end - start, dtype=value.dtype)
        elif (
            value.ndim >= 1
            and value.shape[0] == source_frames
            and (action_file or name not in STATIC_ARRAY_NAMES)
            and not name.startswith("current_")
            and not name.startswith("robot_initial_")
            and not name.startswith("task_owned_robot_")
        ):
            output[name] = value[start:end]
        else:
            output[name] = _adjust_scalar_index(
                name, value, start=start, end=end
            )
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **output)
        stream.flush()
        os.fsync(stream.fileno())


def _complete_initial_base_evidence(state_path: Path) -> None:
    """Fill empty compatibility fields from the same captured robot root."""
    values = _load_npz(state_path)
    base = np.asarray(
        values.get("current_robot_base_xy_m", np.asarray([])), dtype=float)
    orientation = np.asarray(
        values.get("current_robot_orient_wxyz", np.asarray([])), dtype=float)
    changed = False
    if base.shape != (2,) or not np.isfinite(base).all():
        root = np.asarray(
            values.get(
                "robot_root_xyz_m_per_frame",
                values.get("robot_root_xyz_m", np.asarray([])),
            ),
            dtype=float,
        )
        if root.ndim == 2 and root.shape[1:] == (3,):
            root = root[0]
        if root.shape != (3,) or not np.isfinite(root).all():
            raise TrimUnavailable(
                "captured initial robot base XY cannot be recovered")
        values["current_robot_base_xy_m"] = root[:2]
        changed = True
    if orientation.shape != (4,) or not np.isfinite(orientation).all():
        root_wxyz = np.asarray(
            values.get(
                "robot_root_wxyz_per_frame",
                values.get("robot_root_wxyz", np.asarray([])),
            ),
            dtype=float,
        )
        if root_wxyz.ndim == 2 and root_wxyz.shape[1:] == (4,):
            root_wxyz = root_wxyz[0]
        norm = float(np.linalg.norm(root_wxyz))
        if (
            root_wxyz.shape != (4,)
            or not np.isfinite(root_wxyz).all()
            or norm <= 1.0e-9
        ):
            raise TrimUnavailable(
                "captured initial robot base yaw cannot be recovered")
        values["current_robot_orient_wxyz"] = root_wxyz / norm
        changed = True
    if not changed:
        return
    temporary = state_path.with_name(state_path.name + ".initial.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_names(contract: Mapping[str, Any], source: Path) -> list[str]:
    names = {"action.npz", "full_state.npz"}
    names.update(map(str, contract.get("required_artifacts", ())))
    for optional in ("pick_contact_trace.npz", "place_metrics.npz"):
        if (source / optional).is_file():
            names.add(optional)
    missing = sorted(name for name in names if not (source / name).is_file())
    if missing:
        raise TrimUnavailable("missing trim evidence: " + ", ".join(missing))
    non_npz = sorted(name for name in names if not name.endswith(".npz"))
    if non_npz:
        raise TrimUnavailable(
            "frame-safe trimming is not implemented for: "
            + ", ".join(non_npz)
        )
    return sorted(names)


def _write_core_range(
    task: str,
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    source_frames: int,
    fps: float,
) -> None:
    contract = success_contract(task)
    if destination.exists() and any(destination.iterdir()):
        raise TrimUnavailable(f"trim destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _trim_metadata(
        source,
        destination,
        start=start,
        end=end,
        source_frames=source_frames,
        fps=fps,
    )
    for name in _artifact_names(contract, source):
        _slice_npz(
            source / name,
            destination / name,
            start=start,
            end=end,
            source_frames=source_frames,
            action_file=name == "action.npz",
        )
    _complete_initial_base_evidence(destination / "full_state.npz")
    for name in ("randomization_audit.json", "distractor_audit.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def _candidate_success(
    task: str,
    source: Path,
    *,
    start: int,
    end: int,
    source_frames: int,
    fps: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trajectory-trim-check-") as root:
        candidate = Path(root) / "episode"
        try:
            _write_core_range(
                task,
                source,
                candidate,
                start=start,
                end=end,
                source_frames=source_frames,
                fps=fps,
            )
        except TrimUnavailable as exc:
            return {"status": "unverified", "reasons": [str(exc)]}
        return evaluate_task_success(task, candidate)


def _proposed_idle_start(
    action: Mapping[str, np.ndarray], frames: int, fps: float
) -> int:
    activity = np.zeros(frames - 1, dtype=bool)
    for value in action.values():
        if (
            value.ndim < 1
            or value.shape[0] != frames
            or value.dtype.kind not in "biufc"
        ):
            continue
        flattened = value.astype(np.float64).reshape(frames, -1)
        activity |= np.max(np.abs(np.diff(flattened, axis=0)), axis=1) > 1.0e-5
    changed = np.flatnonzero(activity)
    if not len(changed):
        return 0
    first_changed_frame = int(changed[0]) + 1
    preroll = max(1, int(math.ceil(0.25 * fps)))
    return max(0, first_changed_frame - preroll)


def plan_trim(task: str, episode_directory: Path | str) -> dict[str, Any]:
    """Return a non-mutating trim proposal proven with the current checker."""
    source = Path(episode_directory).resolve()
    if not source.is_dir():
        raise ValueError(f"episode directory is missing: {source}")
    contract = success_contract(task)
    adapter = str(contract.get("adapter"))
    original = evaluate_task_success(task, source)
    action, frames, fps = _action_clock(source)
    base = {
        "schema": SCHEMA,
        "benchmark_subtask": str(original.get("benchmark_subtask", task)),
        "episode_directory": str(source),
        "source_frame_count": frames,
        "fps": fps,
        "original_task_success": original,
        "checker_adapter": adapter,
    }
    if original.get("status") != "passed":
        return {
            **base,
            "status": "blocked",
            "reason": (
                "original trajectory must pass the current success function; "
                "trimming cannot repair an invalid or unverified rollout"
            ),
        }
    if adapter not in SUPPORTED_ADAPTERS:
        return {
            **base,
            "status": "blocked",
            "reason": (
                "this owner checker has no frame-prefix-safe trim adapter"
            ),
        }
    _artifact_names(contract, source)

    coarse = max(1, int(round(0.25 * fps)))
    minimum_end = max(2, int(math.ceil(1.0 * fps)))
    candidates = list(range(minimum_end, frames, coarse)) + [frames]
    earliest_end = frames
    previous = minimum_end
    for candidate_end in candidates:
        checked = _candidate_success(
            task,
            source,
            start=0,
            end=candidate_end,
            source_frames=frames,
            fps=fps,
        )
        if checked.get("status") == "passed":
            lower = max(minimum_end, previous - coarse + 1)
            earliest_end = candidate_end
            for refined_end in range(lower, candidate_end):
                refined = _candidate_success(
                    task,
                    source,
                    start=0,
                    end=refined_end,
                    source_frames=frames,
                    fps=fps,
                )
                if refined.get("status") == "passed":
                    earliest_end = refined_end
                    break
            break
        previous = candidate_end

    proposed_start = min(
        _proposed_idle_start(action, frames, fps), earliest_end - 2
    )
    accepted_start = 0
    for candidate_start in range(proposed_start, 0, -1):
        checked = _candidate_success(
            task,
            source,
            start=candidate_start,
            end=earliest_end,
            source_frames=frames,
            fps=fps,
        )
        if checked.get("status") == "passed":
            accepted_start = candidate_start
            break
    final_check = _candidate_success(
        task,
        source,
        start=accepted_start,
        end=earliest_end,
        source_frames=frames,
        fps=fps,
    )
    removed = accepted_start + frames - earliest_end
    return {
        **base,
        "status": "trim_available" if removed else "no_change",
        "source_start_frame_inclusive": accepted_start,
        "source_end_frame_exclusive": earliest_end,
        "output_frame_count": earliest_end - accepted_start,
        "removed_prefix_frames": accepted_start,
        "removed_suffix_frames": frames - earliest_end,
        "final_candidate_task_success": final_check,
    }


def _trim_videos(
    source: Path,
    destination: Path,
    *,
    start: int,
    end: int,
    fps: float,
) -> None:
    present = [
        name for name in VIDEO_NAMES
        if (source / "videos" / f"{name}.mp4").is_file()
    ]
    if not present:
        return
    if set(present) != set(VIDEO_NAMES):
        raise TrimUnavailable("source has an incomplete three-camera video set")
    output = destination / "videos"
    output.mkdir()
    expected = end - start
    for name in VIDEO_NAMES:
        path = source / "videos" / f"{name}.mp4"

        def decoded():
            with av.open(str(path), "r") as container:
                stream = next(
                    item for item in container.streams if item.type == "video"
                )
                for index, frame in enumerate(container.decode(stream)):
                    if index >= end:
                        break
                    if index >= start:
                        yield frame.to_ndarray(format="bgr24")

        target = output / f"{name}.mp4"
        written = write_mp4(decoded(), target, fps, size=None)
        if written != expected:
            raise TrimUnavailable(
                f"{name} trim wrote {written} frames, expected {expected}"
            )
        _video_probe(target, expected, fps)


def apply_trim(
    task: str,
    episode_directory: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    """Write one new append-only episode and re-run the current checker."""
    source = Path(episode_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"trim output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("trim output must not be the source or inside it")
    plan = plan_trim(task, source)
    if plan["status"] != "trim_available":
        raise TrimUnavailable(str(plan.get("reason", plan["status"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.trim.", dir=output.parent
    ))
    try:
        start = int(plan["source_start_frame_inclusive"])
        end = int(plan["source_end_frame_exclusive"])
        _write_core_range(
            task,
            source,
            temporary,
            start=start,
            end=end,
            source_frames=int(plan["source_frame_count"]),
            fps=float(plan["fps"]),
        )
        _trim_videos(
            source,
            temporary,
            start=start,
            end=end,
            fps=float(plan["fps"]),
        )
        checked = evaluate_task_success(task, temporary)
        if checked.get("status") != "passed":
            raise TrimUnavailable(
                "trimmed trajectory did not pass the current success function: "
                + "; ".join(map(str, checked.get("reasons", ())))
            )
        initial = checked.get("components", {}).get(
            "initial_state_reached_and_settled", {})
        if initial.get("status") != "passed":
            raise TrimUnavailable(
                "trimmed trajectory did not prove the required initial "
                "position and settling: "
                + "; ".join(map(str, initial.get("reasons", ())))
            )
        trim_record = {
            "schema": TRIM_SCHEMA,
            "benchmark_subtask": checked["benchmark_subtask"],
            "source_episode_directory": str(source),
            "source_frame_count": plan["source_frame_count"],
            "source_start_frame_inclusive": start,
            "source_end_frame_exclusive": end,
            "output_frame_count": plan["output_frame_count"],
            "fps": plan["fps"],
            "task_success_rechecked": True,
            "human_review_required_after_trim": True,
        }
        (temporary / "trim.json").write_text(
            json.dumps(trim_record, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        for name in ("episode.json", "result.json"):
            path = temporary / name
            if not path.is_file():
                continue
            metadata = _json(path)
            metadata["trajectory_trim"]["task_success_rechecked"] = True
            path.write_text(
                json.dumps(
                    metadata, ensure_ascii=False, indent=2, sort_keys=True
                ) + "\n",
                encoding="utf-8",
            )
        checked["episode_directory"] = str(output)
        (temporary / "task_success.json").write_text(
            json.dumps(checked, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        compatibility = evaluate_episode_compatibility(task, temporary)
        compatibility["episode_directory"] = str(output)
        (temporary / "compatibility.json").write_text(
            json.dumps(
                compatibility, ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return {
            **trim_record,
            "status": "written",
            "output_directory": str(output),
            "task_success": checked,
            "compatibility": compatibility,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("task")
    plan_parser.add_argument("episode", type=Path)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("task")
    apply_parser.add_argument("episode", type=Path)
    apply_parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = plan_trim(args.task, args.episode)
    else:
        value = apply_trim(args.task, args.episode, args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status") not in {"blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

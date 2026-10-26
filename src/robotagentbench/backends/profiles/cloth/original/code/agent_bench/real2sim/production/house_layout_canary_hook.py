#!/usr/bin/env python3
"""Explicit post-initializer House-layout hook for physics canaries.

The ordinary reproduce path never reads this module.  A canary opts in with a
small JSON request named by ``REAL2SIM_HOUSE_LAYOUT_CANARY_REQUEST``.  The
playback action is mapped first; after the task initializer and scene
randomizers finish, the same recipe is authored onto the working House scene.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

try:
    from real2sim.teleop.machine_paths import MESH311_PYTHON
except ImportError:  # Imported by the direct teleop runtime.
    from machine_paths import MESH311_PYTHON

try:
    from . import house_layout
except ImportError:  # Direct script/import from the teleop entrypoint.
    import house_layout


SCHEMA = "real2sim.house_layout_cross_family_hook.v1"
REQUEST_SCHEMA = "real2sim.house_layout_cross_family_request.v1"
FINAL_REQUEST_SCHEMA = "real2sim.house_final_layout_task_canary_request.v1"
REQUEST_ENV = "REAL2SIM_HOUSE_LAYOUT_CANARY_REQUEST"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        # Isaac runs under a different uid inside the simulator container.
        # NamedTemporaryFile defaults to 0600, unlike the ordinary published
        # playback trajectory, so restore the cross-runtime read contract
        # before the atomic replacement.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_request(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    effective = os.environ if environ is None else environ
    raw_path = str(effective.get(REQUEST_ENV, "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in {
        REQUEST_SCHEMA, FINAL_REQUEST_SCHEMA,
    }:
        raise ValueError(f"invalid House-layout canary request: {path}")
    task_key = value.get("task_key")
    task_frame = value.get("task_frame")
    core_ordinal = value.get("core_ordinal")
    if (
        not isinstance(task_key, str)
        or not task_key.startswith("benchmark_")
        or not isinstance(task_frame, str)
        or not task_frame
        or isinstance(core_ordinal, bool)
        or not isinstance(core_ordinal, int)
        or not 0 <= core_ordinal < house_layout.LAYOUT_COUNT
    ):
        raise ValueError("layout request needs task_key/frame and core 0..49")
    catalog = Path(str(value.get(
        "catalog", house_layout.DEFAULT_LAYOUTS_PATH))).resolve()
    evidence = Path(str(value.get("evidence_path", ""))).resolve()
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    if not str(value.get("evidence_path", "")).strip():
        raise ValueError("layout request needs an evidence_path")
    extra = value.get("extra_prim_frames", {})
    if not isinstance(extra, dict):
        raise ValueError("extra_prim_frames must be an object")
    if value["schema"] == FINAL_REQUEST_SCHEMA:
        record, _final_catalog, recipe = _final_layout_context(
            catalog, str(value.get("layout_id", "")))
        expected_layout_id = str(record["id"])
        layout_ordinal = int(record["ordinal"])
        shell_kind = str(record["shell_kind"])
    else:
        recipe = house_layout.scheduled_layout_for_core(
            task_key, core_ordinal, catalog)
        expected_layout_id = value.get("layout_id")
        if (
            expected_layout_id is not None
            and str(expected_layout_id) != recipe.layout_id
        ):
            raise ValueError(
                f"scheduled layout is {recipe.layout_id}, not {expected_layout_id}"
            )
        expected_layout_id = recipe.layout_id
        layout_ordinal = recipe.ordinal
        shell_kind = "canonical_original_house"
    house_layout.transform_for_frame(recipe, task_frame)
    for prim_path, frame in extra.items():
        if not isinstance(prim_path, str) or not prim_path.startswith("/World/"):
            raise ValueError(f"invalid extra layout prim {prim_path!r}")
        if not isinstance(frame, str) or not frame:
            raise ValueError(f"invalid frame for extra layout prim {prim_path}")
        house_layout.transform_for_frame(recipe, frame)
    return {
        **value,
        "request_path": str(path),
        "catalog": str(catalog),
        "evidence_path": str(evidence),
        "extra_prim_frames": extra,
        "layout_id": expected_layout_id,
        "layout_ordinal": layout_ordinal,
        "resolved_transform_layout_id": recipe.layout_id,
        "resolved_transform_layout_ordinal": recipe.ordinal,
        "shell_kind": shell_kind,
    }


def _final_layout_context(
    catalog_path: Path, layout_id: str,
) -> tuple[dict[str, Any], dict[str, Any], house_layout.HouseLayout]:
    try:
        from . import prepare_house_final_layouts as final_layouts
        from . import house_topology_visual
    except ImportError:
        import prepare_house_final_layouts as final_layouts
        import house_topology_visual

    catalog = final_layouts.load_catalog(catalog_path)
    rows = [row for row in catalog["layouts"] if row["id"] == layout_id]
    if len(rows) != 1:
        raise KeyError(f"unknown final House layout {layout_id!r}")
    record = dict(rows[0])
    if record["shell_kind"] == "generated_full_house_topology":
        candidate, _ = house_topology_visual.candidate_for_id(
            layout_id, catalog)
        layout = house_topology_visual.topology_layout(
            candidate, catalog["interaction_modules"])
    else:
        layout = house_layout.layout_for_ordinal(
            int(record["source_layout_ordinal"]))
    return record, catalog, layout


def _layout_from_request(request: Mapping[str, Any]) -> house_layout.HouseLayout:
    if request.get("schema") == FINAL_REQUEST_SCHEMA:
        return _final_layout_context(
            Path(str(request["catalog"])), str(request["layout_id"])
        )[2]
    return house_layout.scheduled_layout_for_core(
        str(request["task_key"]), int(request["core_ordinal"]),
        Path(str(request["catalog"])),
    )


def transform_frame_points(
    points: Any, request: Mapping[str, Any],
) -> np.ndarray:
    """Map catalog-derived task-frame points for layout-aware validators."""
    recipe = _layout_from_request(request)
    transform = house_layout.transform_for_frame(
        recipe, str(request["task_frame"]))
    return house_layout.transform_points(points, transform)


def transform_frame_zone(
    zone: Any, request: Mapping[str, Any],
) -> np.ndarray:
    """Map an axis-aligned ``(cx, cy, hx, hy, bottom_z)`` task zone."""
    source = np.asarray(zone, dtype=float)
    if (
        source.shape != (5,)
        or not np.isfinite(source).all()
        or np.any(source[2:4] <= 0.0)
    ):
        raise ValueError("task zone must be one finite positive 5-vector")
    cx, cy, hx, hy, bottom_z = source
    corners = np.asarray([
        [cx + sx * hx, cy + sy * hy]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ])
    mapped = transform_frame_points(corners, request)
    lower = np.min(mapped, axis=0)
    upper = np.max(mapped, axis=0)
    return np.asarray([
        *(0.5 * (lower + upper)),
        *(0.5 * (upper - lower)),
        bottom_z,
    ])


def transform_playback_action(
    trajectory: Path | str, request: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(trajectory).resolve()
    source = _load_npz(path)
    recipe = _layout_from_request(request)
    transformed = house_layout.transform_action_arrays(
        source, recipe, frame=str(request["task_frame"]))
    unchanged = sorted(set(source) - {"base_xyyaw"})
    if not all(
        np.array_equal(source[name], transformed[name]) for name in unchanged
    ):
        raise RuntimeError("House layout changed a non-base action channel")
    _atomic_npz(path, transformed)
    return {
        "schema": "real2sim.house_layout_cross_family_action.v1",
        "trajectory": str(path),
        "layout_id": str(request["layout_id"]),
        "layout_ordinal": int(request["layout_ordinal"]),
        "resolved_transform_layout_id": recipe.layout_id,
        "task_frame": str(request["task_frame"]),
        "task_cluster": house_layout.cluster_for_frame(
            recipe, str(request["task_frame"])),
        "transform": {
            "pivot_xy_m": list(house_layout.transform_for_frame(
                recipe, str(request["task_frame"])).pivot_xy_m),
            "translate_xy_m": list(house_layout.transform_for_frame(
                recipe, str(request["task_frame"])).translate_xy_m),
            "yaw_deg": house_layout.transform_for_frame(
                recipe, str(request["task_frame"])).yaw_deg,
        },
        "base_xyyaw_transformed": True,
        "non_base_channels_unchanged": True,
        "unchanged_channels": unchanged,
    }


def apply_working_scene(
    scene: Path | str, request: Mapping[str, Any],
) -> dict[str, Any]:
    scene_path = Path(scene).resolve()
    extra_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="house-layout-extra-",
            suffix=".json", delete=False,
        ) as stream:
            extra_path = Path(stream.name)
            json.dump(
                request.get("extra_prim_frames", {}), stream,
                ensure_ascii=False, sort_keys=True,
            )
            stream.write("\n")
        if (
            request.get("schema") == FINAL_REQUEST_SCHEMA
            and request.get("shell_kind") == "generated_full_house_topology"
        ):
            command = [
                str(MESH311_PYTHON),
                str(HERE / "house_topology_visual.py"), "apply-overlay",
                "--scene", str(scene_path),
                "--topology-id", str(request["layout_id"]),
                "--catalog", str(request["catalog"]),
                "--task-frame", str(request["task_frame"]),
                "--robot-path", "/World/Robot",
                "--extra-prim-frames", str(extra_path),
                "--asset-root", str(REPO),
                "--physics-overlay",
            ]
        else:
            recipe = _layout_from_request(request)
            seed = _selection_seed_for_layout(
                str(request["task_key"]), recipe.ordinal)
            command = [
                str(MESH311_PYTHON),
                str(HERE / "house_layout.py"), "apply-scene",
                "--scene", str(scene_path),
                "--task-key", str(request["task_key"]),
                "--core-ordinal", "0",
                "--layout-seed", str(seed),
                "--catalog", str(house_layout.DEFAULT_LAYOUTS_PATH),
                "--task-frame", str(request["task_frame"]),
                "--extra-prim-frames", str(extra_path),
            ]
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=300, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "House layout scene application failed:\n"
                + completed.stdout[-4000:]
            )
        result = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break
        if result is None:
            raise RuntimeError(
                "House layout scene application returned invalid JSON:\n"
                + completed.stdout[-4000:]
            )
        if request.get("schema") == FINAL_REQUEST_SCHEMA and request.get(
            "shell_kind") == "generated_full_house_topology":
            moved = result.get("module_scene_overlay", {})
            valid = bool(
                result.get("ok") is True
                and result.get("topology_id") == request["layout_id"]
                and result.get("physics_overlay_mode") is True
                and result.get("disabled_canonical_collision_paths")
                and moved.get("coordinate_transform_applied") is True
                and moved.get("source_xform_ops_preserved") is True
            )
            normalized = {
                **result,
                "layout_id": request["layout_id"],
                "layout_ordinal": request["layout_ordinal"],
                "task_key": request["task_key"],
                "task_frame": request["task_frame"],
                "coordinate_transform_applied": moved.get(
                    "coordinate_transform_applied"),
                "source_xform_ops_preserved": moved.get(
                    "source_xform_ops_preserved"),
            }
        else:
            valid = bool(
                result.get("ok") is True
                and result.get("layout_id")
                == request["resolved_transform_layout_id"]
                and result.get("task_key") == request["task_key"]
                and result.get("coordinate_transform_applied") is True
                and result.get("source_xform_ops_preserved") is True
            )
            normalized = {
                **result,
                "layout_id": request["layout_id"],
                "layout_ordinal": request["layout_ordinal"],
                "resolved_transform_layout_id": result.get("layout_id"),
            }
        if not valid:
            raise RuntimeError(f"House layout scene evidence drifted: {result}")
        visual_review = request.get("visual_review")
        if visual_review is not None:
            if (
                request.get("schema") != FINAL_REQUEST_SCHEMA
                or request.get("shell_kind")
                != "generated_full_house_topology"
                or not isinstance(visual_review, Mapping)
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                    str(visual_review.get("task_id", "")),
                ) is None
                or isinstance(visual_review.get("gen_seed"), bool)
                or not isinstance(visual_review.get("gen_seed"), int)
                or int(visual_review["gen_seed"]) < 1_000_000
                or str(visual_review.get("style", "")) not in {
                    "warm_oak_linen",
                    "cool_gray_black",
                    "light_scandi",
                    "dark_walnut",
                }
            ):
                raise ValueError("invalid explicit topology visual-review request")
            visual_command = [
                str(MESH311_PYTHON),
                str(HERE / "appearance.py"),
                "--task", str(visual_review["task_id"]),
                "--gen-seed", str(visual_review["gen_seed"]),
                "--style", str(visual_review["style"]),
                "--topology-id", str(request["layout_id"]),
                "--topology-catalog", str(request["catalog"]),
                "--scene", str(scene_path),
            ]
            visual_completed = subprocess.run(
                visual_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
                env={
                    **os.environ,
                    "REAL2SIM_APPEARANCE_PROFILE_V2": "1",
                },
            )
            visual_result = None
            for line in reversed(visual_completed.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    visual_result = candidate
                    break
            appearance = (
                visual_result.get("appearance", {})
                if isinstance(visual_result, dict) else {}
            )
            topology = (
                appearance.get("topology_visual", {})
                if isinstance(appearance, Mapping) else {}
            )
            if (
                visual_completed.returncode != 0
                or not isinstance(visual_result, dict)
                or visual_result.get("ok") is not True
                or appearance.get("physics_effect") != "none"
                or appearance.get("furniture_transforms_unchanged") is not True
                or appearance.get("semantic_prims_untouched") is not True
                or topology.get("topology_id") != request["layout_id"]
                or not topology.get("light_paths")
            ):
                raise RuntimeError(
                    "topology visual-review application failed:\n"
                    + visual_completed.stdout[-4000:]
                )
            normalized["visual_review"] = {
                "task_id": str(visual_review["task_id"]),
                "gen_seed": int(visual_review["gen_seed"]),
                "style": str(visual_review["style"]),
                "physics_effect": "none",
                "light_paths": list(topology["light_paths"]),
                "room_light_intensities": list(
                    topology["room_light_intensities"]),
                "maximum_decor_transform_error_m": topology[
                    "maximum_decor_transform_error_m"],
            }
        return normalized
    finally:
        if extra_path is not None:
            extra_path.unlink(missing_ok=True)


def _selection_seed_for_layout(task_key: str, layout_ordinal: int) -> int:
    for seed in range(50_900_000, 50_910_000):
        if house_layout.task_layout_permutation(
            task_key, seed=seed)[0] == layout_ordinal:
            return seed
    raise RuntimeError(f"could not schedule House layout {layout_ordinal}")


def write_evidence(
    request: Mapping[str, Any], *, action: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "ok": True,
        "task_key": request["task_key"],
        "task_frame": request["task_frame"],
        "core_ordinal": request["core_ordinal"],
        "layout_id": request["layout_id"],
        "layout_ordinal": request["layout_ordinal"],
        "catalog": request["catalog"],
        "request_path": request["request_path"],
        "action_application": dict(action),
        "scene_application": dict(scene),
        "single_variable": "house_layout",
        "remote_hosts_contacted": False,
    }
    _atomic_json(Path(str(request["evidence_path"])), result)
    return result

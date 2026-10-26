#!/usr/bin/env python3
"""Coordinate transforms for the 50 House layout candidates.

The layout catalog is deliberately a small data file, not 50 copied USD
scenes.  A single-task trajectory chooses the furniture cluster it belongs to
and applies that cluster's planar rigid transform to the robot, action and
captured rigid-body state.  Joint-space channels are copied unchanged.

The catalog status is ``2d_candidate``.  These helpers only preserve coordinate
relationships; they do not imply that a layout passed live navigation, USD
collision or physics replay.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .house_paths import SOURCE_BASELINE
except ImportError:  # direct script execution
    from house_paths import SOURCE_BASELINE


HERE = Path(__file__).resolve().parent
DEFAULT_LAYOUTS_PATH = HERE / "house_layouts.json"
LAYOUT_SCHEMA = "real2sim.house_layouts.v1"
LAYOUT_STATUS = "2d_candidate"
LAYOUT_COUNT = 50
LAYOUT_SCHEDULE_SCHEMA = "real2sim.house_layout_schedule.v1"
LAYOUT_SCHEDULE_SEED_BASE = 50_000_000
BENCHMARK_TASK_KEY = re.compile(r"^benchmark_(\d+)_(\d+)$")


@dataclass(frozen=True)
class SE2:
    """One planar rigid transform about a world-space pivot."""

    pivot_xy_m: tuple[float, float]
    translate_xy_m: tuple[float, float]
    yaw_deg: float

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)


@dataclass(frozen=True)
class HouseLayout:
    """One explicit large-room recipe of furniture-cluster transforms."""

    ordinal: int
    layout_id: str
    status: str
    transforms: Mapping[str, SE2]
    frame_aliases: Mapping[str, str | None] = field(default_factory=dict)
    cluster_prim_roots: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict)


@dataclass(frozen=True)
class StationPose:
    key: str
    xy_m: tuple[float, float]
    yaw_deg: float

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        half = math.radians(self.yaw_deg) * 0.5
        return (math.cos(half), 0.0, 0.0, math.sin(half))


@dataclass(frozen=True)
class PlanarFixture:
    """Conservative rectangular footprint used only by the 2D preflight."""

    fixture_id: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    yaw_deg: float
    cluster: str | None
    layout_movable: bool


def _pair(values: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{label} must contain two numbers")
    pair = float(values[0]), float(values[1])
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{label} must be finite")
    return pair


def load_layouts(path: Path | str = DEFAULT_LAYOUTS_PATH) -> tuple[HouseLayout, ...]:
    """Load the explicit catalog and enforce its one important invariant."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != LAYOUT_SCHEMA:
        raise ValueError(f"expected {LAYOUT_SCHEMA}")
    if manifest.get("status") != LAYOUT_STATUS:
        raise ValueError(f"layout catalog must have status {LAYOUT_STATUS}")
    if manifest.get("layout_count") != LAYOUT_COUNT:
        raise ValueError(f"layout_count must be exactly {LAYOUT_COUNT}")
    preflight_report = manifest.get("preflight_report")
    if not isinstance(preflight_report, dict) or any(
        preflight_report.get(name) != expected
        for name, expected in {
            "result": "passed_2d_navigation_candidates",
            "evaluated_layouts": LAYOUT_COUNT,
            "planar_geometry_passed": LAYOUT_COUNT,
            "grid_navigation_passed": LAYOUT_COUNT,
            "live_usd_collision": "not_run",
            "physics_replay": "not_run",
            "production_acceptance": "pending",
        }.items()
    ):
        raise ValueError("layout catalog candidate preflight report is invalid")

    clusters = manifest.get("clusters")
    rows = manifest.get("layouts")
    if not isinstance(clusters, dict) or not isinstance(rows, list):
        raise ValueError("layout catalog must define clusters and layouts")
    if len(rows) != LAYOUT_COUNT:
        raise ValueError(f"layout catalog must contain exactly {LAYOUT_COUNT} layouts")

    cluster_pivots: dict[str, tuple[float, float]] = {}
    cluster_prim_roots: dict[str, tuple[str, ...]] = {}
    all_prim_roots: set[str] = set()
    for cluster_id, cluster in clusters.items():
        if not isinstance(cluster, dict):
            raise ValueError(f"cluster {cluster_id!r} must be an object")
        cluster_pivots[str(cluster_id)] = _pair(
            cluster.get("pivot_xy_m"), label=f"{cluster_id}.pivot_xy_m")
        prim_roots = cluster.get("prim_roots")
        if not isinstance(prim_roots, list) or any(
            not isinstance(item, str) or not item.startswith("/World/")
            for item in prim_roots
        ):
            raise ValueError(
                f"{cluster_id}.prim_roots must be absolute /World paths")
        roots = tuple(prim_roots)
        if len(set(roots)) != len(roots):
            raise ValueError(f"{cluster_id}.prim_roots contains duplicates")
        overlap = all_prim_roots.intersection(roots)
        if overlap:
            raise ValueError(
                "cluster prim roots must be disjoint: "
                + ", ".join(sorted(overlap)))
        all_prim_roots.update(roots)
        cluster_prim_roots[str(cluster_id)] = roots

    raw_aliases = manifest.get("frame_aliases")
    if not isinstance(raw_aliases, dict):
        raise ValueError("layout catalog must define frame_aliases")
    frame_aliases: dict[str, str | None] = {}
    for frame_name, cluster_id in raw_aliases.items():
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError("frame aliases must have non-empty string keys")
        if cluster_id is not None and cluster_id not in cluster_pivots:
            raise ValueError(
                f"frame {frame_name!r} aliases unknown cluster {cluster_id!r}")
        frame_aliases[frame_name] = cluster_id

    expected_ordinals = list(range(1, LAYOUT_COUNT + 1))
    ordinals = [row.get("ordinal") for row in rows]
    ids = [row.get("id") for row in rows]
    if ordinals != expected_ordinals:
        raise ValueError("layout ordinals must be the ordered range 1..50")
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != LAYOUT_COUNT:
        raise ValueError("layout IDs must be 50 unique non-empty strings")

    layouts: list[HouseLayout] = []
    signatures: set[tuple[tuple[str, float, float, float], ...]] = set()
    cluster_ids = set(cluster_pivots)
    for row in rows:
        if row.get("status") != LAYOUT_STATUS:
            raise ValueError(f"layout {row['id']} must have status {LAYOUT_STATUS}")
        for field_name in ("design", "name", "theme", "notes"):
            if not isinstance(row.get(field_name), str) or not row[field_name]:
                raise ValueError(
                    f"layout {row['id']} must define non-empty {field_name}")
        recipes = row.get("cluster_transforms")
        if not isinstance(recipes, dict) or set(recipes) != cluster_ids:
            raise ValueError(
                f"layout {row['id']} must define every cluster transform")
        preflight = row.get("preflight")
        if not isinstance(preflight, dict) or preflight != {
            "planar_geometry": "passed",
            "grid_navigation": "passed",
            "live_usd_collision": "not_run",
            "physics_replay": "not_run",
        }:
            raise ValueError(
                f"layout {row['id']} must report the exact candidate preflight")
        transforms: dict[str, SE2] = {}
        signature: list[tuple[str, float, float, float]] = []
        for cluster_id in sorted(cluster_ids):
            recipe = recipes[cluster_id]
            if not isinstance(recipe, dict):
                raise ValueError(
                    f"layout {row['id']} cluster {cluster_id} must be an object")
            if set(recipe) != {"translate_xy_m", "yaw_deg"}:
                raise ValueError(
                    f"layout {row['id']} cluster {cluster_id} has wrong fields")
            translate = _pair(
                recipe["translate_xy_m"],
                label=f"{row['id']}.{cluster_id}.translate_xy_m",
            )
            yaw_deg = float(recipe["yaw_deg"])
            if not math.isfinite(yaw_deg):
                raise ValueError(f"layout {row['id']} contains a non-finite yaw")
            transforms[cluster_id] = SE2(
                cluster_pivots[cluster_id], translate, yaw_deg)
            signature.append((
                cluster_id,
                translate[0],
                translate[1],
                yaw_deg % 360.0,
            ))
        normalized_signature = tuple(signature)
        if normalized_signature in signatures:
            raise ValueError("layout transforms must be unique")
        signatures.add(normalized_signature)
        layouts.append(
            HouseLayout(
                ordinal=int(row["ordinal"]),
                layout_id=str(row["id"]),
                status=str(row["status"]),
                transforms=transforms,
                frame_aliases=frame_aliases,
                cluster_prim_roots=cluster_prim_roots,
            )
        )
    return tuple(layouts)


def layout_for_ordinal(
    ordinal: int, path: Path | str = DEFAULT_LAYOUTS_PATH
) -> HouseLayout:
    """Return a 1-based House layout (1 through 50)."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("layout ordinal must be an integer")
    if ordinal < 1 or ordinal > LAYOUT_COUNT:
        raise ValueError("layout ordinal must be in 1..50")
    return load_layouts(path)[ordinal - 1]


def layout_for_id(
    layout_id: str, path: Path | str = DEFAULT_LAYOUTS_PATH
) -> HouseLayout:
    """Resolve one explicit recipe by its stable human-readable ID."""

    matches = [layout for layout in load_layouts(path) if layout.layout_id == layout_id]
    if len(matches) != 1:
        raise KeyError(f"unknown House layout {layout_id!r}")
    return matches[0]


def layout_for_core_ordinal(
    core_ordinal: int, path: Path | str = DEFAULT_LAYOUTS_PATH
) -> HouseLayout:
    """Map the 50 trajectory ordinals ``0..49`` one-to-one to layouts."""

    if isinstance(core_ordinal, bool) or not isinstance(core_ordinal, int):
        raise TypeError("core ordinal must be an integer")
    if core_ordinal < 0 or core_ordinal >= LAYOUT_COUNT:
        raise ValueError("core ordinal must be in 0..49")
    return layout_for_ordinal(core_ordinal + 1, path)


def task_layout_schedule_seed(task_key: str) -> int:
    """Return the explicit integer seed for one task's layout permutation.

    The task ID is already numeric, so the schedule uses direct integer
    arithmetic.
    """

    match = BENCHMARK_TASK_KEY.fullmatch(str(task_key))
    if match is None:
        raise ValueError(f"invalid benchmark task key: {task_key!r}")
    family, subtask = (int(value) for value in match.groups())
    return LAYOUT_SCHEDULE_SEED_BASE + family * 10_000 + subtask


def task_layout_permutation(
    task_key: str,
    *,
    seed: int | None = None,
) -> tuple[int, ...]:
    """Shuffle layout ordinals 1..50 once with an explicit integer seed."""

    effective_seed = (
        task_layout_schedule_seed(task_key) if seed is None else seed
    )
    if isinstance(effective_seed, bool) or not isinstance(effective_seed, int):
        raise TypeError("layout schedule seed must be an integer")
    ordinals = list(range(1, LAYOUT_COUNT + 1))
    random.Random(effective_seed).shuffle(ordinals)
    return tuple(ordinals)


def scheduled_layout_for_core(
    task_key: str,
    core_ordinal: int,
    path: Path | str = DEFAULT_LAYOUTS_PATH,
    *,
    seed: int | None = None,
) -> HouseLayout:
    """Select one task-shuffled layout while covering all 50 exactly once."""

    if isinstance(core_ordinal, bool) or not isinstance(core_ordinal, int):
        raise TypeError("core ordinal must be an integer")
    if core_ordinal < 0 or core_ordinal >= LAYOUT_COUNT:
        raise ValueError("core ordinal must be in 0..49")
    permutation = task_layout_permutation(task_key, seed=seed)
    return layout_for_ordinal(permutation[core_ordinal], path)


def _layout(value: HouseLayout | int) -> HouseLayout:
    return layout_for_ordinal(value) if isinstance(value, int) else value


def _transform(value: HouseLayout | int, cluster: str) -> SE2:
    layout = _layout(value)
    try:
        return layout.transforms[cluster]
    except KeyError as exc:
        raise KeyError(f"layout {layout.layout_id} has no cluster {cluster!r}") from exc


IDENTITY = SE2((0.0, 0.0), (0.0, 0.0), 0.0)


def cluster_for_frame(layout: HouseLayout | int, frame: str) -> str | None:
    """Resolve a task frame to its moving furniture cluster.

    Frames explicitly mapped to ``null`` stay in the canonical world frame.
    Unknown names are rejected so a typo cannot silently desynchronise the
    scene and trajectory.
    """

    effective = _layout(layout)
    if frame not in effective.frame_aliases:
        raise KeyError(
            f"layout {effective.layout_id} has no task frame {frame!r}")
    return effective.frame_aliases[frame]


def transform_for_frame(layout: HouseLayout | int, frame: str) -> SE2:
    effective = _layout(layout)
    cluster = cluster_for_frame(effective, frame)
    return IDENTITY if cluster is None else _transform(effective, cluster)


def _selected_transform(
    layout: HouseLayout | int, *, cluster: str | None, frame: str | None
) -> SE2:
    if (cluster is None) == (frame is None):
        raise ValueError("provide exactly one of cluster or frame")
    return (
        _transform(layout, cluster)
        if cluster is not None
        else transform_for_frame(layout, str(frame))
    )


def transform_points(points: Any, transform: SE2) -> np.ndarray:
    """Apply an SE(2) transform to ``(..., 2|3)`` points; Z is unchanged."""

    source = np.asarray(points)
    if source.ndim == 0 or source.shape[-1] not in (2, 3):
        raise ValueError("points must have a final dimension of 2 or 3")
    result = np.array(source, dtype=np.result_type(source.dtype, np.float64), copy=True)
    dx = result[..., 0] - transform.pivot_xy_m[0]
    dy = result[..., 1] - transform.pivot_xy_m[1]
    cosine = math.cos(transform.yaw_rad)
    sine = math.sin(transform.yaw_rad)
    result[..., 0] = (
        transform.pivot_xy_m[0]
        + cosine * dx
        - sine * dy
        + transform.translate_xy_m[0]
    )
    result[..., 1] = (
        transform.pivot_xy_m[1]
        + sine * dx
        + cosine * dy
        + transform.translate_xy_m[1]
    )
    return result


def transform_yaws(yaw_rad: Any, transform: SE2) -> np.ndarray:
    """Rotate absolute world-frame yaw values, in radians."""

    return np.asarray(yaw_rad, dtype=np.float64) + transform.yaw_rad


def transform_quaternions_wxyz(quaternions: Any, transform: SE2) -> np.ndarray:
    """Left-compose WXYZ quaternions with the transform's world-Z rotation."""

    source = np.asarray(quaternions)
    if source.ndim == 0 or source.shape[-1] != 4:
        raise ValueError("quaternions must have a final dimension of 4")
    value = np.asarray(source, dtype=np.float64)
    half = transform.yaw_rad * 0.5
    zw, zz = math.cos(half), math.sin(half)
    w, x, y, z = np.moveaxis(value, -1, 0)
    return np.stack(
        (
            zw * w - zz * z,
            zw * x - zz * y,
            zw * y + zz * x,
            zw * z + zz * w,
        ),
        axis=-1,
    )


def _rotate_vectors(vectors: Any, transform: SE2) -> np.ndarray:
    source = np.asarray(vectors)
    if source.ndim == 0 or source.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of 3")
    result = np.asarray(source, dtype=np.float64).copy()
    x = result[..., 0].copy()
    y = result[..., 1].copy()
    cosine = math.cos(transform.yaw_rad)
    sine = math.sin(transform.yaw_rad)
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


def rotate_vectors(vectors: Any, transform: SE2) -> np.ndarray:
    """Rotate world-frame vectors without applying translation."""

    return _rotate_vectors(vectors, transform)


def _copied_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in arrays.items()}


def _restore_dtype(value: np.ndarray, original: Any) -> np.ndarray:
    return value.astype(np.asarray(original).dtype, copy=False)


def transform_action_arrays(
    arrays: Mapping[str, Any], layout: HouseLayout | int, *,
    cluster: str | None = None, frame: str | None = None,
) -> dict[str, np.ndarray]:
    """Transform world-frame ``base_xyyaw`` and copy every other action channel."""

    result = _copied_arrays(arrays)
    if "base_xyyaw" not in arrays:
        raise KeyError("action arrays do not contain base_xyyaw")
    base = np.asarray(arrays["base_xyyaw"])
    if base.ndim == 0 or base.shape[-1] != 3:
        raise ValueError("base_xyyaw must have a final dimension of 3")
    transform = _selected_transform(layout, cluster=cluster, frame=frame)
    transformed = np.asarray(base, dtype=np.float64).copy()
    transformed[..., :2] = transform_points(base[..., :2], transform)
    transformed[..., 2] = transform_yaws(base[..., 2], transform)
    result["base_xyyaw"] = _restore_dtype(transformed, base)
    return result


def transform_state_arrays(
    arrays: Mapping[str, Any], layout: HouseLayout | int, *,
    cluster: str | None = None, frame: str | None = None,
) -> dict[str, np.ndarray]:
    """Transform one single-task capture into the selected cluster frame.

    Robot roots and every captured prop are moved together.  Linear and
    angular world-frame velocities are rotated.  Joint positions, joint
    velocities, commands, frame indices and metadata are copied unchanged.
    """

    result = _copied_arrays(arrays)
    transform = _selected_transform(layout, cluster=cluster, frame=frame)

    for key in ("robot_root_xyz_m", "robot_root_xyz_m_per_frame"):
        if key in arrays:
            value = transform_points(arrays[key], transform)
            result[key] = _restore_dtype(value, arrays[key])
    for key in ("robot_root_wxyz", "robot_root_wxyz_per_frame"):
        if key in arrays:
            value = transform_quaternions_wxyz(arrays[key], transform)
            result[key] = _restore_dtype(value, arrays[key])
    for key in (
        "robot_root_velocity_mps_radps",
        "robot_root_velocity_mps_radps_per_frame",
    ):
        if key not in arrays:
            continue
        source = np.asarray(arrays[key])
        if source.ndim == 0 or source.shape[-1] != 6:
            raise ValueError(f"{key} must have a final dimension of 6")
        value = np.asarray(source, dtype=np.float64).copy()
        value[..., :3] = _rotate_vectors(source[..., :3], transform)
        value[..., 3:] = _rotate_vectors(source[..., 3:], transform)
        result[key] = _restore_dtype(value, source)

    explicit_prop_frames = "rigid_body_layout_frames" in arrays
    prop_transforms = [transform]
    if explicit_prop_frames:
        frames = np.asarray(arrays["rigid_body_layout_frames"]).astype(str)
        if frames.ndim != 1:
            raise ValueError("rigid_body_layout_frames must be a 1D array")
        prop_transforms = [transform_for_frame(layout, value) for value in frames]

    def transform_props(key: str, operation: Any, width: int) -> None:
        if key not in arrays:
            return
        source = np.asarray(arrays[key])
        if source.ndim < 2 or source.shape[-1] != width:
            raise ValueError(f"{key} must end in (body, {width})")
        body_count = source.shape[-2]
        if body_count > 1 and not explicit_prop_frames:
            raise ValueError(
                "multi-body state requires rigid_body_layout_frames")
        transforms = (
            prop_transforms * body_count
            if len(prop_transforms) == 1
            else prop_transforms
        )
        if len(transforms) != body_count:
            raise ValueError(
                "rigid_body_layout_frames must match the prop body axis")
        value = np.asarray(source, dtype=np.float64).copy()
        for body_index, body_transform in enumerate(transforms):
            value[..., body_index, :] = operation(
                source[..., body_index, :], body_transform)
        result[key] = _restore_dtype(value, source)

    transform_props("prop_xyz_m", transform_points, 3)
    transform_props("prop_wxyz", transform_quaternions_wxyz, 4)

    def rotate_spatial(value: Any, body_transform: SE2) -> np.ndarray:
        source = np.asarray(value)
        transformed = np.asarray(source, dtype=np.float64).copy()
        transformed[..., :3] = _rotate_vectors(source[..., :3], body_transform)
        transformed[..., 3:] = _rotate_vectors(source[..., 3:], body_transform)
        return transformed

    transform_props("prop_velocity_mps_radps", rotate_spatial, 6)
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def stage_action_state_overlay(
    *, action_source: Path, state_source: Path, destination: Path,
    layout: HouseLayout | int, frame: str,
) -> dict[str, Any]:
    """Write transformed scratch copies while leaving the accepted core alone."""

    effective = _layout(layout)
    action_destination = destination / "action.npz"
    state_destination = destination / "full_state.npz"
    _write_npz(
        action_destination,
        transform_action_arrays(
            _load_npz(action_source), effective, frame=frame),
    )
    _write_npz(
        state_destination,
        transform_state_arrays(
            _load_npz(state_source), effective, frame=frame),
    )
    return {
        "layout_id": effective.layout_id,
        "layout_ordinal": effective.ordinal,
        "layout_status": effective.status,
        "task_frame": frame,
        "task_cluster": cluster_for_frame(effective, frame),
        "action": action_destination,
        "state": state_destination,
        "source_action_state_unchanged": True,
        "scratch_action_state_transformed": True,
        "arm_and_joint_channels_unchanged": True,
    }


def _is_identity(transform: SE2) -> bool:
    return bool(
        transform.pivot_xy_m == (0.0, 0.0)
        and transform.translate_xy_m == (0.0, 0.0)
        and transform.yaw_deg % 360.0 == 0.0
    )


def apply_scene_overlay(
    scene: Path | str, layout: HouseLayout | int, *, frame: str,
    robot_path: str | None = "/World/Robot",
    extra_prim_frames: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Author the layout into one disposable working USD scene.

    This is a coordinate authoring operation only.  It deliberately records
    ``2d_candidate`` and performs no navigation, collision, or physics claim.
    ``pxr`` is imported here so the array helpers remain usable from the base
    Python environment; the visual worker calls this through mesh311.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    scene_path = Path(scene).resolve()
    effective = _layout(layout)
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"could not open USD scene {scene_path}")

    def matrix_for(transform: SE2):
        cosine = math.cos(transform.yaw_rad)
        sine = math.sin(transform.yaw_rad)
        pivot_x, pivot_y = transform.pivot_xy_m
        translate_x, translate_y = transform.translate_xy_m
        offset_x = pivot_x - cosine * pivot_x + sine * pivot_y + translate_x
        offset_y = pivot_y - sine * pivot_x - cosine * pivot_y + translate_y
        return Gf.Matrix4d(
            cosine, sine, 0.0, 0.0,
            -sine, cosine, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            offset_x, offset_y, 0.0, 1.0,
        )

    authored: list[dict[str, Any]] = []

    def author_collision_cube_pose(
        prim: Any, transform: SE2, role: str,
    ) -> bool:
        """Bake SE(2) into a collision Cube's decomposed xform stack.

        PhysX does not preserve the live shape of a non-uniformly scaled
        ``PhysicsCollisionAPI`` Cube when a matrix op is layered above it.
        Keep its authored scale and express the planar pose with ordinary
        translate/rotate ops instead.  Cluster collision overlays such as
        SeatOverlay are direct children of the identity ``/World/Assets``
        frame, which makes the world and local planar deltas identical.
        """

        if (
            not prim.IsA(UsdGeom.Cube)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            return False
        xformable = UsdGeom.Xformable(prim)
        original_ops = list(xformable.GetOrderedXformOps())
        marker = [
            op for op in original_ops
            if op.GetOpName() == "xformOp:rotateZ:houseLayout"
        ]
        if len(marker) > 1:
            raise RuntimeError(
                f"multiple House collision-layout ops on {prim.GetPath()}"
            )
        if marker:
            authored.append({"path": str(prim.GetPath()), "role": role})
            return True
        translate_ops = [
            op for op in original_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        scale_ops = [
            op for op in original_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ]
        if len(translate_ops) != 1 or len(scale_ops) != 1:
            raise ValueError(
                f"collision Cube {prim.GetPath()} needs one translate and "
                "one scale op"
            )
        parent_world = xformable.ComputeParentToWorldTransform(
            Usd.TimeCode.Default())
        if not Gf.IsClose(parent_world, Gf.Matrix4d(1.0), 1e-9):
            raise ValueError(
                f"collision Cube {prim.GetPath()} must have an identity "
                "parent frame"
            )
        source_translation = translate_ops[0].Get(Usd.TimeCode.Default())
        transformed_xy = transform_points(
            (float(source_translation[0]), float(source_translation[1])),
            transform,
        )
        with Usd.EditContext(stage, stage.GetRootLayer()):
            rotate_op = xformable.AddRotateZOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="houseLayout",
            )
            translate_ops[0].Set(Gf.Vec3d(
                float(transformed_xy[0]),
                float(transformed_xy[1]),
                float(source_translation[2]),
            ))
            insertion = original_ops.index(translate_ops[0]) + 1
            ordered_ops = [
                *original_ops[:insertion],
                rotate_op,
                *original_ops[insertion:],
            ]
            if (
                not xformable.SetXformOpOrder(
                    ordered_ops, xformable.GetResetXformStack()
                )
                or not rotate_op.Set(float(transform.yaw_deg))
            ):
                raise RuntimeError(
                    f"could not author collision layout on {prim.GetPath()}"
                )
        authored.append({"path": str(prim.GetPath()), "role": role})
        return True

    def author_world_transform(path: str, transform: SE2, role: str) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"layout prim is missing from working scene: {path}")
        if _is_identity(transform):
            return
        if author_collision_cube_pose(prim, transform, role):
            return
        xformable = UsdGeom.Xformable(prim)
        matching_ops = [
            existing
            for existing in xformable.GetOrderedXformOps()
            if existing.GetOpName() == "xformOp:transform:houseLayout"
        ]
        if len(matching_ops) > 1:
            raise RuntimeError(
                f"multiple House layout transform ops on {path}"
            )
        original_ops = list(xformable.GetOrderedXformOps())
        reset_xform_stack = xformable.GetResetXformStack()
        parent_world = xformable.ComputeParentToWorldTransform(
            Usd.TimeCode.Default())
        # Keep authored translate/rotate/scale ops intact.  In particular,
        # PhysX collision cubes such as SeatOverlay depend on their explicit
        # non-uniform scale op; collapsing the stack into one equivalent
        # matrix preserves the USD AABB but does not preserve the live
        # collision shape.  USD composes the first ordered op outermost in
        # this row-vector convention, so conjugate the desired world SE(2)
        # through the parent and prepend it to the existing local stack.
        local_overlay = (
            parent_world
            * matrix_for(transform)
            * parent_world.GetInverse()
        )
        if matching_ops:
            property_path = matching_ops[0].GetAttr().GetPath()
            if stage.GetRootLayer().GetPropertyAtPath(property_path) is None:
                # A clone may inherit the source prim's named House op through
                # an internal USD reference.  That inherited matrix belongs
                # to the source object's frame, not this clone's requested
                # extra_prim_frame.  Author a local value opinion so the clone
                # receives its own correct cluster transform.
                with Usd.EditContext(stage, stage.GetRootLayer()):
                    if not matching_ops[0].Set(local_overlay):
                        raise RuntimeError(
                            f"could not override inherited House layout on {path}"
                        )
            # A root-layer value means a disposable scene is being retried
            # after this exact prim already received the selected overlay.
            authored.append({"path": path, "role": role})
            return
        with Usd.EditContext(stage, stage.GetRootLayer()):
            op = xformable.AddTransformOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="houseLayout",
            )
            if (
                not xformable.SetXformOpOrder(
                    [op, *original_ops], reset_xform_stack
                )
                or not op.Set(local_overlay)
            ):
                raise RuntimeError(f"could not author House layout on {path}")
        authored.append({"path": path, "role": role})

    for cluster_id, transform in effective.transforms.items():
        for prim_root in effective.cluster_prim_roots.get(cluster_id, ()):
            author_world_transform(prim_root, transform, cluster_id)
    task_transform = transform_for_frame(effective, frame)
    if robot_path:
        author_world_transform(robot_path, task_transform, "task_robot")
    inherited: list[dict[str, str]] = []
    extra_frames = dict(extra_prim_frames or {})
    cluster_roots = {
        prim_root: cluster_id
        for cluster_id, prim_roots in effective.cluster_prim_roots.items()
        for prim_root in prim_roots
    }
    for path, prim_frame in sorted(extra_frames.items()):
        if not isinstance(path, str) or not path.startswith("/World/"):
            raise ValueError(f"extra layout prim path is invalid: {path!r}")
        if not isinstance(prim_frame, str) or not prim_frame:
            raise ValueError(f"extra layout frame is invalid for {path}")
        transform = transform_for_frame(effective, prim_frame)
        owners = [
            (root, cluster_id)
            for root, cluster_id in cluster_roots.items()
            if path == root or path.startswith(root.rstrip("/") + "/")
        ]
        if owners:
            root, cluster_id = max(owners, key=lambda item: len(item[0]))
            if transform != effective.transforms[cluster_id]:
                raise ValueError(
                    f"extra layout prim {path} inherits {cluster_id}, not "
                    f"frame {prim_frame}"
                )
            inherited.append({
                "path": path,
                "frame": prim_frame,
                "cluster_root": root,
            })
            continue
        if robot_path and path == robot_path:
            raise ValueError("robot path cannot also be an extra layout prim")
        author_world_transform(path, transform, f"episode_{prim_frame}")
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save House layout scene {scene_path}")
    return {
        "ok": True,
        "scene": str(scene_path),
        "layout_id": effective.layout_id,
        "layout_ordinal": effective.ordinal,
        "layout_status": effective.status,
        "task_frame": frame,
        "task_cluster": cluster_for_frame(effective, frame),
        "authored_prims": authored,
        "extra_prim_frames": extra_frames,
        "extra_inherited_prims": inherited,
        "coordinate_transform_applied": True,
        "source_xform_ops_preserved": True,
        "live_navigation_validated": False,
        "live_collision_validated": False,
        "physics_replay_validated": False,
    }


def usd_smoke_all(
    source_scene: Path | str = SOURCE_BASELINE,
    catalog_path: Path | str = DEFAULT_LAYOUTS_PATH,
    *,
    task_frame: str = "coffee_table",
) -> dict[str, Any]:
    """Author, save, and reopen all 50 recipes in disposable USD layers.

    Each temporary root layer sublayers the source scene by absolute path.  The
    layout opinions are therefore written only to the temporary root while all
    relative asset paths continue to resolve from their canonical source
    layers.  This check proves USD coordinate authoring, not collision or
    physics behavior.
    """

    try:
        from pxr import Sdf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "usd-smoke requires a Python environment with pxr (for example mesh311)"
        ) from exc

    source_path = Path(source_scene).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"House source baseline does not exist: {source_path}")
    layouts = load_layouts(catalog_path)
    source_before = source_path.read_bytes()
    passed_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    temporary_root: Path | None = None

    with tempfile.TemporaryDirectory(prefix="house-layout-usd-smoke-") as raw_temp:
        temporary_root = Path(raw_temp)
        for layout in layouts:
            scratch_scene = temporary_root / f"{layout.layout_id}.usda"
            try:
                root_layer = Sdf.Layer.CreateNew(str(scratch_scene))
                if root_layer is None:
                    raise RuntimeError("could not create disposable USD root layer")
                root_layer.subLayerPaths = [str(source_path)]
                if not root_layer.Save():
                    raise RuntimeError("could not save disposable USD root layer")

                result = apply_scene_overlay(
                    scratch_scene,
                    layout,
                    frame=task_frame,
                    robot_path="/World/Robot",
                )
                expected_paths = {
                    path
                    for paths in layout.cluster_prim_roots.values()
                    for path in paths
                }
                expected_paths.add("/World/Robot")
                authored_paths = {
                    row["path"] for row in result["authored_prims"]
                }
                if authored_paths != expected_paths:
                    missing = sorted(expected_paths - authored_paths)
                    extra = sorted(authored_paths - expected_paths)
                    raise RuntimeError(
                        f"authored prim set differs: missing={missing}, extra={extra}"
                    )

                reopened = Usd.Stage.Open(
                    str(scratch_scene), load=Usd.Stage.LoadNone)
                if reopened is None:
                    raise RuntimeError("could not reopen disposable USD scene")
                reopened_root = reopened.GetRootLayer()
                missing_opinions = []
                for prim_path in sorted(expected_paths):
                    prim = reopened.GetPrimAtPath(prim_path)
                    layout_ops = (
                        [
                            op for op in UsdGeom.Xformable(
                                prim).GetOrderedXformOps()
                            if op.GetOpName().endswith(":houseLayout")
                        ]
                        if prim else []
                    )
                    attribute = layout_ops[0].GetAttr() if len(layout_ops) == 1 else None
                    opinion = (
                        reopened_root.GetPropertyAtPath(
                            attribute.GetPath())
                        if attribute else None
                    )
                    if (
                        not prim
                        or opinion is None
                        or not attribute
                        or not attribute.HasAuthoredValueOpinion()
                        or attribute.Get() is None
                    ):
                        missing_opinions.append(prim_path)
                if missing_opinions:
                    raise RuntimeError(
                        "reopened scene lacks layout opinions on: "
                        + ", ".join(missing_opinions)
                    )
                passed_rows.append(
                    {
                        "layout_id": layout.layout_id,
                        "layout_ordinal": layout.ordinal,
                        "authored_prims": len(expected_paths),
                        "author_save_reopen": "passed",
                    }
                )
            except Exception as exc:
                failed_rows.append(
                    {
                        "layout_id": layout.layout_id,
                        "layout_ordinal": layout.ordinal,
                        "error": str(exc),
                    }
                )

    source_unchanged = source_path.read_bytes() == source_before
    temporary_files_removed = bool(
        temporary_root is not None and not temporary_root.exists())
    passed = (
        len(layouts) == LAYOUT_COUNT
        and len(passed_rows) == LAYOUT_COUNT
        and not failed_rows
        and source_unchanged
        and temporary_files_removed
    )
    authored_counts = sorted({
        int(row["authored_prims"]) for row in passed_rows
    })
    return {
        "schema": "real2sim.house_layout_usd_smoke.v1",
        "candidate_status": LAYOUT_STATUS,
        "passed": passed,
        "result": (
            "passed_usd_author_save_reopen"
            if passed
            else "failed_usd_author_save_reopen"
        ),
        "source_scene": str(source_path),
        "evaluated_layouts": len(layouts),
        "author_save_reopen_passed": len(passed_rows),
        "authored_prims_per_layout": (
            authored_counts[0] if len(authored_counts) == 1 else authored_counts
        ),
        "source_baseline_unchanged": source_unchanged,
        "runtime_scene_touched": False,
        "temporary_files_removed": temporary_files_removed,
        "failed_layouts": failed_rows,
        "layouts": passed_rows,
        "live_usd_collision": "not_run",
        "physics_replay": "not_run",
        "production_acceptance": "pending",
    }


def _robot_positions() -> Mapping[str, Any]:
    teleop = HERE.parent / "teleop"
    if str(teleop) not in sys.path:
        sys.path.insert(0, str(teleop))
    from robot_positions import ROBOT_POSITIONS

    return ROBOT_POSITIONS


def station_pose(
    station_key: str,
    layout: HouseLayout | int,
    *,
    catalog_path: Path | str = DEFAULT_LAYOUTS_PATH,
) -> StationPose:
    """Return a named robot station in the selected layout's world frame."""

    effective = layout_for_ordinal(layout, catalog_path) if isinstance(layout, int) else layout
    positions = _robot_positions()
    try:
        source = positions[station_key]
    except KeyError as exc:
        raise KeyError(f"unknown robot station {station_key!r}") from exc

    manifest = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    owners = [
        cluster_id
        for cluster_id, cluster in manifest["clusters"].items()
        if station_key in cluster.get("station_keys", [])
    ]
    if not owners:
        return StationPose(station_key, tuple(source.xy_m), float(source.yaw_deg))
    if len(owners) != 1:
        raise ValueError(f"robot station {station_key!r} belongs to multiple clusters")
    transform = effective.transforms[owners[0]]
    xy = transform_points(source.xy_m, transform)
    return StationPose(
        station_key,
        (float(xy[0]), float(xy[1])),
        float(source.yaw_deg) + transform.yaw_deg,
    )


def _fixture_corners(
    fixture: PlanarFixture, inflate_m: float = 0.0
) -> tuple[tuple[float, float], ...]:
    half_x = fixture.size_xy_m[0] * 0.5 + inflate_m
    half_y = fixture.size_xy_m[1] * 0.5 + inflate_m
    cosine = math.cos(math.radians(fixture.yaw_deg))
    sine = math.sin(math.radians(fixture.yaw_deg))
    return tuple(
        (
            fixture.center_xy_m[0] + cosine * x - sine * y,
            fixture.center_xy_m[1] + sine * x + cosine * y,
        )
        for x, y in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    )


def _point_in_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ):
            inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(
        0.0,
        min(
            1.0,
            (
                (point[0] - start[0]) * dx
                + (point[1] - start[1]) * dy
            )
            / denominator,
        ),
    )
    return math.hypot(
        point[0] - (start[0] + t * dx),
        point[1] - (start[1] + t * dy),
    )


def _polygon_boundary_distance(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> float:
    return min(
        _point_segment_distance(point, polygon[index - 1], polygon[index])
        for index in range(len(polygon))
    )


def _point_to_fixture_distance(
    point: tuple[float, float], fixture: PlanarFixture
) -> float:
    angle = math.radians(-fixture.yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = point[0] - fixture.center_xy_m[0]
    dy = point[1] - fixture.center_xy_m[1]
    local_x = cosine * dx - sine * dy
    local_y = sine * dx + cosine * dy
    outside_x = max(abs(local_x) - fixture.size_xy_m[0] * 0.5, 0.0)
    outside_y = max(abs(local_y) - fixture.size_xy_m[1] * 0.5, 0.0)
    return math.hypot(outside_x, outside_y)


def _fixtures_overlap(
    left: PlanarFixture, right: PlanarFixture, gap_m: float
) -> bool:
    left_points = _fixture_corners(left, gap_m * 0.5)
    right_points = _fixture_corners(right, gap_m * 0.5)
    axes: list[tuple[float, float]] = []
    for points in (left_points, right_points):
        for index in range(2):
            edge_x = points[index + 1][0] - points[index][0]
            edge_y = points[index + 1][1] - points[index][1]
            length = math.hypot(edge_x, edge_y)
            axes.append((-edge_y / length, edge_x / length))
    for axis in axes:
        left_projection = [x * axis[0] + y * axis[1] for x, y in left_points]
        right_projection = [x * axis[0] + y * axis[1] for x, y in right_points]
        if (
            max(left_projection) <= min(right_projection) + 1e-12
            or max(right_projection) <= min(left_projection) + 1e-12
        ):
            return False
    return True


def _preflight_inputs(
    catalog_path: Path | str,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[PlanarFixture, ...],
    Mapping[str, tuple[float, float]],
    Mapping[str, float],
    Mapping[str, str],
]:
    manifest = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    geometry = manifest.get("preflight_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("layout catalog must define preflight_geometry")
    raw_floor = geometry.get("floor_polygon_xy_m")
    if not isinstance(raw_floor, list) or len(raw_floor) < 3:
        raise ValueError("preflight floor polygon must contain at least three points")
    floor = tuple(
        _pair(point, label=f"preflight floor point {index}")
        for index, point in enumerate(raw_floor)
    )

    raw_portals = geometry.get("navigation_portals_xy_m")
    if not isinstance(raw_portals, dict) or "main_hub" not in raw_portals:
        raise ValueError("preflight navigation portals must include main_hub")
    portals = {
        str(key): _pair(value, label=f"navigation portal {key}")
        for key, value in raw_portals.items()
    }

    required_rules = (
        "robot_base_radius_m",
        "carried_object_margin_m",
        "minimum_route_clearance_m",
        "grid_resolution_m",
        "station_bridge_radius_m",
        "portal_snap_radius_m",
        "minimum_fixture_gap_m",
    )
    raw_rules = geometry.get("rules")
    if not isinstance(raw_rules, dict):
        raise ValueError("preflight geometry must define rules")
    rules = {key: float(raw_rules[key]) for key in required_rules}
    if any(not math.isfinite(value) or value < 0.0 for value in rules.values()):
        raise ValueError("preflight rules must be finite non-negative numbers")
    if rules["grid_resolution_m"] == 0.0:
        raise ValueError("preflight grid resolution must be positive")

    raw_fixtures = geometry.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("preflight geometry must define fixtures")
    cluster_ids = set(manifest.get("clusters", {}))
    fixtures: list[PlanarFixture] = []
    fixture_ids: set[str] = set()
    for index, row in enumerate(raw_fixtures):
        if not isinstance(row, dict):
            raise ValueError(f"preflight fixture {index} must be an object")
        fixture_id = row.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError(f"preflight fixture {index} needs an ID")
        if fixture_id in fixture_ids:
            raise ValueError(f"duplicate preflight fixture {fixture_id!r}")
        fixture_ids.add(fixture_id)
        cluster = row.get("cluster")
        if cluster is not None and cluster not in cluster_ids:
            raise ValueError(
                f"preflight fixture {fixture_id!r} uses unknown cluster {cluster!r}")
        size = _pair(row.get("size_xy_m"), label=f"{fixture_id}.size_xy_m")
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError(f"preflight fixture {fixture_id!r} needs positive size")
        if not isinstance(row.get("layout_movable"), bool):
            raise ValueError(
                f"preflight fixture {fixture_id!r} must declare layout_movable")
        fixtures.append(
            PlanarFixture(
                fixture_id=fixture_id,
                center_xy_m=_pair(
                    row.get("center_xy_m"), label=f"{fixture_id}.center_xy_m"),
                size_xy_m=size,
                yaw_deg=float(row.get("yaw_deg", 0.0)),
                cluster=cluster,
                layout_movable=bool(row["layout_movable"]),
            )
        )

    station_owners: dict[str, str] = {}
    for cluster_id, cluster in manifest["clusters"].items():
        for station_key in cluster.get("station_keys", []):
            if station_key in station_owners:
                raise ValueError(
                    f"robot station {station_key!r} belongs to multiple clusters")
            station_owners[station_key] = cluster_id
    return floor, tuple(fixtures), portals, rules, station_owners


def _layout_preflight_geometry(
    layout: HouseLayout,
    catalog_path: Path | str,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[PlanarFixture, ...],
    Mapping[str, tuple[float, float]],
    Mapping[str, float],
    Mapping[str, StationPose],
]:
    floor, fixtures, portals, rules, station_owners = _preflight_inputs(
        catalog_path)
    transformed_fixtures = []
    for fixture in fixtures:
        if fixture.cluster is None:
            transformed_fixtures.append(fixture)
            continue
        transform = layout.transforms[fixture.cluster]
        center = transform_points(fixture.center_xy_m, transform)
        transformed_fixtures.append(
            PlanarFixture(
                fixture_id=fixture.fixture_id,
                center_xy_m=(float(center[0]), float(center[1])),
                size_xy_m=fixture.size_xy_m,
                yaw_deg=fixture.yaw_deg + transform.yaw_deg,
                cluster=fixture.cluster,
                layout_movable=fixture.layout_movable,
            )
        )

    stations: dict[str, StationPose] = {}
    for key, source in _robot_positions().items():
        cluster_id = station_owners.get(key)
        if cluster_id is None:
            stations[key] = StationPose(key, tuple(source.xy_m), float(source.yaw_deg))
            continue
        transform = layout.transforms[cluster_id]
        xy = transform_points(source.xy_m, transform)
        stations[key] = StationPose(
            key,
            (float(xy[0]), float(xy[1])),
            float(source.yaw_deg) + transform.yaw_deg,
        )
    return floor, tuple(transformed_fixtures), portals, rules, stations


def _grid_navigation_audit(
    floor: Sequence[tuple[float, float]],
    fixtures: Sequence[PlanarFixture],
    stations: Mapping[str, StationPose],
    portals: Mapping[str, tuple[float, float]],
    rules: Mapping[str, float],
) -> dict[str, Any]:
    resolution = rules["grid_resolution_m"]
    inflation = (
        rules["robot_base_radius_m"]
        + rules["carried_object_margin_m"]
        + rules["minimum_route_clearance_m"]
    )
    min_x = math.floor(min(point[0] for point in floor) / resolution) * resolution
    max_x = math.ceil(max(point[0] for point in floor) / resolution) * resolution
    min_y = math.floor(min(point[1] for point in floor) / resolution) * resolution
    max_y = math.ceil(max(point[1] for point in floor) / resolution) * resolution
    width = int(round((max_x - min_x) / resolution)) + 1
    height = int(round((max_y - min_y) / resolution)) + 1

    def grid_point(cell: tuple[int, int]) -> tuple[float, float]:
        return min_x + cell[0] * resolution, min_y + cell[1] * resolution

    def is_free(point: tuple[float, float]) -> bool:
        return bool(
            _point_in_polygon(point, floor)
            and _polygon_boundary_distance(point, floor) + 1e-12 >= inflation
            and all(
                _point_to_fixture_distance(point, fixture) + 1e-12 >= inflation
                for fixture in fixtures
            )
        )

    free_cells = {
        (ix, iy)
        for ix in range(width)
        for iy in range(height)
        if is_free(grid_point((ix, iy)))
    }
    ordered_free_cells = sorted(free_cells)

    def nearest_cell(
        point: tuple[float, float], radius: float
    ) -> tuple[tuple[int, int] | None, float]:
        best: tuple[int, int] | None = None
        best_distance = float("inf")
        for cell in ordered_free_cells:
            candidate = grid_point(cell)
            distance = math.hypot(candidate[0] - point[0], candidate[1] - point[1])
            if distance < best_distance:
                best, best_distance = cell, distance
        if best_distance > radius + 1e-12:
            return None, best_distance
        return best, best_distance

    station_cells = {
        key: nearest_cell(pose.xy_m, rules["station_bridge_radius_m"])
        for key, pose in stations.items()
    }
    portal_cells = {
        key: nearest_cell(point, rules["portal_snap_radius_m"])
        for key, point in portals.items()
    }
    start = portal_cells["main_hub"][0]
    reached: set[tuple[int, int]] = set()
    if start is not None:
        reached.add(start)
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = cell[0] + dx, cell[1] + dy
                if neighbor in free_cells and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)

    def endpoints(
        poses: Mapping[str, tuple[float, float]],
        cells: Mapping[str, tuple[tuple[int, int] | None, float]],
    ) -> dict[str, Any]:
        rows = {}
        for key, point in poses.items():
            cell, snap_distance = cells[key]
            rows[key] = {
                "xy_m": list(point),
                "nearest_free_xy_m": list(grid_point(cell)) if cell else None,
                "bridge_or_snap_distance_m": (
                    snap_distance if math.isfinite(snap_distance) else None),
                "reachable": cell is not None and cell in reached,
            }
        return rows

    station_points = {key: pose.xy_m for key, pose in stations.items()}
    station_rows = endpoints(station_points, station_cells)
    portal_rows = endpoints(portals, portal_cells)
    passed = bool(
        start is not None
        and all(row["reachable"] for row in station_rows.values())
        and all(row["reachable"] for row in portal_rows.values())
    )
    return {
        "passed": passed,
        "inflation_radius_m": inflation,
        "grid_resolution_m": resolution,
        "free_cell_count": len(free_cells),
        "reachable_free_cell_count": len(reached),
        "portals": portal_rows,
        "stations": station_rows,
    }


def preflight_layout(
    layout: HouseLayout | int,
    catalog_path: Path | str = DEFAULT_LAYOUTS_PATH,
) -> dict[str, Any]:
    """Recompute conservative planar and navigation checks for one candidate."""

    effective = (
        layout_for_ordinal(layout, catalog_path)
        if isinstance(layout, int)
        else layout
    )
    floor, fixtures, portals, rules, stations = _layout_preflight_geometry(
        effective, catalog_path)

    containment_errors = []
    for fixture in fixtures:
        if not fixture.layout_movable:
            continue
        outside = [
            list(corner)
            for corner in _fixture_corners(fixture)
            if not _point_in_polygon(corner, floor)
        ]
        if outside:
            containment_errors.append(
                {"fixture_id": fixture.fixture_id, "outside_corners": outside})

    overlap_errors = []
    for index, left in enumerate(fixtures):
        for right in fixtures[index + 1:]:
            if left.cluster is not None and left.cluster == right.cluster:
                continue
            if not (left.layout_movable or right.layout_movable):
                continue
            if _fixtures_overlap(left, right, rules["minimum_fixture_gap_m"]):
                overlap_errors.append([left.fixture_id, right.fixture_id])

    station_errors = []
    for station in stations.values():
        if not _point_in_polygon(station.xy_m, floor):
            station_errors.append(
                {"station": station.key, "reason": "outside_floor"})
            continue
        blockers = [
            fixture.fixture_id
            for fixture in fixtures
            if _point_to_fixture_distance(station.xy_m, fixture) <= 1e-9
        ]
        if blockers:
            station_errors.append(
                {
                    "station": station.key,
                    "reason": "inside_fixture",
                    "fixtures": blockers,
                }
            )

    planar_checks = {
        "all_audited_fixture_footprints_inside_floor": not containment_errors,
        "no_intergroup_fixture_overlap": not overlap_errors,
        "all_station_origins_inside_floor_and_outside_fixtures": (
            not station_errors),
    }
    navigation = _grid_navigation_audit(
        floor, fixtures, stations, portals, rules)
    planar_passed = all(planar_checks.values())
    return {
        "layout_id": effective.layout_id,
        "layout_ordinal": effective.ordinal,
        "candidate_status": LAYOUT_STATUS,
        "passed": bool(planar_passed and navigation["passed"]),
        "planar_geometry": {
            "passed": planar_passed,
            "checks": planar_checks,
            "errors": {
                "containment": containment_errors,
                "overlap": overlap_errors,
                "station": station_errors,
            },
        },
        "grid_navigation": navigation,
        "live_usd_collision": "not_run",
        "physics_replay": "not_run",
        "production_acceptance": "pending",
    }


def preflight_all(
    catalog_path: Path | str = DEFAULT_LAYOUTS_PATH,
) -> dict[str, Any]:
    """Recompute the 2D/grid gate for every explicit House layout recipe."""

    audits = [
        preflight_layout(layout, catalog_path)
        for layout in load_layouts(catalog_path)
    ]
    planar_passed = sum(
        bool(row["planar_geometry"]["passed"]) for row in audits)
    grid_passed = sum(bool(row["grid_navigation"]["passed"]) for row in audits)
    failed_layouts = []
    layout_rows = []
    for row in audits:
        summary = {
            "layout_id": row["layout_id"],
            "layout_ordinal": row["layout_ordinal"],
            "planar_geometry": (
                "passed" if row["planar_geometry"]["passed"] else "failed"),
            "grid_navigation": (
                "passed" if row["grid_navigation"]["passed"] else "failed"),
            "free_cell_count": row["grid_navigation"]["free_cell_count"],
            "reachable_free_cell_count": row["grid_navigation"][
                "reachable_free_cell_count"],
        }
        layout_rows.append(summary)
        if not row["passed"]:
            failed_layouts.append(
                {
                    **summary,
                    "planar_errors": row["planar_geometry"]["errors"],
                    "unreachable_portals": [
                        key
                        for key, endpoint in row["grid_navigation"]["portals"].items()
                        if not endpoint["reachable"]
                    ],
                    "unreachable_stations": [
                        key
                        for key, endpoint in row["grid_navigation"]["stations"].items()
                        if not endpoint["reachable"]
                    ],
                }
            )
    passed = len(audits) == LAYOUT_COUNT and not failed_layouts
    return {
        "schema": "real2sim.house_layout_preflight.v1",
        "candidate_status": LAYOUT_STATUS,
        "passed": passed,
        "result": (
            "passed_2d_navigation_candidates"
            if passed
            else "failed_2d_navigation_preflight"
        ),
        "evaluated_layouts": len(audits),
        "planar_geometry_passed": planar_passed,
        "grid_navigation_passed": grid_passed,
        "failed_layouts": failed_layouts,
        "layouts": layout_rows,
        "live_usd_collision": "not_run",
        "physics_replay": "not_run",
        "production_acceptance": "pending",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    apply_parser = subparsers.add_parser(
        "apply-scene", help="author one candidate into a working USD scene")
    apply_parser.add_argument("--scene", type=Path, required=True)
    apply_parser.add_argument("--task-key", required=True)
    apply_parser.add_argument("--core-ordinal", type=int, required=True)
    apply_parser.add_argument("--layout-seed", type=int)
    apply_parser.add_argument(
        "--layout-id",
        help=(
            "apply one explicit source recipe; used by reviewed delivery "
            "catalogs whose final ordinal differs from the source ordinal"
        ),
    )
    apply_parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_LAYOUTS_PATH,
        help="explicit catalog used by candidate-only layout canaries",
    )
    apply_parser.add_argument("--task-frame", required=True)
    apply_parser.add_argument("--robot-path", default="/World/Robot")
    apply_parser.add_argument("--extra-prim-frames", type=Path)
    preflight_parser = subparsers.add_parser(
        "preflight", help="recompute all conservative 2D/grid candidate checks")
    preflight_parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_LAYOUTS_PATH)
    smoke_parser = subparsers.add_parser(
        "usd-smoke", help="author, save, and reopen all 50 disposable USD layouts")
    smoke_parser.add_argument(
        "--source-scene", type=Path, default=SOURCE_BASELINE)
    smoke_parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_LAYOUTS_PATH)
    smoke_parser.add_argument("--task-frame", default="coffee_table")
    args = parser.parse_args(argv)
    if args.command == "apply-scene":
        extra_prim_frames = None
        if args.extra_prim_frames is not None:
            extra_prim_frames = json.loads(
                args.extra_prim_frames.read_text(encoding="utf-8")
            )
            if not isinstance(extra_prim_frames, dict):
                raise ValueError("extra prim frames must be a JSON object")
        selected_layout = (
            layout_for_id(args.layout_id, args.catalog)
            if args.layout_id is not None
            else scheduled_layout_for_core(
                args.task_key,
                args.core_ordinal,
                args.catalog,
                seed=args.layout_seed,
            )
        )
        result = apply_scene_overlay(
            args.scene,
            selected_layout,
            frame=args.task_frame,
            robot_path=args.robot_path,
            extra_prim_frames=extra_prim_frames,
        )
        result.update({
            "layout_schedule_schema": LAYOUT_SCHEDULE_SCHEMA,
            "layout_schedule_seed": (
                task_layout_schedule_seed(args.task_key)
                if args.layout_seed is None else args.layout_seed
            ),
            "layout_selection_mode": (
                "explicit_layout_id" if args.layout_id is not None
                else "task_seeded_permutation"
            ),
            "task_key": args.task_key,
            "catalog": str(args.catalog.resolve()),
        })
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = preflight_all(args.catalog)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.command == "usd-smoke":
        result = usd_smoke_all(
            args.source_scene,
            args.catalog,
            task_frame=args.task_frame,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 2
    for item in load_layouts():
        print(item.ordinal, item.layout_id, item.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

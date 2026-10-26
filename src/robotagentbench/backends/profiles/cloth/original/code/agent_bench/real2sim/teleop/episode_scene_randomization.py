#!/usr/bin/env python3
"""Deterministic per-episode distractor layout for current real2sim scenes.

The planner is pure Python and can be tested without USD.  The CLI application
path is run with the mesh311 interpreter and is deliberately limited to small
props.  It snapshots the canonical furniture transforms, authors the episode
props, verifies that the target/protected supports and furniture did not move,
and writes an atomic JSON sidecar next to the scene.

The policy version, benchmark task id, and exact REAL episode UUID or benchmark
seed determine the random stream. The same inputs therefore produce the same
layout, while different episodes receive different object order, XY jitter,
and yaw. Git records policy source changes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


POLICY_PATH = Path(__file__).with_name("episode_distractor_policy_v1.json")
# v2 2026-08-04: denser refreshes on the two big surfaces (coffee_table and
# sofa desired_count 3 -> 5) per the family-7 scale-up direction; the sampler
# keeps an explicit minimum_count floor, so a shrunken per-episode pool
# degrades gracefully instead of silently truncating.
# v3 2026-08-04 (late): EmptyCan (unsealed sprite specialization) + two
# clones + PaperBallClone1 join the pool, and tasks 29/30 may see same-type
# clones of their own target (real tidy-table scenes hold 1-3 cups and 1-2
# paper balls; the recorded action still manipulates the original prim).
# v4 2026-08-05: tasks 29 and 78 join the randomization scope, and all three
# tidy-table tasks share the same real-footage companion set (two crumpled
# papers + three disposable cups) so every produced episode looks like the
# recordings. The version changed, so layouts differ from v3 renders.
# v5 2026-08-06: strict high-density floors are coffee table >=5, sofa >=5,
# side table >=3.  Tidy-table tasks may additionally carry two same-type
# clones next to the one designated target (three visually identical primary
# assets total); those clones do not count toward the distractor floor.
# v6 2026-08-07: shelf picks reserve a gripper-width target corridor.  When
# the full foreground cannot fit around that corridor, the planner omits the
# mathematically smallest number of stock items instead of reducing the
# collision gap.  Assets known to be physically unstable may remain targetable
# while being excluded from other tasks' production distractor inventory.
# v10 2026-08-07: the CoffeeTable carries eight distractors in two semantic
# world-Y halves.  The bin-facing half draws only the four reviewed trash
# classes; the shelf-facing half draws from every production-approved shelf
# beverage/snack foreground type.  Sofa and side-table density stay unchanged.
# v11 2026-08-08: coke_mini_fridge_pack remains a retained authored bundle but
# has no public runtime release approval.  Exclude it from production
# background selection instead of weakening the fail-closed approval gate.
# v12 2026-08-08: pring_g and starbucks remain targetable shelf stock but the
# canonical House prims have no enabled collision shape, so they cannot enter
# the dynamic production distractor pool.
POLICY_VERSION = "episode-distractor-v12-2026-08-08"
from machine_paths import DEMO_ROOT as DEFAULT_DEMO_ROOT  # noqa: E402
from machine_paths import RECORD_ROOT as DEFAULT_RECORD_ROOT  # noqa: E402
from machine_paths import MESH311_PYTHON  # noqa: E402
import group_shift  # noqa: E402
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
HAND_VALUES = frozenset({"l", "r", "?"})


def load_policy(path: Path | str = POLICY_PATH) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "real2sim.episode_distractor_policy.v1"
        or payload.get("status") != "current"
    ):
        raise ValueError("episode distractor policy is not current v1")
    return payload


def _seed_from_explicit_identity(*parts: str) -> int:
    return random.Random("\0".join(parts)).getrandbits(64)


def eligible_task_ids(policy: Mapping | None = None) -> frozenset[str]:
    effective = policy or load_policy()
    return frozenset(
        str(value)
        for key in ("eligible_small_task_ids", "eligible_production_task_ids")
        for value in effective.get(key, ())
    )


def randomization_enabled(
    task_id: str,
    policy: Mapping | None = None,
) -> bool:
    override = os.environ.get("REAL2SIM_SCENE_RANDOMIZATION", "").strip()
    if override:
        if override == "0":
            return False
        if override != "1":
            raise ValueError(
                "REAL2SIM_SCENE_RANDOMIZATION must be 0, 1, or unset"
            )
    return str(task_id) in eligible_task_ids(policy)


def _identity_token(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
) -> str:
    if seed_override is not None and str(seed_override).strip():
        token = str(seed_override).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", token):
            raise ValueError(
                "SCENE_RANDOMIZATION_SEED contains unsafe characters"
            )
        return "benchmark:" + token
    uid = str(episode_uid).strip().lower()
    if uid:
        if not UUID_RE.fullmatch(uid):
            raise ValueError(
                "episode scene randomization requires an exact UUID"
            )
        return "real:" + uid
    return "manual:" + str(task_id)


def randomization_identity(
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    *,
    policy: Mapping | None = None,
) -> dict:
    effective = policy or load_policy()
    token = _identity_token(
        task_id=str(task_id),
        episode_uid=episode_uid,
        seed_override=seed_override,
    )
    seed = _seed_from_explicit_identity(
        POLICY_VERSION, "clutter", str(task_id), token)
    return {
        "schema": "real2sim.episode_scene_randomization.identity.v1",
        "enabled": str(task_id) in eligible_task_ids(effective),
        "policy_version": POLICY_VERSION,
        "task_id": str(task_id),
        "episode_token": token,
        "seed_u64": seed,
    }


def infer_grasp_hand_from_hdf5(path: Path | str) -> dict:
    """Resolve the moving arm directly from the recorded 14-DoF commands."""
    import h5py
    import numpy as np

    source = Path(path)
    with h5py.File(source, "r") as stream:
        action = np.asarray(stream["/action/joint/position"], dtype=float)
    if action.ndim != 2 or action.shape[1] != 14 or action.shape[0] < 2:
        raise ValueError(
            f"invalid 14-DoF action trajectory in {source}"
        )
    if not np.isfinite(action).all():
        raise ValueError(f"non-finite action trajectory in {source}")
    scores = []
    for arm_slice in (slice(0, 7), slice(7, 14)):
        arm = action[:, arm_slice]
        scores.append(float(np.max(np.linalg.norm(arm - arm[0], axis=1))))
    dominant = max(scores)
    other = min(scores)
    hand = "?"
    if dominant >= 0.05 and dominant >= max(5.0 * other, other + 0.05):
        hand = "l" if scores[0] > scores[1] else "r"
    return {
        "schema": "real2sim.episode_grasp_hand_audit.v1",
        "hand": hand,
        "left_arm_max_delta_rad": scores[0],
        "right_arm_max_delta_rad": scores[1],
        "method": "maximum 7-DoF command displacement from frame zero",
        "hdf5": str(source),
    }


def audit_episode_hand(
    episode_uid: str,
    *,
    record_root: Path | str = DEFAULT_RECORD_ROOT,
) -> dict | None:
    uid = str(episode_uid).strip().lower()
    if not UUID_RE.fullmatch(uid):
        return None
    path = Path(record_root) / uid / "record" / "raw_joints.h5"
    if not path.is_file():
        return None
    return infer_grasp_hand_from_hdf5(path)


def context_object_descriptors(
    context: Mapping | None,
    *,
    policy: Mapping | None = None,
) -> list[dict]:
    """Convert initializer secondary-object bounds into planner reservations."""
    if not isinstance(context, Mapping):
        return []
    objects = context.get("secondary_objects")
    if not isinstance(objects, Mapping):
        return []
    effective = policy or load_policy()
    result = []
    for name, raw_bounds in objects.items():
        if not isinstance(raw_bounds, Mapping):
            continue
        minimum = raw_bounds.get("min")
        maximum = raw_bounds.get("max")
        if (
            not isinstance(minimum, Sequence)
            or isinstance(minimum, (str, bytes))
            or not isinstance(maximum, Sequence)
            or isinstance(maximum, (str, bytes))
            or len(minimum) < 3
            or len(maximum) < 3
        ):
            continue
        values = [float(value) for value in (*minimum[:3], *maximum[:3])]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"secondary context {name!r} has non-finite bounds"
            )
        xmin, ymin, zmin, xmax, ymax, zmax = values
        if xmax <= xmin or ymax <= ymin or zmax <= zmin:
            raise ValueError(
                f"secondary context {name!r} has invalid bounds"
            )
        descriptor = {
            "id": f"initializer-context:{name}",
            "x_m": (xmin + xmax) / 2.0,
            "y_m": (ymin + ymax) / 2.0,
            "z_m": (zmin + zmax) / 2.0,
            "width_m": xmax - xmin,
            "depth_m": ymax - ymin,
            "height_m": zmax - zmin,
            "counts_toward_density": True,
        }
        for surface_name, surface in effective["surfaces"].items():
            sxmin, sxmax, symin, symax = map(
                float, surface["bounds_xy_m"]
            )
            top = float(surface["surface_z_m"])
            if (
                sxmin <= descriptor["x_m"] <= sxmax
                and symin <= descriptor["y_m"] <= symax
                and zmin <= top + 0.05
                and zmax >= top - 0.02
            ):
                descriptor["surface"] = surface_name
                break
        result.append(descriptor)
    return result


def reconcile_hand(
    initializer_hand: str,
    hand_audit: Mapping | None,
) -> tuple[str, dict]:
    initialized = str(initializer_hand or "?").strip().lower()
    if initialized not in HAND_VALUES:
        initialized = "?"
    audited = (
        str(hand_audit.get("hand", "?")).strip().lower()
        if hand_audit else "?"
    )
    resolved = audited if audited in {"l", "r"} else initialized
    return resolved, {
        "initializer_hand": initialized,
        "audited_hand": audited,
        "resolved_hand": resolved,
        "initializer_mismatch_corrected": (
            audited in {"l", "r"}
            and initialized in {"l", "r"}
            and audited != initialized
        ),
    }


@dataclass(frozen=True)
class OBB:
    cx: float
    cy: float
    width: float
    depth: float
    yaw_deg: float
    label: str = ""

    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        angle = math.radians(self.yaw_deg)
        u = (math.cos(angle), math.sin(angle))
        v = (-math.sin(angle), math.cos(angle))
        return u, v

    def corners(self) -> tuple[tuple[float, float], ...]:
        u, v = self.axes()
        hx = self.width / 2.0
        hy = self.depth / 2.0
        return tuple(
            (
                self.cx + sx * hx * u[0] + sy * hy * v[0],
                self.cy + sx * hx * u[1] + sy * hy * v[1],
            )
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        )


def _projection_radius(box: OBB, axis: tuple[float, float]) -> float:
    u, v = box.axes()
    return (
        box.width / 2.0 * abs(u[0] * axis[0] + u[1] * axis[1])
        + box.depth / 2.0 * abs(v[0] * axis[0] + v[1] * axis[1])
    )


def obb_clear(a: OBB, b: OBB, minimum_gap_m: float) -> bool:
    delta = (b.cx - a.cx, b.cy - a.cy)
    for axis in (*a.axes(), *b.axes()):
        distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        separation = (
            _projection_radius(a, axis)
            + _projection_radius(b, axis)
            + minimum_gap_m
        )
        if distance >= separation - 1e-12:
            return True
    return False


def obb_inside(
    box: OBB,
    bounds_xy: Sequence[float],
    edge_margin_m: float,
) -> bool:
    xmin, xmax, ymin, ymax = map(float, bounds_xy)
    return all(
        xmin + edge_margin_m <= x <= xmax - edge_margin_m
        and ymin + edge_margin_m <= y <= ymax - edge_margin_m
        for x, y in box.corners()
    )


def _interval_clear(
    center: float,
    half_width: float,
    intervals: Sequence[tuple[float, float]],
    gap: float,
) -> bool:
    left = center - half_width
    right = center + half_width
    return all(
        right + gap <= other_left + 1e-12
        or left >= other_right + gap - 1e-12
        for other_left, other_right in intervals
    )


def pack_shelf_row(
    items: Sequence[Mapping],
    *,
    x_bounds: Sequence[float],
    y_range: Sequence[float],
    minimum_gap_m: float,
    rng: random.Random,
    reserved_intervals: Sequence[tuple[float, float]] = (),
    max_omissions: int = 0,
    omitted_prim_paths: list[str] | None = None,
) -> list[dict]:
    """Place the densest feasible item set in one graspable front row.

    Build a complete packing before sampling any slack.  Greedily choosing
    independent random slots can strand an otherwise valid final item, which
    makes valid episode seeds fail nondeterministically.  If a protected target
    corridor makes the full row impossible, search omission counts in ascending
    order and never weaken ``minimum_gap_m``.
    """
    xmin, xmax = map(float, x_bounds)
    ymin, ymax = map(float, y_range)
    order = [dict(item) for item in items]
    rng.shuffle(order)
    clipped = sorted(
        (
            max(xmin, float(left)),
            min(xmax, float(right)),
        )
        for left, right in reserved_intervals
        if float(right) > xmin and float(left) < xmax
    )
    merged: list[list[float]] = []
    for left, right in clipped:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    segments: list[tuple[float, float]] = []
    cursor = xmin
    for left, right in merged:
        safe_right = left - minimum_gap_m
        if safe_right > cursor:
            segments.append((cursor, safe_right))
        cursor = max(cursor, right + minimum_gap_m)
    if cursor < xmax:
        segments.append((cursor, xmax))
    if not segments:
        raise ValueError("reserved shelf intervals consume the whole row")

    def required_width(indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        return (
            sum(float(order[index]["width_m"]) for index in indices)
            + minimum_gap_m * (len(indices) - 1)
        )

    # The current policy has at most one reserved target and therefore two
    # segments, but this exhaustive assignment remains valid for future rows
    # with a few protected objects.  Shelf rows intentionally stay small.
    segment_count = len(segments)
    maximum = max(0, min(int(max_omissions), len(order)))
    candidates: list[
        tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]
    ] = []
    for omission_count in range(maximum + 1):
        for omitted in itertools.combinations(
            range(len(order)), omission_count
        ):
            omitted_set = set(omitted)
            retained = [
                index for index in range(len(order))
                if index not in omitted_set
            ]
            assignment_count = segment_count ** len(retained)
            if assignment_count > 250_000:
                raise ValueError("shelf packing search exceeds the safety bound")
            for encoded in range(assignment_count):
                value = encoded
                buckets: list[list[int]] = [[] for _ in segments]
                for index in retained:
                    bucket = value % segment_count
                    value //= segment_count
                    buckets[bucket].append(index)
                if all(
                    required_width(bucket) <= right - left + 1e-12
                    for bucket, (left, right) in zip(buckets, segments)
                ):
                    candidates.append((
                        tuple(omitted),
                        tuple(tuple(bucket) for bucket in buckets),
                    ))
        if candidates:
            break
    if not candidates:
        raise ValueError(
            "cannot pack shelf items around the target within the "
            f"{maximum}-item omission limit"
        )
    omitted, selected = rng.choice(candidates)
    if omitted_prim_paths is not None:
        omitted_prim_paths.extend(
            str(order[index]["prim_path"]) for index in omitted
        )
    placed: list[dict] = []
    for indices, (left, right) in zip(selected, segments):
        if not indices:
            continue
        sequence = list(indices)
        rng.shuffle(sequence)
        slack = right - left - required_width(sequence)
        # Positive exponential samples normalized to the available slack give
        # episode-varying but mathematically safe leading/inter-item/trailing
        # whitespace.
        samples = [rng.expovariate(1.0) for _ in range(len(sequence) + 1)]
        scale = slack / sum(samples) if slack > 0.0 else 0.0
        gaps = [sample * scale for sample in samples]
        item_cursor = left + gaps[0]
        for position, index in enumerate(sequence):
            item = order[index]
            width = float(item["width_m"])
            x = item_cursor + width / 2.0
            placed.append({
                **item,
                "x_m": x,
                "y_m": rng.uniform(ymin, ymax),
                "yaw_deg": rng.uniform(-180.0, 180.0),
                "role": "interactive_foreground",
                "dynamic": True,
            })
            item_cursor += (
                width
                + minimum_gap_m
                + gaps[position + 1]
            )
    return sorted(placed, key=lambda item: item["x_m"])


def _background_row(
    items: Sequence[Mapping],
    *,
    x_bounds: Sequence[float],
    y_m: float,
    rng: random.Random,
) -> list[dict]:
    order = [dict(item) for item in items]
    rng.shuffle(order)
    widths = [float(item["width_m"]) for item in order]
    total = sum(widths)
    xmin, xmax = map(float, x_bounds)
    slack = xmax - xmin - total
    if slack < 0:
        raise ValueError("shelf background stock does not fit")
    gap = slack / (len(order) + 1)
    cursor = xmin + gap
    result = []
    for item, width in zip(order, widths):
        result.append({
            **item,
            "x_m": cursor + width / 2.0,
            "y_m": float(y_m),
            "yaw_deg": float(item.get("yaw_deg", 0.0)),
            "role": "physical_background_stock",
            "dynamic": True,
        })
        cursor += width + gap
    return result


def _sample_surface_yaw(
    item: Mapping,
    surface_name: str,
    rng: random.Random,
) -> float:
    if surface_name == "sofa":
        base = float(item.get("sofa_yaw_base_deg", 90.0))
        return base + rng.uniform(-18.0, 18.0) + rng.choice((0.0, 180.0))
    return rng.uniform(-180.0, 180.0)


def place_surface_objects(
    items: Sequence[Mapping],
    *,
    surface_name: str,
    surface: Mapping,
    rng: random.Random,
    reserved: Sequence[OBB] = (),
) -> list[dict]:
    bounds = tuple(map(float, surface["bounds_xy_m"]))
    edge_margin = float(surface["edge_margin_m"])
    minimum_gap = float(surface["minimum_gap_m"])
    desired = int(surface["desired_count"])
    candidates = [dict(item) for item in items]
    candidates = [
        item for item in candidates
        if surface_name in item.get(
            "allowed_surfaces", ("coffee_table", "sofa")
        )
    ]
    # Select optional objects once per episode, before any physical rollout.
    probability = float(surface.get("presence_probability", 1.0))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("surface presence_probability must be within [0, 1]")
    if probability < 1.0:
        candidates = [item for item in candidates if rng.random() < probability]
        desired = min(desired, len(candidates))
    rng.shuffle(candidates)
    if surface_name == "side_table":
        # The side table is only 325 mm deep.  With a target and its protected
        # NW approach lane, placing a large notebook first can consume every
        # remaining pocket even though three small props fit.  Shuffle first
        # for seeded tie-breaking, then allocate the smallest footprints
        # first so the strict three-distractor floor is physically feasible.
        candidates.sort(key=lambda item: (
            float(item["width_m"]) * float(item["depth_m"]),
            max(float(item["width_m"]), float(item["depth_m"])),
        ))

    def placement_for(item, x, y, yaw):
        rest_modes = tuple(item.get("rest_modes", ("flat",)))
        rest_mode = rng.choice(rest_modes)
        rotate_x = float(item.get("rotate_x_deg", 0.0))
        rotate_y = float(item.get("rotate_y_deg", 0.0))
        if rest_mode == "side":
            rotate_x, rotate_y = -90.0, -90.0
        elif rest_mode == "upright":
            rotate_x, rotate_y = 0.0, 0.0
        return {
            **item,
            "surface": surface_name,
            "placement_bounds_xy_m": list(bounds),
            "placement_edge_margin_m": edge_margin,
            "x_m": x,
            "y_m": y,
            "surface_z_m": float(surface["surface_z_m"]),
            "yaw_deg": yaw,
            "rotate_x_deg": rotate_x,
            "rotate_y_deg": rotate_y,
            "rest_mode": rest_mode,
            "role": "interactive_distractor",
            "dynamic": True,
        }

    # Without a target reservation, use a seeded one-row pack on the narrow
    # side table.  Pure rejection sampling can strand two props at unlucky
    # positions even though three fit easily; the row retains random object
    # selection, gaps, depth and 0/180 orientation while guaranteeing space.
    if surface_name == "side_table" and not reserved and len(candidates) >= desired:
        selected = candidates[:desired]
        xmin, xmax, ymin, ymax = bounds
        usable_width = xmax - xmin - 2.0 * edge_margin
        occupied_width = sum(float(item["width_m"]) for item in selected)
        mandatory_gaps = minimum_gap * max(0, desired - 1)
        slack = usable_width - occupied_width - mandatory_gaps
        if slack >= 0.0:
            weights = [rng.random() + 0.1 for _ in range(desired + 1)]
            total_weight = sum(weights)
            extras = [slack * value / total_weight for value in weights]
            cursor = xmin + edge_margin + extras[0]
            placed = []
            for index, item in enumerate(selected):
                width = float(item["width_m"])
                depth = float(item["depth_m"])
                x = cursor + width / 2.0
                y = rng.uniform(
                    ymin + edge_margin + depth / 2.0,
                    ymax - edge_margin - depth / 2.0,
                )
                yaw = rng.choice((0.0, 180.0))
                placed.append(placement_for(item, x, y, yaw))
                cursor += width
                if index < desired - 1:
                    cursor += minimum_gap + extras[index + 1]
            return placed
    placed: list[dict] = []
    occupied = list(reserved)
    for item in candidates:
        if len(placed) >= desired:
            break
        for _attempt in range(1200):
            yaw = _sample_surface_yaw(item, surface_name, rng)
            xmin, xmax, ymin, ymax = bounds
            x = rng.uniform(xmin + edge_margin, xmax - edge_margin)
            y = rng.uniform(ymin + edge_margin, ymax - edge_margin)
            box = OBB(
                x,
                y,
                float(item["width_m"]),
                float(item["depth_m"]),
                yaw,
                str(item["id"]),
            )
            if not obb_inside(box, bounds, edge_margin):
                continue
            if not all(
                obb_clear(box, other, minimum_gap)
                for other in occupied
            ):
                continue
            placed.append(placement_for(item, x, y, yaw))
            occupied.append(box)
            break
    # The strict floor is fail-closed: a geometrically infeasible episode is
    # rejected instead of silently producing a lower-density sample.
    minimum = surface.get("minimum_count")
    minimum = desired if minimum is None else int(minimum)
    if len(placed) < min(minimum, desired):
        raise ValueError(
            f"could place only {len(placed)}/{desired} objects on "
            f"{surface_name} (minimum {minimum})"
        )
    return placed


def _coffee_table_semantic_config(policy: Mapping) -> Mapping | None:
    config = policy.get("coffee_table_semantic_refresh")
    if not isinstance(config, Mapping) or config.get("enabled") is not True:
        return None
    if config.get("axis") != "canonical_world_x":
        raise ValueError("CoffeeTable must split canonical X, parallel to its long edge")
    table = policy["surfaces"]["coffee_table"]
    total = int(config["total_distractor_count"])
    if total != int(table["desired_count"]) or total != sum(
        int(config[name]["desired_count"]) for name in ("trash_group", "shelf_group")
    ):
        raise ValueError("CoffeeTable semantic group counts must equal its desired count")
    count_range = config.get("distractor_count_range")
    if count_range is not None:
        if (not isinstance(count_range, (list, tuple)) or len(count_range) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in count_range)
            or not 2 <= count_range[0] <= count_range[1]
            or sum(count_range) != 2 * total):
            raise ValueError("CoffeeTable count range must have the configured mean")
    xmin, xmax, ymin, ymax = map(float, table["bounds_xy_m"])
    split = float(config["split_x_m"])
    if (abs(split - (xmin + xmax) / 2) > 1e-8
        or ymax - ymin <= xmax - xmin
        or list(config["sofa_side_bounds_xy_m"]) != [split, xmax, ymin, ymax]
        or list(config["tablecloth_side_bounds_xy_m"]) != [xmin, split, ymin, ymax]):
        raise ValueError("CoffeeTable workspaces must equally partition its short axis")
    return config


def _table_shelf_item_candidates(policy: Mapping) -> list[dict]:
    candidates: list[dict] = []
    shelf = policy["shelf"]
    for category, key in (
        ("beverage", "beverage_foreground"),
        ("snack", "snack_foreground"),
    ):
        for source in shelf[key]:
            if source.get("production_distractor", True) is False:
                continue
            source_id = str(source["id"])
            candidates.append({
                **dict(source),
                "id": f"TableShelf_{source_id}",
                "prim_path": (
                    f"/World/EpisodeDistractors/TableShelf_{source_id}"
                ),
                "source_prim_path": str(source["prim_path"]),
                "semantic_class": "shelf_item",
                "shelf_item_kind": category,
                "shelf_inventory_id": source_id,
                "rest_modes": tuple(
                    source.get("table_rest_modes", ("upright",))
                ),
                "allowed_surfaces": ("coffee_table",),
            })
    if not candidates:
        raise ValueError("CoffeeTable shelf-item candidate pool is empty")
    return candidates


def _placement_boxes(items: Sequence[Mapping]) -> list[OBB]:
    return [
        OBB(
            float(item["x_m"]),
            float(item["y_m"]),
            float(item["width_m"]),
            float(item["depth_m"]),
            float(item["yaw_deg"]),
            str(item["id"]),
        )
        for item in items
    ]


def place_coffee_table_semantic_objects(
    policy: Mapping,
    *,
    rng: random.Random,
    reserved: Sequence[OBB] = (),
    existing_density: int = 0,
) -> tuple[list[dict], dict]:
    """Place both cleanup object groups inside the sofa-side workspace."""
    config = _coffee_table_semantic_config(policy)
    if config is None:
        raise ValueError("CoffeeTable semantic refresh is disabled")
    table = dict(policy["surfaces"]["coffee_table"])
    count_range = config.get("distractor_count_range")
    total = (rng.randint(*count_range) if count_range is not None
             else int(config["total_distractor_count"]))
    needed = max(0, total - int(existing_density))
    if count_range is not None:
        # Keep both semantic groups while allowing odd totals without a bias
        # toward either group. Sampling is fixed once for this episode.
        bin_goal = needed // 2 + (rng.randrange(2) if needed % 2 else 0)
    else:
        bin_goal = min(int(config["trash_group"]["desired_count"]), needed)
    shelf_goal = needed - bin_goal
    if count_range is None and shelf_goal > int(config["shelf_group"]["desired_count"]):
        raise ValueError("CoffeeTable semantic refresh exceeds shelf-item quota")

    bin_candidates = [
        {
            **dict(item),
            "allowed_surfaces": ("coffee_table",),
        }
        for item in config["trash_group"]["candidates"]
    ]
    if len(bin_candidates) < bin_goal:
        raise ValueError("CoffeeTable trash candidate pool is too small")
    bin_surface = dict(table)
    bin_surface["bounds_xy_m"] = list(
        map(float, config["sofa_side_bounds_xy_m"])
    )
    bin_surface["desired_count"] = bin_goal
    bin_surface["minimum_count"] = bin_goal
    shelf_candidates = _table_shelf_item_candidates(policy)
    if len(shelf_candidates) < shelf_goal:
        raise ValueError("CoffeeTable shelf-item candidate pool is too small")
    shelf_surface = dict(table)
    shelf_surface["bounds_xy_m"] = list(
        map(float, config["sofa_side_bounds_xy_m"])
    )
    shelf_surface["desired_count"] = shelf_goal
    shelf_surface["minimum_count"] = shelf_goal
    placed_bin: list[dict] = []
    placed_shelf: list[dict] = []
    placement_error: ValueError | None = None
    # Retry both object groups together within the same sofa-side half. A
    # greedy trash placement can otherwise prevent a valid shelf-item packing.
    # This is
    # still fully deterministic because every retry consumes the seeded RNG.
    for _attempt in range(32):
        try:
            placed_bin = place_surface_objects(
                bin_candidates,
                surface_name="coffee_table",
                surface=bin_surface,
                rng=rng,
                reserved=reserved,
            ) if bin_goal else []
            placed_shelf = place_surface_objects(
                shelf_candidates,
                surface_name="coffee_table",
                surface=shelf_surface,
                rng=rng,
                reserved=[*reserved, *_placement_boxes(placed_bin)],
            ) if shelf_goal else []
            placement_error = None
            break
        except ValueError as exc:
            placement_error = exc
    if placement_error is not None:
        raise ValueError(
            "CoffeeTable semantic zones have no valid joint packing after "
            f"32 deterministic attempts: {placement_error}"
        ) from placement_error
    for item in placed_bin:
        item["semantic_zone"] = "sofa_side"
        item["semantic_group"] = "trash_group"
    for item in placed_shelf:
        item["semantic_zone"] = "sofa_side"
        item["semantic_group"] = "shelf_group"

    placed = [*placed_bin, *placed_shelf]
    if len(placed) != needed:
        raise ValueError(
            f"CoffeeTable semantic refresh placed {len(placed)}/{needed}"
        )
    audit = {
        "enabled": True,
        "split_axis": "canonical_world_x",
        "split_x_m": float(config["split_x_m"]),
        "workspace": "sofa_side",
        "bounds_xy_m": list(config["sofa_side_bounds_xy_m"]),
        "target_distractor_count": total,
        "configured_mean_distractor_count": int(config["total_distractor_count"]),
        "distractor_count_range": list(count_range) if count_range is not None else [total, total],
        "existing_context_count": int(existing_density),
        "generated_count": len(placed),
        "trash_group_count": len(placed_bin),
        "shelf_group_count": len(placed_shelf),
        "trash_kinds": sorted(
            str(item["trash_kind"]) for item in placed_bin
        ),
        "shelf_inventory_ids": sorted(
            str(item["shelf_inventory_id"]) for item in placed_shelf
        ),
    }
    return placed, audit


def _target_reservations(
    target: Mapping | None,
    policy: Mapping,
) -> dict[str, list[OBB]]:
    reservations = {name: [] for name in policy["surfaces"]}
    if not target:
        return reservations
    surface = target.get("surface")
    if surface not in reservations:
        return reservations
    width = max(0.04, float(target.get("width_m", 0.08)))
    depth = max(0.04, float(target.get("depth_m", 0.08)))
    x = float(target["x_m"])
    y = float(target["y_m"])
    # 2026-08-05: a 100/120 mm skirt is not enough once the denser v2/v3
    # layouts can drop a 350 mm prop 200 mm away — its far end then lies
    # across the pinch, and the recorded close grabs clutter instead of the
    # target (the 07810daa umbrella case).  The keep-out is sized for the
    # LARGEST prop in the pool so any neighbour's body stays clear of the
    # grasp, not just its centre.
    # 2026-08-05: the keep-out is a MODEST skirt again.  Sizing it for the
    # largest prop in the pool pushed same-kind companions far away, while
    # the real recordings show the two crumpled papers lying right next to
    # each other.  Oversized props are kept off the target's surface
    # instead (see BULKY_PROP_M below), which is what actually blocked the
    # approach; small neighbours may sit close, as they do in the footage.
    skirt_x, skirt_y = 0.05, 0.06
    reservations[surface].append(
        OBB(x, y, width + 2 * skirt_x, depth + 2 * skirt_y, 0.0,
            "target_clearance")
    )
    bounds = policy["surfaces"][surface]["bounds_xy_m"]
    if surface == "coffee_table":
        xmax = float(bounds[1])
        lane_width = max(0.18, depth + 0.16)
        reservations[surface].append(
            OBB(
                (x + xmax) / 2.0,
                y,
                max(0.02, xmax - x),
                lane_width,
                0.0,
                "right/east-hand-approach-lane",
            )
        )
    elif surface == "side_table":
        # The shared side-table station stands in the CoffeeTable/TvTable
        # corridor NW of the table and approaches along (+x, -y) (init_pick
        # "small_table" family, 2026-08-03).  Reserve the diagonal corridor
        # from the table's NW corner to the target.
        xmin = float(bounds[0])
        ymax = float(bounds[3])
        direction = (xmin - x, ymax - y)
        length = math.hypot(*direction)
        lane_width = max(0.20, depth + 0.18)
        reservations[surface].append(
            OBB(
                (x + xmin) / 2.0,
                (y + ymax) / 2.0,
                max(0.02, length),
                lane_width,
                math.degrees(math.atan2(direction[1], direction[0])),
                "side-table-NW-approach-lane",
            )
        )
    else:
        xmin = float(bounds[0])
        lane_depth = max(0.20, depth + 0.18)
        reservations[surface].append(
            OBB(
                (xmin + x) / 2.0,
                y,
                max(0.02, x - xmin),
                lane_depth,
                0.0,
                "sofa-robot-approach-lane",
            )
        )
    return reservations


def _place_zone_reservation(
    task_id: str,
    zone: Sequence[float] | None,
    policy: Mapping,
) -> OBB | None:
    if str(task_id) not in {
        "14", "16", "17", "18", "19", "20", "27",
        "41", "42", "43", "44", "45", "46",
        "86", "87", "88", "89",
    }:
        return None
    # The task zone is often the entire CoffeeTable acceptance envelope.  A
    # smaller central landing corridor keeps the recorded release reachable
    # while still allowing clutter along the perimeter.
    if zone and len(zone) >= 4:
        cx, cy = float(zone[0]), float(zone[1])
    else:
        bounds = policy["surfaces"]["coffee_table"]["bounds_xy_m"]
        cx = (float(bounds[0]) + float(bounds[1])) / 2.0
        cy = (float(bounds[2]) + float(bounds[3])) / 2.0
    return OBB(cx, cy, 0.38, 0.50, 0.0, "place-landing-corridor")


def validate_shelf_hand_half(
    *,
    target_x_m: float,
    hand: str,
    shelf: Mapping,
) -> None:
    resolved = str(hand).lower()
    if resolved not in {"l", "r"}:
        return
    # The policy stores the shelf centre for the canonical group placement;
    # under a group shift the robot and its targets moved together, so the
    # half-split follows the shifted frame while the shelf itself stays put.
    center = float(shelf["center_x_m"]) + group_shift.GROUP_SHIFT_XY_M[0]
    deadband = float(shelf["hand_half_deadband_m"])
    if resolved == "l" and target_x_m > center - deadband:
        raise ValueError(
            f"left-hand shelf target {target_x_m:.4f} m is not in "
            "the shelf's left/world-X-minus half"
        )
    if resolved == "r" and target_x_m < center + deadband:
        raise ValueError(
            f"right-hand shelf target {target_x_m:.4f} m is not in "
            "the shelf's right/world-X-plus half"
        )


def build_layout_plan(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    hand: str = "?",
    target: Mapping | None = None,
    place_zone: Sequence[float] | None = None,
    protected_prim_paths: Iterable[str] = (),
    protected_objects: Sequence[Mapping] = (),
    policy: Mapping | None = None,
) -> dict:
    effective = dict(policy or load_policy())
    identity = randomization_identity(
        str(task_id),
        episode_uid,
        seed_override,
        policy=effective,
    )
    if not identity["enabled"]:
        return {
            **identity,
            "schema": "real2sim.episode_scene_randomization.plan.v1",
            "objects": [],
            "interactive_prim_paths": [],
            "reason": "small task is outside the current distractor scope",
        }
    rng = random.Random(identity["seed_u64"])
    shelf = effective["shelf"]
    target_path = str(target.get("prim_path", "")) if target else ""
    target_surface = str(target.get("surface", "")) if target else ""
    target_x = float(target["x_m"]) if target and "x_m" in target else None
    beverage_paths = {
        str(item["prim_path"])
        for item in shelf["beverage_foreground"]
    }
    snack_paths = {
        str(item["prim_path"])
        for item in shelf["snack_foreground"]
    }
    beverage_reserved = []
    snack_reserved = []
    hand_half_exception = str(task_id) in {
        str(value)
        for value in shelf.get(
            "hand_half_exception_small_task_ids", ()
        )
    }
    target_clearance_half = float(
        shelf.get("target_grasp_clearance_half_width_m", 0.025)
    )
    maximum_target_omissions = int(
        shelf.get(
            "maximum_foreground_omissions_for_target_clearance", 0
        )
    )
    swept_interval_by_task = shelf.get(
        "target_swept_x_interval_m_by_small_task", {}
    )
    swept_interval = swept_interval_by_task.get(str(task_id))
    if swept_interval is not None:
        if (
            not isinstance(swept_interval, Sequence)
            or isinstance(swept_interval, (str, bytes))
            or len(swept_interval) != 2
        ):
            raise ValueError(
                "target swept shelf interval must contain [xmin, xmax]"
            )
        swept_interval = tuple(map(float, swept_interval))
        if swept_interval[0] >= swept_interval[1]:
            raise ValueError(
                "target swept shelf interval must have xmin < xmax"
            )
    if (
        target_path in beverage_paths
        and target_surface == "shelf_beverage"
        and target_x is not None
    ):
        if not hand_half_exception:
            validate_shelf_hand_half(
                target_x_m=target_x, hand=hand, shelf=shelf
            )
        half = max(
            target_clearance_half,
            float(target.get("width_m", 0.07)) / 2.0,
        )
        beverage_reserved.append((target_x - half, target_x + half))
    if (
        target_path in snack_paths
        and target_surface == "shelf_snack"
        and target_x is not None
    ):
        if not hand_half_exception:
            validate_shelf_hand_half(
                target_x_m=target_x, hand=hand, shelf=shelf
            )
        half = max(
            target_clearance_half,
            float(target.get("width_m", 0.10)) / 2.0,
        )
        snack_reserved.append(
            swept_interval
            if swept_interval is not None
            else (target_x - half, target_x + half)
        )

    beverages = [
        item for item in shelf["beverage_foreground"]
        if (
            str(item["prim_path"]) != target_path
            and item.get("production_distractor", True) is not False
        )
    ]
    snacks = [
        item for item in shelf["snack_foreground"]
        if (
            str(item["prim_path"]) != target_path
            and item.get("production_distractor", True) is not False
        )
    ]
    objects = []
    shelf_omitted_paths: list[str] = []
    objects.extend(pack_shelf_row(
        beverages,
        x_bounds=shelf["inner_x_bounds_m"],
        y_range=shelf["front_local_y_range_m"],
        minimum_gap_m=float(shelf["minimum_foreground_edge_gap_m"]),
        rng=rng,
        reserved_intervals=beverage_reserved,
        max_omissions=(
            maximum_target_omissions if beverage_reserved else 0
        ),
        omitted_prim_paths=shelf_omitted_paths,
    ))
    for item in objects:
        if item["prim_path"] in beverage_paths:
            item["surface"] = "shelf_beverage"
            item["surface_z_m"] = float(
                shelf["beverage_surface_z_m"])
    snack_layout = pack_shelf_row(
        snacks,
        x_bounds=shelf["inner_x_bounds_m"],
        y_range=shelf["front_local_y_range_m"],
        minimum_gap_m=float(shelf["minimum_foreground_edge_gap_m"]),
        rng=rng,
        reserved_intervals=snack_reserved,
        max_omissions=(
            maximum_target_omissions if snack_reserved else 0
        ),
        omitted_prim_paths=shelf_omitted_paths,
    )
    for item in snack_layout:
        item["surface"] = "shelf_snack"
        item["surface_z_m"] = float(shelf["snack_surface_z_m"])
    objects.extend(snack_layout)
    beverage_background = _background_row(
        [
            item for item in shelf["beverage_background"]
            if item.get("production_distractor", True) is not False
        ],
        x_bounds=shelf["inner_x_bounds_m"],
        y_m=float(shelf["background_local_y_m"]),
        rng=rng,
    )
    for item in beverage_background:
        item["surface"] = "shelf_beverage"
        item["surface_z_m"] = float(shelf["beverage_surface_z_m"])
    objects.extend(beverage_background)
    snack_background = _background_row(
        [
            item for item in shelf["snack_background"]
            if item.get("production_distractor", True) is not False
        ],
        x_bounds=shelf["inner_x_bounds_m"],
        y_m=float(shelf["background_local_y_m"]),
        rng=rng,
    )
    for item in snack_background:
        item["surface"] = "shelf_snack"
        item["surface_z_m"] = float(shelf["snack_surface_z_m"])
    objects.extend(snack_background)
    # The TOP board (above the drinks) is stocked in every real frame and was
    # empty in sim -- pure set dressing, but an empty tier reads as a different
    # room on camera. Same approved-asset background-row mechanics as the two
    # rows above; optional so older policy files stay valid.
    if shelf.get("top_background"):
        top_background = _background_row(
            [
                item for item in shelf["top_background"]
                if item.get("production_distractor", True) is not False
            ],
            x_bounds=shelf["inner_x_bounds_m"],
            # The top tier has no interactive foreground, so its dressing sits
            # near the FRONT edge like the real stock does; parked at the deep
            # background y it is above the head camera's view cone and the
            # tier still reads empty on camera.
            y_m=float(shelf.get("top_local_y_m",
                                shelf["background_local_y_m"])),
            rng=rng,
        )
        for item in top_background:
            item["surface"] = "shelf_top"
            item["surface_z_m"] = float(shelf["top_surface_z_m"])
            # Generated prims do not inherit ShelfItems' room translation.
            item["coordinate_frame_prim"] = "/World/Assets/ShelfItems"
        objects.extend(top_background)

    protected = set(map(str, protected_prim_paths))
    if target_path:
        protected.add(target_path)
    # Identity guard: a clone that references the target's own asset puts a
    # second, visually identical object on a surface while the instruction
    # names exactly one (small Task 33's striped paper cup) — ambiguous by
    # construction, so by default it never enters the pool.  The tidy-table
    # multiplicity tasks are the reviewed exception (2026-08-04 direction):
    # their real scenes genuinely hold 1-3 cups / 1-2 paper balls, so
    # same-type clones are allowed and the recorded action still manipulates
    # the original prim.
    semantic_table_config = _coffee_table_semantic_config(effective)
    allow_same_type = str(task_id) in set(map(str, effective.get(
        "target_source_clone_allowed_task_ids", ())))
    # A prop whose body spans more than this cannot share the target's
    # surface: its far end reaches across the grasp approach even when its
    # centre is cleanly outside the target skirt (the 350 mm umbrella lying
    # 200 mm from the paper, 07810daa).
    BULKY_PROP_M = 0.25
    surface_pool = [
        item for item in effective["surface_objects"]
        if str(item["prim_path"]) not in protected
        and (
            allow_same_type
            or not (
                target_path
                and str(item.get("source_prim_path", "")) == target_path
            )
        )
    ]
    rng.shuffle(surface_pool)
    reservations = _target_reservations(target, effective)
    for descriptor in protected_objects:
        surface_name = str(descriptor.get("surface", ""))
        if surface_name not in reservations:
            continue
        reservations[surface_name].append(OBB(
            float(descriptor["x_m"]),
            float(descriptor["y_m"]),
            max(0.04, float(descriptor["width_m"])) + 0.06,
            max(0.04, float(descriptor["depth_m"])) + 0.06,
            0.0,
            "protected-support-clearance",
        ))
    landing = _place_zone_reservation(task_id, place_zone, effective)
    if landing is not None:
        reservations["coffee_table"].append(landing)
    # Clutter surfaces in a FIXED allocation order.  The side table's
    # PHYSICAL top is small, so only the small-prop subset is allowed
    # there; it draws first, while the pool still holds small items —
    # coffee table and sofa accept everything and fill fine from the
    # remainder.  Object identities stay disjoint across surfaces —
    # clone cups are independent rigid bodies, not multiple transforms
    # of one prim.
    clutter_surfaces = [
        name for name in ("side_table", "coffee_table", "sofa")
        if name in effective["surfaces"]
    ]
    existing_density = {
        surface_name: sum(
            descriptor.get("counts_toward_density") is True
            and descriptor.get("surface") == surface_name
            for descriptor in protected_objects
        )
        for surface_name in clutter_surfaces
    }
    protected_density = dict(existing_density)
    # 2026-08-05 direction: the tidy-table recordings ALWAYS show the same
    # companion set (two crumpled papers, three disposable cups), so those
    # ids are required on the coffee table every episode rather than sampled
    # — only their poses vary.  They are drawn first, from the same pool and
    # through the same clearance solver, and they count toward the surface
    # density so the total stays at desired_count.
    required_ids = [
        str(value)
        for value in effective.get(
            "required_companions_by_task", {}).get(str(task_id), ())
    ] if semantic_table_config is None else []
    primary_clone_ids = set(map(str, effective.get(
        "primary_asset_clone_ids_by_task", {}).get(str(task_id), ()))) \
        if semantic_table_config is None else set()
    maximum_primary_instances = int(effective.get(
        "primary_asset_max_instances_including_target", 1))
    if maximum_primary_instances < 1:
        raise ValueError("primary asset instance cap must be positive")
    if len(primary_clone_ids) + 1 > maximum_primary_instances:
        raise ValueError(
            f"task {task_id} requests {len(primary_clone_ids) + 1} primary "
            f"instances, cap is {maximum_primary_instances}"
        )
    if not primary_clone_ids.issubset(required_ids):
        raise ValueError(
            f"task {task_id} primary clones must be required companions"
        )
    used: set[str] = set()
    if required_ids:
        by_id = {str(item["id"]): item for item in surface_pool}
        missing_required = [item_id for item_id in required_ids
                            if item_id not in by_id]
        if missing_required:
            raise ValueError(
                f"task {task_id} required companions are unavailable: "
                f"{missing_required}"
            )
        required_items = [by_id[item_id] for item_id in required_ids]
        if required_items:
            required_surface = dict(effective["surfaces"]["coffee_table"])
            required_surface["desired_count"] = len(required_items)
            required_surface["minimum_count"] = len(required_items)
            placed_required = place_surface_objects(
                required_items,
                surface_name="coffee_table",
                surface=required_surface,
                rng=rng,
                reserved=reservations["coffee_table"],
            )
            for item in placed_required:
                if str(item["id"]) in primary_clone_ids:
                    item["role"] = "primary_asset_clone"
            for item in placed_required:
                reservations["coffee_table"].append(OBB(
                    float(item["x_m"]), float(item["y_m"]),
                    float(item["width_m"]) + 0.02,
                    float(item["depth_m"]) + 0.02,
                    math.radians(float(item.get("yaw_deg", 0.0))),
                    f"required-{item['id']}",
                ))
            used.update(str(item["id"]) for item in placed_required)
            existing_density["coffee_table"] = (
                existing_density.get("coffee_table", 0)
                + len(placed_required))
            objects.extend(placed_required)
    coffee_table_semantic_audit = {"enabled": False}
    for surface_name in clutter_surfaces:
        if surface_name == "coffee_table" and semantic_table_config is not None:
            placed_here, coffee_table_semantic_audit = (
                place_coffee_table_semantic_objects(
                    effective,
                    rng=rng,
                    reserved=reservations[surface_name],
                    existing_density=existing_density[surface_name],
                )
            )
            used.update(str(item["id"]) for item in placed_here)
            objects.extend(placed_here)
            continue
        pool = [
            item for item in surface_pool
            if str(item["id"]) not in used
            and surface_name in item.get(
                "allowed_surfaces", ("coffee_table", "sofa")
            )
            and not (
                surface_name == target_surface == "coffee_table"
                and max(float(item.get("width_m", 0.0)),
                        float(item.get("depth_m", 0.0))) > BULKY_PROP_M
            )
        ]
        surface_config = dict(effective["surfaces"][surface_name])
        base_count = int(surface_config["desired_count"])
        if (
            target_surface == surface_name
            and "desired_count_when_target_here" in surface_config
        ):
            # A small physical top that also hosts the episode target (plus
            # its clearance and approach lane) cannot honour the full
            # clutter quota; the target itself is that surface's
            # interactive object.
            base_count = int(
                surface_config["desired_count_when_target_here"])
        surface_config["desired_count"] = max(
            0,
            base_count - existing_density[surface_name],
        )
        placed_here = place_surface_objects(
            pool,
            surface_name=surface_name,
            surface=surface_config,
            rng=rng,
            reserved=reservations[surface_name],
        )
        used.update(str(item["id"]) for item in placed_here)
        objects.extend(placed_here)
    surface_distractor_counts = {}
    surface_primary_clone_counts = {}
    for surface_name in clutter_surfaces:
        generated_distractors = sum(
            item.get("surface") == surface_name
            and item.get("role") == "interactive_distractor"
            for item in objects
        )
        primary_clones = sum(
            item.get("surface") == surface_name
            and item.get("role") == "primary_asset_clone"
            for item in objects
        )
        observed_distractors = (
            int(protected_density.get(surface_name, 0))
            + generated_distractors
        )
        minimum_distractors = int(effective["surfaces"][surface_name].get(
            "minimum_distractor_count",
            effective["surfaces"][surface_name].get(
                "minimum_count",
                effective["surfaces"][surface_name]["desired_count"],
            ),
        ))
        if observed_distractors < minimum_distractors:
            raise ValueError(
                f"surface {surface_name} has {observed_distractors} "
                f"distractors, minimum is {minimum_distractors}"
            )
        surface_distractor_counts[surface_name] = observed_distractors
        surface_primary_clone_counts[surface_name] = primary_clones
    visual_source_counts: dict[str, int] = {}
    for item in objects:
        if (
            item.get("surface") not in clutter_surfaces
            or item.get("dynamic") is not True
        ):
            continue
        source = str(item.get("source_prim_path", item["prim_path"]))
        visual_source_counts[source] = visual_source_counts.get(source, 0) + 1
    excessive = {
        source: count for source, count in visual_source_counts.items()
        if count > maximum_primary_instances
    }
    if excessive:
        raise ValueError(
            f"surface asset multiplicity exceeds cap "
            f"{maximum_primary_instances}: {excessive}"
        )
    interactive = sorted(
        str(item["prim_path"])
        for item in objects
        if item.get("dynamic") is True
    )
    return {
        **identity,
        "schema": "real2sim.episode_scene_randomization.plan.v1",
        "hand": str(hand).lower(),
        "target_prim_path": target_path or None,
        "target_surface": target_surface or None,
        "protected_prim_paths": sorted(protected),
        "protected_object_descriptors": [
            dict(value) for value in protected_objects
        ],
        "objects": objects,
        "interactive_prim_paths": interactive,
        "shelf_pending_background_assets": list(
            shelf.get("pending_background_assets", ())
        ),
        "constraints": {
            "shelf_foreground_single_row": True,
            "shelf_category_separation": True,
            "shelf_target_clearance_half_width_m": (
                target_clearance_half if (
                    beverage_reserved or snack_reserved
                ) else None
            ),
            "shelf_target_reserved_x_intervals_m": [
                [float(left), float(right)]
                for left, right in (
                    beverage_reserved + snack_reserved
                )
            ],
            "shelf_task_swept_interval_applied": (
                swept_interval is not None
                and bool(beverage_reserved or snack_reserved)
            ),
            "shelf_omitted_foreground_prim_paths": sorted(
                shelf_omitted_paths
            ),
            "target_hand_half_checked": (
                target_surface in {"shelf_beverage", "shelf_snack"}
                and str(hand).lower() in {"l", "r"}
                and not hand_half_exception
            ),
            "target_hand_half_exception": hand_half_exception,
            "target_and_approach_reserved": bool(target),
            "coffee_table_place_landing_reserved": landing is not None,
            "coffee_table_semantic_refresh": coffee_table_semantic_audit,
            "protected_existing_context_count": protected_density,
            "surface_distractor_counts_including_context": (
                surface_distractor_counts
            ),
            "surface_primary_asset_clone_counts": (
                surface_primary_clone_counts
            ),
            "primary_asset_instances_including_target": (
                1 + sum(surface_primary_clone_counts.values())
                if primary_clone_ids else 1
            ),
            "primary_asset_max_instances_including_target": (
                maximum_primary_instances
            ),
            "surface_visual_source_counts": visual_source_counts,
            "furniture_transform_authorship_forbidden": True,
        },
    }


def _matrix_values(matrix) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _world_bound(stage, prim_path: str) -> tuple[float, ...] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or not prim.IsActive():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = bound.GetMin()
    maximum = bound.GetMax()
    values = tuple(map(float, (*minimum, *maximum)))
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def shelf_member_surface(
    descriptor: Mapping,
    shelf: Mapping,
) -> str | None:
    """Classify a foreground shelf member as on-shelf, or None when staged
    away from the shelf.

    The recorded-hand half rule and foreground row reservations only make
    sense for a target that is physically resting on its shelf level (a pick
    grasp).  A place-task target is staged in-hand near its landing zone;
    foreground-list membership alone must not re-label it as an on-shelf
    grasp target, or the hand-half validation fails against a staging
    position the shelf rule was never reviewed for.
    """
    prim_path = str(descriptor.get("prim_path", ""))
    beverage = {
        str(item["prim_path"])
        for item in shelf["beverage_foreground"]
    }
    snack = {
        str(item["prim_path"])
        for item in shelf["snack_foreground"]
    }
    if prim_path in beverage:
        level_z = float(shelf["beverage_surface_z_m"])
        name = "shelf_beverage"
    elif prim_path in snack:
        level_z = float(shelf["snack_surface_z_m"])
        name = "shelf_snack"
    else:
        return None
    inner_x = shelf["inner_x_bounds_m"]
    x = float(descriptor["x_m"])
    bottom_z = (
        float(descriptor["z_m"])
        - float(descriptor.get("height_m", 0.0)) / 2.0
    )
    if not (float(inner_x[0]) - 0.05 <= x <= float(inner_x[1]) + 0.05):
        return None
    if abs(bottom_z - level_z) > 0.05:
        return None
    return name


def _target_descriptor(
    stage,
    target_prim_path: str,
    policy: Mapping,
) -> dict | None:
    bound = _world_bound(stage, target_prim_path)
    if bound is None:
        return None
    xmin, ymin, zmin, xmax, ymax, zmax = bound
    descriptor = {
        "prim_path": target_prim_path,
        "x_m": (xmin + xmax) / 2.0,
        "y_m": (ymin + ymax) / 2.0,
        "z_m": (zmin + zmax) / 2.0,
        "width_m": xmax - xmin,
        "depth_m": ymax - ymin,
        "height_m": zmax - zmin,
    }
    sofa = policy["surfaces"]["sofa"]
    whole = sofa.get("whole_sofa_bounds_xy_m")
    if whole and (whole[0] <= descriptor["x_m"] <= whole[1]
                  and whole[2] <= descriptor["y_m"] <= whole[3]
                  and zmin <= float(sofa["surface_z_m"]) + 0.16
                  and zmax >= float(sofa["surface_z_m"]) - 0.02):
        x0, x1, y0, y1 = sofa["bounds_xy_m"]
        if not (x0 <= xmin and xmax <= x1 and y0 <= ymin and ymax <= y1):
            raise ValueError(f"sofa interaction object outside middle two seats: {target_prim_path}; regenerate initial placement")
    for name, surface in policy["surfaces"].items():
        sxmin, sxmax, symin, symax = map(
            float, surface["bounds_xy_m"])
        top = float(surface["surface_z_m"])
        if (
            sxmin <= descriptor["x_m"] <= sxmax
            and symin <= descriptor["y_m"] <= symax
            and zmin <= top + 0.16
            and zmax >= top - 0.02
        ):
            descriptor["surface"] = name
            return descriptor
    member_surface = shelf_member_surface(descriptor, policy["shelf"])
    if member_surface is not None:
        descriptor["surface"] = member_surface
    return descriptor


def _find_xform_op(prim, op_type):
    from pxr import UsdGeom

    for operation in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if operation.GetOpType() == op_type:
            return operation
    return None


def _set_pose(stage, placement: Mapping, demo_root: Path) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    path = str(placement["prim_path"])
    prim = stage.GetPrimAtPath(path)
    source_asset = placement.get("source_asset")
    source_prim = placement.get("source_prim_path")
    if source_asset:
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().SetReferences([
            Sdf.Reference(str(demo_root / str(source_asset)))
        ])
    elif source_prim and (
        not prim or not prim.IsValid()
    ):
        source = stage.GetPrimAtPath(str(source_prim))
        if not source or not source.IsValid():
            raise ValueError(
                f"clone source prim is missing: {source_prim}")
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().SetReferences([
            Sdf.Reference(
                assetPath="",
                primPath=Sdf.Path(str(source_prim)),
            )
        ])
    if not prim or not prim.IsValid():
        raise ValueError(f"randomized prop is missing: {path}")
    prim.SetActive(True)
    xform = UsdGeom.Xformable(prim)
    if source_prim and any(
        op.GetOpType() == UsdGeom.XformOp.TypeTransform
        for op in xform.GetOrderedXformOps()
    ):
        # Recorded-start targets carry a matrix op. A clone must retain its
        # size, but not inherit that target's position/rotation ahead of the
        # newly sampled placement ops. Never author the source target itself.
        scale = Gf.Transform(xform.GetLocalTransformation()).GetScale()
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(opSuffix="randomPlacement")
        xform.AddRotateXYZOp(opSuffix="randomPlacement")
        xform.AddScaleOp(opSuffix="randomPlacement").Set(Gf.Vec3f(*scale))
    translate = _find_xform_op(prim, UsdGeom.XformOp.TypeTranslate)
    if translate is None:
        translate = xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble)
    rotate_xyz = _find_xform_op(prim, UsdGeom.XformOp.TypeRotateXYZ)
    orient = _find_xform_op(prim, UsdGeom.XformOp.TypeOrient)
    rotate_x = float(placement.get("rotate_x_deg", 0.0))
    rotate_y = float(placement.get("rotate_y_deg", 0.0))
    yaw = float(placement["yaw_deg"])
    if rotate_xyz is not None:
        rotate_xyz.Set(Gf.Vec3f(rotate_x, rotate_y, yaw))
    elif orient is not None:
        rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw)
        quaternion = rotation.GetQuat()
        orient.Set(Gf.Quatf(
            quaternion.GetReal(),
            Gf.Vec3f(*quaternion.GetImaginary()),
        ))
    else:
        rotate_xyz = xform.AddRotateXYZOp()
        rotate_xyz.Set(Gf.Vec3f(rotate_x, rotate_y, yaw))
    surface = str(placement["surface"])
    if surface == "sofa":
        production_path = str(Path(__file__).resolve().parents[1] / "production")
        if production_path not in sys.path:
            sys.path.insert(0, production_path)
        from visual_scene_geometry import visible_bounds
        get_bound = visible_bounds
    else:
        get_bound = _world_bound
    x = float(placement["x_m"])
    y = (
        float(placement["y_m"])
        + float(placement.get("world_y_offset_m", 0.0))
    )
    top = float(placement["surface_z_m"])
    initial_z = top + .05
    if placement.get("coordinate_frame_prim"):
        from pxr import Usd
        frame = stage.GetPrimAtPath(str(placement["coordinate_frame_prim"]))
        if not frame or not frame.IsActive():
            raise ValueError(f"placement coordinate frame missing: {placement['coordinate_frame_prim']}")
        frame_matrix = UsdGeom.Xformable(frame).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = frame_matrix.Transform(Gf.Vec3d(x, y, 0.0))
        parent_matrix = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local = parent_matrix.GetInverse().Transform(Gf.Vec3d(world[0], world[1], top + .05))
        x, y = float(local[0]), float(local[1])
        initial_z = float(local[2])
    translate.Set(Gf.Vec3d(x, y, initial_z))
    bound = get_bound(stage, path)
    if bound is None:
        raise ValueError(f"randomized prop has no finite bound: {path}")
    if surface in {"coffee_table", "sofa"}:
        # Policy XY denotes the visible footprint center, not an imported
        # asset's arbitrary pivot. Correct it before physics and before the
        # full-geometry workspace check; the recorded target is never moved.
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
        intended = parent.Transform(Gf.Vec3d(x, y, initial_z))
        correction = parent.GetInverse().TransformDir(Gf.Vec3d(
            intended[0] - (bound[0] + bound[3]) / 2,
            intended[1] - (bound[1] + bound[4]) / 2, 0))
        current = translate.Get()
        translate.Set(Gf.Vec3d(current[0] + correction[0],
                              current[1] + correction[1], current[2]))
        bound = get_bound(stage, path)
    dz = top + (0.003 if placement.get("dynamic") else 0.001) - bound[2]
    current = translate.Get()
    translate.Set(type(current)(
        float(current[0]),
        float(current[1]),
        float(current[2]) + dz,
    ))
    if surface == "sofa":
        sofa = load_policy()["surfaces"]["sofa"]
        if placement.get("coordinate_frame_prim"):
            raise ValueError("sofa placement needs a validated sofa-local bounds adapter")
        xmin, xmax, ymin, ymax = placement.get("placement_bounds_xy_m", sofa["bounds_xy_m"])
        visible = get_bound(stage, path)
        margin = float(placement.get("placement_edge_margin_m", sofa["edge_margin_m"]))
        if not (xmin + margin <= visible[0] and visible[3] <= xmax - margin
                and ymin + margin <= visible[1] and visible[4] <= ymax - margin):
            raise ValueError(f"sofa prop outside middle two seats: {path}: {visible}")
    from physical_distractors import ensure_collision_geometry
    ensure_collision_geometry(stage, path)
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    if not rigid:
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(
        not bool(placement.get("dynamic")))
    rigid.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
    if not mass:
        mass = UsdPhysics.MassAPI.Apply(prim)
    if placement.get("mass_kg") is not None:
        mass.CreateMassAttr(float(placement["mass_kg"]))
    return path


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            # The Isaac container runs under a different uid and consumes this
            # plan before enabling contact capture. NamedTemporaryFile is 0600.
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_plan_to_stage(
    *,
    scene_path: Path | str,
    plan: Mapping,
    target_prim_path: str = "",
    protected_prim_paths: Iterable[str] = (),
    policy: Mapping | None = None,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> dict:
    """Apply a verified plan without ever authoring furniture transforms."""
    from pxr import Usd, UsdGeom
    from shelf_asset_policy import (
        asset_has_approval,
        verify_refined_bundle,
    )

    effective = policy or load_policy()
    verified_background_assets = {}
    for asset_id in sorted({
        str(item["asset_id"])
        for item in plan.get("objects", ())
        if item.get("asset_id")
    }):
        approved, approval_reason, approval_revision = asset_has_approval(
            asset_id
        )
        if not approved:
            raise ValueError(
                f"background asset is not approved: {asset_id}: "
                f"{approval_reason}"
            )
        pinned, pin_reason, bundle_revision = verify_refined_bundle(
            asset_id, demo_root
        )
        if not pinned:
            raise ValueError(
                f"background asset bundle is not reviewed: {asset_id}: "
                f"{pin_reason}"
            )
        verified_background_assets[asset_id] = {
            "approval_revision": approval_revision,
            "bundle_revision": bundle_revision,
        }
    scene = Path(scene_path)
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"could not open scene: {scene}")
    furniture_paths = tuple(
        str(value)
        for value in effective["protected_furniture_prims"]
    )
    furniture_before = {}
    for path in furniture_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"protected furniture prim is missing: {path}")
        furniture_before[path] = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
        )
    protected = set(map(str, protected_prim_paths))
    if target_prim_path:
        protected.add(str(target_prim_path))
    protected_before = {}
    for path in protected:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            protected_before[path] = {
                "active": prim.IsActive(),
                "matrix": _matrix_values(
                    UsdGeom.Xformable(
                        prim).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())
                ),
            }

    # Episode-generated roots are safe to rebuild because no canonical asset
    # or furniture is authored below them.
    for generated in (
        "/World/EpisodeDistractors",
        "/World/EpisodeBackground",
    ):
        if stage.GetPrimAtPath(generated):
            stage.RemovePrim(generated)
    stage.DefinePrim("/World/EpisodeDistractors", "Xform")
    stage.DefinePrim("/World/EpisodeBackground", "Xform")

    from shelf_support_geometry import validate_shelf_supports
    validate_shelf_supports(stage, policy=effective)

    surface_paths = {
        str(item["prim_path"])
        for item in effective["surface_objects"]
        if not str(item["prim_path"]).startswith(
            "/World/EpisodeDistractors/")
    }
    for path in surface_paths:
        if path in protected:
            continue
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)
    # Tray/CoffeeBeans are episode props, not furniture.  Keep the tray only
    # for the four tray-destination tasks; otherwise clear both before adding
    # the randomized interactive set.
    task_id = str(plan.get("task_id", ""))
    for path in ("/World/Assets/Tray", "/World/Assets/CoffeeBeans"):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or path in protected:
            continue
        prim.SetActive(
            path == "/World/Assets/Tray"
            and task_id in {"42", "43", "44", "46"}
        )

    applied = []
    for placement in plan.get("objects", ()):
        path = str(placement["prim_path"])
        if path in protected:
            continue
        applied.append(_set_pose(
            stage, placement, Path(demo_root)))

    from coffee_table_partition import validate_cleanup_objects
    table_workspace = validate_cleanup_objects(stage, [
        str(item["prim_path"]) for item in plan.get("objects", ())
        if item.get("surface") == "coffee_table" and str(item["prim_path"]) in applied
    ])

    from physical_distractors import validate_bodies
    physical_bodies = validate_bodies(stage, applied)

    support_evidence = validate_shelf_supports(
        stage, policy=effective,
        placements=[p for p in plan.get("objects", ()) if str(p["prim_path"]) in applied],
    )

    for path, expected in furniture_before.items():
        prim = stage.GetPrimAtPath(path)
        observed = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
        )
        if observed != expected:
            raise ValueError(
                f"episode randomization moved protected furniture {path}"
            )
    for path, expected in protected_before.items():
        prim = stage.GetPrimAtPath(path)
        if (
            not prim
            or not prim.IsValid()
            or prim.IsActive() != expected["active"]
            or _matrix_values(
                UsdGeom.Xformable(
                    prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
            ) != expected["matrix"]
        ):
            raise ValueError(
                f"episode randomization changed target/support {path}"
            )
    stage.GetRootLayer().Save()
    # The clean-session sidecar is playback's authoritative isolation list.
    # Randomized props are selected after session open, so authorizing only
    # their USD active flags would let playback deactivate them again.  Extend
    # this same session with exactly the applied bounded set.
    from session_scene import extend_session_activation
    session_extension = extend_session_activation(
        applied, authored_poses_replaced=True
    )
    manifest = {
        **dict(plan),
        "schema": "real2sim.episode_scene_randomization.applied.v1",
        "scene_path": str(scene.resolve()),
        "applied_prim_paths": sorted(applied),
        "physical_bodies": physical_bodies,
        "coffee_table_workspace": table_workspace,
        "physics_rollout_performed": False,
        "verified_background_assets": verified_background_assets,
        "shelf_support_geometry": support_evidence,
        "furniture_transforms_unchanged": True,
        "target_and_support_transforms_unchanged": True,
        "session_activation_extension": session_extension,
    }
    sidecar = scene.with_suffix(
        scene.suffix + ".episode_randomization.json")
    _atomic_json(sidecar, manifest)
    manifest["sidecar_path"] = str(sidecar)
    return manifest


def run_randomization_subprocess(
    *,
    scene_path: Path | str,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    target_prim_path: str = "",
    hand: str = "?",
    place_zone: Sequence[float] | None = None,
    protected_prim_paths: Iterable[str] = (),
    protected_objects: Sequence[Mapping] = (),
    policy_path: Path | str = POLICY_PATH,
    timeout: int = 180,
) -> dict:
    """Apply with the USD-capable interpreter and return its JSON result."""
    command = [
        str(MESH311_PYTHON),
        str(Path(__file__).resolve()),
        "--scene",
        str(scene_path),
        "--task-id",
        str(task_id),
        "--hand",
        str(hand or "?").lower(),
        "--policy",
        str(policy_path),
    ]
    if episode_uid:
        command.extend(["--episode-uid", str(episode_uid)])
    if seed_override is not None and str(seed_override).strip():
        command.extend(["--seed", str(seed_override)])
    if target_prim_path:
        command.extend(["--target-prim", str(target_prim_path)])
    if place_zone is not None:
        command.extend([
            "--place-zone-json",
            json.dumps(list(place_zone), separators=(",", ":")),
        ])
    for path in sorted(set(map(str, protected_prim_paths))):
        command.extend(["--protected-prim", path])
    for descriptor in protected_objects:
        command.extend([
            "--protected-object-json",
            json.dumps(dict(descriptor), separators=(",", ":")),
        ])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "episode randomizer returned invalid JSON: "
            + (result.stdout + result.stderr)[-800:]
        ) from exc
    if result.returncode != 0 or payload.get("ok") is not True:
        raise RuntimeError(
            "episode randomization failed: "
            + str(payload.get("reason", payload))
        )
    return payload["randomization"]


def _parse_zone(raw: str) -> list[float] | None:
    if not raw:
        return None
    value = json.loads(raw)
    if (
        not isinstance(value, list)
        or len(value) < 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError("place zone must be a finite JSON number list")
    return [float(item) for item in value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-uid", default="")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--target-prim", default="")
    parser.add_argument("--hand", default="?")
    parser.add_argument("--place-zone-json", default="")
    parser.add_argument("--protected-prim", action="append", default=[])
    parser.add_argument(
        "--protected-object-json", action="append", default=[]
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        hand = str(args.hand).strip().lower()
        if hand not in HAND_VALUES:
            raise ValueError("hand must be l, r, or ?")
        target = None
        protected_objects = [
            json.loads(value) for value in args.protected_object_json
        ]
        # Open the stage only on the USD application path.  This keeps policy
        # and planner unit tests independent of the pxr package.
        if args.target_prim or args.protected_prim:
            from pxr import Usd

            stage = Usd.Stage.Open(str(args.scene))
            if not stage:
                raise ValueError(f"could not open scene: {args.scene}")
        if args.target_prim:
            target = _target_descriptor(
                stage, args.target_prim, policy)
            if target is None:
                raise ValueError(
                    f"target prim is missing or unbounded: "
                    f"{args.target_prim}"
                )
        for path in args.protected_prim:
            descriptor = _target_descriptor(stage, path, policy)
            if descriptor is not None:
                protected_objects.append(descriptor)
        zone = _parse_zone(args.place_zone_json)
        plan = build_layout_plan(
            task_id=str(args.task_id),
            episode_uid=str(args.episode_uid),
            seed_override=args.seed,
            hand=hand,
            target=target,
            place_zone=zone,
            protected_prim_paths=args.protected_prim,
            protected_objects=protected_objects,
            policy=policy,
        )
        if args.dry_run:
            result = plan
        else:
            result = apply_plan_to_stage(
                scene_path=args.scene,
                plan=plan,
                target_prim_path=args.target_prim,
                protected_prim_paths=args.protected_prim,
                policy=policy,
            )
        print(json.dumps(
            {"ok": True, "randomization": result},
            sort_keys=True,
        ))
        return 0
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

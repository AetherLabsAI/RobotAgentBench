#!/usr/bin/env python3
"""Load and statically audit the final explicit full-House layouts.

The catalog is a layout manifest, not a task registry.  Selected layouts are
materialized only in a transient House session layer.
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = HERE / "house_final_layouts.json"
SCHEMA = "real2sim.house_final_layout_candidates.v3"
STATUS = "canary_candidate"
CANDIDATE_COUNT = 50
WALL_THICKNESS_M = 0.12
FURNITURE_TO_WALL_GAP_M = 0.02
ROOM_INTERIOR_INSET_M = WALL_THICKNESS_M / 2.0 + FURNITURE_TO_WALL_GAP_M


def _numbers(values: Any, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _inside_rect(
    point: tuple[float, float], rect: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    x, y, width, height = rect
    return bool(
        x + margin <= point[0] <= x + width - margin
        and y + margin <= point[1] <= y + height - margin
    )


def _rect_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    gap: float = 0.0,
) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return not (
        lx + lw + gap <= rx
        or rx + rw + gap <= lx
        or ly + lh + gap <= ry
        or ry + rh + gap <= ly
    )


def _rotated_size(size: tuple[float, float], yaw_deg: float) -> tuple[float, float]:
    radians = math.radians(yaw_deg)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))
    return (
        size[0] * cosine + size[1] * sine,
        size[0] * sine + size[1] * cosine,
    )


def _rotated_footprint_bounds(
    module: Mapping[str, Any], yaw_deg: float,
) -> tuple[float, float, float, float]:
    """Return the exact rotated AABB relative to the module task pivot."""

    bounds = _numbers(
        module.get("footprint_bounds_xy_m"), 4, "module footprint bounds")
    min_x, min_y, max_x, max_y = bounds
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("module footprint bounds must have positive area")
    angle = math.radians(float(yaw_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = [
        (
            cosine * x - sine * y,
            sine * x + cosine * y,
        )
        for x in (min_x, max_x)
        for y in (min_y, max_y)
    ]
    return (
        min(point[0] for point in rotated),
        min(point[1] for point in rotated),
        max(point[0] for point in rotated),
        max(point[1] for point in rotated),
    )


def _fixture_rect(
    module: Mapping[str, Any], pose: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    center = _numbers(pose.get("center_xy_m"), 2, "module center")
    min_x, min_y, max_x, max_y = _rotated_footprint_bounds(
        module, float(pose.get("yaw_deg", 0.0)))
    return (
        center[0] + min_x,
        center[1] + min_y,
        max_x - min_x,
        max_y - min_y,
    )


def _station_point(
    module: Mapping[str, Any], pose: Mapping[str, Any],
) -> tuple[float, float]:
    explicit = pose.get("interaction_station_xy_m")
    if explicit is not None:
        values = _numbers(explicit, 2, "interaction_station_xy_m")
        return float(values[0]), float(values[1])
    center = _numbers(pose.get("center_xy_m"), 2, "module center")
    distance = float(module.get("station_offset_m", 0.8))
    yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
    # Every module's canonical interaction face points toward local -Y.
    local_x, local_y = 0.0, -distance
    return (
        center[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
        center[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
    )


def _floor_rects(candidate: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    rects = [
        _numbers(space.get("rect_xywh_m"), 4, f"space {space.get('id')} rect")
        for space in candidate.get("spaces", ())
    ]
    for corridor in candidate.get("corridors", ()):
        rects.extend(
            _numbers(rect, 4, f"corridor {corridor.get('id')} rect")
            for rect in corridor.get("rects_xywh_m", ())
        )
    return [tuple(rect) for rect in rects]


def _grid_navigation(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    *,
    resolution: float = 0.20,
    robot_radius: float = 0.30,
) -> dict[str, Any]:
    floor = _floor_rects(candidate)
    min_x = min(rect[0] for rect in floor)
    min_y = min(rect[1] for rect in floor)
    max_x = max(rect[0] + rect[2] for rect in floor)
    max_y = max(rect[1] + rect[3] for rect in floor)
    columns = int(math.ceil((max_x - min_x) / resolution)) + 1
    rows = int(math.ceil((max_y - min_y) / resolution)) + 1
    poses = candidate["module_poses"]
    obstacles = []
    for module_id, pose in poses.items():
        x, y, width, height = _fixture_rect(modules[module_id], pose)
        obstacles.append((
            x - robot_radius,
            y - robot_radius,
            width + 2.0 * robot_radius,
            height + 2.0 * robot_radius,
        ))

    def point(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            min_x + cell[0] * resolution,
            min_y + cell[1] * resolution,
        )

    def free(cell: tuple[int, int]) -> bool:
        xy = point(cell)
        if not any(_inside_rect(xy, rect, robot_radius) for rect in floor):
            return False
        return not any(_inside_rect(xy, obstacle) for obstacle in obstacles)

    free_cells = {
        (column, row)
        for column in range(columns)
        for row in range(rows)
        if free((column, row))
    }
    entry = _numbers(candidate.get("entry_xy_m"), 2, "entry_xy_m")

    def nearest_cell(xy: tuple[float, float]) -> tuple[int, int] | None:
        if not free_cells:
            return None
        return min(
            free_cells,
            key=lambda cell: math.dist(point(cell), xy),
        )

    source = nearest_cell(tuple(entry))
    visited: set[tuple[int, int]] = set()
    if source is not None:
        queue = deque([source])
        visited.add(source)
        while queue:
            column, row = queue.popleft()
            for neighbor in (
                (column - 1, row), (column + 1, row),
                (column, row - 1), (column, row + 1),
            ):
                if neighbor in free_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    station_rows: dict[str, Any] = {}
    for module_id, pose in poses.items():
        station = _station_point(modules[module_id], pose)
        cell = nearest_cell(station)
        bridge = None if cell is None else math.dist(station, point(cell))
        station_rows[module_id] = {
            "xy_m": [round(value, 4) for value in station],
            "reachable": cell in visited if cell is not None else False,
            "bridge_distance_m": None if bridge is None else round(bridge, 4),
        }
    return {
        "passed": bool(
            source is not None
            and visited
            and all(row["reachable"] for row in station_rows.values())
        ),
        "resolution_m": resolution,
        "robot_radius_m": robot_radius,
        "free_cell_count": len(free_cells),
        "reachable_free_cell_count": len(visited),
        "reachable_fraction": (
            round(len(visited) / len(free_cells), 6) if free_cells else 0.0
        ),
        "stations": station_rows,
    }


def audit_candidate(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    spaces = list(candidate.get("spaces", ()))
    space_by_id = {str(space.get("id", "")): space for space in spaces}
    if len(space_by_id) != len(spaces):
        errors.append("space IDs must be unique")
    required_kinds = {"living", "kitchen", "dining", "storage", "entry"}
    found_kinds = {str(space.get("kind", "")) for space in spaces}
    if not required_kinds.issubset(found_kinds):
        errors.append("missing required functional spaces")

    poses = candidate.get("module_poses")
    if not isinstance(poses, dict) or set(poses) != set(modules):
        errors.append("module_poses must place every interaction module exactly once")
        poses = {}
    fixture_rows: dict[str, tuple[float, float, float, float]] = {}
    station_rows: dict[str, tuple[float, float]] = {}
    for module_id, pose in poses.items():
        space_id = str(pose.get("space_id", ""))
        if space_id not in space_by_id:
            errors.append(f"{module_id} uses unknown space {space_id}")
            continue
        module = modules[module_id]
        bounds = _numbers(
            module.get("footprint_bounds_xy_m"), 4,
            f"{module_id} footprint bounds",
        )
        size = _numbers(
            module.get("fixture_size_xy_m"), 2,
            f"{module_id} fixture size",
        )
        measured_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if any(
            not math.isclose(recorded, measured, abs_tol=1.0e-6)
            for recorded, measured in zip(size, measured_size, strict=True)
        ):
            errors.append(
                f"{module_id} fixture size does not match exact footprint bounds"
            )
        allowed_kinds = set(map(str, module.get("allowed_space_kinds", ())))
        kind = str(space_by_id[space_id].get("kind", ""))
        if kind not in allowed_kinds:
            errors.append(f"{module_id} cannot be placed in {kind}")
        room = _numbers(
            space_by_id[space_id].get("rect_xywh_m"), 4,
            f"space {space_id} rect",
        )
        fixture = _fixture_rect(module, pose)
        fixture_rows[module_id] = fixture
        corners = (
            (fixture[0], fixture[1]),
            (fixture[0] + fixture[2], fixture[1]),
            (fixture[0], fixture[1] + fixture[3]),
            (fixture[0] + fixture[2], fixture[1] + fixture[3]),
        )
        if not all(
            _inside_rect(corner, tuple(room), ROOM_INTERIOR_INSET_M - 2.0e-6)
            for corner in corners
        ):
            errors.append(f"{module_id} fixture leaves its assigned room")
        attachment = pose.get("wall_attachment")
        if attachment is not None:
            side = str(attachment.get("side", ""))
            recorded_gap = float(attachment.get("furniture_to_wall_gap_m", -1.0))
            if side == "north":
                observed_gap = room[1] + room[3] - WALL_THICKNESS_M / 2.0 - (
                    fixture[1] + fixture[3]
                )
            elif side == "south":
                observed_gap = fixture[1] - (room[1] + WALL_THICKNESS_M / 2.0)
            elif side == "east":
                observed_gap = room[0] + room[2] - WALL_THICKNESS_M / 2.0 - (
                    fixture[0] + fixture[2]
                )
            elif side == "west":
                observed_gap = fixture[0] - (room[0] + WALL_THICKNESS_M / 2.0)
            else:
                errors.append(f"{module_id} has invalid wall attachment side")
                observed_gap = recorded_gap
            if (
                not math.isclose(
                    recorded_gap, FURNITURE_TO_WALL_GAP_M, abs_tol=1.0e-9
                )
                or not math.isclose(
                    observed_gap, recorded_gap, abs_tol=2.0e-6
                )
            ):
                errors.append(f"{module_id} is not flush to its assigned wall")
        station = _station_point(module, pose)
        station_rows[module_id] = station
        # Wall-mounted modules can have a task station whose tangential axis
        # remains close to the wall (the TV cabinet is the important case).
        # Require the declared point to be physically inside the room here;
        # the conservative inflated grid below remains the authority for
        # whether the robot can actually reach or bridge to that station.
        if not _inside_rect(station, tuple(room), ROOM_INTERIOR_INSET_M):
            errors.append(f"{module_id} station leaves its assigned room")

    module_ids = sorted(fixture_rows)
    collisions = []
    for index, left_id in enumerate(module_ids):
        for right_id in module_ids[index + 1:]:
            if _rect_overlap(fixture_rows[left_id], fixture_rows[right_id], 0.10):
                collisions.append([left_id, right_id])
    if collisions:
        errors.append("fixture collisions: " + repr(collisions))

    graph = {space_id: set() for space_id in space_by_id}
    for corridor in candidate.get("corridors", ()):
        endpoints = list(map(str, corridor.get("connects", ())))
        if len(endpoints) != 2 or any(item not in graph for item in endpoints):
            errors.append(f"invalid corridor {corridor.get('id')}")
            continue
        if float(corridor.get("clear_width_m", 0.0)) < 1.0:
            errors.append(f"corridor {corridor.get('id')} is too narrow")
        graph[endpoints[0]].add(endpoints[1])
        graph[endpoints[1]].add(endpoints[0])
    reached: set[str] = set()
    if graph:
        queue = deque([next(iter(graph))])
        while queue:
            node = queue.popleft()
            if node in reached:
                continue
            reached.add(node)
            queue.extend(graph[node] - reached)
    if reached != set(graph):
        errors.append("space/corridor graph is disconnected")

    navigation = _grid_navigation(candidate, modules) if not errors else {
        "passed": False,
        "skipped_due_to_geometry_errors": True,
    }
    if not navigation["passed"]:
        errors.append("conservative grid navigation did not reach every module station")
    return {
        "passed": not errors,
        "errors": errors,
        "space_graph_connected": reached == set(graph),
        "fixture_collision_count": len(collisions),
        "grid_navigation": navigation,
    }


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    if catalog.get("status") != STATUS:
        raise ValueError(f"candidate status must be {STATUS}")
    if catalog.get("layout_count") != CANDIDATE_COUNT:
        raise ValueError("candidate count must be exactly 50")
    modules = catalog.get("interaction_modules")
    candidates = catalog.get("candidates")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("catalog must define interaction_modules")
    if not isinstance(candidates, list) or len(candidates) != CANDIDATE_COUNT:
        raise ValueError("catalog must contain exactly 50 candidates")
    expected_ordinals = list(range(1, CANDIDATE_COUNT + 1))
    if [row.get("ordinal") for row in candidates] != expected_ordinals:
        raise ValueError("candidate ordinals must be 1..50")
    ids = [row.get("id") for row in candidates]
    if len(set(ids)) != CANDIDATE_COUNT or any(not item for item in ids):
        raise ValueError("candidate IDs must be 50 unique non-empty values")
    signatures = set()
    for row in candidates:
        audit = audit_candidate(row, modules)
        effective_errors = [
            error for error in audit["errors"]
            if error != (
                "conservative grid navigation did not reach every module "
                "station"
            )
        ]
        if effective_errors:
            raise ValueError(f"{row['id']} failed audit: {effective_errors}")
        recorded = row.get("preflight")
        if (
            not isinstance(recorded, dict)
            or recorded.get("planar_geometry") != "passed"
            or recorded.get("space_graph") != "passed"
            or recorded.get("exact_station_continuous_base_sweep") != "passed"
            or row.get("strict_base_navigation", {}).get("passed") is not True
        ):
            raise ValueError(f"{row['id']} has invalid final preflight record")
        signature = json.dumps(
            {
                "spaces": row["spaces"],
                "corridors": row["corridors"],
                "module_poses": row["module_poses"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in signatures:
            raise ValueError("full-House candidates must be structurally unique")
        signatures.add(signature)
    return catalog


def audit_all(path: Path | str = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    catalog = load_catalog(path)
    modules = catalog["interaction_modules"]
    rows = [audit_candidate(row, modules) for row in catalog["candidates"]]
    for index, row in enumerate(rows):
        errors = [
            error for error in row["errors"]
            if error != (
                "conservative grid navigation did not reach every module station"
            )
        ]
        rows[index] = {**row, "errors": errors, "passed": not errors}
    strict_reachable_fractions = []
    for candidate in catalog["candidates"]:
        stations = candidate["strict_base_navigation"]["stations"]
        strict_reachable_fractions.append(
            sum(
                station.get("exact_station_occupiable") is True
                and station.get("reachable") is True
                for station in stations.values()
            ) / len(stations)
        )
    return {
        "schema": "real2sim.house_final_layout_static_audit.v2",
        "status": catalog["status"],
        "candidate_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "minimum_exact_station_reachable_fraction": min(
            strict_reachable_fractions),
        "rows": rows,
    }

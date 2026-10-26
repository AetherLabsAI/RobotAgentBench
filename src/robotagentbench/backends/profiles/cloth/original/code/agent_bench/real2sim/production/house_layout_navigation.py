#!/usr/bin/env python3
"""Strict continuous base-envelope navigation audit for House candidates.

The older topology preflight intentionally answered only a coarse planning
question.  It snapped an interaction station to the nearest free grid cell,
which can hide a station behind another fixture.  This module is the stricter
design-review gate: the entry and every authored station must themselves fit
the complete navigation envelope, and every grid edge plus the short exact
endpoint connections is checked as a continuous swept segment.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Mapping, Sequence

try:
    from . import house_topology
except ImportError:  # Direct script execution.
    import house_topology


GRID_RESOLUTION_M = 0.05
ROBOT_BASE_RADIUS_M = 0.30
CARRIED_OBJECT_MARGIN_M = 0.04
MINIMUM_ROUTE_CLEARANCE_M = 0.03
NAVIGATION_ENVELOPE_RADIUS_M = (
    ROBOT_BASE_RADIUS_M
    + CARRIED_OBJECT_MARGIN_M
    + MINIMUM_ROUTE_CLEARANCE_M
)
_EPSILON = 1.0e-9


Point = tuple[float, float]
Rect = tuple[float, float, float, float]
Segment = tuple[Point, Point]
Cell = tuple[int, int]


def _point_in_floor(point: Point, floor: Sequence[Rect]) -> bool:
    return any(house_topology._inside_rect(point, rect) for rect in floor)


def _floor_boundary_segments(floor: Sequence[Rect]) -> tuple[Segment, ...]:
    """Return the exact exterior boundary of an axis-aligned rectangle union."""

    xs = sorted({value for x, _y, width, _height in floor for value in (x, x + width)})
    ys = sorted({value for _x, y, _width, height in floor for value in (y, y + height)})
    occupied: set[Cell] = set()
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            center = (
                (xs[xi] + xs[xi + 1]) / 2.0,
                (ys[yi] + ys[yi + 1]) / 2.0,
            )
            if _point_in_floor(center, floor):
                occupied.add((xi, yi))

    segments: list[Segment] = []
    for xi, yi in occupied:
        x0, x1 = xs[xi], xs[xi + 1]
        y0, y1 = ys[yi], ys[yi + 1]
        if (xi - 1, yi) not in occupied:
            segments.append(((x0, y0), (x0, y1)))
        if (xi + 1, yi) not in occupied:
            segments.append(((x1, y0), (x1, y1)))
        if (xi, yi - 1) not in occupied:
            segments.append(((x0, y0), (x1, y0)))
        if (xi, yi + 1) not in occupied:
            segments.append(((x0, y1), (x1, y1)))
    return tuple(segments)


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-18:
        return math.dist(point, start)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    closest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.dist(point, closest)


def _orientation(left: Point, right: Point, point: Point) -> float:
    return (
        (right[0] - left[0]) * (point[1] - left[1])
        - (right[1] - left[1]) * (point[0] - left[0])
    )


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    return bool(
        min(start[0], end[0]) - _EPSILON <= point[0]
        <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON <= point[1]
        <= max(start[1], end[1]) + _EPSILON
        and abs(_orientation(start, end, point)) <= _EPSILON
    )


def _segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    if (
        (o1 > _EPSILON and o2 < -_EPSILON or o1 < -_EPSILON and o2 > _EPSILON)
        and (o3 > _EPSILON and o4 < -_EPSILON or o3 < -_EPSILON and o4 > _EPSILON)
    ):
        return True
    return bool(
        (abs(o1) <= _EPSILON and _on_segment(b0, a0, a1))
        or (abs(o2) <= _EPSILON and _on_segment(b1, a0, a1))
        or (abs(o3) <= _EPSILON and _on_segment(a0, b0, b1))
        or (abs(o4) <= _EPSILON and _on_segment(a1, b0, b1))
    )


def _segment_to_segment_distance(
    a0: Point, a1: Point, b0: Point, b1: Point,
) -> float:
    if _segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_to_segment_distance(a0, b0, b1),
        _point_to_segment_distance(a1, b0, b1),
        _point_to_segment_distance(b0, a0, a1),
        _point_to_segment_distance(b1, a0, a1),
    )


def _point_to_rect_distance(point: Point, rect: Rect) -> float:
    x, y, width, height = rect
    dx = max(x - point[0], 0.0, point[0] - (x + width))
    dy = max(y - point[1], 0.0, point[1] - (y + height))
    return math.hypot(dx, dy)


def _segment_intersects_rect(start: Point, end: Point, rect: Rect) -> bool:
    x, y, width, height = rect
    if house_topology._inside_rect(start, rect) or house_topology._inside_rect(end, rect):
        return True
    corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
    return any(
        _segments_intersect(start, end, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def _segment_to_rect_distance(start: Point, end: Point, rect: Rect) -> float:
    if _segment_intersects_rect(start, end, rect):
        return 0.0
    x, y, width, height = rect
    corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
    return min(
        _point_to_rect_distance(start, rect),
        _point_to_rect_distance(end, rect),
        *(_point_to_segment_distance(corner, start, end) for corner in corners),
    )


def _station_point(module: Mapping[str, Any], pose: Mapping[str, Any]) -> Point:
    """Use an explicit interaction station when a design defines one."""

    explicit = pose.get("interaction_station_xy_m")
    if explicit is not None:
        values = house_topology._numbers(explicit, 2, "interaction_station_xy_m")
        return float(values[0]), float(values[1])
    return house_topology._station_point(module, pose)


def audit_candidate_navigation(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
    *,
    resolution_m: float = GRID_RESOLUTION_M,
    envelope_radius_m: float = NAVIGATION_ENVELOPE_RADIUS_M,
    include_paths: bool = False,
) -> dict[str, Any]:
    """Plan exact-entry-to-exact-station paths with swept disk clearance."""

    if resolution_m <= 0.0 or envelope_radius_m <= 0.0:
        raise ValueError("navigation resolution and envelope radius must be positive")
    floor = house_topology._floor_rects(candidate)
    boundary = _floor_boundary_segments(floor)
    poses = candidate["module_poses"]
    obstacles: dict[str, Rect] = {
        module_id: house_topology._fixture_rect(modules[module_id], pose)
        for module_id, pose in poses.items()
    }
    min_x = min(rect[0] for rect in floor)
    min_y = min(rect[1] for rect in floor)
    max_x = max(rect[0] + rect[2] for rect in floor)
    max_y = max(rect[1] + rect[3] for rect in floor)
    columns = int(math.floor((max_x - min_x) / resolution_m)) + 1
    rows = int(math.floor((max_y - min_y) / resolution_m)) + 1

    def point(cell: Cell) -> Point:
        return min_x + cell[0] * resolution_m, min_y + cell[1] * resolution_m

    def point_clearance(xy: Point) -> tuple[bool, str | None, float]:
        if not _point_in_floor(xy, floor):
            return False, "outside_floor", 0.0
        wall_clearance = min(
            _point_to_segment_distance(xy, start, end)
            for start, end in boundary
        )
        closest_id, obstacle_clearance = min(
            (
                (module_id, _point_to_rect_distance(xy, rect))
                for module_id, rect in obstacles.items()
            ),
            key=lambda row: row[1],
        )
        clearance = min(wall_clearance, obstacle_clearance)
        if wall_clearance + _EPSILON < envelope_radius_m:
            return False, "wall", clearance
        if obstacle_clearance + _EPSILON < envelope_radius_m:
            return False, closest_id, clearance
        return True, None, clearance

    def segment_geometry_free(start: Point, end: Point) -> bool:
        if any(
            _segment_to_segment_distance(start, end, wall_start, wall_end)
            + _EPSILON < envelope_radius_m
            for wall_start, wall_end in boundary
        ):
            return False
        return all(
            _segment_to_rect_distance(start, end, rect) + _EPSILON
            >= envelope_radius_m
            for rect in obstacles.values()
        )

    def segment_free(start: Point, end: Point) -> bool:
        start_ok, _start_blocker, _start_clearance = point_clearance(start)
        end_ok, _end_blocker, _end_clearance = point_clearance(end)
        return bool(
            start_ok and end_ok and segment_geometry_free(start, end)
        )

    free_clearances: dict[Cell, float] = {}
    for column in range(columns):
        for row in range(rows):
            cell = (column, row)
            free, _blocker, clearance = point_clearance(point(cell))
            if free:
                free_clearances[cell] = clearance
    free_cells = set(free_clearances)

    edge_cache: dict[tuple[Cell, Cell], bool] = {}

    def edge_free(left: Cell, right: Cell) -> bool:
        key = (left, right) if left < right else (right, left)
        if key not in edge_cache:
            length = math.dist(point(left), point(right))
            # Clearance is 1-Lipschitz.  If both endpoints have another half
            # edge of margin, every point on the segment is proven clear and
            # the expensive exact boundary sweep is unnecessary.
            if min(free_clearances[left], free_clearances[right]) + _EPSILON >= (
                envelope_radius_m + length / 2.0
            ):
                edge_cache[key] = True
            else:
                edge_cache[key] = segment_geometry_free(
                    point(left), point(right))
        return edge_cache[key]

    def neighbors(cell: Cell):
        for dx, dy in (
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ):
            neighbor = cell[0] + dx, cell[1] + dy
            if neighbor in free_cells and edge_free(cell, neighbor):
                yield neighbor

    def connector(xy: Point) -> tuple[Cell | None, float | None]:
        ok, _blocker, _clearance = point_clearance(xy)
        if not ok:
            return None, None
        column = round((xy[0] - min_x) / resolution_m)
        row = round((xy[1] - min_y) / resolution_m)
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (column + dx, row + dy)
                if cell in free_cells and segment_free(xy, point(cell)):
                    candidates.append((math.dist(xy, point(cell)), cell))
        if not candidates:
            return None, None
        distance, cell = min(candidates)
        # This is only the sub-grid connection actually swept by the disk,
        # never a semantic bridge to a different work position.
        if distance > math.sqrt(2.0) * resolution_m + _EPSILON:
            return None, None
        return cell, distance

    entry_values = house_topology._numbers(candidate.get("entry_xy_m"), 2, "entry_xy_m")
    entry = float(entry_values[0]), float(entry_values[1])
    source, entry_connection = connector(entry)
    parents: dict[Cell, Cell | None] = {}
    if source is not None:
        parents[source] = None
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors(current):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)

    station_rows: dict[str, Any] = {}
    for module_id, pose in poses.items():
        station = _station_point(modules[module_id], pose)
        station_ok, blocker, station_clearance = point_clearance(station)
        target, station_connection = connector(station)
        reachable = bool(
            station_ok and target is not None and target in parents
        )
        path: list[Point] = []
        path_length = None
        if reachable and target is not None:
            cells = []
            cursor: Cell | None = target
            while cursor is not None:
                cells.append(cursor)
                cursor = parents[cursor]
            cells.reverse()
            path = [entry, *(point(cell) for cell in cells), station]
            path_length = sum(
                math.dist(start, end) for start, end in zip(path, path[1:])
            )
        row: dict[str, Any] = {
            "xy_m": [round(value, 6) for value in station],
            "exact_station_occupiable": station_ok,
            "reachable": reachable,
            "blocking_geometry": blocker,
            "station_clearance_m": round(station_clearance, 6),
            "subgrid_connection_distance_m": (
                round(station_connection, 6)
                if station_connection is not None else None
            ),
            "path_length_m": round(path_length, 6) if path_length is not None else None,
        }
        if include_paths:
            row["path_xy_m"] = [
                [round(value, 6) for value in waypoint] for waypoint in path
            ]
        station_rows[module_id] = row

    entry_ok, entry_blocker, entry_clearance = point_clearance(entry)
    failed = sorted(
        module_id for module_id, row in station_rows.items()
        if not row["reachable"]
    )
    return {
        "passed": bool(entry_ok and source is not None and not failed),
        "method": "exact_endpoint_continuous_swept_disk_grid_v1",
        "grid_resolution_m": resolution_m,
        "robot_base_radius_m": ROBOT_BASE_RADIUS_M,
        "carried_object_margin_m": CARRIED_OBJECT_MARGIN_M,
        "minimum_route_clearance_m": MINIMUM_ROUTE_CLEARANCE_M,
        "navigation_envelope_radius_m": envelope_radius_m,
        "semantic_station_bridge_allowed": False,
        "entry": {
            "xy_m": [round(value, 6) for value in entry],
            "exact_entry_occupiable": entry_ok,
            "blocking_geometry": entry_blocker,
            "entry_clearance_m": round(entry_clearance, 6),
            "subgrid_connection_distance_m": (
                round(entry_connection, 6) if entry_connection is not None else None
            ),
        },
        "free_cell_count": len(free_cells),
        "reachable_free_cell_count": len(parents),
        "failed_station_ids": failed,
        "stations": dict(sorted(station_rows.items())),
    }

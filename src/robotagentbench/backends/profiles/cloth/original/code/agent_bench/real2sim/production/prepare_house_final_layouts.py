#!/usr/bin/env python3
"""Build the reviewed 5x10 compact House layout candidate set.

The file produced here is a layout-delivery manifest, not a task registry and
not a replacement for ``house_layouts.json``.  Five compact open-plan
living/dining shells each receive ten explicit arrangements whose exact work
stations pass the strict mobile-base navigation gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import house_layout, house_layout_navigation, house_topology
    from . import house_topology_modules
except ImportError:  # Direct script execution.
    import house_layout
    import house_layout_navigation
    import house_topology
    import house_topology_modules


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "house_final_layouts.json"
SCHEMA = "real2sim.house_final_layout_candidates.v3"
STATUS = "canary_candidate"
SCHEDULE_SCHEMA = "real2sim.house_final_layout_schedule.v1"
LAYOUT_COUNT = 50
GROUP_COUNT = 5
PALETTES = (
    "warm_oak_linen", "cool_gray_black", "light_scandi", "dark_walnut",
)

# These shells deliberately stay close to the canonical 40.56 m2 House.
# Their union areas are roughly 53.8--55.0 m2.  The shallow dining extension
# along the south edge preserves a generous circulation and robot lane around
# the dining/packing zone; the rooms remain only about 33--36% larger than the
# original.  Kitchen/storage/entry are semantic zones inside one
# continuous living+dining room; optional bays/nooks still change the visible
# shell.
COMPACT_SHELLS: tuple[dict[str, Any], ...] = (
    {"id": "compact_open_rectangle", "label": "Compact Open Plan"},
    {
        "id": "compact_dining_bay", "label": "Dining Bay",
        "extension": {
            "id": "dining_bay", "kind": "dining",
            # Keep the north kitchen wall continuous behind all three
            # appliances.  The bay extends from the otherwise unused south
            # dining edge instead.
            "rect_xywh_m": [2.9, -5.0, 1.5, 1.4],
        },
    },
    {
        "id": "compact_living_bay", "label": "Living Bay",
        "extension": {
            "id": "living_bay", "kind": "living",
            # A pronounced west-side bay changes the living-room silhouette
            # without removing the north/south media-wall alternatives.
            "rect_xywh_m": [-5.7, 1.4, 1.4, 1.4],
        },
    },
    {
        "id": "compact_entry_alcove", "label": "Entry Alcove",
        "extension": {
            "id": "entry_alcove", "kind": "entry",
            # Preserve the south-wall segment supporting the coat rack.
            "rect_xywh_m": [-2.0, -4.48, 1.0, 1.68],
        },
    },
    {
        "id": "compact_corner_nook", "label": "Offset Dining Nook",
        "extension": {
            "id": "corner_nook", "kind": "dining",
            # A south-west dining projection complements the south-east bay
            # above while preserving every cardinal media-wall option.
            "rect_xywh_m": [0.40, -5.0, 1.4, 1.4],
        },
    },
)

COMPACT_LAYOUT_PROFILES: tuple[dict[str, Any], ...] = (
    # Two meaningful arrangements for each cardinal living-room direction.
    # The previous revision kept the television on the west wall in all eight
    # profiles and only perturbed appliances or a table by a few centimetres.
    {"living_orientation": "tv_west", "living_variant": 0},
    {
        "living_orientation": "tv_west", "living_variant": 1,
        "appliance_rotation": 1,
    },
    {"living_orientation": "tv_south", "living_variant": 0},
    {
        "living_orientation": "tv_south", "living_variant": 1,
        "appliance_rotation": 2, "dining_center_x_m": 1.30,
        "dining_center_y_m": -0.80,
    },
    {"living_orientation": "tv_north", "living_variant": 0},
    {
        "living_orientation": "tv_north", "living_variant": 1,
        "appliance_rotation": 1,
    },
    {"living_orientation": "tv_east", "living_variant": 0},
    {
        "living_orientation": "tv_east", "living_variant": 1,
        "appliance_rotation": 2,
    },
    {
        "living_orientation": "tv_west", "living_variant": 2,
        "appliance_rotation": 2,
        "sofa_center_x_m": 0.20, "coffee_center_x_m": -1.70,
        "side_table_center_x_m": -1.035,
        "dining_center_x_m": 1.30,
    },
    {
        "living_orientation": "tv_east", "living_variant": 2,
        "appliance_rotation": 1,
        "sofa_center_x_m": -1.50, "coffee_center_x_m": 1.00,
        "side_table_center_x_m": -2.655,
        "dining_center_x_m": 1.30,
    },
)

LIVING_ORIENTATION_LABELS = {
    "tv_west": "TV west / sofa east",
    "tv_south": "TV south / sofa north",
    "tv_north": "TV north / sofa south",
    "tv_east": "TV east / sofa west",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _floor_geometry(candidate: Mapping[str, Any]) -> dict[str, float]:
    slabs = house_topology._floor_rects(candidate)
    xs = sorted({value for x, _y, width, _height in slabs for value in (x, x + width)})
    ys = sorted({value for _x, y, _width, height in slabs for value in (y, y + height)})
    occupied: set[tuple[int, int]] = set()
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            center = ((xs[xi] + xs[xi + 1]) / 2.0, (ys[yi] + ys[yi + 1]) / 2.0)
            if any(house_topology._inside_rect(center, rect) for rect in slabs):
                occupied.add((xi, yi))
    area = sum(
        (xs[xi + 1] - xs[xi]) * (ys[yi + 1] - ys[yi])
        for xi, yi in occupied
    )
    perimeter = 0.0
    for xi, yi in occupied:
        for neighbor, length in (
            ((xi - 1, yi), ys[yi + 1] - ys[yi]),
            ((xi + 1, yi), ys[yi + 1] - ys[yi]),
            ((xi, yi - 1), xs[xi + 1] - xs[xi]),
            ((xi, yi + 1), xs[xi + 1] - xs[xi]),
        ):
            if neighbor not in occupied:
                perimeter += length
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    compactness = area / bbox_area
    wall_count = len(house_topology_visual_walls(candidate))
    score = compactness * 100.0 - perimeter / area * 10.0 - wall_count * 0.05
    return {
        "floor_area_m2": round(area, 4),
        "bbox_compactness": round(compactness, 6),
        "exterior_perimeter_m": round(perimeter, 4),
        "exterior_wall_segment_count": wall_count,
        "selection_score": round(score, 6),
    }


def house_topology_visual_walls(candidate: Mapping[str, Any]) -> list[Any]:
    # Local import avoids making topology visual/pxr a generator dependency.
    try:
        from . import house_topology_visual
    except ImportError:
        import house_topology_visual
    return house_topology_visual.exterior_wall_segments(candidate)


def _canary_plan(ordinal: int, index_in_group: int) -> dict[str, Any]:
    return {
        "task_key": "benchmark_7_2",
        "accepted_core_ordinal": 0,
        "palette": PALETTES[index_in_group % len(PALETTES)],
        "visual_seed": 50_800_000 + ordinal,
        "state_replay_required": True,
        "physics_rollout_required": False,
    }


def _compact_spaces(shell: Mapping[str, Any]) -> list[dict[str, Any]]:
    spaces = [
        {"id": "living", "kind": "living", "rect_xywh_m": [-4.3, -2.8, 5.9, 5.6]},
        # Overlapping semantic media zone for east-wall living arrangements.
        # It adds no floor geometry; it lets the TV remain a living-room
        # fixture while using the real exterior wall on the dining side of
        # the continuous open plan.
        {
            "id": "living_east_media", "kind": "living",
            "rect_xywh_m": [0.4, -2.8, 4.3, 5.6],
        },
        # Extend the dining side 0.8 m south to preserve a robot-width
        # circulation lane past the dining and freestanding packing zones.
        {"id": "dining", "kind": "dining", "rect_xywh_m": [0.4, -3.6, 4.3, 6.4]},
        # These overlapping semantic zones do not create separate rooms.  They
        # bind task modules to the kitchen wall, storage niche and entry strip
        # inside the continuous living+dining shell.
        {"id": "kitchen", "kind": "kitchen", "rect_xywh_m": [0.4, -2.8, 4.3, 5.6]},
        {"id": "storage", "kind": "storage", "rect_xywh_m": [-4.3, 0.7, 2.2, 2.1]},
        {"id": "entry", "kind": "entry", "rect_xywh_m": [-2.1, -2.8, 2.1, 2.1]},
    ]
    extension = shell.get("extension")
    if extension is not None:
        spaces.append(copy.deepcopy(extension))
    return spaces


def _compact_corridors(shell: Mapping[str, Any]) -> list[dict[str, Any]]:
    corridors = [
        {
            "id": "open_living_media", "connects": [
                "living", "living_east_media",
            ],
            "clear_width_m": 1.2, "rects_xywh_m": [[0.4, 0.8, 1.2, 1.2]],
        },
        {
            "id": "open_living_dining", "connects": ["living", "dining"],
            "clear_width_m": 1.2, "rects_xywh_m": [[0.4, -0.6, 1.2, 1.2]],
        },
        {
            "id": "open_dining_kitchen", "connects": ["dining", "kitchen"],
            "clear_width_m": 1.2, "rects_xywh_m": [[1.0, -0.6, 1.2, 1.2]],
        },
        {
            "id": "storage_niche", "connects": ["living", "storage"],
            "clear_width_m": 1.2, "rects_xywh_m": [[-3.8, 1.0, 1.2, 1.2]],
        },
        {
            "id": "entry_niche", "connects": ["living", "entry"],
            "clear_width_m": 1.2, "rects_xywh_m": [[-1.7, -2.0, 1.2, 1.2]],
        },
    ]
    extension = shell.get("extension")
    if extension is not None:
        link_rects = {
            "dining_bay": [3.05, -4.2, 1.2, 1.2],
            "living_bay": [-4.7, 1.5, 1.2, 1.2],
            "entry_alcove": [-2.0, -3.4, 1.0, 1.0],
            "corner_nook": [0.45, -4.2, 1.2, 1.2],
        }
        corridors.append({
            "id": f"open_{extension['id']}",
            "connects": [
                "living" if extension["kind"] in {"living", "entry"}
                else "dining",
                extension["id"],
            ],
            "clear_width_m": (
                1.0 if extension["id"] == "entry_alcove" else 1.2
            ),
            "rects_xywh_m": [link_rects[str(extension["id"])]],
        })
    return corridors


def _centered_module_pose(
    module_id: str, space_id: str, fixture_center_xy_m: Sequence[float],
    yaw_deg: float, *, placement_role: str | None = None,
) -> dict[str, Any]:
    bounds = house_topology._rotated_footprint_bounds(
        house_topology_modules.MODULES[module_id], yaw_deg)
    footprint_center = (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )
    pose: dict[str, Any] = {
        "space_id": space_id,
        "center_xy_m": [
            round(float(fixture_center_xy_m[0]) - footprint_center[0], 6),
            round(float(fixture_center_xy_m[1]) - footprint_center[1], 6),
        ],
        "yaw_deg": float(yaw_deg),
    }
    if placement_role is not None:
        pose["placement_role"] = placement_role
    return pose


def _wall_module_pose_at(
    rect: Sequence[float], side: str, tangential_center_m: float,
    module_id: str, space_id: str, yaw_deg: float,
    *, placement_role: str | None = None,
) -> dict[str, Any]:
    x, y, width, height = map(float, rect)
    min_x, min_y, max_x, max_y = house_topology._rotated_footprint_bounds(
        house_topology_modules.MODULES[module_id], yaw_deg)
    inset = house_topology.ROOM_INTERIOR_INSET_M
    footprint_center_x = (min_x + max_x) / 2.0
    footprint_center_y = (min_y + max_y) / 2.0
    if side == "west":
        center = (x + inset - min_x, tangential_center_m - footprint_center_y)
    elif side == "east":
        center = (
            x + width - inset - max_x,
            tangential_center_m - footprint_center_y,
        )
    elif side == "south":
        center = (tangential_center_m - footprint_center_x, y + inset - min_y)
    elif side == "north":
        center = (
            tangential_center_m - footprint_center_x,
            y + height - inset - max_y,
        )
    else:
        raise ValueError(f"unknown wall side {side}")
    pose: dict[str, Any] = {
        "space_id": space_id,
        "center_xy_m": [round(center[0], 6), round(center[1], 6)],
        "yaw_deg": float(yaw_deg),
        "wall_attachment": {
            "side": side,
            "furniture_to_wall_gap_m": house_topology.FURNITURE_TO_WALL_GAP_M,
        },
    }
    if placement_role is not None:
        pose["placement_role"] = placement_role
    return pose


def _wall_module_pose(
    rect: Sequence[float], side: str, fraction: float,
    module_id: str, space_id: str,
) -> dict[str, Any]:
    """Place one fixture by fractional position along a real exterior wall."""

    x, y, width, height = map(float, rect)
    tangential_center = (
        x + width * float(fraction)
        if side in {"north", "south"}
        else y + height * float(fraction)
    )
    yaw = {
        "north": 0.0,
        "south": 180.0,
        "east": 270.0,
        "west": 90.0,
    }[side]
    return _wall_module_pose_at(
        rect, side, tangential_center, module_id, space_id, yaw)


def _center_module_pose_in_rect(
    rect: Sequence[float], module_id: str, space_id: str, yaw_deg: float,
    offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    x, y, width, height = map(float, rect)
    return _centered_module_pose(
        module_id,
        space_id,
        [
            x + width / 2.0 + float(offset_xy_m[0]),
            y + height / 2.0 + float(offset_xy_m[1]),
        ],
        yaw_deg,
    )


def _interaction_station_on_side(
    pose: dict[str, Any], module_id: str, side: str, *,
    tangent_fraction: float = 0.5, fixture_gap_m: float = 0.50,
) -> None:
    """Author an exact, world-space base station beside a fixture.

    Fixture yaw describes the visible asset orientation and cannot also be
    overloaded to encode which side a mobile base should approach.  Keeping
    the station explicit prevents a rotated TV or sofa from silently moving
    its work position behind the fixture.
    """

    x, y, width, height = house_topology._fixture_rect(
        house_topology_modules.MODULES[module_id], pose)
    fraction = float(tangent_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("station tangent fraction must be within 0..1")
    if side == "west":
        station = (x - fixture_gap_m, y + height * fraction)
    elif side == "east":
        station = (x + width + fixture_gap_m, y + height * fraction)
    elif side == "south":
        station = (x + width * fraction, y - fixture_gap_m)
    elif side == "north":
        station = (x + width * fraction, y + height + fixture_gap_m)
    else:
        raise ValueError(f"unknown interaction station side {side}")
    pose["interaction_station_xy_m"] = [
        round(station[0], 6), round(station[1], 6),
    ]
    pose["interaction_station_side"] = side
    pose["interaction_station_fixture_gap_m"] = fixture_gap_m


def _author_compact_interaction_stations(
    poses: dict[str, Any], orientation: str, variant: int,
) -> None:
    """Keep all work positions on deliberately open fixture faces."""

    style = variant % 2

    for module_id, pose in poses.items():
        attachment = pose.get("wall_attachment")
        if attachment is None:
            continue
        approach_side = {
            "west": "east", "east": "west",
            "south": "north", "north": "south",
        }[str(attachment["side"])]
        _interaction_station_on_side(pose, module_id, approach_side)

    # The east counter and the rightmost north-wall appliance meet at a
    # corner.  Their usable faces are long enough to keep both work positions
    # away from that corner; author that choice instead of accepting a point
    # beside the neighboring fixture.  The counter's upper section also stays
    # clear of the freestanding tote.
    _interaction_station_on_side(
        poses["kitchen_counter_cluster"], "kitchen_counter_cluster", "west",
        tangent_fraction=0.85,
    )
    north_wall_modules = [
        module_id for module_id, pose in poses.items()
        if pose.get("wall_attachment", {}).get("side") == "north"
        and module_id != "snack_shelf_cluster"
    ]
    if north_wall_modules:
        eastmost = max(
            north_wall_modules,
            key=lambda module_id: house_topology._fixture_rect(
                house_topology_modules.MODULES[module_id], poses[module_id]
            )[0],
        )
        _interaction_station_on_side(
            poses[eastmost], eastmost, "south", tangent_fraction=0.20)
    _interaction_station_on_side(
        poses["coat_rack_cluster"], "coat_rack_cluster", "north",
        tangent_fraction=0.15,
    )

    living = {
        "tv_west": {
            "sofa": ("west", (0.18, 0.25)[style]),
            "coffee": ("south", 0.50),
            "side": ("west", 0.50),
        },
        "tv_south": {
            "sofa": ("south", (0.18, 0.82)[style]),
            "coffee": ("east", 0.50),
            "side": ("north", 0.50),
        },
        "tv_north": {
            "sofa": ("north", (0.83, 0.05)[style]),
            "coffee": (("west", "east")[style], 0.50),
            "side": ("north", 0.50),
        },
        "tv_east": {
            "sofa": ("east", (0.18, 0.82)[style]),
            "coffee": ("south", (0.35, 0.65)[style]),
            "side": ("south", 0.50),
        },
    }[orientation]
    for key, module_id in (
        ("sofa", "sofa_cluster"),
        ("coffee", "coffee_table_cluster"),
        ("side", "side_table_cluster"),
    ):
        side, fraction = living[key]
        _interaction_station_on_side(
            poses[module_id], module_id, side, tangent_fraction=fraction)

    if orientation == "tv_south":
        _interaction_station_on_side(
            poses["tv_cluster"], "tv_cluster", "north",
            tangent_fraction=(0.0, 1.0)[style],
        )
    elif orientation == "tv_north":
        _interaction_station_on_side(
            poses["tv_cluster"], "tv_cluster", "south",
            tangent_fraction=(1.0, 0.0)[style],
        )

    # South-side access keeps the base out of the kitchen wall and leaves the
    # open dining aisle available for both freestanding operation zones.
    _interaction_station_on_side(
        poses["dining_table_cluster"], "dining_table_cluster",
        "north" if orientation == "tv_east" else "south",
    )
    _interaction_station_on_side(
        poses["tote_cluster"], "tote_cluster", "south")


def _compact_module_poses(
    spaces: Mapping[str, Sequence[float]], arrangement: int,
) -> dict[str, Any]:
    profile = COMPACT_LAYOUT_PROFILES[arrangement]
    orientation = str(profile["living_orientation"])
    variant = int(profile["living_variant"])
    style = variant % 2
    appliance_rotation = int(profile.get("appliance_rotation", 0))
    dining_center_x = float(profile.get("dining_center_x_m", 1.6))
    dining_center_y = float(profile.get("dining_center_y_m", -0.6))
    snack_fraction = float(profile.get("snack_shelf_fraction", 0.35))

    if orientation == "tv_west":
        axis_y = (0.35, 0.48)[style]
        sofa_center_x = float(profile.get("sofa_center_x_m", 0.0))
        coffee_center_x = float(profile.get(
            "coffee_center_x_m", (-2.0, -2.25)[style]))
        side_table_center_x = float(profile.get(
            "side_table_center_x_m", -1.235))
        result = {
            "sofa_cluster": _centered_module_pose(
                "sofa_cluster", "living", [sofa_center_x, axis_y], 0.0,
                placement_role="sofa_centerline_facing_media_wall",
            ),
            "tv_cluster": _wall_module_pose_at(
                spaces["living"], "west", axis_y, "tv_cluster", "living", 0.0,
                placement_role="media_wall_centerline_facing_sofa",
            ),
            "coffee_table_cluster": _centered_module_pose(
                "coffee_table_cluster", "living",
                [coffee_center_x, axis_y], 0.0,
                placement_role="coffee_table_between_sofa_and_tv",
            ),
            "side_table_cluster": _centered_module_pose(
                "side_table_cluster", "living",
                [side_table_center_x, axis_y + 1.495], 180.0,
                placement_role="end_table_beside_sofa",
            ),
        }
    elif orientation == "tv_south":
        axis_x = -2.87
        sofa_y = (0.40, 0.55)[style]
        coffee_y = (-1.35, -1.20)[style]
        coffee_offset_x = (0.27, -0.27)[style]
        result = {
            "sofa_cluster": _centered_module_pose(
                "sofa_cluster", "living", [axis_x, sofa_y], 90.0,
                placement_role="sofa_centerline_facing_media_wall",
            ),
            "tv_cluster": _wall_module_pose_at(
                spaces["living"], "south", axis_x, "tv_cluster", "living", 90.0,
                placement_role="media_wall_centerline_facing_sofa",
            ),
            "coffee_table_cluster": _centered_module_pose(
                "coffee_table_cluster", "living",
                [axis_x + coffee_offset_x, coffee_y], 90.0,
                placement_role="coffee_table_between_sofa_and_tv",
            ),
            "side_table_cluster": _centered_module_pose(
                "side_table_cluster", "living",
                [-0.685, sofa_y], 180.0,
                placement_role="end_table_beside_sofa",
            ),
        }
    elif orientation == "tv_north":
        axis_x = -1.90
        sofa_y = (-0.25, -0.35)[style]
        coffee_y = (1.45, 1.45)[style]
        coffee_offset_x = (-0.34, 0.34)[style]
        result = {
            "sofa_cluster": _centered_module_pose(
                "sofa_cluster", "living", [axis_x, sofa_y], 270.0,
                placement_role="sofa_centerline_facing_media_wall",
            ),
            "tv_cluster": _wall_module_pose_at(
                spaces["living"], "north", axis_x, "tv_cluster", "living", 270.0,
                placement_role="media_wall_centerline_facing_sofa",
            ),
            "coffee_table_cluster": _centered_module_pose(
                "coffee_table_cluster", "living",
                [axis_x + coffee_offset_x, coffee_y], 270.0,
                placement_role="coffee_table_between_sofa_and_tv",
            ),
            "side_table_cluster": _centered_module_pose(
                "side_table_cluster", "living",
                [0.265, sofa_y], 180.0,
                placement_role="end_table_beside_sofa",
            ),
        }
    elif orientation == "tv_east":
        axis_y = (0.63, 0.67)[style]
        sofa_x = float(profile.get(
            "sofa_center_x_m", (-1.0, -0.8)[style]))
        coffee_x = float(profile.get(
            "coffee_center_x_m", (1.45, 1.65)[style]))
        side_table_x = float(profile.get(
            "side_table_center_x_m", (-2.155, -1.955)[style]))
        coffee_offset_y = -0.35
        result = {
            "sofa_cluster": _centered_module_pose(
                "sofa_cluster", "living", [sofa_x, axis_y], 180.0,
                placement_role="sofa_centerline_facing_media_wall",
            ),
            "tv_cluster": _wall_module_pose_at(
                spaces["living_east_media"], "east", axis_y,
                "tv_cluster", "living_east_media", 180.0,
                placement_role="media_wall_centerline_facing_sofa",
            ),
            "coffee_table_cluster": _centered_module_pose(
                "coffee_table_cluster", "living_east_media",
                [coffee_x, axis_y + coffee_offset_y], 180.0,
                placement_role="coffee_table_between_sofa_and_tv",
            ),
            "side_table_cluster": _centered_module_pose(
                "side_table_cluster", "living",
                [side_table_x, axis_y], 270.0,
                placement_role="end_table_beside_sofa",
            ),
        }
    else:
        raise ValueError(f"unknown living orientation {orientation}")
    appliances = [
        "coffee_machine_cluster", "air_fryer_cluster", "microwave_cluster",
    ]
    appliances = (
        appliances[appliance_rotation:] + appliances[:appliance_rotation]
    )
    if orientation == "tv_east":
        kitchen_x, _kitchen_y, _kitchen_width, _kitchen_height = map(
            float, spaces["kitchen"])
        cursor = kitchen_x + house_topology.ROOM_INTERIOR_INSET_M
        for module_id in appliances + ["refrigerator_cluster"]:
            bounds = house_topology._rotated_footprint_bounds(
                house_topology_modules.MODULES[module_id], 0.0)
            width = bounds[2] - bounds[0]
            fixture_center_x = cursor + width / 2.0
            result[module_id] = _wall_module_pose_at(
                spaces["kitchen"], "north", fixture_center_x,
                module_id, "kitchen", 0.0)
            cursor += width + 0.12
    else:
        for module_id, fraction in zip(
            appliances, (0.14, 0.45, 0.76), strict=True
        ):
            result[module_id] = _wall_module_pose(
                spaces["kitchen"], "north", fraction, module_id, "kitchen")
    if orientation == "tv_east":
        refrigerator_pose = result["refrigerator_cluster"]
        counter_pose = _wall_module_pose(
            spaces["kitchen"], "east", 0.226786,
            "kitchen_counter_cluster", "kitchen")
    else:
        refrigerator_pose = _wall_module_pose(
            spaces["kitchen"], "east", 0.25,
            "refrigerator_cluster", "kitchen")
        counter_pose = _wall_module_pose(
            spaces["kitchen"], "east", 0.60,
            "kitchen_counter_cluster", "kitchen")
    if orientation == "tv_east":
        dining_pose = _centered_module_pose(
            "dining_table_cluster", "dining",
            [(2.60, 2.80)[style], -3.0], 270.0)
    else:
        dining_pose = _center_module_pose_in_rect(
            spaces["dining"], "dining_table_cluster", "dining", 0.0,
            (dining_center_x - 2.55, dining_center_y + 0.40),
        )
    snack_pose = _wall_module_pose(
        spaces["storage"], "north", snack_fraction,
        "snack_shelf_cluster", "storage")
    result.update({
        "refrigerator_cluster": refrigerator_pose,
        "kitchen_counter_cluster": counter_pose,
        "dining_table_cluster": dining_pose,
        # Preserve Task-73's canonical pose.  Rotating the soft tote changed
        # bottom contact by 12 mm, and translating it made the same delivered
        # action intermittently settle below the strict bottom gate.  It stays
        # a natural freestanding packing zone; the dining table moves west to
        # provide clearance instead of moving this soft physics reference.
        "tote_cluster": {
            "space_id": "dining",
            "center_xy_m": [2.785, -0.135],
            "yaw_deg": 0.0,
            "placement_role": "freestanding_packing_tote_with_south_access",
        },
        "snack_shelf_cluster": snack_pose,
        "coat_rack_cluster": _wall_module_pose(
            spaces["entry"], "south", 0.75,
            "coat_rack_cluster", "entry"),
    })
    _author_compact_interaction_stations(result, orientation, variant)
    return result


def _living_room_functional_design(
    module_poses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the visible sofa/TV/coffee-table relationship is conventional."""

    fixtures = {
        module_id: house_topology._fixture_rect(
            house_topology_modules.MODULES[module_id], module_poses[module_id]
        )
        for module_id in (
            "sofa_cluster", "tv_cluster", "coffee_table_cluster",
            "side_table_cluster",
        )
    }
    centers = {
        module_id: (
            rect[0] + rect[2] / 2.0,
            rect[1] + rect[3] / 2.0,
        )
        for module_id, rect in fixtures.items()
    }
    sofa = fixtures["sofa_cluster"]
    television = fixtures["tv_cluster"]
    coffee = fixtures["coffee_table_cluster"]
    sofa_xy = centers["sofa_cluster"]
    tv_xy = centers["tv_cluster"]
    coffee_xy = centers["coffee_table_cluster"]
    tv_to_sofa = math.degrees(math.atan2(
        sofa_xy[1] - tv_xy[1], sofa_xy[0] - tv_xy[0]))
    sofa_to_tv = math.degrees(math.atan2(
        tv_xy[1] - sofa_xy[1], tv_xy[0] - sofa_xy[0]))
    sofa_facing_error = _angular_error_deg(
        sofa_to_tv, float(module_poses["sofa_cluster"]["yaw_deg"]) + 180.0)
    tv_facing_error = _angular_error_deg(
        tv_to_sofa, float(module_poses["tv_cluster"]["yaw_deg"]))
    delta_x = sofa_xy[0] - tv_xy[0]
    delta_y = sofa_xy[1] - tv_xy[1]
    viewing_distance = math.hypot(delta_x, delta_y)
    if abs(delta_x) >= abs(delta_y):
        axis = 0
        viewing_axis = "west_east"
    else:
        axis = 1
        viewing_axis = "south_north"
    perpendicular = 1 - axis
    centerline_error = abs(sofa_xy[perpendicular] - tv_xy[perpendicular])
    coffee_centerline_offset = abs(
        coffee_xy[perpendicular]
        - (sofa_xy[perpendicular] + tv_xy[perpendicular]) / 2.0
    )

    def interval(rect: Sequence[float]) -> tuple[float, float]:
        start = float(rect[axis])
        return start, start + float(rect[axis + 2])

    tv_interval = interval(television)
    coffee_interval = interval(coffee)
    sofa_interval = interval(sofa)
    if sofa_xy[axis] > tv_xy[axis]:
        coffee_between = bool(
            tv_interval[1] < coffee_interval[0]
            and coffee_interval[1] < sofa_interval[0]
        )
    else:
        coffee_between = bool(
            sofa_interval[1] < coffee_interval[0]
            and coffee_interval[1] < tv_interval[0]
        )
    parallel_long_axes = _angular_error_deg(
        float(module_poses["sofa_cluster"]["yaw_deg"]),
        float(module_poses["tv_cluster"]["yaw_deg"]),
    ) <= 1.0e-6
    tv_attachment = module_poses["tv_cluster"].get("wall_attachment", {})
    tv_wall_side = str(tv_attachment.get("side", ""))
    expected_tv_yaw = {
        "west": 0.0, "south": 90.0, "east": 180.0, "north": 270.0,
    }.get(tv_wall_side)
    tv_faces_inward_from_wall = bool(
        expected_tv_yaw is not None
        and _angular_error_deg(
            float(module_poses["tv_cluster"]["yaw_deg"]), expected_tv_yaw
        ) <= 1.0e-6
    )
    sofa_side_gap = _fixture_gap_m(
        fixtures["sofa_cluster"], fixtures["side_table_cluster"])
    side_table = module_poses["side_table_cluster"]
    side_table_beside_sofa = bool(
        0.05 <= sofa_side_gap <= 0.50
        and side_table.get("placement_role")
        == "end_table_beside_sofa"
        and "wall_attachment" not in side_table
    )
    passed = bool(
        centerline_error <= 1.0e-6
        and coffee_centerline_offset <= 0.35
        and viewing_distance >= 2.8
        and sofa_facing_error <= 10.0
        and tv_facing_error <= 10.0
        and coffee_between
        and parallel_long_axes
        and tv_faces_inward_from_wall
        and side_table_beside_sofa
    )
    return {
        "passed": passed,
        "sofa_faces_tv": sofa_facing_error <= 10.0,
        "sofa_facing_error_deg": round(sofa_facing_error, 6),
        "tv_faces_sofa": tv_facing_error <= 10.0,
        "tv_facing_error_deg": round(tv_facing_error, 6),
        "viewing_axis": viewing_axis,
        "sofa_tv_centerline_error_m": round(centerline_error, 6),
        "sofa_tv_viewing_distance_m": round(viewing_distance, 6),
        "coffee_table_centerline_offset_m": round(
            coffee_centerline_offset, 6),
        "sofa_and_tv_long_axes_parallel": parallel_long_axes,
        "coffee_table_strictly_between_sofa_and_tv": coffee_between,
        "tv_media_wall_side": tv_wall_side,
        "tv_flush_to_declared_media_wall": tv_wall_side in {
            "west", "south", "east", "north",
        },
        "tv_faces_inward_from_media_wall": tv_faces_inward_from_wall,
        "side_table_beside_sofa": side_table_beside_sofa,
        "sofa_side_table_fixture_gap_m": round(sofa_side_gap, 6),
        "side_table_is_not_misclassified_as_wall_storage": (
            "wall_attachment" not in side_table
        ),
    }


def _fixture_gap_m(
    left: Sequence[float], right: Sequence[float],
) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    dx = max(rx - (lx + lw), lx - (rx + rw), 0.0)
    dy = max(ry - (ly + lh), ly - (ry + rh), 0.0)
    return math.hypot(dx, dy)


def _exterior_wall_support(
    candidate: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check that every declared wall fixture has a real exterior wall.

    A semantic room rectangle is not sufficient evidence: an overlapping bay
    can remove that portion of the union boundary and leave a nominally
    wall-attached object floating in an opening.  Require one authored exterior
    wall segment to cover the complete fixture projection.
    """

    spaces = {
        str(row["id"]): tuple(map(float, row["rect_xywh_m"]))
        for row in candidate["spaces"]
    }
    walls = house_topology_visual_walls(candidate)
    rows: dict[str, bool] = {}
    tolerance = 1.0e-6
    for module_id, pose in candidate["module_poses"].items():
        attachment = pose.get("wall_attachment")
        if attachment is None:
            continue
        x, y, width, height = house_topology._fixture_rect(
            modules[module_id], pose)
        room_x, room_y, room_width, room_height = spaces[
            str(pose["space_id"])
        ]
        side = str(attachment["side"])
        if side in {"north", "south"}:
            axis = "x"
            fixed = room_y + room_height if side == "north" else room_y
            interval = (x, x + width)
        else:
            axis = "y"
            fixed = room_x + room_width if side == "east" else room_x
            interval = (y, y + height)
        rows[str(module_id)] = any(
            wall_axis == axis
            and math.isclose(wall_fixed, fixed, abs_tol=tolerance)
            and wall_start <= interval[0] + tolerance
            and wall_end >= interval[1] - tolerance
            for wall_axis, wall_fixed, wall_start, wall_end in walls
        )
    return {
        "all_wall_attached_modules_have_real_exterior_wall_support": all(
            rows.values()),
        "exterior_wall_support_by_module": dict(sorted(rows.items())),
    }


def _angular_error_deg(actual: float, expected: float) -> float:
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def build_catalog() -> dict[str, Any]:
    interaction_modules = house_topology_modules.MODULES
    layouts: list[dict[str, Any]] = []
    topology_candidates: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for group_index, shell in enumerate(COMPACT_SHELLS, start=1):
        shell_id = str(shell["id"])
        group_id = f"topology_{group_index}_{shell_id}"
        group_layout_ids = []
        space_rows = _compact_spaces(shell)
        spaces = {row["id"]: row["rect_xywh_m"] for row in space_rows}
        for arrangement in range(len(COMPACT_LAYOUT_PROFILES)):
            ordinal = len(layouts) + 1
            final_id = f"final_house_layout_{ordinal:02d}"
            profile = COMPACT_LAYOUT_PROFILES[arrangement]
            living_orientation = str(profile["living_orientation"])
            living_variant = int(profile["living_variant"])
            candidate = {
                "ordinal": ordinal,
                "id": final_id,
                "name": (
                    f"{shell['label']} / "
                    f"{LIVING_ORIENTATION_LABELS[living_orientation]} / "
                    f"relationship {living_variant + 1}"
                ),
                "archetype": shell_id,
                "variant": (
                    f"{living_orientation}_relationship_{living_variant + 1}"
                ),
                "living_orientation": living_orientation,
                "living_relationship_variant": living_variant + 1,
                "shell_strategy": (
                    "compact_open_plan_living_dining_generated_in_session_layer"
                ),
                "spaces": copy.deepcopy(space_rows),
                "corridors": _compact_corridors(shell),
                # The previous point overlapped the coat-rack footprint.  This
                # is the exact base-center entry, not a portal that may snap.
                "entry_xy_m": [0.80, -2.80],
                "module_poses": _compact_module_poses(spaces, arrangement),
                "open_plan_design": {
                    "primary_rooms": ["living", "dining"],
                    "kitchen_storage_entry_are_embedded_zones": True,
                    "living_room_cardinal_orientation": living_orientation,
                    "coffee_table_may_be_laterally_offset": True,
                    "target_area_m2": [53.0, 56.0],
                    "canonical_house_area_m2": 40.56019936,
                },
            }
            candidate["living_room_functional_design"] = (
                _living_room_functional_design(candidate["module_poses"])
            )
            tote_is_freestanding = bool(
                candidate["module_poses"]["tote_cluster"].get(
                    "placement_role")
                == "freestanding_packing_tote_with_south_access"
                and "wall_attachment" not in candidate[
                    "module_poses"]["tote_cluster"]
            )
            wall_support = _exterior_wall_support(
                candidate, interaction_modules)
            candidate["household_rationality"] = {
                **candidate["living_room_functional_design"],
                "scope": "generated_whole_house",
                "tote_is_freestanding_packing_zone": tote_is_freestanding,
                "wall_attached_module_ids": sorted(
                    module_id
                    for module_id, pose in candidate["module_poses"].items()
                    if "wall_attachment" in pose
                ),
                **wall_support,
                "passed": bool(
                    candidate["living_room_functional_design"]["passed"]
                    and tote_is_freestanding
                    and wall_support[
                        "all_wall_attached_modules_have_real_exterior_wall_support"
                    ]
                ),
            }
            if not candidate["household_rationality"]["passed"]:
                raise RuntimeError(
                    f"final candidate {final_id} fails the generated whole-house "
                    "rationality gate"
                )
            audit = house_topology.audit_candidate(
                candidate, interaction_modules)
            non_navigation_errors = [
                error for error in audit["errors"]
                if error != "conservative grid navigation did not reach every module station"
            ]
            strict_navigation = house_layout_navigation.audit_candidate_navigation(
                candidate, interaction_modules)
            if non_navigation_errors or not strict_navigation["passed"]:
                raise RuntimeError(
                    f"final candidate {final_id} failed: "
                    f"geometry={non_navigation_errors}, "
                    f"strict_navigation={strict_navigation['failed_station_ids']}"
                )
            navigation = audit["grid_navigation"]
            candidate["strict_base_navigation"] = strict_navigation
            candidate["preflight"] = {
                "planar_geometry": "passed",
                "space_graph": "passed",
                "legacy_snap_grid_navigation": (
                    "passed" if navigation["passed"] else "failed_superseded"
                ),
                "exact_station_continuous_base_sweep": "passed",
                "usd_shell_authoring": "pending_final_canary",
                "live_physx_navigation": "not_run",
                "physics_replay": "not_run",
            }
            topology_candidates.append(candidate)
            group_layout_ids.append(final_id)
            layouts.append({
                "ordinal": ordinal,
                "id": final_id,
                "group_id": group_id,
                "index_in_group": arrangement + 1,
                "shell_kind": "generated_full_house_topology",
                "compact_shell_template_id": shell_id,
                "archetype": shell_id,
                "arrangement_index": arrangement,
                "living_orientation": living_orientation,
                "living_relationship_variant": living_variant + 1,
                "static_audit": {
                    "passed": True,
                    "fixture_collision_count": audit["fixture_collision_count"],
                    "free_cell_count": strict_navigation["free_cell_count"],
                    "reachable_free_cell_count": strict_navigation[
                        "reachable_free_cell_count"],
                    "reachable_fraction": round(
                        strict_navigation["reachable_free_cell_count"]
                        / strict_navigation["free_cell_count"], 6,
                    ),
                    "semantic_station_bridge_allowed": False,
                    "maximum_subgrid_connection_distance_m": max(
                        row["subgrid_connection_distance_m"]
                        for row in strict_navigation["stations"].values()
                    ),
                },
                "household_rationality": copy.deepcopy(
                    candidate["household_rationality"]),
                "canary_plan": _canary_plan(ordinal, arrangement),
            })
        groups.append({
            "ordinal": group_index,
            "id": group_id,
            "label": shell["label"],
            "archetype": shell_id,
            "compact_shell_template_id": shell_id,
            "compact_shell_geometry": _floor_geometry(
                topology_candidates[-1]),
            "layout_count": len(COMPACT_LAYOUT_PROFILES),
            "layout_ids": group_layout_ids,
            "selection_rule": (
                "compact 53-56 m2 open living/dining shell; ten arrangements "
                "cover four cardinal TV/sofa orientations and pass exact "
                "station continuous base-envelope sweeps"
            ),
        })

    if len(layouts) != LAYOUT_COUNT or len(groups) != GROUP_COUNT:
        raise RuntimeError("final delivery must contain 50 layouts in five groups")
    if len(topology_candidates) != 50:
        raise RuntimeError("final delivery must contain exactly 50 generated-shell layouts")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "layout_count": LAYOUT_COUNT,
        "topology_group_count": GROUP_COUNT,
        "allocation": "5 generated shells x 10 strictly reachable layouts",
        "design_revision": (
            "compact_open_plan_living_dining_v12_all_50_exact_base_reachability"
        ),
        "generated_shell_area_target_m2": [53.0, 56.0],
        "canonical_house_area_m2": 40.56019936,
        "schedule_permutation_source": "house_layout.task_layout_permutation",
        "interaction_module_source": (
            "real2sim/production/house_topology_modules.py"
        ),
        "task_identity_source_unchanged": True,
        "production_layout_catalog_replaced": False,
        "reasonableness_gate": {
            "required_layout_pass_count": 50,
            "sofa_facing_error_max_deg": 10.0,
            "sofa_tv_viewing_distance_min_m": 2.8,
            "coffee_table_between_sofa_and_tv": True,
            "coffee_table_centerline_offset_max_m": 0.35,
            "generated_living_orientations": sorted(
                LIVING_ORIENTATION_LABELS),
            "sofa_side_table_fixture_gap_m": [0.05, 0.50],
            "generated_wall_fixture_gap_m": (
                house_topology.FURNITURE_TO_WALL_GAP_M
            ),
            "generated_wall_attached_module_ids": sorted({
                "air_fryer_cluster", "coat_rack_cluster",
                "coffee_machine_cluster", "kitchen_counter_cluster",
                "microwave_cluster", "refrigerator_cluster",
                "snack_shelf_cluster", "tv_cluster",
            }),
            "generated_tote_is_freestanding_packing_zone": True,
            "generated_wall_fixtures_require_real_exterior_wall_support": True,
            "robot_base_radius_m": (
                house_layout_navigation.ROBOT_BASE_RADIUS_M),
            "carried_object_margin_m": (
                house_layout_navigation.CARRIED_OBJECT_MARGIN_M),
            "minimum_route_clearance_m": (
                house_layout_navigation.MINIMUM_ROUTE_CLEARANCE_M),
            "navigation_envelope_radius_m": (
                house_layout_navigation.NAVIGATION_ENVELOPE_RADIUS_M),
            "grid_resolution_m": house_layout_navigation.GRID_RESOLUTION_M,
            "exact_station_occupancy_required": True,
            "semantic_station_bridge_allowed": False,
            "continuous_swept_segment_collision_check_required": True,
        },
        "interaction_modules": interaction_modules,
        "groups": groups,
        "layouts": layouts,
        "candidates": topology_candidates,
        "canary_acceptance": {
            "required_layout_pass_count": 50,
            "required_transform_audit": True,
            "required_state_driven_camera_count": 3,
            "required_visual_qc": True,
            "required_generated_usd_author_save_reopen": True,
            "live_collision_and_physics_scope": (
                "planar continuous base-envelope sweep passed; live PhysX "
                "chassis rollout remains pending and is not implied"
            ),
            "previous_design_revision_evidence_reusable": False,
        },
    }


def validate(
    catalog: Mapping[str, Any], *, recompute_strict_navigation: bool = False,
) -> dict[str, Any]:
    errors = []
    layouts = catalog.get("layouts", [])
    groups = catalog.get("groups", [])
    if catalog.get("schema") != SCHEMA or catalog.get("status") != STATUS:
        errors.append("schema or status mismatch")
    if len(layouts) != LAYOUT_COUNT:
        errors.append("layout count is not 50")
    if len(groups) != GROUP_COUNT:
        errors.append("topology group count is not 5")
    if [row.get("ordinal") for row in layouts] != list(range(1, 51)):
        errors.append("layout ordinals are not 1..50")
    if len({row.get("id") for row in layouts}) != LAYOUT_COUNT:
        errors.append("layout IDs are not unique")
    if any(
        row.get("shell_kind") != "generated_full_house_topology"
        for row in layouts
    ):
        errors.append("every final layout must use a generated full-House shell")
    if len(catalog.get("candidates", [])) != LAYOUT_COUNT:
        errors.append("candidate count is not 50")
    if catalog.get("interaction_module_source") != (
        "real2sim/production/house_topology_modules.py"
    ):
        errors.append("interaction module source is not canonical")
    if catalog.get("schedule_permutation_source") != (
        "house_layout.task_layout_permutation"
    ):
        errors.append("layout schedule permutation source drifted")
    if [row.get("layout_count") for row in groups] != [10, 10, 10, 10, 10]:
        errors.append("group allocation is not 10+10+10+10+10")
    by_id = {row["id"]: row for row in layouts}
    rationality_passes = sum(
        row.get("household_rationality", {}).get("passed") is True
        for row in layouts
    )
    if rationality_passes != LAYOUT_COUNT:
        errors.append(
            f"household rationality passes are {rationality_passes}, not 50")
    for group in groups:
        if any(item not in by_id for item in group["layout_ids"]):
            errors.append(f"group {group['id']} references an unknown layout")
    modules = catalog.get("interaction_modules", {})
    generated_areas = []
    orientation_counts = {key: 0 for key in LIVING_ORIENTATION_LABELS}
    for candidate in catalog.get("candidates", []):
        orientation = str(candidate.get("living_orientation", ""))
        if orientation in orientation_counts:
            orientation_counts[orientation] += 1
        else:
            errors.append(
                f"{candidate.get('id')}: unknown living orientation {orientation}"
            )
        audit = house_topology.audit_candidate(candidate, modules)
        non_navigation_errors = [
            error for error in audit["errors"]
            if error != "conservative grid navigation did not reach every module station"
        ]
        if non_navigation_errors:
            errors.append(f"{candidate['id']}: {non_navigation_errors}")
        strict_navigation = candidate.get("strict_base_navigation", {})
        station_rows = strict_navigation.get("stations", {})
        if (
            strict_navigation.get("passed") is not True
            or strict_navigation.get("failed_station_ids") != []
            or strict_navigation.get("semantic_station_bridge_allowed") is not False
            or set(station_rows) != set(modules)
            or any(
                row.get("exact_station_occupiable") is not True
                or row.get("reachable") is not True
                for row in station_rows.values()
            )
        ):
            errors.append(
                f"{candidate['id']}: stored strict base navigation did not pass"
            )
        if recompute_strict_navigation:
            expected_navigation = (
                house_layout_navigation.audit_candidate_navigation(
                    candidate, modules)
            )
            if expected_navigation != strict_navigation:
                errors.append(
                    f"{candidate['id']}: strict base navigation audit drifted"
                )
        geometry = _floor_geometry(candidate)
        area = float(geometry["floor_area_m2"])
        generated_areas.append(area)
        if not 53.0 <= area <= 56.0:
            errors.append(
                f"{candidate['id']}: compact floor area {area} is outside "
                "53..56 m2")
        design = candidate.get("open_plan_design", {})
        if (
            design.get("primary_rooms") != ["living", "dining"]
            or design.get("kitchen_storage_entry_are_embedded_zones") is not True
            or design.get("living_room_cardinal_orientation") != orientation
            or design.get("coffee_table_may_be_laterally_offset") is not True
        ):
            errors.append(f"{candidate['id']}: open-plan design contract drifted")
        functional = _living_room_functional_design(candidate["module_poses"])
        if functional != candidate.get("living_room_functional_design"):
            errors.append(
                f"{candidate['id']}: sofa/TV functional design contract drifted"
            )
        tote_is_freestanding = bool(
            candidate["module_poses"]["tote_cluster"].get(
                "placement_role")
            == "freestanding_packing_tote_with_south_access"
            and "wall_attachment" not in candidate[
                "module_poses"]["tote_cluster"]
        )
        wall_support = _exterior_wall_support(candidate, modules)
        expected_rationality = {
            **functional,
            "scope": "generated_whole_house",
            "tote_is_freestanding_packing_zone": tote_is_freestanding,
            "wall_attached_module_ids": sorted(
                module_id
                for module_id, pose in candidate["module_poses"].items()
                if "wall_attachment" in pose
            ),
            **wall_support,
            "passed": bool(
                functional["passed"]
                and tote_is_freestanding
                and wall_support[
                    "all_wall_attached_modules_have_real_exterior_wall_support"
                ]
            ),
        }
        if expected_rationality != candidate.get("household_rationality"):
            errors.append(
                f"{candidate['id']}: household rationality contract drifted")
        required_wall_modules = set(catalog.get(
            "reasonableness_gate", {}
        ).get("generated_wall_attached_module_ids", ()))
        if set(expected_rationality["wall_attached_module_ids"]) != required_wall_modules:
            errors.append(
                f"{candidate['id']}: required wall-fixture attachments drifted")
        if expected_rationality[
            "tote_is_freestanding_packing_zone"] is not True:
            errors.append(
                f"{candidate['id']}: tote packing-zone placement drifted")
        if expected_rationality[
            "all_wall_attached_modules_have_real_exterior_wall_support"
        ] is not True:
            errors.append(
                f"{candidate['id']}: wall fixture lacks a real exterior wall")
        layout_row = by_id.get(candidate["id"], {})
        if layout_row.get("household_rationality") != expected_rationality:
            errors.append(
                f"{candidate['id']}: layout rationality projection drifted")
        if layout_row.get("living_orientation") != orientation:
            errors.append(
                f"{candidate['id']}: layout orientation projection drifted")
    if orientation_counts != {
        "tv_west": 15, "tv_south": 10, "tv_north": 10, "tv_east": 15,
    }:
        errors.append(
            f"generated living orientation counts drifted: {orientation_counts}"
        )
    return {
        "schema": "real2sim.house_final_layout_validation.v3",
        "passed": not errors,
        "layout_count": len(layouts),
        "topology_group_count": len(groups),
        "generated_shell_layout_count": len(catalog.get("candidates", [])),
        "strict_base_navigation_passed": sum(
            candidate.get("strict_base_navigation", {}).get("passed") is True
            for candidate in catalog.get("candidates", [])
        ),
        "household_rationality_passed": rationality_passes,
        "minimum_generated_floor_area_m2": (
            min(generated_areas) if generated_areas else None),
        "maximum_generated_floor_area_m2": (
            max(generated_areas) if generated_areas else None),
        "maximum_area_ratio_to_canonical": (
            round(max(generated_areas) / 40.56019936, 6)
            if generated_areas else None
        ),
        "errors": errors,
    }


def load_catalog(path: Path | str = DEFAULT_OUTPUT) -> dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    report = validate(catalog)
    if not report["passed"]:
        raise ValueError(f"invalid final House layout catalog: {report['errors']}")
    return catalog


def scheduled_layout_record(
    task_key: str,
    core_ordinal: int,
    catalog: Mapping[str, Any] | Path | str = DEFAULT_OUTPUT,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select every final layout exactly once across core ordinals 0..49.

    The final delivery reuses the established task-seeded permutation instead
    of creating a second hidden scheduler.  Only the catalog projected through
    that permutation changes when this opt-in delivery is selected.
    """

    if isinstance(core_ordinal, bool) or not isinstance(core_ordinal, int):
        raise TypeError("core ordinal must be an integer")
    if not 0 <= core_ordinal < LAYOUT_COUNT:
        raise ValueError("core ordinal must be in 0..49")
    if isinstance(catalog, (str, Path)):
        effective = json.loads(Path(catalog).read_text(encoding="utf-8"))
    else:
        # Mapping callers (notably the 50x4 worker) pass an already validated
        # catalog so the expensive 40-candidate grid audit is not repeated for
        # every core and every palette.
        effective = dict(catalog)
    if (
        effective.get("schema") != SCHEMA
        or len(effective.get("layouts", [])) != LAYOUT_COUNT
        or [row.get("ordinal") for row in effective.get("layouts", [])]
        != list(range(1, LAYOUT_COUNT + 1))
    ):
        raise ValueError("invalid final House layout schedule catalog")
    permutation = house_layout.task_layout_permutation(task_key, seed=seed)
    selected_ordinal = permutation[core_ordinal]
    record = dict(effective["layouts"][selected_ordinal - 1])
    if record.get("ordinal") != selected_ordinal:
        raise ValueError("final House layout catalog is not ordinal-addressable")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    catalog = build_catalog()
    report = validate(catalog)
    if not report["passed"]:
        raise RuntimeError(report["errors"])
    _atomic_json(args.output.resolve(), catalog)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

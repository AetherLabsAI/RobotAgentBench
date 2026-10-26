"""Resolve task-owned placement checks in a captured House layout frame."""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
from . import house_layout, house_topology_visual, prepare_house_final_layouts
from .score_task6_code_control import mode
from .benchmark_eval_registry import resolve


def score_kwargs(task_id, state_path, evidence):
    if evidence is None:
        return {}
    task_id = str(task_id)
    frame = resolve(task_id)["task"]["runtime"]["environment"]["layout_frame"]
    key = "benchmark_" + task_id.replace(".", "_")
    scene = evidence.get("scene", {})
    action = evidence.get("action", {})
    identity = evidence.get("layout_id")
    if (evidence.get("schema") != "real2sim.task6_physical_layout.v1"
            or evidence.get("task_key") != key
            or evidence.get("task_frame") != frame
            or action.get("task_frame") != frame
            or action.get("layout_id") != identity
            or scene.get("topology_id") != identity
            or scene.get("physics_overlay_mode") is not True
            or scene.get("usd_author_save_reopen") != "passed"):
        raise ValueError("Task6 physical layout/action/scene binding disagrees")
    catalog = prepare_house_final_layouts.load_catalog()
    candidate, _ = house_topology_visual.candidate_for_id(identity, catalog)
    recipe = house_topology_visual.topology_layout(candidate, catalog["interaction_modules"])
    transform = house_layout.transform_for_frame(recipe, frame)
    expected = dict(pivot_xy_m=list(transform.pivot_xy_m),
                    translate_xy_m=list(transform.translate_xy_m), yaw_deg=transform.yaw_deg)
    reported = action.get("transform", {})
    if set(reported) != set(expected) or any(
            np.asarray(reported[k]).shape != np.asarray(v).shape
            or not np.allclose(reported[k], v, atol=1e-9, rtol=0)
            for k, v in expected.items()):
        raise ValueError("Task6 action transform disagrees with current House recipe")
    with np.load(Path(evidence["source_action"]), allow_pickle=False) as z:
        original = dict(z)
    mapped = house_layout.transform_action_arrays(original, recipe, frame=frame)
    with np.load(Path(evidence["executed_action"]), allow_pickle=False) as z:
        executed = dict(z)
    if mapped.keys() != executed.keys() or any(
            not np.array_equal(mapped[k], executed[k]) for k in mapped):
        raise ValueError("Task6 executed action does not match its preserved source mapping")
    from ..teleop.physical_distractors import read_capture
    with np.load(state_path, allow_pickle=False) as state:
        capture = read_capture(state)
    static = capture.get("static_collision_transforms", {})
    if not any(p.startswith("/World/HouseTopologyCandidate/Shell/Floors/") for p in static):
        raise ValueError("Task6 capture lacks the generated collision shell")
    selected = mode(task_id)
    zone = np.asarray(selected.destination_zone)
    center = house_layout.transform_points(np.array([zone[0], zone[1], zone[4]]), transform)
    angle = np.deg2rad(transform.yaw_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array(list(itertools.product(*zip(
        selected.furniture_aabb_min_xyz_m, selected.furniture_aabb_max_xyz_m))))
    corners = house_layout.transform_points(corners, transform)
    return dict(expected_destination_zone=[*center[:2], *(np.abs(rotation) @ zone[2:4]), zone[4]],
                expected_furniture_aabb_min_xyz_m=corners.min(axis=0),
                expected_furniture_aabb_max_xyz_m=corners.max(axis=0))

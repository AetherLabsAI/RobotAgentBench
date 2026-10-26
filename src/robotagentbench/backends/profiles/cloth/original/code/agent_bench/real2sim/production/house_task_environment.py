#!/usr/bin/env python3
"""Resolve one compact House environment view for a benchmark task.

This is not another task registry.  The caller supplies the task definition
from the Sheet-backed catalog; this module only answers which canonical
interaction frame and named robot station that definition uses.  The named
stations continue to live in ``real2sim/teleop/robot_positions.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


try:
    from .house_paths import SOURCE_BASELINE, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_BASELINE, WORKING_SCENE

BASELINE = SOURCE_BASELINE
WORKING_SCENE_NAME = "scene_room.usda"


FRAME_MODULES = {
    "coffee_table": "coffee_table_cluster",
    "sofa": "sofa_cluster",
    "side_table": "side_table_cluster",
    "tv_cabinet": "tv_cluster",
    "snack_shelf": "snack_shelf_cluster",
    "tote": "tote_cluster",
    "coat_rack": "coat_rack_cluster",
}

# A task that moves an object between two interaction areas cannot be
# validated by replaying only its destination manipulation.  These routes are
# part of the task's resolved House environment, not a layout catalog or a
# second task registry.
MULTI_MODULE_ROUTES: dict[str, tuple[str, str]] = {
    **{
        f"3.{number}": ("snack_shelf_cluster", "coffee_table_cluster")
        for number in range(2, 31, 2)
        if number != 18
    },
    "5.2": ("coffee_table_cluster", "tote_cluster"),
    "5.4": ("kitchen_counter_cluster", "tote_cluster"),
    "5.6": ("sofa_cluster", "tote_cluster"),
    "5.10": ("coffee_table_cluster", "tote_cluster"),
    "5.12": ("sofa_cluster", "tote_cluster"),
    "6.2": ("sofa_cluster", "coffee_table_cluster"),
    "6.3": ("sofa_cluster", "side_table_cluster"),
    "6.5": ("sofa_cluster", "coffee_table_cluster"),
    "6.7": ("sofa_cluster", "coat_rack_cluster"),
    "6.9": ("sofa_cluster", "coffee_table_cluster"),
    "6.10": ("sofa_cluster", "tv_cluster"),
    "6.15": ("sofa_cluster", "coffee_table_cluster"),
    "6.16": ("sofa_cluster", "tote_cluster"),
    "9.2": ("refrigerator_cluster", "microwave_cluster"),
    "9.4": ("refrigerator_cluster", "microwave_cluster"),
    "10.2": ("microwave_cluster", "dining_table_cluster"),
    "10.3": ("microwave_cluster", "coffee_table_cluster"),
    "10.5": ("microwave_cluster", "dining_table_cluster"),
    "10.6": ("microwave_cluster", "coffee_table_cluster"),
    "10.8": ("microwave_cluster", "dining_table_cluster"),
    "10.9": ("microwave_cluster", "coffee_table_cluster"),
}


def _contains(text: str, *values: str) -> bool:
    return any(value in text for value in values)


def _frame_from_prim(path: str) -> str | None:
    for prefix, frame in (
        ("/World/Assets/TableclothBasket", "coffee_table"),
        ("/World/Assets/CoffeeTable", "coffee_table"),
        ("/World/Assets/TrashBin", "coffee_table"),
        ("/World/Assets/Sofa", "sofa"),
        ("/World/Assets/SmallCoffeeTable", "side_table"),
        ("/World/Assets/NightStand", "side_table"),
        ("/World/Assets/TvTable", "tv_cabinet"),
        ("/World/Assets/Tv", "tv_cabinet"),
        ("/World/Assets/ShelvingUnit", "snack_shelf"),
        ("/World/Assets/Tier", "snack_shelf"),
        ("/World/Task73/Tote", "tote"),
        ("/World/Background/Shell/CounterTop", "kitchen"),
    ):
        if path.startswith(prefix):
            return frame
    return None


def _bound_frame(task_id: str) -> str | None:
    """Use a delivered runtime binding before interpreting Sheet prose."""
    try:
        from .recovered_scene_materialization import binding
    except ImportError:
        try:
            from recovered_scene_materialization import binding
        except ImportError:
            return None
    try:
        value = binding(task_id)
    except KeyError:
        return None
    for key in ("support_prim", "target_prim"):
        frame = _frame_from_prim(str(value.get(key, "")))
        if frame is not None:
            return frame
    return None


def _episode_destination_binding(
    task: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the recovered open-destination view when one exists."""
    try:
        from .recovered_scene_materialization import binding
    except ImportError:
        try:
            from recovered_scene_materialization import binding
        except ImportError:
            return None
    try:
        value = binding(
            str(task["id"]), task.get("episode_destination"),
        )
    except KeyError:
        return None
    if value.get("episode_destination_required") is not True:
        return None
    return value


def _frame(task: Mapping[str, Any]) -> str:
    if str(task["id"]).startswith("8."):
        return "coffee_table"
    bound = _bound_frame(str(task["id"]))
    if bound is not None:
        return bound
    action_target = " ".join(map(str, (
        task.get("action", ""),
        task.get("target_or_destination", ""),
    )))
    text = " ".join(map(str, (
        task.get("instruction", ""),
        action_target,
    )))
    # The family name can mention a sofa while the concrete task acts on the
    # CoffeeTable drawer.  Concrete action geometry wins over family context.
    if "茶几抽屉" in action_target:
        return "coffee_table"
    if _contains(text, "炸篮", "空气炸锅", "微波炉", "咖啡机", "豆仓", "冰箱"):
        return "kitchen"
    if _contains(text, "收纳篮", "装包", "临时区"):
        return "tote"
    if "移动电源" in text:
        return "kitchen"
    if "水瓶" in text:
        return "sofa"
    # Match the specific surfaces before the generic 茶几 token.
    if _contains(text, "小茶几", "边柜"):
        return "side_table"
    if "电视柜" in text:
        return "tv_cabinet"
    if "衣架" in text:
        return "coat_rack"
    if "零食架" in text:
        return "snack_shelf"
    if "沙发" in text:
        return "sofa"
    if _contains(text, "茶几", "垃圾桶", "桌布"):
        return "coffee_table"
    if _contains(text, "便当盒", "饭盒"):
        return "kitchen"
    if "餐桌" in text:
        return "kitchen"
    return "coffee_table"


def _station(task: Mapping[str, Any], frame: str) -> str:
    task_id = str(task["id"])
    action = str(task.get("action", ""))
    target = str(task.get("target_or_destination", ""))
    combined = f"{action} {target}"
    if task_id.startswith("8."):
        return "tablecloth"
    recovered = _recovered_station(task_id)
    if recovered:
        return recovered
    if task_id.startswith("3."):
        subtask = int(task_id.split(".", 1)[1])
        return "shelf_pick" if subtask % 2 else "shelf_place_table"
    if "茶几抽屉" in combined or "抽屉" in combined:
        # Both drawer producers face the drawer front; the diagonal
        # coffee_table_drawer_bill station is kept only for small Task 28
        # real-data replay (init_place.DRAWER_PLACE).
        return (
            "coffee_table_drawer_wallet"
            if "钱包" in combined
            else "coffee_table_drawer_bill_two_hand"
        )
    if task_id.startswith("7."):
        if task_id == "7.2":
            return "family7_paper_bin"
        if task_id in {"7.4", "7.8", "7.10"}:
            return "family7_tidy_bin"
        return "sidetable_pick" if frame == "side_table" else "table_pick"
    if frame == "snack_shelf":
        return "shelf_pick"
    if frame == "sofa":
        return "sofa_pick"
    if frame == "side_table":
        return "sidetable_pick"
    if frame == "tv_cabinet":
        return "tv_cabinet_pick"
    if frame == "coat_rack":
        return "coat_rack_hang"
    if frame in {"kitchen", "tote"}:
        return "kitchen_table_place"
    # Coffee-table placement faces the table like table_pick (6.9 producer,
    # 6.2 replay source).  The diagonal table_place station remains only for
    # small-Task real-data replay in init_place.
    return "table_pick"


def _recovered_station(task_id: str) -> str:
    try:
        from .recovered_task_contract import TASKS
    except ImportError:
        try:
            from recovered_task_contract import TASKS
        except ImportError:
            return ""
    entry = TASKS.get(task_id, {})
    return str(entry.get("robot_station", ""))


def _station_pose(key: str) -> dict[str, Any]:
    from robot_positions import robot_position

    pose = robot_position(key)
    return {
        "key": key,
        "xy_m": [float(value) for value in pose.xy_m],
        "yaw_deg": float(pose.yaw_deg),
        "quat_wxyz": [float(value) for value in pose.quat_wxyz],
        "anchor": pose.anchor,
    }


def _item(text: str) -> str:
    for token, value in (
        ("牛排", "beef_box"),
        ("剩饭", "meal_box"),
        ("沙拉", "salad_box"),
        ("便当盒", "salad_box"),
        ("墨镜盒", "glasses_case"),
        ("钱包", "wallet"),
        ("遥控器", "remote"),
        ("书籍", "book"),
        ("帽子", "hat"),
        ("纸巾", "tissue"),
        ("包装", "food_wrapper"),
        ("易拉罐", "can"),
        ("杯", "cup"),
    ):
        if token in text:
            return value
    return text.split("→", 1)[0].split("@", 1)[0].strip() or "task_item"


def scenario_choices(task: Mapping[str, Any], frame: str) -> dict[str, str]:
    """Return only genuine scenario alternatives, not task ownership locks."""
    family = int(task["family"])
    task_id = str(task["id"])
    action = str(task.get("action", ""))
    target = str(task.get("target_or_destination", ""))
    text = f"{action} {target}"
    item = _item(target)
    if family == 2:
        sugar = "sugar" if task_id in {"2.2", "2.3"} else "no_sugar"
        return {"family2_bread": sugar}
    if family == 3:
        return {"family3_item": item}
    if family == 4 and "→" not in text:
        return {f"family4_location:{item}": frame}
    if family == 6 and str(task.get("action", "")).startswith("将"):
        return {f"family6_destination:{item}": frame}
    if family == 7:
        return {"family7_focus": task_id}
    if family == 5 and item == "salad_box":
        return {"meal_payload": "salad_box"}
    if family in {9, 10}:
        choices = {"meal_payload": item}
        if family == 10 and "放到" in text:
            choices[f"serve_destination:{item}"] = frame
        return choices
    return {}


def _asset_bindings(task: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve shared payload assets without introducing another registry."""
    target = str(task.get("target_or_destination", ""))
    item = _item(target)
    if item not in {"beef_box", "meal_box", "salad_box"}:
        return {}
    try:
        from .dining_task_contract import ASSETS
    except ImportError:
        from dining_task_contract import ASSETS
    asset = ASSETS[item]
    # Recovered Task5 owns its user-selected physical payload; other dining
    # tasks retain their own asset bindings.
    try:
        from .recovered_task_contract import ASSETS as recovered_assets, TASKS
    except ImportError:
        from recovered_task_contract import ASSETS as recovered_assets, TASKS
    recovered = TASKS.get(str(task["id"]))
    if recovered and recovered["target"] in recovered_assets:
        asset = recovered_assets[recovered["target"]]
    return {
        "primary_payload": {
            "type": item,
            "asset": str(asset),
            "compatibility_aliases": (
                ["lunchbox"] if item == "salad_box" else []
            ),
        }
    }


def _single_layout_module(task_id: str, frame: str | None) -> str | None:
    if frame in FRAME_MODULES:
        return FRAME_MODULES[str(frame)]
    if frame is None:
        return None
    if task_id == "1.1":
        return "coffee_machine_cluster"
    if task_id.startswith("2."):
        return "air_fryer_cluster"
    if task_id == "5.3":
        return "kitchen_counter_cluster"
    if task_id in {"9.1", "9.3", "9.5"}:
        return "refrigerator_cluster"
    if task_id in {"9.2", "9.4", "10.1", "10.4", "10.7"}:
        return "microwave_cluster"
    if task_id in {"10.2", "10.5", "10.8"}:
        return "dining_table_cluster"
    raise ValueError(
        f"cannot resolve a House layout module for task {task_id} "
        f"in frame {frame!r}"
    )


def _layout_replay_contract(
    task: Mapping[str, Any], frame: str | None, *, destination_required: bool,
) -> dict[str, Any]:
    """Project task semantics into the current final-layout replay contract."""

    task_id = str(task["id"])
    if destination_required:
        mode = "runtime_destination_required"
        modules: list[str] = []
    elif task_id == "4.17":
        mode = "fixed_destination_requires_initially_held"
        modules = ["tv_cluster"]
    elif task_id in MULTI_MODULE_ROUTES:
        mode = "multi_module_requires_full_route"
        modules = list(MULTI_MODULE_ROUTES[task_id])
    else:
        mode = "single_module_same_action"
        module = _single_layout_module(task_id, frame)
        modules = [] if module is None else [module]
    return {
        "schema": "real2sim.house_layout_replay_contract.v1",
        "mode": mode,
        "modules": modules,
        "source": "Sheet task semantics resolved by house_task_environment",
    }


def resolve(task: Mapping[str, Any]) -> dict[str, Any]:
    """Add canonical scene/frame/station facts to one catalog definition."""
    destination = _episode_destination_binding(task)
    if destination is None:
        frame: str | None = _frame(task)
        station = _station(task, frame)
        station_pose: dict[str, Any] | None = _station_pose(station)
    elif destination.get("runtime_supported") is True:
        frame = str(destination["layout_frame"])
        station_pose = _station_pose(str(destination["robot_station"]))
    else:
        frame = None
        station_pose = None
    destination_selection = (
        {
            "required": True,
            "selected": bool(destination.get("destination_selected")),
            "allowed_modes": list(destination["allowed_destination_modes"]),
            "runtime_supported": bool(destination.get("runtime_supported")),
            "value": destination.get("episode_destination"),
            "blocker": str(destination.get("unsupported_reason", "")),
        }
        if destination is not None else {
            "required": False,
            "selected": False,
            "allowed_modes": [],
            "runtime_supported": True,
            "value": None,
            "blocker": "",
        }
    )
    return {
        "kind": "canonical_house_scene",
        "scene": str(WORKING_SCENE),
        "source_baseline": str(BASELINE),
        "working_scene_basename": WORKING_SCENE_NAME,
        "runtime_scene_basename": WORKING_SCENE_NAME,
        "layout_frame": frame,
        "robot_station": station_pose,
        "destination_selection": destination_selection,
        "layout_replay": _layout_replay_contract(
            task, frame,
            destination_required=destination_selection["required"],
        ),
        "scenario_choices": scenario_choices(task, frame or ""),
        "asset_bindings": _asset_bindings(task),
        "episode_state_source": (
            "owner action/full_state contribution loaded by the central "
            "episode adapter"
        ),
        "owner_initializer_mode": "single_task_producer",
    }

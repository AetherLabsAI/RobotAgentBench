"""Read-only initial-state planning. No scene edits, simulation imports or defaults."""
import json
import math
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
FRIDGE = "/World/Background/Shell/RefrigeratorRecovered"
TARGET = "/World/Assets/MicrowaveBeefbox"
LEGACY_SOURCE_TARGET = "/World/Assets/BeefTray"
ROBOT = "/World/Robot"


def definition():
    return json.loads((TASK_DIR / "definition.json").read_text())


def relative_xy_yaw(robot_xy_yaw, fixture_xy_yaw):
    """World Z-up planar poses, meters/degrees; NOT fridge asset's Y-up frame."""
    values = tuple(robot_xy_yaw) + tuple(fixture_xy_yaw)
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected two finite (x, y, yaw_degrees) poses")
    rx, ry, ryaw, fx, fy, fyaw = values
    a = math.radians(fyaw)
    dx, dy = rx - fx, ry - fy
    return [math.cos(a)*dx + math.sin(a)*dy,
            -math.sin(a)*dx + math.cos(a)*dy,
            (ryaw - fyaw + 180.0) % 360.0 - 180.0]


def source_plan():
    """Source facts only. This is not an accepted/resettable episode."""
    snapshot = json.loads((TASK_DIR / "sources/l1_snapshot.json").read_text())
    recovered = json.loads((TASK_DIR / "sources/initial_frame0.json").read_text())
    layout = snapshot["reports"]["placement_layout.json"]["data"]["beef"]
    robot = [2.865, 0.928, 53.0]  # mw_studio_server.py T95_BASE
    fixture = [3.1730380596772965, 2.038976182815812, -2.0326655]
    return {
        "accepted": False,
        "robot_world_xy_yaw_m_deg": robot,
        "fixture_world_xy_yaw_m_deg": fixture,
        "robot_relative_planar_xy_yaw_m_deg": relative_xy_yaw(robot, fixture),
        "relative_frame": "world-Z-up fixture yaw frame; not asset-local Y-up",
        "target_path": TARGET,
        "legacy_source_target_path": LEGACY_SOURCE_TARGET,
        "asset_selection": "shared beef_box_standard243_v1; baked geometry, instance scale=1",
        "source_layout_sample": layout,
        "target_source_frame": "right-door local; Y vertical, units meters",
        "target_source_local_position_m": [layout["x"], 0.815, layout["z"]],
        "target_local_quaternion_wxyz": recovered["target_local_quaternion_wxyz"],
        "robot_joint_positions": recovered["positions"],
        "pending": [
            "Validate L1 candidate placement against the shared baked-size box geometry",
            "Isaac settle validation, support and handle reachability",
        ],
    }


def require_accepted_initial(plan):
    """Prevent source notes from being mistaken for a validated reset state."""
    if plan.get("accepted") is not True:
        raise ValueError("Fridge initial state has not passed Isaac validation")
    if plan.get("target_local_quaternion_wxyz") is None or plan.get("robot_joint_positions") is None:
        raise ValueError("Incomplete initial state")
    return plan


def recovered_robot_initialization():
    """Existing reset contract; all values come from the audited L1 initial frame."""
    from arena_api.task_runtime import RobotInitialization
    data = json.loads((TASK_DIR / "sources/initial_frame0.json").read_text())
    return RobotInitialization(positions=dict(data["positions"]),
                               hold_targets=dict(data["hold_targets"]))

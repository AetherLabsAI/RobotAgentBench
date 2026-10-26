"""Per-episode init for place tasks (including 28 bill->CoffeeTable drawer).
In-hand start (same mechanism as throw-28): the recorded episode begins with the object
already held; align_grasp's longest-hold midpoint IS the in-hand pose. Place the target
there, closed-loop settle pins it between the pads, the recorded trajectory carries it to
the calibrated surface/bin destination and releases.

    PLACE_TASK=27 conda run -n base python real2sim/teleop/init_place.py
"""
import json, math, os, re, subprocess, sys, tempfile
from pathlib import Path

from managed_place_lock import PLACE_LOCK, acquire_or_inherit_lock
from managed_subprocess import run_argv
from bag_place_contract import (
    BAG_PLACE_EXISTING_BOTTLE_PATH,
    BAG_PLACE_SCENE_FILENAME,
    BAG_PLACE_SCENE_ROOT,
    BAG_PLACE_STATION,
    BAG_PLACE_TARGETS,
    BAG_PLACE_TOTE_PATH,
)
from dual_trash_bin_layout import RED_CENTER_XY_M, TEAL_CENTER_XY_M
import group_shift
from robot_positions import (
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    KITCHEN_TABLE_INTERACTION_SHIFT_XY_M,
    robot_position,
    translate_interval,
    translate_xy,
    translate_xyz,
    translate_zone_xy,
)
from shelf_asset_policy import (
    require_ready_scene_assets,
    task_asset_decision,
    usd_scene_filter_source,
)
from drawer_articulation import (
    drawer_articulation_code,
    drawer_reset_code,
)

from machine_paths import (
    CONTAINER_NAME, MESH311_PYTHON, RENDER_GPU, isaac_python_command,
)
from machine_paths import DEMO_ROOT as _DEMO_ROOT
DEMO = str(_DEMO_ROOT)
INSTANCE = os.environ.get("INSTANCE", "")
SCENE = f"{DEMO}/scene_room{INSTANCE}.usda"
CAM_PREVIEW = f"{DEMO}/cam_preview{INSTANCE}"
TRAY_TEXTURE = "./usdz/tray_coffee_baseColor.jpg"
CURRENT_CONTEXT_SCRIPT = (
    Path(__file__).with_name("init_place_current_context.py")
)
CURRENT_CONTEXT_TASKS = {"20", "27", "28", "42"}
PARK_MARKER = "REAL2SIM_PLACE_PARK_COMPLETE"
PLACE_MARKER = "REAL2SIM_PLACE_SCENE_COMPLETE"
SCENE_FILTER = usd_scene_filter_source("s")

BASE = robot_position("shelf_place_table").xy_m
YAW = robot_position("shelf_place_table").yaw_deg


def _private_base(xy, layout_delta_xy):
    """Task-private solved station: furniture layout delta, then group shift.

    Stations that come from ``robot_position()`` inherit the group shift
    there; these literals were solved from real head frames at the canonical
    group placement, so they carry the same shift explicitly.
    """
    return group_shift.apply_xy(translate_xy(xy, layout_delta_xy))

# zone: (cx, cy, half_x, half_y, surface_z) where the object must come to rest.
# CoffeeTable is the living-room tea table, not the separate dining table.
# Its 0.8 x 1.2 m footprint and 0.45 m height remain authoritative; the
# scene-level furniture layout transaction moves its centre to satisfy the
# measured wall/table/sofa gaps.
COFFEE_TABLE_CENTER_XY = COFFEE_TABLE_CURRENT_CENTER_XY_M
COFFEE_TABLE_WORLD_SIZE = (0.8, 1.2, 0.45)
TRAY_VISIBLE_TOP_Z = 0.47
TRAY_INTERIOR_SUPPORT_Z = 0.455337
TABLE_ZONE = group_shift.apply_zone((
    COFFEE_TABLE_CENTER_XY[0],
    COFFEE_TABLE_CENTER_XY[1],
    COFFEE_TABLE_WORLD_SIZE[0] / 2.0,
    COFFEE_TABLE_WORLD_SIZE[1] / 2.0,
    COFFEE_TABLE_WORLD_SIZE[2],
))
TRAY_ZONE = translate_zone_xy(
    (-1.80, -0.78, 0.19, 0.13, TRAY_INTERIOR_SUPPORT_Z),
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
)  # 0.470 m is the rim; the visible/colliding interior is 0.455337 m.
SMALL_ZONE = group_shift.apply_zone(
    (-2.115, -1.99, 0.48, 0.63, 0.409))  # 深色小桌 (t14/15)
# The physical drawer both drawer tasks use is black_box_1, the box NEAREST
# the robot (user-identified from the recordings, and confirmed by the solved
# station: under it the recorded pull pinch lands on this box's front face —
# see drawer_articulation.py and station_solve.py).
T28_DRAWER_BODY_PATH = "/World/Assets/CoffeeTable/black_box_1"
T28_DRAWER_SIDE_COLLIDER_PATH = (
    T28_DRAWER_BODY_PATH
    + "/box/Geom/Meshes/FAB_converted_model/root/"
    "GLTF_SceneRootNode/Box_1/Object_4/Object_0"
)
T28_DRAWER_BOTTOM_COLLIDER_PATH = (
    T28_DRAWER_BODY_PATH
    + "/box/Geom/Meshes/FAB_converted_model/root/"
    "GLTF_SceneRootNode/Box_1/Bottom_0/Object_6/Object_1"
)
T28_DRAWER_WORLD_BOUNDS = {
    # Native CLOSED black_box_2 bounds in the shared 2026-07-28 furniture
    # layout.  The articulated drawer starts pulled out by a per-episode lip
    # and may be height-scaled to the FK lip height, so consumers must treat
    # these as the closed-identity reference, not the frame-0 pose.
    "min": (
        -1.860582009,
        -0.629711968,
        0.199993230,
    ),
    "max": (
        -1.550025237,
        -0.405577761,
        0.337311269,
    ),
}
T28_DRAWER_INTERIOR_BOTTOM_Z = 0.203662484
T28_DRAWER_ZONE = (
    (
        T28_DRAWER_WORLD_BOUNDS["min"][0]
        + T28_DRAWER_WORLD_BOUNDS["max"][0]
    ) / 2.0,
    (
        T28_DRAWER_WORLD_BOUNDS["min"][1]
        + T28_DRAWER_WORLD_BOUNDS["max"][1]
    ) / 2.0,
    0.120,
    0.080,
    T28_DRAWER_WORLD_BOUNDS["max"][2],
)
T28_DRAWER_DESTINATION = {
    "kind": "bin",
    "zone": T28_DRAWER_ZONE,
    "bottom_z": T28_DRAWER_INTERIOR_BOTTOM_Z,
    "prim_path": T28_DRAWER_BODY_PATH,
    "collider_paths": (
        T28_DRAWER_SIDE_COLLIDER_PATH,
        T28_DRAWER_BOTTOM_COLLIDER_PATH,
    ),
    "measured_bounds": T28_DRAWER_WORLD_BOUNDS,
}
T73_TOTE_ZONE = translate_zone_xy(
    (1.915, -0.505, 0.19575, 0.06075, 1.16255),
    KITCHEN_TABLE_INTERACTION_SHIFT_XY_M,
)
T73_TOTE_DESTINATION = {
    "kind": "bin",
    "zone": T73_TOTE_ZONE,
    "bottom_z": 0.920,
    "prim_path": "/World/Task73/Tote",
}
BAG_PLACE_SCENE = f"{DEMO}/{BAG_PLACE_SCENE_FILENAME}"
T73_SCENE = BAG_PLACE_SCENE
TASK42_FULL_ENVELOPE_WIDTH = 0.085
# src = where the object came FROM. It决定机器人停哪:"sofa"(书/本子/钱包/墨镜盒…)= 沙发与茶几之间、
# 垃圾桶旁、斜着面向茶几;"shelf"(零食/饮料)= 货架与茶几之间、正对茶几。CoffeeTable 可见的
# 12 个 task 都从自己的真机 head 帧解出显式 base/yaw；14/15 的 exemplar 是另一张小木桌，与
# CoffeeTable override 冲突，故保留已标注 fallback，不能伪称已解 —— 历史审计见
# diagnostics/legacy_plans_pre_current_contract_20260727/BASE_SOLUTIONS.md。
TASKS = {
    "27": dict(prim="Book", root="/World/Assets", zone=TABLE_ZONE, src="sofa",
               base=_private_base(
                   (-1.391, -1.329), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=129.1),
    # The exact REAL episode carries the green Sales Order/bill into the first
    # open black storage drawer on the CoffeeTable.  The red-bin interpretation
    # was an old semantic error and is forbidden by the current contract.
    "28": dict(prim="Note", root="/World/Assets", zone=T28_DRAWER_ZONE, src="sofa",
               destination=T28_DRAWER_DESTINATION,
               # Station solved from the REAL head frame: the table corners
               # in the recording fix the base pose (station_solve.py, 14.9 px
               # RMS over three corners).  The retired station stood ~26 cm too
               # far forward and ~6 deg off, which is what made the replayed
               # pull hand slap the drawer front instead of hooking its lip.
               base=(-1.1925, -1.1752),
               yaw=127.82),
    "41": dict(prim="honeymaid", root="/World/Assets/Tier2Items", zone=TABLE_ZONE, src="shelf",
               base=_private_base(
                   (-2.087, 0.655), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-88.2, up_axis=2,
               # +0.09 is the measured usable edge seat: +0.110 raises
               # proximal cage loads to 44 N and +0.118 loses bilateral
               # contact under every bounded squeeze.
               #
               # 2026-08-02: this was keyed as seat_mid and no run could ever
               # pass the pre-seat check, which reported margins(mm)=[111.0,
               # -34.0]. honeymaid's envelope is 125x45x235 mm, so the mid
               # (width) half-extent is 62.5 mm: seating 90 mm off centre puts
               # the pad 27.5 mm outside the box. The numbers above only
               # describe the LONG axis, whose half-extent is 117.5 mm — +0.118
               # "loses bilateral contact" is exactly the pad centre clearing
               # the 117.5 mm top edge. It is an edge grasp near the top of a
               # 235 mm tall box, which is also how the real episode holds it.
               # 2026-08-02 sweep: the seat offset is NOT the lever for the
               # release load gate. -0.090 / +0.090 / +0.100 / +0.110 all give
               # exactly 0.0 N on the proximal cage across three samples, while
               # the link4 pads read 26-29 N outer and 52-60 N inner in every
               # one of them. The note above ("0.110 raises proximal cage loads
               # to 44 N") does not reproduce at any of the four. The box is
               # pinched at the fingertips, so the three outer and three inner
               # phalanges it never touches cannot carry load — and the gate at
               # playback_video3.py:6600 requires >= 5 N there. Restored to the
               # reviewed value, which also keeps the largest pre-seat margin
               # (21.0 mm against 1.0 mm at 0.110).
               seat_long=0.090, release_retreat=0.04),                # place packs
    "42": dict(prim="pring_bbq", root="/World/Assets/Tier2Items", zone=TRAY_ZONE, src="shelf",
               base=_private_base(
                   (-2.088, 0.657), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-88.2,
               tray=translate_xy(
                   (-2.055, -0.084), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               # UNRESOLVED 2026-08-03, needs a human call. This declares a
               # 75 mm can, but the runtime measures its active collision
               # envelope as he(mm)=(42,42,115), i.e. 84 mm -- the static bound
               # off the inactive prim reads 75 mm, which is where this value
               # came from. At 0.075 the jaw pre-opens to 76 mm and strikes an
               # 84 mm can instead of going around it: frame-0 height 0.493
               # (lying on the table), 0 bilateral frames. Raising it to the
               # full envelope (0.085) does NOT fix it either -- the runtime
               # then refuses the run outright ("physical clamp failed:
               # sustained target-only bilateral contact was not observed"),
               # which is the correct fail-closed answer but still no grasp.
               # So the 75/84 mismatch is real and this value is suspect, but
               # neither number produces a grasp; the cylinder needs a contact
               # model decision, not another width. Left at the calibrated
               # value the test contract pins.
               grip_width=0.075),                                     # place pringles
    # Delivered beef-bar rollouts use this near-side tray position. Keep the
    # physical initializer and destination scorer on the same descriptor.
    "43": dict(prim="chomps_r", root="/World/Assets/Tier2Items", zone=TRAY_ZONE, src="shelf",
               base=_private_base(
                   (-2.085, 0.666), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-88.2,
               tray=translate_xy(
                   (-1.993, -0.170), COFFEE_TABLE_LAYOUT_SHIFT_XY_M)),  # place beef bar
    "44": dict(prim="skit_littles", root="/World/Assets/Tier2Items", zone=TRAY_ZONE, src="shelf",
               base=_private_base(
                   (-2.099, 0.650), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-88.2,
               tray=translate_xy(
                   (-2.130, -0.053), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               seat_long=0.03),                                       # place littles
    # Owner-confirmed benchmark 3.24: same CoffeeTable destination and same
    # robot station as every other shelf-to-CoffeeTable place.  The station
    # faces the table with the shelf behind the robot; furniture never moves.
    "45": dict(
        prim="oreo_gf",
        root="/World/Assets/Tier2Items",
        zone=TABLE_ZONE,
        src="shelf",
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    "46": dict(prim="zbar_t", root="/World/Assets/Tier2Items", zone=TRAY_ZONE, src="shelf",
               base=_private_base(
                   (-1.926, 0.723), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-89.6,
               tray=translate_xy(
                   (-2.124, -0.048), COFFEE_TABLE_LAYOUT_SHIFT_XY_M)),  # place zbar
    # Task-local layer imported from l2.  The soft tote, staged held target,
    # and existing bottle move as one interaction cluster with the resized
    # dining counter's north edge.  The exact holding hand is resolved from
    # each episode rather than copied from the small Task 73 bottle example.
    **{
        task_id: dict(
            prim=target["prim"],
            root=target["root"],
            zone=T73_TOTE_ZONE,
            destination=T73_TOTE_DESTINATION,
            src="bag_place",
            base=robot_position(BAG_PLACE_STATION).xy_m,
            yaw=robot_position(BAG_PLACE_STATION).yaw_deg,
            hand=None,
            up_axis=target.get("up_axis"),
            seat_long=target.get(
                "seat_long_m",
                0.080 if task_id == "73" else 0.000,
            ),
            seat_mid=0.000,
            grasp_long_plane_rotation_deg=target.get(
                "grasp_long_plane_rotation_deg"
            ),
            approach_offset_xyz=target.get("approach_offset_xyz_m"),
            staging_xyz=target["staging_xyz_m"],
            scene=BAG_PLACE_SCENE,
            policy_scene=SCENE,
            task_local_scene=True,
        )
        for task_id, target in BAG_PLACE_TARGETS.items()
    },
    # 老批次 place (14-20): 饮料→桌 / 书平放
    # 用户指令 2026-07-21: 不再用小黑桌, 所有 snack place 统一以大 CoffeeTable 为目的地。
    # t14/15 原 SMALL_ZONE+南向base(15/15 in 小黑桌设置, 已归档 .bak_smalltable) →
    # 改 TABLE_ZONE；但两集 exemplar 仍显示小木桌，与该 override 冲突，所以暂时明确使用 fallback，
    # 不复制 t16/17 的 pose 冒充逐集解。待场景语义裁决后再按目标桌重新标定。
    "14": dict(prim="gatorade_m", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf"),
    "15": dict(prim="gatorade_t", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf"),
    "16": dict(prim="gatorade_m", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf",
               base=_private_base(
                   (-2.073, 0.636), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-91.6),
    "17": dict(prim="gatorade_t", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf",
               base=_private_base(
                   (-2.076, 0.636), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-91.3),
    "18": dict(prim="sprite_c1", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf",
               base=_private_base(
                   (-2.056, 0.642), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-91.9),
    "19": dict(prim="diet_coke", root="/World/Assets/ShelfItems", zone=TABLE_ZONE, src="shelf",
               base=_private_base(
                   (-2.050, 0.650), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
               yaw=-91.1),
    # New 2026-07-29 snack-place recordings.  Start from the canonical
    # shelf-to-CoffeeTable station; any visual tuning must move this robot
    # station and its robot-relative trajectory, never the furniture.
    "86": dict(
        prim="good2grow", root="/World/Assets/ShelfItems",
        zone=TABLE_ZONE, src="shelf",
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    "87": dict(
        prim="grapefruit_can", root="/World/Assets/ShelfItems",
        zone=TABLE_ZONE, src="shelf",
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    "88": dict(
        prim="cayman_jacked", root="/World/Assets/ShelfItems",
        zone=TABLE_ZONE, src="shelf",
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    "89": dict(
        prim="starbucks", root="/World/Assets/ShelfItems",
        zone=TABLE_ZONE, src="shelf",
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    # t20 真实内容(两集结尾抽帧核实 2026-07-21): 粉书**平放**到茶几面西南角(同 t27 语义,
    # 老批录制)。"斜靠桌南缘"假设来自 0464909e 5.2s 废录截帧, 被数据推翻。
    # zone = TABLE_ZONE 但 hy 仍比共用 zone 放宽 0.04 m：保持原任务的边缘容差。
    # (取证: 0786d9c4 a4 平放桌面 fz=0.432 xy(-2.04,-0.93), 差 5mm 被原 hy 误杀)。
    # band 保持全局默认: 0464909e 类废录锚点 (-1.11,·) 应 skip-init 快速跳过。
    "20": dict(
        prim="Book",
        root="/World/Assets",
        zone=group_shift.apply_zone((
            COFFEE_TABLE_CENTER_XY[0],
            COFFEE_TABLE_CENTER_XY[1],
            COFFEE_TABLE_WORLD_SIZE[0] / 2.0,
            COFFEE_TABLE_WORLD_SIZE[1] / 2.0 + 0.04,
            COFFEE_TABLE_WORLD_SIZE[2],
        )),
        src="sofa",
        base=_private_base(
            (-1.245, -1.139), COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
        yaw=144.6),
}

# These benchmark Task 6 rows have no robot-template place ID.  Rows 6.3,
# 6.9, 6.10 and 6.15 reverse real picks from the same physical destination;
# row 6.7 directly replays the real small Task 70 hang.  Keep the benchmark
# identity explicit and move only the robot station/trajectory.
from task6_code_control_contract import MODES as _TASK6_CC_MODES
for _task6_subtask in ("6.3", "6.5", "6.7", "6.9", "6.10", "6.15"):
    _task6_mode = _TASK6_CC_MODES[_task6_subtask]
    _task6_station = robot_position(_task6_mode.robot_station)
    TASKS[_task6_mode.runtime_task_id] = dict(
        prim=_task6_mode.target_path.rsplit("/", 1)[-1],
        root=_task6_mode.target_path.rsplit("/", 1)[0],
        zone=_task6_mode.destination_zone,
        destination={
            "kind": _task6_mode.destination_kind,
            "zone": _task6_mode.destination_zone,
            "bottom_z": _task6_mode.destination_bottom_z,
            "prim_path": _task6_mode.destination_path,
        },
        src=(
            "task6_direct_place"
            if _task6_subtask == "6.7" else "task6_reverse_pick"
        ),
        base=_task6_station.xy_m,
        yaw=_task6_station.yaw_deg,
        hand=_task6_mode.hand,
        up_axis=_task6_mode.grasp_up_axis,
        grip_width=_task6_mode.grip_width_m,
        operated_drawer_destination=(_task6_subtask in {"6.5", "6.15"}),
        # Covers only the living-room interaction volume and exists to reject
        # a wrong station/action binding, not to tune a target pose.
        band=(
            dict(x=(-1.5, 0.2), y=(1.0, 2.6), z=(0.50, 1.60))
            if _task6_subtask == "6.7" else (
                dict(x=(-3.0, -1.3), y=(-2.8, -1.0), z=(0.18, 1.60))
                if _task6_subtask == "6.3" else
                dict(x=(-4.2, -0.6), y=(-1.4, 1.0), z=(0.18, 1.60))
            )
        ),
    )


def coffee_table_calibration_from_environment(environment=None):
    """Return an optional exact CoffeeTable ``(x, y, z, bottom_z)`` override."""
    environment = os.environ if environment is None else environment
    keys = (
        "COFFEE_TABLE_WORLD_X_M",
        "COFFEE_TABLE_WORLD_Y_M",
        "COFFEE_TABLE_WORLD_Z_M",
    )
    values = tuple(environment.get(key) for key in keys)
    bottom_text = environment.get("COFFEE_TABLE_BOTTOM_Z_M")
    if not any(value is not None for value in (*values, bottom_text)):
        return None
    if not all(value is not None for value in values):
        raise ValueError(
            "COFFEE_TABLE_WORLD_X_M, COFFEE_TABLE_WORLD_Y_M and "
            "COFFEE_TABLE_WORLD_Z_M must be supplied together"
        )
    try:
        x_size, y_size, z_size = (float(value) for value in values)
        bottom_z = 0.0 if bottom_text is None else float(bottom_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CoffeeTable dimensions and bottom height must be finite numbers"
        ) from exc
    if (
        not all(math.isfinite(value)
                for value in (x_size, y_size, z_size, bottom_z))
        or not 0.5 <= x_size <= 1.2
        or not 0.7 <= y_size <= 3.2
        or not 0.25 <= z_size <= 0.7
        or not -0.1 <= bottom_z <= 0.2
    ):
        raise ValueError(
            "CoffeeTable X/Y/Z dimensions must be within "
            "[0.5,1.2] x [0.7,3.2] x [0.25,0.7] m and bottom_z within "
            "[-0.1,0.2] m"
        )
    return (x_size, y_size, z_size, bottom_z)


def destination_for_task(task_id, coffee_table_calibration=None):
    """Return the complete, effective destination descriptor for a place task.

    Tray task entries store the per-episode measured tray centre separately from
    the shared tray footprint/height.  Task 28 carries an audited CoffeeTable
    drawer descriptor with exact native colliders, measured world bounds,
    conservative interior, and physical bottom.  Initialisation and scoring
    both consume this function.
    """
    tid = str(task_id)
    if tid not in TASKS:
        raise KeyError(f"unknown place task {tid}")
    cfg = TASKS[tid]
    if not group_shift.is_identity() and (
        "destination" in cfg
        or "tray" in cfg
        or cfg.get("zone") is TRAY_ZONE
        or cfg.get("zone") is T73_TOTE_ZONE
    ):
        group_shift.require_identity(
            f"place task {tid}'s destination (tray/drawer/tote)")
    if "destination" in cfg:
        destination = cfg["destination"]
        resolved = {
            "kind": str(destination["kind"]),
            "zone": tuple(destination["zone"]),
            "bottom_z": float(destination["bottom_z"]),
            "prim_path": str(destination["prim_path"]),
        }
        if "collider_paths" in destination:
            resolved["collider_paths"] = tuple(
                str(path) for path in destination["collider_paths"])
        if "measured_bounds" in destination:
            resolved["measured_bounds"] = {
                "min": tuple(destination["measured_bounds"]["min"]),
                "max": tuple(destination["measured_bounds"]["max"]),
            }
        return resolved
    zone = tuple(cfg["zone"])
    if "tray" in cfg:
        tray_x, tray_y = cfg["tray"]
        zone = (float(tray_x), float(tray_y), *zone[2:])
    result = {
        "kind": "surface",
        "zone": zone,
        "bottom_z": float(zone[4]),
        "prim_path": (
            "/World/Assets/Tray"
            if "tray" in cfg else "/World/Assets/CoffeeTable"
        ),
    }
    if (
        coffee_table_calibration is not None
        and result["prim_path"] == "/World/Assets/CoffeeTable"
    ):
        x_size, y_size, z_size, bottom_z = coffee_table_calibration
        surface_z = bottom_z + z_size
        result["zone"] = (
            COFFEE_TABLE_CENTER_XY[0],
            COFFEE_TABLE_CENTER_XY[1],
            x_size / 2.0,
            y_size / 2.0,
            surface_z,
        )
        result["bottom_z"] = surface_z
    return result


def zone_for_task(task_id, coffee_table_calibration=None):
    """Compatibility accessor backed by :func:`destination_for_task`."""
    return destination_for_task(
        task_id, coffee_table_calibration=coffee_table_calibration
    )["zone"]


# The align result is expressed in room-world coordinates.  These limits were
# calibrated before the accepted CoffeeTable transaction, so move the complete
# interval with the same canonical XY delta as the robot-relative trajectory.
# In particular, the +Y room-shell correction moves valid shelf-origin anchors
# above the legacy y=0.20 ceiling.  Expanding only that ceiling would silently
# admit stale anchors; translating both ends preserves the reviewed envelope.
BAND = dict(
    x=group_shift.apply_interval(translate_interval(
        (-2.90, -1.30), COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0]), "x"),
    y=group_shift.apply_interval(translate_interval(
        (-2.40, 0.20), COFFEE_TABLE_LAYOUT_SHIFT_XY_M[1]), "y"),
    z=(0.25, 1.10),
)

# Current user-reviewed room layout: every task that retains these two bin
# assets uses one fixed fireplace-side pair.  Red is the adjacent +X member,
# i.e. the side closer to the robot.
BINS_TWIN = {
    "TrashBinRed": RED_CENTER_XY_M,
    "TrashBinTeal": TEAL_CENTER_XY_M,
}
BINS_T28 = BINS_TWIN
for _t in (
    "14", "15", "16", "17", "18", "19", "20", "27",
    "41", "42", "43", "44", "46", "86", "87", "88", "89",
):
    TASKS[_t].setdefault("bins", BINS_TWIN)
TASKS["28"]["bins"] = BINS_T28

# --- 2026-08-02: one station per semantic group -----------------------------
# Each place task above carries its own solved base/yaw, so thirteen tasks that
# do the same thing -- carry an object from the shelf and set it on the coffee
# table -- stood in thirteen different places, spread over 173 mm and 3.7 deg.
# The per-task solutions were not noise: 16-19 all lean one way (-1.1 to -1.9
# deg) and 41-44 all lean the other (+1.8 deg), which is two calibration
# batches each drifting as a batch. Nine private corrections hid that drift
# instead of fixing it.
#
# The shared station is not a new number: robot_positions already declares it,
# tasks 86-89 already use it, and it sits within 4 mm / 0.4 deg of the median
# of the nine private values. Group identity now beats per-task tuning; a task
# that genuinely needs its own pose has to say why here.
SHELF_TO_TABLE_PLACE = (
    "16", "17", "18", "19", "41", "42", "43", "44", "45", "46",
    "86", "87", "88", "89",
)
# Placing an object ON the coffee table. Task 28 is deliberately NOT here: it
# opens a drawer, which needs the hand at that drawer's face, and folding it in
# put its object 144 mm outside the target zone.
COFFEE_TABLE_PLACE = ("20", "27")
# One station per drawer, because they are different drawers.
DRAWER_PLACE = {"28": "coffee_table_drawer_bill"}


def _group_station(name):
    """Shared station for a place group, with a search override.

    PLACE_STATION_<NAME>="x,y,yaw" moves the whole group at once so a station
    can be searched by measurement instead of argued about. It exists because
    the nine private stations these tasks used to carry were each solved from
    their own real frame-0 view, which makes any single replacement a claim
    that has to be tested on all of them rather than reasoned from one video.
    """
    pose = robot_position(name)
    raw = os.environ.get(f"PLACE_STATION_{name.upper()}", "").strip()
    if not raw:
        return pose.xy_m, pose.yaw_deg
    try:
        x, y, yaw = (float(v) for v in raw.split(","))
    except ValueError:
        raise SystemExit(
            f"PLACE_STATION_{name.upper()} must be 'x,y,yaw', got {raw!r}")
    return (x, y), yaw


for _group, _station in ((SHELF_TO_TABLE_PLACE, "shelf_place_table"),
                         (COFFEE_TABLE_PLACE, "table_place")):
    _xy, _yaw = _group_station(_station)
    for _t in _group:
        TASKS[_t]["base"] = _xy
        TASKS[_t]["yaw"] = _yaw
for _t, _station in DRAWER_PLACE.items():
    _xy, _yaw = _group_station(_station)
    TASKS[_t]["base"] = _xy
    TASKS[_t]["yaw"] = _yaw

# per-source FALLBACK parked base (x,y) + yaw(deg). Every resolved task above carries its own
# solved base/yaw；当前已知 fallback 仅为语义冲突尚未裁决的 14/15。
#
# HOW THE PER-TASK VALUES WERE OBTAINED (2026-07-24, replaces the old guess-and-render loop):
#   base_sweep.py prints CAMBASE = the head camera's pose in the BASE frame at this episode's
#   frame-0 joints (waist 30deg / head 25deg for every place episode, so one CAMBASE serves all).
#   The historical solve_base.py fit back-projected CoffeeTable corners onto
#   z=0.423.  That plane is audit provenance only: current task poses apply the
#   reviewed furniture transaction and the authoritative top is z=0.450 m.
#   Anchoring on the TABLE (not the room) is deliberate: it also reproduces WHERE ON THE TABLE the
#   carried object lands, which is what the place task is judged on. 但 2026-07-26 small Task 18
#   的真实桌尺寸已确认为 0.8 x 1.2 x 0.45 m；旧 SIM 只有
#   0.643838 x 1.003000 x 0.360000 m，旧 RGB 桌角解会把尺寸误差吸收到 base。
#   重新拟合真实外边并用精确 RGB K/distortion 做 PnP 得到 55.585° down、
#   相机距桌面 0.5835 m。尺寸修正已进入共享场景；后续家具布局又将同尺寸茶几
#   平移到 x=-1.950，因此旧绝对位姿/视频仅作历史证据。
SRC_BASE = {
    # BETWEEN the shelf (y=+1.6) and the coffee table (y=+0.1 near edge), facing the table (-y),
    # standoff ~0.55 m. (The pre-07-24 value y=+0.30 stood 0.34 m too close: the sim head view had
    # the table overflowing the frame while the real one shows the whole table plus floor.)
    "shelf": dict(
        base=robot_position("shelf_place_table").xy_m,
        yaw=robot_position("shelf_place_table").yaw_deg,
    ),
    # ANGLED between the sofa and the coffee table, right beside the twin bins, facing the table's
    # SE corner — this is where the robot actually stands after picking off the sofa (t20/27/28 all
    # solve to x -1.25..-1.39, y -1.13..-1.33, yaw 129-145). The old value (-2.10,-1.45)/+90deg was
    # the shelf pose mirrored to the south and matched no real frame.
    "sofa": dict(
        base=robot_position("table_place").xy_m,
        yaw=robot_position("table_place").yaw_deg,
    ),
}


def run(command, t=None):
    return run_argv(command, timeout=t)


def drawer_pull_crossing_frame(hand="r"):
    """Close-completion frame of the drawer-pull hand's first rising grip.

    The simulated linkage needs a few frames from the 0.5 crossing to real
    pinch force; aligning the drawer front face to the crossing-frame FK
    loses the first centimetres of the pull while the pads are still
    closing.  Report at the first frame the rising closure reaches 0.9 (the
    teleoperator closes before pulling, so the hand has barely moved by
    then).  Small Task 28's left hand holds the bill from frame 0; the pull
    is the right hand's first rising closure."""
    import numpy as np
    trajectory = np.load(f"{DEMO}/playback_traj{INSTANCE}.npz")
    grip = np.asarray(
        trajectory["rgrip" if hand == "r" else "lgrip"], float)
    above = grip >= 0.5
    for index in range(1, len(above)):
        if above[index] and not above[index - 1]:
            for closed in range(index, len(grip)):
                if grip[closed] >= 0.9:
                    return int(closed)
                if grip[closed] < 0.5:
                    break
            return int(index)
    return None


def run_mesh_script(source, timeout=120):
    """Run a unique host-side USD edit script and always clean up its temp file."""
    descriptor, name = tempfile.mkstemp(
        prefix="real2sim_init_place_", suffix=".py", dir="/tmp")
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        return run([str(MESH311_PYTHON), str(path)], t=timeout)
    finally:
        path.unlink(missing_ok=True)


def clear_container_align_marker(instance=INSTANCE):
    """Clear the container-owned align marker without changing tree modes."""
    marker = (
        f"/geniesim/main/demo_sim/cam_preview{instance}/align.txt"
    )
    return run(
        [
            "docker", "exec", CONTAINER_NAME,
            "rm", "-f", marker,
        ],
        t=30,
    )


def main():
    tid = os.environ.get("PLACE_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if tid not in TASKS:
        print(json.dumps({"ok": False, "reason": f"unknown PLACE_TASK {tid}"}))
        return 2
    cfg = TASKS[tid]
    asset = task_asset_decision(tid)
    if asset.applicable and not asset.usable:
        print(json.dumps({
            "ok": False,
            "task": tid,
            "family": "place",
            "target": cfg["prim"],
            "asset_id": asset.asset_id,
            "asset_status": asset.status,
            "artifact_url": asset.artifact_url,
            "reason": f"shelf asset blocked: {asset.reason}",
        }))
        return 10
    grip_width = cfg.get("grip_width")
    try:
        if grip_width is not None:
            grip_width = float(grip_width)
            if (not math.isfinite(grip_width)
                    or not 0.0 < grip_width <= TASK42_FULL_ENVELOPE_WIDTH):
                raise ValueError(
                    "grip_width must be finite and in (0, 0.085] metres")
    except (TypeError, ValueError) as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"invalid place grip width: {exc}",
        }))
        return 2
    grasp_mode = os.environ.get("GRASP_MODE", "")
    if (not re.fullmatch(r"[A-Za-z0-9_-]*", INSTANCE)
            or grasp_mode not in ("", "rise")):
        print(json.dumps({
            "ok": False,
            "reason": "unsafe/unsupported INSTANCE or GRASP_MODE",
        }))
        return 2
    try:
        if asset.applicable:
            asset = task_asset_decision(
                tid, verify_runtime=True, scene_path=SCENE
            )
            if not asset.usable:
                raise RuntimeError(asset.reason)
        require_ready_scene_assets(SCENE)
    except RuntimeError as exc:
        print(json.dumps({
            "ok": False,
            "task": tid,
            "family": "place",
            "target": cfg["prim"],
            "asset_id": asset.asset_id,
            "asset_status": (
                asset.status
                if asset.applicable
                else "scene_whitelist_failed"
            ),
            "reason": f"shelf asset verification failed: {exc}",
        }))
        return 10
    try:
        lock_stream = acquire_or_inherit_lock(PLACE_LOCK)
    except BlockingIOError:
        print(json.dumps({
            "ok": False,
            "reason": "another managed place render owns the shared Isaac lock",
        }))
        return 7
    tp = f"{cfg['root']}/{cfg['prim']}"
    if cfg.get("task_local_scene"):
        scene_path = str(cfg["scene"])
        destination = destination_for_task(tid)
        expected_tote = (
            T73_TOTE_ZONE[0],
            T73_TOTE_ZONE[1],
            T73_TOTE_DESTINATION["bottom_z"],
        )
        expected_base = (*cfg["base"], 0.0)
        expected_yaw_rad = math.radians(float(cfg["yaw"]))
        expected_robot_quat = (
            math.cos(expected_yaw_rad / 2.0),
            0.0,
            0.0,
            math.sin(expected_yaw_rad / 2.0),
        )
        expected_staging = tuple(cfg["staging_xyz"])
        expected_existing = translate_xyz(
            (1.995, -0.525, 0.925),
            KITCHEN_TABLE_INTERACTION_SHIFT_XY_M,
        )
        verify_bag_place = f"""from pxr import Gf, Usd, UsdGeom, UsdPhysics
s = Usd.Stage.Open({scene_path!r})
assert s, 'could not open bag-place scene'
assert s.GetRootLayer().identifier.endswith('/scene_room.usda'), \
    'bag-place runtime must be the canonical scene_room.usda root'
assert not any('scene_room_task' in path \
               for path in s.GetRootLayer().subLayerPaths), \
    'canonical root may not sublayer a task-specific scene root'
task = s.GetPrimAtPath({BAG_PLACE_SCENE_ROOT!r})
tote = s.GetPrimAtPath({BAG_PLACE_TOTE_PATH!r})
target = s.GetPrimAtPath({tp!r})
existing_bottle = s.GetPrimAtPath({BAG_PLACE_EXISTING_BOTTLE_PATH!r})
simulation = s.GetPrimAtPath({(BAG_PLACE_TOTE_PATH + '/Simulation')!r})
visual = s.GetPrimAtPath({(BAG_PLACE_TOTE_PATH + '/Visual/Geometry')!r})
robot = s.GetPrimAtPath('/World/Robot')
assert all(p and p.IsValid() for p in (
    task, tote, target, existing_bottle, simulation, visual, robot
)), 'bag-place scene is missing an approved required prim'
robot.GetAttribute('xformOp:translate').Set(
    Gf.Vec3d(*{expected_base!r}))
robot.GetAttribute('xformOp:orient').Set(
    Gf.Quatf(*{expected_robot_quat!r}))
target.GetAttribute('physics:kinematicEnabled').Set(True)
s.GetRootLayer().Save()
assert task.GetCustomDataByKey('acceptedToteRevision') == \
    'v97-endon-seat080', 'bag-place tote revision changed'
assert task.GetCustomDataByKey('layoutAdaptation') == \
    'kitchen-north-edge-shift-+0.87,+0.37', \
    'bag-place dining-table layout adaptation changed'
assert tote.GetCustomDataByKey('task73AcceptedAssetRevision') == \
    'v51-readable-droop-position-aligned', \
    'bag-place tote asset revision changed'
assert bool(tote.GetCustomDataByKey('manualVisualSkinning')), \
    'bag-place tote lost manual visual skinning'
assert target.IsActive(), 'bag-place target is inactive'
assert target.HasAPI(UsdPhysics.RigidBodyAPI), \
    'bag-place target is not a rigid body'
kinematic = target.GetAttribute('physics:kinematicEnabled')
assert kinematic and bool(kinematic.Get()) is True, \
    'bag-place target must remain staged until physical seating'
assert existing_bottle.HasAPI(UsdPhysics.RigidBodyAPI), \
    'bag-place existing bottle is not a rigid body'
existing_kinematic = existing_bottle.GetAttribute(
    'physics:kinematicEnabled')
assert not existing_kinematic or bool(existing_kinematic.Get()) is False, \
    'bag-place existing bottle must remain dynamically physical'
assert task.GetCustomDataByKey('realInitialContents') == \
    {BAG_PLACE_EXISTING_BOTTLE_PATH!r}, \
    'bag-place real initial contents changed'
scale_ops = [
    op for op in UsdGeom.Xformable(tote).GetOrderedXformOps()
    if op.GetOpType() == UsdGeom.XformOp.TypeScale
]
assert len(scale_ops) == 1, 'bag-place tote must have one scale op'
scale = scale_ops[0].Get()
assert all(abs(float(scale[i]) - value) <= 1e-6
           for i, value in enumerate((1.35, 1.5, 1.05))), \
    'bag-place tote scale changed'
def translation(prim):
    values = [
        op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    assert values, str(prim.GetPath()) + ' has no translate op'
    return values[-1]
def close(actual, expected):
    return all(abs(float(actual[i]) - value) <= 1e-6
               for i, value in enumerate(expected))
assert close(translation(tote), {expected_tote!r}), \
    'bag-place tote layout position changed'
assert close(translation(robot), {expected_base!r}), \
    'bag-place robot layout position changed'
assert close(translation(target), {expected_staging!r}), \
    'bag-place target staging position changed'
assert close(translation(existing_bottle), {expected_existing!r}), \
    'bag-place existing bottle layout position changed'
assert not any(
    prim.IsA(UsdPhysics.Joint)
    for prim in Usd.PrimRange(task)
), 'bag-place scene must not contain an attachment/joint shortcut'
print('{PLACE_MARKER}')
"""
        try:
            verified = run_mesh_script(verify_bag_place)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(json.dumps({
                "ok": False,
                "task": tid,
                "family": "place",
                "reason": (
                    "bag-place scene verification failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }))
            return 6
        ok = (
            verified.returncode == 0
            and PLACE_MARKER in verified.stdout.splitlines()
        )
        out = {
            "ok": ok,
            "task": tid,
            "family": "place",
            "target": cfg["prim"],
            "track_prim": tp,
            "hand": cfg["hand"],
            "zone": destination["zone"],
            "destination": destination,
            "kin": None,
            "base": list(cfg["base"]),
            "yaw": float(cfg["yaw"]),
            "up_axis": cfg["up_axis"],
            "seat_mid": cfg.get("seat_mid"),
            "seat_long": cfg["seat_long"],
            "grasp_long_plane_rotation_deg": cfg.get(
                "grasp_long_plane_rotation_deg"
            ),
            "approach_offset_xyz": cfg.get("approach_offset_xyz"),
            "grip_width": None,
            "release_retreat": None,
            "asset_id": asset.asset_id,
            "asset_status": asset.status,
            "asset_revision": asset.asset_revision,
            "scene": scene_path,
        }
        if not ok:
            out["reason"] = (
                "bag-place scene verification failed: "
                + (verified.stdout + verified.stderr)[-240:]
            )
        print(json.dumps(out))
        return 0 if ok else 6
    global BASE, YAW
    # per-TASK parked base (solved from that episode's real head frame); falls back to the family
    # default. FIXED_BASE / FIXED_YAW env override for calibration.
    sb = SRC_BASE.get(cfg.get("src", "shelf"), SRC_BASE["shelf"])
    try:
        if "FIXED_BASE" in os.environ:
            BASE = tuple(
                float(value)
                for value in os.environ["FIXED_BASE"].split(",")
            )
        else:
            BASE = tuple(cfg.get("base", sb["base"]))
        YAW = float(os.environ.get("FIXED_YAW", cfg.get("yaw", sb["yaw"])))
        ih_yaw = float(os.environ.get("IH_YAW", "0"))
        fixed_roll_rad = float(os.environ.get("FIXED_ROLL_RAD", "0"))
        if (len(BASE) != 2
                or not all(math.isfinite(value) for value in (
                    *BASE, YAW, ih_yaw, fixed_roll_rad))):
            raise ValueError(
                "base/yaw/roll values must be finite and BASE must have two values")
    except (TypeError, ValueError) as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"invalid place calibration environment: {exc}",
        }))
        return 2
    try:
        coffee_table_calibration = (
            coffee_table_calibration_from_environment()
        )
    except ValueError as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"invalid CoffeeTable size calibration: {exc}",
        }))
        return 2
    destination = destination_for_task(
        tid, coffee_table_calibration=coffee_table_calibration
    )
    ih_top_text = os.environ.get("IH_TOP", "0").lower()
    if ih_top_text not in ("", "0", "1", "false", "true", "no", "yes"):
        print(json.dumps({
            "ok": False,
            "reason": f"invalid IH_TOP boolean {ih_top_text!r}",
        }))
        return 2
    ih_top = ih_top_text in ("1", "true", "yes")
    _h = math.radians(YAW) / 2.0
    _r = fixed_roll_rad / 2.0
    # yaw * roll: the task-owned roll code is carried by the authored robot
    # root quaternion while remaining below PhysX float-normal resolution.
    QUAT = (
        math.cos(_h) * math.cos(_r),
        math.cos(_h) * math.sin(_r),
        math.sin(_h) * math.sin(_r),
        math.sin(_h) * math.cos(_r),
    )

    park = f"""from pxr import Usd, UsdGeom, Gf
s = Usd.Stage.Open('{SCENE}')
assert s, 'could not open scene'
{SCENE_FILTER}
rob = s.GetPrimAtPath('/World/Robot')
assert rob and rob.IsValid(), 'missing robot'
translated = False
for op in UsdGeom.Xformable(rob).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d({BASE[0]}, {BASE[1]}, 0))
        translated = True
assert translated, 'robot has no translate op'
orient = rob.GetAttribute('xformOp:orient')
assert orient and orient.IsValid(), 'robot has no orient attribute'
assert orient.Set(Gf.Quatf({QUAT[0]}, {QUAT[1]}, {QUAT[2]}, {QUAT[3]})), 'robot orient write failed'
assert s.GetRootLayer().Save(), 'park scene save failed'
print('{PARK_MARKER}')
"""
    try:
        park_result = run_mesh_script(park)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"park execution failed: {type(exc).__name__}: {exc}",
        }))
        return 3
    if (park_result.returncode != 0
            or PARK_MARKER not in park_result.stdout.splitlines()):
        print(json.dumps({
            "ok": False,
            "reason": "park failed: " + (
                park_result.stdout + park_result.stderr)[-240:],
        }))
        return 3

    f = f"{CAM_PREVIEW}/align.txt"
    reuse_align_control = os.environ.get(
        "REAL2SIM_REUSE_ALIGN_RESULT", "")
    if reuse_align_control not in ("", "1"):
        print(json.dumps({
            "ok": False,
            "reason": "REAL2SIM_REUSE_ALIGN_RESULT must be exactly 1",
        }))
        return 4
    reuse_align_result = reuse_align_control == "1"
    if reuse_align_result:
        try:
            align_bytes = Path(f).read_bytes()
        except OSError as exc:
            print(json.dumps({
                "ok": False,
                "reason": f"cannot read pinned align result: {exc}",
            }))
            return 4
        try:
            pinned_align_text = align_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            print(json.dumps({
                "ok": False,
                "reason": f"pinned align result is not UTF-8: {exc}",
            }))
            return 4
        if (
            "REAL2SIM_ALIGN_V2" not in pinned_align_text.splitlines()
            or "NO grasp" in pinned_align_text
            or not re.search(
                r"midpoint world: \(([-+\d.]+), ([-+\d.]+), "
                r"([-+\d.]+)\)",
                pinned_align_text,
            )
        ):
            print(json.dumps({
                "ok": False,
                "reason": "pinned align result is incomplete",
            }))
            return 4
    try:
        run(
            [
                "docker", "exec", CONTAINER_NAME, "bash", "-lc",
                "pkill -TERM -f "
                "'[/]geniesim/main/demo_sim/usdz/align_grasp.py' "
                "2>/dev/null || true; sleep 2; "
                "pkill -KILL -f "
                "'[/]geniesim/main/demo_sim/usdz/align_grasp.py' "
                "2>/dev/null || true",
            ]
        )
    except OSError:
        # The authoritative align invocation below produces the structured
        # failure; cleanup itself is best-effort and never counts as success.
        pass
    if reuse_align_result:
        align_command = ["true"]
    else:
        try:
            clear_result = clear_container_align_marker()
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(json.dumps({
                "ok": False,
                "reason": (
                    "align marker cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }))
            return 4
        if clear_result.returncode != 0:
            print(json.dumps({
                "ok": False,
                "reason": "align marker cleanup failed: " + (
                    clear_result.stdout + clear_result.stderr)[-240:],
            }))
            return 4
        align_command = ["timeout", "900", "docker", "exec"]
        align_environment = {
            "CUDA_VISIBLE_DEVICES": RENDER_GPU,
            "INSTANCE": INSTANCE,
            "GRASP_MODE": grasp_mode,
            "SCENE_USDA": f"/geniesim/main/demo_sim/scene_room{INSTANCE}.usda",
            "TRAJ_NPZ": f"/geniesim/main/demo_sim/playback_traj{INSTANCE}.npz",
            "CAM_PREVIEW_DIR": f"/geniesim/main/demo_sim/cam_preview{INSTANCE}",
        }
        if tid == "28":
            drawer_pull_frame = drawer_pull_crossing_frame("r")
            if drawer_pull_frame is not None:
                align_environment["ALIGN_EXTRA_REPORTS"] = (
                    f"r:{drawer_pull_frame}")
        for key, value in sorted(align_environment.items()):
            align_command.extend(["--env", f"{key}={value}"])
        align_command.extend([
            CONTAINER_NAME,
            *isaac_python_command(
                "/geniesim/main/demo_sim/usdz/align_grasp.py"),
        ])
    try:
        align_result = run(align_command, t=950)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"align execution failed: {type(exc).__name__}: {exc}",
        }))
        return 4
    if align_result.returncode != 0:
        print(json.dumps({
            "ok": False,
            "reason": "align failed: " + (
                align_result.stdout + align_result.stderr)[-240:],
        }))
        return 4
    txt = (
        pinned_align_text
        if reuse_align_result
        else (open(f).read() if os.path.isfile(f) else "")
    )
    m = re.search(r"midpoint world: \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)", txt)
    if not m or "NO grasp" in txt or "REAL2SIM_ALIGN_V2" not in txt.splitlines():
        print(json.dumps({"ok": False, "reason": "no grasp crossing"}))
        return 4
    gx, gy, gz = map(float, m.groups())
    hand = (re.search(r"hand (\w)", txt) or [None, "?"])[1]
    band = dict(BAND, **cfg.get("band", {}))
    if not (band["x"][0] < gx < band["x"][1] and band["y"][0] < gy < band["y"][1]
            and band["z"][0] < gz < band["z"][1]):
        print(json.dumps({"ok": False, "reason": f"kin anchor off band: ({gx:.2f},{gy:.2f},{gz:.2f})",
                          "hand": hand}))
        return 5

    zone = destination["zone"]
    is_tray = "tray" in cfg                    # 托盘任务才在桌面显示托盘
    tray_xy = zone[:2]                         # 每集实测托盘位(真机帧反投影)
    bins = cfg.get("bins", {})                                # 每集实测垃圾桶位(同上)
    destination_prim_path = destination.get("prim_path", "")
    destination_collider_paths = tuple(
        destination.get("collider_paths", ()))
    destination_bounds = destination.get(
        "measured_bounds", {"min": (0.0, 0.0, 0.0), "max": (0.0, 0.0, 0.0)})
    is_task28_drawer = (
        tid == "28"
        and destination_prim_path == T28_DRAWER_BODY_PATH
        and destination_collider_paths == (
            T28_DRAWER_SIDE_COLLIDER_PATH,
            T28_DRAWER_BOTTOM_COLLIDER_PATH,
        )
    )
    is_task6_operated_drawer = bool(
        cfg.get("operated_drawer_destination")
        and destination_prim_path == T28_DRAWER_BODY_PATH
    )
    drawer_front_x = None
    drawer_wall_top_z = None
    drawer_initial_lip_m = None
    if is_task28_drawer:
        extra_match = re.search(
            r"EXTRA r@[0-9]+ midpoint world: "
            r"\(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)", txt)
        if extra_match:
            drawer_front_x = float(extra_match.group(1))
            drawer_pinch_y = float(extra_match.group(2))
            drawer_wall_top_z = float(extra_match.group(3))
            # Per-episode drawer OPENING (not station): the teleoperator
            # left the drawer pulled out by a different amount in each take
            # ("some stick out a bit"), and the pull hand always pinches the
            # outer top edge of that same front wall.  With the station now
            # solved from the real head frame, this FK reading is a direct
            # measurement of where that front face sat at frame 0.
            _closed_front = T28_DRAWER_WORLD_BOUNDS["max"][0]
            _raw_lip = drawer_front_x - _closed_front
            if not (0.005 <= _raw_lip <= 0.16):
                # Outside the physical staging range the reading is not a
                # front-face measurement (occluded pinch, wrong run); fall
                # back to the nominal teleoperator lip rather than inventing
                # a drawer pose.
                drawer_front_x = _closed_front + 0.024
            drawer_initial_lip_m = round(
                float(drawer_front_x - _closed_front), 4)
    elif is_task6_operated_drawer:
        # Benchmark 6.15 explicitly starts almost closed.  The same native
        # black drawer is dynamic on its world-X rail; the right hand must
        # clamp, pull and return it through physical contact.
        drawer_front_x = T28_DRAWER_WORLD_BOUNDS["max"][0] + 0.020
        drawer_initial_lip_m = 0.020
    table_size_block = ""
    if coffee_table_calibration is not None:
        coffee_table_world_size = coffee_table_calibration[:3]
        coffee_table_bottom_z = coffee_table_calibration[3]
        table_size_block = f"""
# Bounded physical calibration: resize the table to an exact world AABB and
# preserve its audited horizontal centre.  The canonical asset is
# rotated +90 degrees, so local X maps to world Y and local Y maps to world X.
_ct = s.GetPrimAtPath('/World/Assets/CoffeeTable')
assert _ct and _ct.IsValid(), 'missing CoffeeTable'
_ct_rotate_z = _ct.GetAttribute('xformOp:rotateZ')
assert _ct_rotate_z and abs(float(_ct_rotate_z.Get()) - 90.0) < 1e-6, \
    'CoffeeTable diagnostic size requires the canonical +90 degree rotation'
_ct_scale = _ct.GetAttribute('xformOp:scale')
assert _ct_scale and _ct_scale.Get() is not None, 'CoffeeTable has no scale'
_ct_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
_ct_box = _ct_cache.ComputeWorldBound(_ct).ComputeAlignedRange()
_ct_min, _ct_max = _ct_box.GetMin(), _ct_box.GetMax()
_ct_x = float(_ct_max[0] - _ct_min[0])
_ct_y = float(_ct_max[1] - _ct_min[1])
_ct_z = float(_ct_max[2] - _ct_min[2])
assert _ct_x > 0.0 and _ct_y > 0.0 and _ct_z > 0.0, \
    'CoffeeTable has invalid bounds'
_ct_old_scale = _ct_scale.Get()
assert _ct_scale.Set(Gf.Vec3f(
    float(_ct_old_scale[0]) * {coffee_table_world_size[1]!r} / _ct_y,
    float(_ct_old_scale[1]) * {coffee_table_world_size[0]!r} / _ct_x,
    float(_ct_old_scale[2]) * {coffee_table_world_size[2]!r} / _ct_z)), \
    'CoffeeTable scale write failed'
_ct_scaled_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
_ct_scaled = _ct_scaled_cache.ComputeWorldBound(_ct).ComputeAlignedRange()
_ct_scaled_min, _ct_scaled_max = _ct_scaled.GetMin(), _ct_scaled.GetMax()
_ct_translate = _ct.GetAttribute('xformOp:translate')
assert _ct_translate and _ct_translate.Get() is not None, \
    'CoffeeTable has no translate'
_ct_old_translate = _ct_translate.Get()
_ct_delta = (
    {COFFEE_TABLE_CENTER_XY[0]!r}
        - float(_ct_scaled_min[0] + _ct_scaled_max[0]) / 2.0,
    {COFFEE_TABLE_CENTER_XY[1]!r}
        - float(_ct_scaled_min[1] + _ct_scaled_max[1]) / 2.0,
    {coffee_table_bottom_z!r} - float(_ct_scaled_min[2]),
)
assert _ct_translate.Set(type(_ct_old_translate)(
    float(_ct_old_translate[0]) + _ct_delta[0],
    float(_ct_old_translate[1]) + _ct_delta[1],
    float(_ct_old_translate[2]) + _ct_delta[2])), \
    'CoffeeTable translate write failed'
_ct_verify_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
_ct_verify = _ct_verify_cache.ComputeWorldBound(_ct).ComputeAlignedRange()
_ct_verify_min, _ct_verify_max = _ct_verify.GetMin(), _ct_verify.GetMax()
_ct_verify_x = float(_ct_verify_max[0] - _ct_verify_min[0])
_ct_verify_y = float(_ct_verify_max[1] - _ct_verify_min[1])
_ct_verify_z = float(_ct_verify_max[2] - _ct_verify_min[2])
assert abs(_ct_verify_x - {coffee_table_world_size[0]!r}) < 0.002, \
    f'CoffeeTable world X verification failed: {{_ct_verify_x}}'
assert abs(_ct_verify_y - {coffee_table_world_size[1]!r}) < 0.002, \
    f'CoffeeTable world Y verification failed: {{_ct_verify_y}}'
assert abs(_ct_verify_z - {coffee_table_world_size[2]!r}) < 0.002, \
    f'CoffeeTable world Z verification failed: {{_ct_verify_z}}'
assert abs(float(_ct_verify_min[2]) - {coffee_table_bottom_z!r}) < 0.002, \
    f'CoffeeTable bottom verification failed: {{_ct_verify_min[2]}}'
print(
    'CoffeeTable diagnostic world size: '
    f'{{_ct_verify_x:.6f}} x {{_ct_verify_y:.6f}} x '
    f'{{_ct_verify_z:.6f}} m; bottom={{float(_ct_verify_min[2]):.6f}}')
"""
    if is_task28_drawer:
        # Wall lip exactly at the FK pinch height — A/B round 4 measured a
        # +15 mm "bite" lip to be strictly WORSE (taller wall in the pads'
        # descent path pushes the drawer shut; exemplar +0.119 -> -0.019).
        drawer_scene_block = drawer_articulation_code(
            front_face_x=drawer_front_x,
            wall_top_z=drawer_wall_top_z)
    elif is_task6_operated_drawer:
        drawer_scene_block = drawer_articulation_code(
            front_face_x=drawer_front_x,
            wall_top_z=None,
            grip_walls=True,
            fixed=False,
        )
    else:
        drawer_scene_block = drawer_reset_code()
    place = f"""from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import math
s = Usd.Stage.Open('{SCENE}')
assert s, 'could not open scene'
{SCENE_FILTER}
{table_size_block}
p = s.GetPrimAtPath('{tp}')
assert p and p.IsValid(), 'missing {tp}'
if not p.IsActive(): p.SetActive(True)
assert p.CreateAttribute(
    'physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(False), \
    'could not disable target kinematic mode'
for _nm in ('physics:velocity', 'physics:angularVelocity'):
    _a = p.GetAttribute(_nm)
    if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))
c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
lo=[1e9]*3; hi=[-1e9]*3
for pr in Usd.PrimRange(p):
    if pr.GetTypeName()!='Mesh' or not pr.IsActive(): continue
    bb=c.ComputeWorldBound(pr).ComputeAlignedRange()
    if bb.IsEmpty(): continue
    mn,mx=bb.GetMin(),bb.GetMax()
    if not all(math.isfinite(v) for v in (*mn,*mx)): continue
    for i in range(3):
        lo[i]=min(lo[i],mn[i]); hi[i]=max(hi[i],mx[i])
assert all(math.isfinite(v) for v in (*lo,*hi)) and all(lo[i] <= hi[i] for i in range(3)), \
    'target has no finite active mesh bounds'
cur=[(lo[i]+hi[i])/2 for i in range(3)]
ih_yaw = {ih_yaw}
if ih_yaw:
    rotated = False
    for op in UsdGeom.Xformable(p).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            r = op.Get()
            assert op.Set(Gf.Vec3f(r[0], r[1], r[2] + ih_yaw)), \
                'target rotate write failed'
            rotated = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            import math as _mm
            q = op.Get(); h = _mm.radians(ih_yaw)/2
            yq = type(q)(_mm.cos(h), 0, 0, _mm.sin(h))
            assert op.Set(yq * q), 'target orient write failed'
            rotated = True
    assert rotated, 'IH_YAW requested but target has no rotation op'
    # bbox 重算(旋转后)
    lo=[1e9]*3; hi=[-1e9]*3
    c2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render'], useExtentsHint=False)
    for pr in Usd.PrimRange(p):
        if pr.GetTypeName()!='Mesh' or not pr.IsActive(): continue
        bb=c2.ComputeWorldBound(pr).ComputeAlignedRange()
        if bb.IsEmpty(): continue
        mn,mx=bb.GetMin(),bb.GetMax()
        if not all(math.isfinite(v) for v in (*mn,*mx)): continue
        for i in range(3):
            lo[i]=min(lo[i],mn[i]); hi[i]=max(hi[i],mx[i])
    assert all(math.isfinite(v) for v in (*lo,*hi)) and all(lo[i] <= hi[i] for i in range(3)), \
        'rotated target has no finite active mesh bounds'
    cur=[(lo[i]+hi[i])/2 for i in range(3)]
tz = ({gz} + 0.012 - (hi[2]-lo[2])/2) if {ih_top} else {gz}
dx, dy, dz = {gx}-cur[0], {gy}-cur[1], tz-cur[2]
translated = False
for op in UsdGeom.Xformable(p).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        t = op.Get()
        assert op.Set(Gf.Vec3d(t[0]+dx, t[1]+dy, t[2]+dz)), \
            'target translate write failed'
        translated = True
assert translated, 'target has no translate op'
# Retire the old "place tables are empty" assumption.  Episode-bound visible
# context is rebuilt after this transaction; only stale generic plate/beans
# are hidden here.  Tray tasks keep the audited visible tray and align its
# physical top to the current CoffeeTable instead of retaining the old
# 0.423 m table-height transform.
for _dn in ('PaperPlate', 'CoffeeBeans'):
    _dp = s.GetPrimAtPath('/World/Assets/' + _dn)
    if _dp and _dp.IsValid() and _dp.IsActive(): _dp.SetActive(False)
q = s.GetPrimAtPath('/World/Assets/Tray')
if {is_tray}:
    assert q and q.IsValid(), 'missing tray'
    if not q.IsActive(): q.SetActive(True)
    # Geometry/collision stay on the audited japanese_tray asset.  Only its connected base-color
    # texture is overridden to match the real light cream/pink coffee-pattern tray.
    _tex = s.GetPrimAtPath(
        '/World/Assets/Tray/Obj/Materials/OPM0047/tex_base').GetAttribute('inputs:file')
    assert _tex and _tex.IsValid(), 'missing tray base-color input'
    assert _tex.Set(Sdf.AssetPath('{TRAY_TEXTURE}')), 'tray texture write failed'
    _tray_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render'],
        useExtentsHint=False)
    _tray_before = _tray_cache.ComputeWorldBound(q).ComputeAlignedRange()
    assert not _tray_before.IsEmpty(), 'tray has empty bounds'
    _tray_before_top = float(_tray_before.GetMax()[2])
    _tray_target_top = float({TRAY_VISIBLE_TOP_Z!r})
    tray_translated = False
    for op in UsdGeom.Xformable(q).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            assert op.Set(Gf.Vec3d(
                {tray_xy[0]}, {tray_xy[1]},
                float(t[2]) + _tray_target_top - _tray_before_top)), \
                'tray translate write failed'
            tray_translated = True
    assert tray_translated, 'tray has no translate op'
    _tray_verify_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render'],
        useExtentsHint=False)
    _tray_after = _tray_verify_cache.ComputeWorldBound(q).ComputeAlignedRange()
    assert not _tray_after.IsEmpty(), 'translated tray has empty bounds'
    assert abs(float(_tray_after.GetMax()[2]) - _tray_target_top) < 0.001, \
        'visible tray top does not match physical destination surface'
elif q and q.IsValid() and q.IsActive():
    q.SetActive(False)
# riser 收起(上个pick集可能留在桌上挡释放区)
rs = s.GetPrimAtPath('/World/Assets/SeatRiser')
if rs and rs.IsValid() and rs.IsActive(): rs.SetActive(False)
# 垃圾桶按本集实测位摆好(桶在 sofa-place 视角是左前景主体)
for _bn, _bxy in {bins!r}.items():
    _bp = s.GetPrimAtPath('/World/Assets/' + _bn)
    if not (_bp and _bp.IsValid()):
        continue
    if not _bp.IsActive(): _bp.SetActive(True)
    for op in UsdGeom.Xformable(_bp).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get(); op.Set(Gf.Vec3d(_bxy[0], _bxy[1], t[2]))
# small Task 28's destination is the native black storage drawer, now under
# the shared articulated-drawer contract: dynamic body on a world-X prismatic
# slide, open convexDecomposition interior, per-episode FK-derived initial
# lip.  The real pull and the bill drop both resolve through plain physics.
# Every other place task restores the closed kinematic drawer instead.
{drawer_scene_block}
assert s.GetRootLayer().Save(), 'place scene save failed'
print('{PLACE_MARKER}')
"""
    try:
        r2 = run_mesh_script(place)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({
            "ok": False,
            "target": cfg["prim"],
            "hand": hand,
            "zone": zone,
            "reason": f"place execution failed: {type(exc).__name__}: {exc}",
        }))
        return 6
    ok = (
        r2.returncode == 0
        and PLACE_MARKER in r2.stdout.splitlines()
    )
    current_secondary_context = None
    if ok and tid in CURRENT_CONTEXT_TASKS:
        context_returncode = None
        episode_uid = os.environ.get("EPISODE_UID", "")
        try:
            with open(
                Path(__file__).with_name("place_current_context_v1.json"),
                "r", encoding="utf-8",
            ) as _ctx_stream:
                _audited_context_episodes = set(
                    json.load(_ctx_stream).get("episodes", {}))
        except (OSError, ValueError):
            _audited_context_episodes = set()
        if not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}", episode_uid):
            ok = False
            current_secondary_context = {
                "ok": False,
                "reason": (
                    "current place context requires an exact EPISODE_UID"),
            }
        elif episode_uid not in _audited_context_episodes:
            # Reproduce-rate batches run every recorded episode, but only
            # audited episodes carry a frame-0 secondary-object contract.
            # Skip context reconstruction for the rest (physics of the
            # target interaction is unaffected; the missing tabletop
            # clutter is recorded so review can tell these runs apart).
            current_secondary_context = {
                "ok": True,
                "skipped": "episode has no audited frame-0 context entry",
                "episode_uid": episode_uid,
            }
        else:
            try:
                context_result = run(
                    [
                        str(MESH311_PYTHON), str(CURRENT_CONTEXT_SCRIPT),
                        "--scene", SCENE,
                        "--task-id", tid,
                        "--episode-uid", episode_uid,
                    ],
                    t=120,
                )
                context_returncode = context_result.returncode
                current_secondary_context = json.loads(
                    context_result.stdout.strip().splitlines()[-1])
            except (
                OSError,
                subprocess.TimeoutExpired,
                json.JSONDecodeError,
                IndexError,
            ) as exc:
                current_secondary_context = {
                    "ok": False,
                    "reason": (
                        "current place context failed: "
                        f"{type(exc).__name__}: {exc}"),
                }
            if (
                context_returncode != 0
                or current_secondary_context.get("ok") is not True
            ):
                ok = False
    # Keep the family initializer independently usable through
    # make_compare3.py --init-cmd.  init_task.py adds the same uniform fields,
    # but callers that invoke init_place.py directly must not silently lose the
    # task/target identity and fall back to a non-physical playback.
    out = {"ok": ok, "task": tid, "family": "place",
           "target": cfg["prim"], "track_prim": tp,
           "hand": hand, "zone": zone, "destination": destination,
           "kin": [round(gx, 3), round(gy, 3), round(gz, 3)],
           "base": list(BASE), "yaw": round(YAW, 1),
           "up_axis": cfg.get("up_axis"),
           "seat_mid": cfg.get("seat_mid"),
           "seat_long": cfg.get("seat_long", 0.0),
           "grasp_long_plane_rotation_deg": cfg.get(
               "grasp_long_plane_rotation_deg"
           ),
           "approach_offset_xyz": cfg.get("approach_offset_xyz"),
           "grip_width": grip_width,
           "release_retreat": cfg.get("release_retreat"),
           "tray_texture": TRAY_TEXTURE if is_tray else None,
           "asset_id": asset.asset_id,
           "asset_status": asset.status,
           "asset_revision": asset.asset_revision,
           "current_secondary_context": current_secondary_context,
           "drawer": (
               {"front_x": drawer_front_x,
                "wall_top_z": drawer_wall_top_z,
                "initial_lip_m": drawer_initial_lip_m,
                "articulated": (
                    is_task28_drawer or is_task6_operated_drawer),
                "fixed_collision_backed": False}
               if (is_task28_drawer or is_task6_operated_drawer) else None),
           "coffee_table_calibration": (
               list(coffee_table_calibration)
               if coffee_table_calibration is not None else None)}
    if not ok:
        out["reason"] = (
            "place/current-context failed: "
            + (r2.stdout + r2.stderr)[-1600:]
            + " "
            + json.dumps(
                current_secondary_context, ensure_ascii=False)[:800]
        )
    print(json.dumps(out))
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())

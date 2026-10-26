#!/usr/bin/env python3
"""Success checker for pick(book)-class tasks — layered for sim and real.

Why layered: the real logs (raw_joints.h5) contain NO object poses, and the obvious proprio
proxy is DEAD on this hardware — measured on episode 1505fe08, the omnipicker's effector STATE
saturates to 1.000 (fully closed) even while holding the 18 mm book, because the adaptive
linkage absorbs the stroke. So:

  SIM  (object ground truth)  -> objective predicate, implemented here, thresholds grounded in
                                 tonight's measured success/failure signatures.
  REAL (cameras only)         -> the SAME semantic checklist evaluated on head/wrist frames
                                 (VLM/detector or human label); hook stub provided. Do NOT use
                                 gripper stall on the omnipicker.

Sim predicate for PICK, all four must hold:
  1. GRIP     commanded grip >= 0.9 somewhere (the data actually tried to grasp);
  2. LIFT     a window >= hold_s where book_z - support_z >= min_lift with grip closed
              throughout (success measured +73 mm sustained; settle noise <= 18 mm one-off;
              bulldozed failures never left 0 -- min_lift 40 mm splits these cleanly);
  3. ATTACH   during that window the book tracks the gripper: corr(tip_z, book_z) >= 0.6 and
              std(tip_z - book_z) <= 40 mm. Recalibrated 2026-07-13 on 21 genuine 40 cm carries
              (in-hand pivot gives ~20 mm std, corr 0.99); the fake-perch failure this exists to
              catch measured std 123 mm / corr -0.29 — still rejected by an order of magnitude;
  4. NO-BULLDOZE  before the first grip close, book xy displacement <= 80 mm (success run
              slid 49 mm during approach; bulldozed runs 104-105 mm. Scene-specific margin —
              re-tune if the prop or approach changes).

PLACE requires a complete physical sequence, not merely a plausible final pose: bilateral
pad contact while closed, an attached carry, a sustained open/contact-free release with
object-tip separation, and a stable final pose on the target surface.

    conda run -n base python real2sim/teleop/check_success.py            # run on pd_metrics.npz
    conda run -n base python real2sim/teleop/check_success.py --selftest # + bulldoze regression
"""
import argparse
import os
import math
import re

import numpy as np

from dual_trash_bin_layout import (
    BIN_OUTER_DIMENSIONS_XYZ_M,
    RED_CENTER_XY_M,
    TEAL_CENTER_XY_M,
)
import group_shift
from robot_positions import (
    COFFEE_TABLE_CURRENT_AABB_MAX_XYZ_M,
    COFFEE_TABLE_CURRENT_AABB_MIN_XYZ_M,
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_CURRENT_SIZE_XYZ_M,
    COFFEE_TABLE_CURRENT_TOP_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    SHELF_CURRENT_CENTER_XY_M,
    SHELF_LAYOUT_SHIFT_XY_M,
    robot_position,
)

from machine_paths import DEMO_ROOT as _DEMO_ROOT
METRICS = str(_DEMO_ROOT / "cam_preview" / "playback" / "pd_metrics.npz")
FPS = 20.0

# Every task in the current rerender queue whose scene is governed by the
# shared ShelvingUnit/CoffeeTable transaction.  These sets deliberately use
# robot-template small-task IDs, not benchmark family IDs.
CURRENT_SHELF_SUCCESS_TASKS = frozenset({
    "6", "10", "21", "22", "35", "36", "38", "39", "47",
})
STAGED_NEW_SNACK_PLACE_SUCCESS_TASKS = frozenset({
    "86", "87", "88", "89",
})
CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS = frozenset({
    "16", "17", "18", "19", "20", "27", "28",
    "42", "43", "44", "45", "46",
}) | STAGED_NEW_SNACK_PLACE_SUCCESS_TASKS
CURRENT_COFFEE_TABLE_PICK_SUCCESS_TASKS = frozenset({
    "55", "56", "57", "78",
})
CURRENT_COFFEE_TABLE_THROW_SUCCESS_TASKS = frozenset({"29", "30"})
CURRENT_COFFEE_TABLE_DIAGNOSTIC_TASKS = frozenset({"52"})
CURRENT_COFFEE_TABLE_SUCCESS_TASKS = frozenset().union(
    CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS,
    CURRENT_COFFEE_TABLE_PICK_SUCCESS_TASKS,
    CURRENT_COFFEE_TABLE_THROW_SUCCESS_TASKS,
    CURRENT_COFFEE_TABLE_DIAGNOSTIC_TASKS,
)
CURRENT_LAYOUT_SUCCESS_TASKS = frozenset().union(
    CURRENT_SHELF_SUCCESS_TASKS,
    CURRENT_COFFEE_TABLE_SUCCESS_TASKS,
)
CURRENT_LAYOUT_TRACE_SCHEMA = "real2sim.current_scene_layout_trace.v1"


def _quat_wxyz_for_yaw(yaw_deg):
    half = math.radians(float(yaw_deg)) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _current_task_robot_pose(task_id):
    """Return the exact current robot pose consumed by one small task."""
    tid = str(task_id)
    if tid in CURRENT_SHELF_SUCCESS_TASKS:
        station = robot_position("shelf_pick")
        return station.xy_m, station.yaw_deg, station.key
    if tid in CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS:
        import importlib
        cfg = importlib.import_module("init_place").TASKS[tid]
        return tuple(cfg["base"]), float(cfg["yaw"]), "task_calibrated_place"
    if tid in {"55", "56", "57", "78"}:
        station = robot_position("table_pick")
        return station.xy_m, station.yaw_deg, station.key
    if tid == "29":
        # table_pick plus the reviewed bin-follow delta that keeps the
        # recorded releases aligned with the basket-adjacent teal bin.
        group_shift.require_identity("small Task 29 (throw, fixed bins)")
        import importlib
        task29 = importlib.import_module("task29_full_reproduce")
        return tuple(task29.ROBOT_BASE_XY), 180.0, "task29_release_aligned"
    if tid == "30":
        group_shift.require_identity("small Task 30 (throw, fixed bins)")
        import importlib
        task30 = importlib.import_module("task30_throw_reproduce")
        return tuple(task30.TASK30_BASE_XY_M), 180.0, "task30_reviewed"
    if tid == "52":
        group_shift.require_identity("small Task 52 (Newton diagnostic)")
        return (
            (-2.6, COFFEE_TABLE_CURRENT_CENTER_XY_M[1]),
            0.0,
            "task52_newton_diagnostic",
        )
    raise KeyError(f"small Task {tid} is not in the current layout scope")


def current_layout_success_contract(task_id):
    """Machine-readable success/layout contract for one current small task.

    This is the scoring-side view of the current global scene.  Furniture
    values come only from ``robot_positions.py``; place destinations come
    only from ``init_place.destination_for_task``.
    """
    tid = str(task_id)
    if tid not in CURRENT_LAYOUT_SUCCESS_TASKS:
        raise KeyError(f"small Task {tid} is not in the current layout scope")
    robot_xy, robot_yaw, robot_source = _current_task_robot_pose(tid)
    if tid in CURRENT_SHELF_SUCCESS_TASKS:
        furniture = {
            "kind": "shelf",
            "body_path": "/World/Assets/ShelvingUnit",
            "center_xy_m": tuple(SHELF_CURRENT_CENTER_XY_M),
            "layout_shift_xy_m": tuple(SHELF_LAYOUT_SHIFT_XY_M),
        }
    else:
        furniture = {
            "kind": "coffee_table",
            "body_path": "/World/Assets/CoffeeTable",
            "center_xy_m": tuple(COFFEE_TABLE_CURRENT_CENTER_XY_M),
            "size_xyz_m": tuple(COFFEE_TABLE_CURRENT_SIZE_XYZ_M),
            "aabb_min_xyz_m": tuple(
                COFFEE_TABLE_CURRENT_AABB_MIN_XYZ_M),
            "aabb_max_xyz_m": tuple(
                COFFEE_TABLE_CURRENT_AABB_MAX_XYZ_M),
            "top_m": float(COFFEE_TABLE_CURRENT_TOP_M),
            "layout_shift_xy_m": tuple(COFFEE_TABLE_LAYOUT_SHIFT_XY_M),
        }
    contract = {
        "schema": "real2sim.current_layout_success_contract.v1",
        "small_task_id": tid,
        "strict_success_allowed": (
            tid not in CURRENT_COFFEE_TABLE_DIAGNOSTIC_TASKS),
        "furniture": furniture,
        "robot": {
            "source": robot_source,
            "base_xy_m": tuple(robot_xy),
            "yaw_deg": float(robot_yaw),
            "orient_wxyz": _quat_wxyz_for_yaw(robot_yaw),
        },
    }
    if not group_shift.is_identity():
        contract["group_shift_xy_m"] = tuple(group_shift.GROUP_SHIFT_XY_M)
    if tid in CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS:
        contract["destination"] = _place_destination(tid)
    if tid in CURRENT_COFFEE_TABLE_THROW_SUCCESS_TASKS:
        contract["bins"] = {
            "teal_center_xy_m": tuple(TEAL_CENTER_XY_M),
            "red_center_xy_m": tuple(RED_CENTER_XY_M),
            "outer_dimensions_xyz_m": tuple(
                BIN_OUTER_DIMENSIONS_XYZ_M),
            "expected": "teal" if tid == "29" else "red",
        }
    return contract


def current_layout_trace_errors(
    task_id, trace, *, expected_furniture_center_xy_m=None,
):
    """Validate live scene measurements embedded in a current pick trace."""
    tid = str(task_id)
    contract = current_layout_success_contract(tid)
    errors = []

    def text_value(key):
        try:
            raw = np.asarray(trace[key])
            if raw.shape != () or raw.dtype.kind != "U":
                raise ValueError
            return str(raw.item())
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key} must be one Unicode scalar")
            return None

    def numeric_value(key, shape):
        try:
            raw = np.asarray(trace[key])
            if raw.shape != shape or raw.dtype.kind not in "fiu":
                raise ValueError
            value = np.asarray(raw, dtype=float)
            if not np.isfinite(value).all():
                raise ValueError
            return value
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{key} must be one finite numeric array with shape {shape}")
            return None

    schema = text_value("current_scene_layout_schema")
    if schema is not None and schema != CURRENT_LAYOUT_TRACE_SCHEMA:
        errors.append(
            f"current_scene_layout_schema {schema!r} != "
            f"{CURRENT_LAYOUT_TRACE_SCHEMA!r}")
    family = text_value("current_scene_layout_family")
    expected_family = contract["furniture"]["kind"]
    if family is not None and family != expected_family:
        errors.append(
            f"current_scene_layout_family {family!r} != "
            f"{expected_family!r}")
    body_path = text_value("current_furniture_body_path")
    expected_furniture = contract["furniture"]
    if (
        body_path is not None
        and body_path != expected_furniture["body_path"]
    ):
        errors.append(
            f"current_furniture_body_path {body_path!r} != "
            f"{expected_furniture['body_path']!r}")
    actual_min = numeric_value(
        "current_furniture_aabb_min_xyz_m", (3,))
    actual_max = numeric_value(
        "current_furniture_aabb_max_xyz_m", (3,))
    if actual_min is not None and actual_max is not None:
        if np.any(actual_max <= actual_min):
            errors.append("current furniture AABB has no positive width")
        elif expected_furniture["kind"] == "shelf":
            actual_center_xy = 0.5 * (actual_min[:2] + actual_max[:2])
            expected_center = np.asarray(
                expected_furniture["center_xy_m"]
                if expected_furniture_center_xy_m is None
                else expected_furniture_center_xy_m,
                dtype=float,
            )
            if expected_center.shape != (2,) or not np.isfinite(
                expected_center).all():
                errors.append(
                    "expected furniture center must be one finite XY point")
            elif not np.allclose(
                actual_center_xy,
                expected_center,
                atol=0.002,
                rtol=0.0,
            ):
                errors.append(
                    "ShelvingUnit AABB center does not match the current "
                    "global shelf layout")
        elif (
            not np.allclose(
                actual_min,
                np.asarray(
                    expected_furniture["aabb_min_xyz_m"], dtype=float),
                atol=0.002,
                rtol=0.0,
            )
            or not np.allclose(
                actual_max,
                np.asarray(
                    expected_furniture["aabb_max_xyz_m"], dtype=float),
                atol=0.002,
                rtol=0.0,
            )
        ):
            errors.append(
                "CoffeeTable AABB does not match the current global "
                "0.8 x 1.2 x 0.45 m layout")

    try:
        collision_count_raw = np.asarray(
            trace["current_furniture_collision_path_count"])
        if (
            collision_count_raw.shape != ()
            or collision_count_raw.dtype.kind not in "iu"
        ):
            raise ValueError
        collision_count = int(collision_count_raw.item())
    except (KeyError, TypeError, ValueError):
        errors.append(
            "current_furniture_collision_path_count must be one "
            "integer")
    else:
        if collision_count < 1:
            errors.append("current furniture has no enabled collision")
    try:
        visible_raw = np.asarray(trace["current_furniture_visible"])
        visible = (
            visible_raw.shape == ()
            and visible_raw.dtype.kind == "b"
            and bool(visible_raw.item())
        )
    except (KeyError, TypeError, ValueError):
        visible = False
    if not visible:
        errors.append("current furniture is not directly visible")

    # Robot base/yaw are intentionally not layout success gates.  Task-owned
    # producers must perturb the base and bind the applied state/action in
    # their own evidence; requiring the historical calibrated station here
    # made that contract impossible.  The shared checker continues to enforce
    # the furniture, collision, visibility, destination and task semantics.

    if tid in CURRENT_COFFEE_TABLE_THROW_SUCCESS_TASKS:
        for role in ("teal", "red"):
            actual = numeric_value(
                f"current_{role}_bin_center_xy_m", (2,))
            expected = np.asarray(
                contract["bins"][f"{role}_center_xy_m"], dtype=float)
            if (
                actual is not None
                and not np.allclose(
                    actual, expected, atol=1e-8, rtol=0.0)
            ):
                errors.append(
                    f"{role} bin center does not match the current global "
                    "dual-bin layout")
    return errors


DISTRACTOR_AUDIT_SCHEMA = "real2sim.distractor_audit.v1"


def distractor_audit_errors(
        audit, *, task_id=None, episode_uid=None, expected_prims=None):
    """Validate one runtime no-touch audit without trusting summary booleans."""
    expected = (
        None if expected_prims is None
        else sorted(str(path) for path in expected_prims)
    )
    if expected is not None and len(expected) != len(set(expected)):
        return ["expected distractor prims must be unique"]
    if audit is None:
        return (
            ["missing distractor no-touch audit"]
            if expected
            else []
        )
    if not hasattr(audit, "get"):
        return ["distractor audit must be a mapping"]

    errors = []
    if audit.get("schema") != DISTRACTOR_AUDIT_SCHEMA:
        errors.append(
            f"distractor audit schema {audit.get('schema')!r} != "
            f"{DISTRACTOR_AUDIT_SCHEMA!r}")
    if task_id is not None and str(audit.get("task_id")) != str(task_id):
        errors.append(
            f"distractor audit task_id {audit.get('task_id')!r} != "
            f"{str(task_id)!r}")
    if (
        episode_uid is not None
        and str(audit.get("episode_uid")) != str(episode_uid)
    ):
        errors.append(
            f"distractor audit episode_uid {audit.get('episode_uid')!r} != "
            f"{str(episode_uid)!r}")

    prims_raw = audit.get("prims")
    prims = (
        [str(path) for path in prims_raw]
        if isinstance(prims_raw, list)
        else None
    )
    if (
        prims is None
        or not prims
        or len(prims) != len(set(prims))
        or prims != sorted(prims)
    ):
        errors.append(
            "distractor audit prims must be a non-empty sorted unique list")
    elif expected is not None and prims != expected:
        errors.append(
            "distractor audit prims do not match episode randomization")

    try:
        threshold = float(audit["disturbed_threshold_mm"])
        displacement = np.asarray(audit["displacement_mm"], dtype=float)
        maximum = float(audit["max_displacement_mm"])
    except (KeyError, TypeError, ValueError):
        errors.append(
            "distractor audit displacement fields are missing or invalid")
    else:
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or displacement.ndim != 1
            or prims is None
            or len(displacement) != len(prims)
            or not np.isfinite(displacement).all()
            or np.any(displacement < 0.0)
            or not math.isfinite(maximum)
        ):
            errors.append(
                "distractor audit displacement fields are inconsistent")
        else:
            calculated_maximum = float(np.max(displacement))
            if not math.isclose(
                    maximum, calculated_maximum, rel_tol=0.0, abs_tol=0.011):
                errors.append(
                    "distractor audit max_displacement_mm is inconsistent")
            calculated_disturbed = [
                path for path, value in zip(prims, displacement)
                if float(value) > threshold
            ]
            if audit.get("disturbed_prims") != calculated_disturbed:
                errors.append(
                    "distractor audit disturbed_prims is inconsistent")
            if audit.get("untouched") != (not calculated_disturbed):
                errors.append(
                    "distractor audit untouched flag is inconsistent")
            if calculated_disturbed:
                errors.append(
                    "distractor no-touch threshold exceeded for: "
                    + ", ".join(calculated_disturbed))
    return errors


# ---------------------------------------------------------------------------
# Per-task success config. The base predicate below was calibrated on the sofa
# BOOK; different props carry differently, so each task declares its object kind
# and any threshold overrides. Kinds:
#   rigid  — box / tube / can / bottle: the object moves rigidly with the tip, so
#            the standard ATTACH (corr>=0.6 & std<=40mm, or the pivot/firmly path)
#            is the right fake-perch guard.
#   bag    — soft pouch/package (Skittles, OREO): the tracked origin SAGS as the
#            bag deforms in the grip, so a genuine carry shows a larger tip-book
#            std even though it is held. Lift+hold stays the primary signal; ATTACH
#            std is relaxed (a bag can't "perch", so the guard is less load-bearing).
# object dims/type from build_tier2.py; grasp mechanism from the scene colliders.
TASK_CONFIG = {
    # —— 老批 pick(成熟组) ——
    "6":  dict(obj="orange",       kind="rigid", note="货架橙色运动饮料"),
    "8":  dict(obj="coke",         kind="rigid", note="可乐罐, 圆柱; 半悬板缘家族"),
    "9":  dict(obj="blue",         kind="rigid", note="货架蓝色运动饮料"),
    "10": dict(obj="sprite",       kind="rigid", note="货架雪碧"),
    "11": dict(obj="Note",         kind="rigid", note="便签本, flat 捏取"),
    "12": dict(obj="Book",         kind="rigid", note="沙发拿书(左手批)"),
    "13": dict(obj="Book",         kind="rigid", note="沙发拿书(右手批)"),
    "21": dict(obj="sprite",       kind="rigid", note="货架雪碧（新录制）"),
    "22": dict(obj="blue",         kind="rigid", note="货架蓝色运动饮料"),
    "23": dict(obj="orange",       kind="rigid", note="橙汁瓶"),
    # —— tier2 货架 ——
    "35": dict(obj="chomps_r",     kind="bag",   note="beef-sticks pouch (build_tier2 BAG), grip_proxy"),
    "36": dict(obj="pring_bbq",    kind="rigid", note="Pringles tube, col0 body + col1 rim"),
    "37": dict(obj="honeymaid",    kind="rigid", note="Honey Maid box ('pack'), grip_proxy"),
    "38": dict(obj="skit_littles", kind="bag",   note="Skittles Littles closed pouch, exact 150x50x200 mm collider"),
    "39": dict(obj="zbar_t",       kind="rigid", note="Zbar box, grip_proxy"),
    "40": dict(obj="oreo_gf",      kind="bag",   note="OREO package, soft (资产已移除)"),
    # —— 新批 pick (27-60, init_pick.TASKS 同源) ——
    "31": dict(obj="Remote",       kind="rigid", note="沙发遥控器, 光滑薄板(滑脱尾巴)"),
    "32": dict(obj="BaseballCap",  kind="rigid", note="棒球帽, 凸包圆顶(真帽是软布)"),
    "33": dict(obj="PaperCup",     kind="rigid", note="条纹纸杯"),
    "53": dict(obj="Wallet",       kind="rigid", note="沙发钱包；旧 GlassesCase 输出不可沿用"),
    "54": dict(obj="Umbrella",     kind="rigid", note="折叠伞, 圆柱"),
    "55": dict(obj="Wallet",       kind="rigid", note="长桌钱包；旧 GlassesCase 输出不可沿用"),
    "56": dict(obj="Remote",       kind="rigid", note="长桌遥控器"),
    "57": dict(obj="GlassesCase",  kind="rigid", note="黑款未建模, 人字纹代理"),
    "58": dict(obj="Remote",       kind="rigid", note="TV柜遥控器"),
    "59": dict(obj="GlassesCase",  kind="rigid", note="小茶几墨镜盒"),
    "60": dict(obj="Wallet",       kind="rigid", note="小茶几钱包；旧 GlassesCase 输出不可沿用"),
    "61": dict(obj="Remote",       kind="rigid", note="茶几遥控器"),
    "78": dict(obj="RedBullPink",  kind="rigid", note="茶几空易拉罐"),
    "90": dict(obj="Umbrella",     kind="rigid", note="木质边桌折叠伞 (2026-07-31 批)"),
    "91": dict(obj="GlassesCase",  kind="rigid", note="沙发座面墨镜盒 (2026-07-31 批, benchmark 4.1)"),
    "92": dict(obj="Wallet",       kind="rigid", note="茶几抽屉钱包 (2026-07-31 批): 右手拉开 black_box_2 抽屉, 左手取出"),
    "405": dict(obj="GlassesCase", kind="rigid", note="边桌墨镜盒 (benchmark 4.3)"),
}
# per-kind threshold overrides (only what differs from the rigid/book defaults)
KIND_THRESHOLDS = {
    "rigid": {},
    "bag":   dict(attach_std=0.080),   # allow deformation sag; lift+hold still gates the carry
}


def thresholds_for_task(task_id):
    """kwargs to pass check_pick_sim for this task (empty -> book/rigid defaults)."""
    cfg = TASK_CONFIG.get(str(task_id))
    if not cfg:
        return {}
    return dict(KIND_THRESHOLDS.get(cfg["kind"], {}))


def check_pick_sim(book_z, book_xy, tip, lgrip, fps=FPS,
                   min_lift=0.040, hold_s=1.0, attach_corr=0.6, attach_std=0.040,
                   bulldoze_xy=0.080, grip_thresh=0.9,
                   distractors=None, distractor_max=0.080, sim_grip=None):
    """Returns dict(success, reasons, margins). Arrays are per-frame, same length.

    distractors: optional {name: (T,3) world positions} of TASK-IRRELEVANT objects (e.g. the
    Note). Condition 5: none may move more than distractor_max over the whole episode — the
    benchmark's tasks require identifying WHICH object to pick, so disturbing the others is a
    failure even when the target is carried."""
    out = {"success": False, "reasons": [], "margins": {}}
    raw_evidence = {
        "book_z": book_z,
        "book_xy": book_xy,
        "tip": tip,
        "lgrip": lgrip,
    }
    converted = {}
    for name, value in raw_evidence.items():
        if value is None:
            out["reasons"].append(
                "invalid legacy pick evidence: missing " + name)
            return out
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError):
            out["reasons"].append(
                f"invalid legacy pick evidence: {name} is not an array")
            return out
        if raw.dtype.kind not in "fiu":
            out["reasons"].append(
                f"invalid legacy pick evidence: {name} must use a real numeric dtype")
            return out
        converted[name] = np.asarray(raw, dtype=float)
    book_z = converted["book_z"]
    book_xy = converted["book_xy"]
    tip = converted["tip"]
    lgrip = converted["lgrip"]
    if book_z.ndim != 1:
        out["reasons"].append(
            f"invalid legacy pick evidence: book_z shape {book_z.shape} != (T,)")
        return out
    n = len(book_z)
    expected_shapes = {
        "book_xy": (n, 2),
        "tip": (n, 3),
        "lgrip": (n,),
    }
    for name, expected in expected_shapes.items():
        if converted[name].shape != expected:
            out["reasons"].append(
                f"invalid legacy pick evidence: {name} shape "
                f"{converted[name].shape} != {expected}")
            return out
    if n < 2:
        out["reasons"].append(
            "invalid legacy pick evidence: trajectory must contain at least two samples")
        return out
    if not all(np.isfinite(value).all() for value in converted.values()):
        out["reasons"].append(
            "invalid legacy pick evidence: arrays contain NaN or inf")
        return out
    if sim_grip is not None:
        try:
            raw_sim_grip = np.asarray(sim_grip)
            if (raw_sim_grip.dtype.kind not in "fiu"
                    or raw_sim_grip.shape != (n,)):
                raise ValueError
            sim_grip = np.asarray(raw_sim_grip, dtype=float)
            if not np.isfinite(sim_grip).all():
                raise ValueError
        except (TypeError, ValueError):
            out["reasons"].append(
                "invalid legacy pick evidence: sim_grip must be one finite (T,) array")
            return out
    normalized_distractors = None
    if distractors is not None:
        if not hasattr(distractors, "items"):
            out["reasons"].append(
                "invalid legacy pick evidence: distractors must be a mapping")
            return out
        normalized_distractors = {}
        for name, trajectory in distractors.items():
            try:
                raw_trajectory = np.asarray(trajectory)
                if (raw_trajectory.dtype.kind not in "fiu"
                        or raw_trajectory.shape != (n, 3)):
                    raise ValueError
                normalized = np.asarray(raw_trajectory, dtype=float)
                if not np.isfinite(normalized).all():
                    raise ValueError
            except (TypeError, ValueError):
                out["reasons"].append(
                    "invalid legacy pick evidence: distractor "
                    f"{str(name)!r} must be one finite (T,3) array")
                return out
            normalized_distractors[str(name)] = normalized
    distractors = normalized_distractors
    threshold_values = {
        "fps": fps,
        "min_lift": min_lift,
        "hold_s": hold_s,
        "attach_corr": attach_corr,
        "attach_std": attach_std,
        "bulldoze_xy": bulldoze_xy,
        "grip_thresh": grip_thresh,
        "distractor_max": distractor_max,
    }
    try:
        threshold_values = {
            name: float(np.asarray(value))
            for name, value in threshold_values.items()
            if np.asarray(value).shape == ()
            and np.asarray(value).dtype.kind in "fiu"
        }
    except (TypeError, ValueError):
        threshold_values = {}
    if (len(threshold_values) != 8
            or not np.isfinite(tuple(threshold_values.values())).all()
            or threshold_values["fps"] <= 0
            or threshold_values["hold_s"] <= 0
            or threshold_values["min_lift"] < 0
            or threshold_values["attach_std"] < 0
            or threshold_values["bulldoze_xy"] < 0
            or threshold_values["distractor_max"] < 0
            or not -1 <= threshold_values["attach_corr"] <= 1
            or not 0 <= threshold_values["grip_thresh"] <= 1):
        out["reasons"].append(
            "invalid legacy pick thresholds: expected finite scalar values")
        return out
    fps = threshold_values["fps"]
    min_lift = threshold_values["min_lift"]
    hold_s = threshold_values["hold_s"]
    attach_corr = threshold_values["attach_corr"]
    attach_std = threshold_values["attach_std"]
    bulldoze_xy = threshold_values["bulldoze_xy"]
    grip_thresh = threshold_values["grip_thresh"]
    distractor_max = threshold_values["distractor_max"]
    tip_z = tip[:, 2]
    hold_n = max(2, int(hold_s * fps))

    support_z = float(np.median(book_z[: max(5, int(0.5 * fps))]))
    # 抓取门控: 优先用 SIM 手指真实闭合(sim_grip), 而非录制命令(lgrip=agrip)——
    # 后者让 policy 输出 grip=1 就无条件过。审计 BLOCKER。
    _grip_sig = sim_grip if sim_grip is not None else lgrip
    out['margins']['grip_source'] = 'sim' if sim_grip is not None else 'command'
    closed = _grip_sig >= grip_thresh
    if not closed.any():
        out["reasons"].append("no grasp commanded (grip never >= %.1f)" % grip_thresh)
        return out
    t_close = int(np.argmax(closed))

    # 4. no-bulldoze before the grasp
    pre_slide = float(np.max(np.linalg.norm(book_xy[:t_close + 1] - book_xy[0], axis=1))) if t_close > 0 else 0.0
    out["margins"]["pre_grasp_slide_mm"] = pre_slide * 1000
    if pre_slide > bulldoze_xy:
        out["reasons"].append(f"bulldozed: pre-grasp slide {pre_slide*1000:.0f} mm > {bulldoze_xy*1000:.0f}")
        return out

    # 2+3. lifted-and-attached window
    lift = book_z - support_z
    ok_frames = (lift >= min_lift) & closed
    best = cur = start = 0
    best_win = None
    for i, f in enumerate(ok_frames):
        if f:
            cur += 1
            if cur > best:
                best, best_win = cur, (i - cur + 1, i + 1)
        else:
            cur = 0
    out["margins"]["max_lift_mm"] = float(lift.max()) * 1000
    out["margins"]["longest_hold_s"] = best / fps
    if best < hold_n:
        out["reasons"].append(f"no sustained lift: longest {best/fps:.2f} s < {hold_s:.2f} s "
                              f"(max lift {lift.max()*1000:.0f} mm)")
        return out

    w0, w1 = best_win
    bz, tz = book_z[w0:w1], tip_z[w0:w1]
    corr = float(np.corrcoef(bz, tz)[0, 1]) if bz.std() > 1e-6 and tz.std() > 1e-6 else 1.0
    dstd = float(np.std(tz - bz))
    out["margins"]["attach_corr"] = corr
    out["margins"]["attach_std_mm"] = dstd * 1000
    # attached if EITHER: high vertical coupling (corr) with modest gap std, OR a very tight
    # gap std on its own — a rigidly held object that PIVOTS as the tip rises (box tilted out
    # of the shelf) anti-correlates in z (corr<0) yet tracks the tip rigidly (tiny std). std is
    # the real rigid-coupling measure; corr only means anything when there's vertical travel.
    firmly = (corr >= 0.9 and dstd <= 0.060)   # 真刚性耦合: 原标定60mm(真实摩擦下物体指间微晃30-50mm正常); 删除了
    # 'or dstd<=0.025' 自动过(搁置/楔取无相关也能通过)——审计标为 hack 向量
    if (corr < attach_corr or dstd > attach_std) and not firmly:
        out["reasons"].append(f"not attached: corr {corr:.2f} (need {attach_corr}) "
                              f"std {dstd*1000:.0f} mm (max {attach_std*1000:.0f})")
        return out

    # 5. task-irrelevant objects undisturbed
    if distractors:
        for name, tr in distractors.items():
            tr = np.asarray(tr, float)
            if tr.ndim == 2 and len(tr) > 1:
                dmax = float(np.max(np.linalg.norm(tr[:, :2] - tr[0, :2], axis=1)))
                out["margins"][f"{name}_move_mm"] = round(dmax * 1000, 1)
                if dmax > distractor_max:
                    out["reasons"].append(
                        f"distractor {name} disturbed {dmax*1000:.0f} mm > {distractor_max*1000:.0f} mm")
                    out["margins"][f"{name}_disturbed"] = 1.0
        if any(k.endswith("_disturbed") for k in out["margins"]):
            return out

    out["success"] = True
    out["reasons"].append(f"lifted {lift[w0:w1].max()*1000:.0f} mm for {best/fps:.2f} s, "
                          f"attached (corr {corr:.2f}, std {dstd*1000:.1f} mm), "
                          f"pre-grasp slide {pre_slide*1000:.0f} mm")
    return out


def check_pick_real_stub(head_frames, wrist_frames):
    """Real-side hook: same checklist on camera frames. NOT proprio — the omnipicker's gripper
    state saturates closed while holding (measured), so vision or human label is the only
    trustworthy real signal. Suggested VLM checks per sampled frame pair:
      pre:  'book resting on the seat/support'  ->  post-grasp: 'book held in gripper, clear of
      the support'  ->  end (pick): 'book still in gripper' / (place): 'book resting at target'.
    Returns None here; wire to a VLM judge (routes verifier) when needed."""
    raise NotImplementedError


def _selftest():
    """Regression fixtures reconstructed from tonight's measured runs."""
    n, fps = 234, 20.0
    t = np.arange(n)
    # SUCCESS shape (pdm5): slide 49mm pre-grasp, close at 111, lift to 73mm until ~150, release
    lg = (t >= 111).astype(float)
    bz = np.full(n, 0.464)
    bz[112:150] = 0.464 + np.minimum((t[112:150] - 112) * 0.01, 0.073)
    bz[150:] = 0.464
    tz = bz + 0.06 + np.random.default_rng(0).normal(0, 0.002, n)   # tip rides 6cm above, tight
    tip = np.stack([np.zeros(n), np.zeros(n), tz], 1)
    xy = np.zeros((n, 2)); xy[60:111, 0] = np.linspace(0, 0.049, 51); xy[111:, 0] = 0.049
    r = check_pick_sim(bz, xy, tip, lg, fps)
    assert r["success"], f"success fixture failed: {r}"
    print("selftest SUCCESS fixture -> PASS:", r["reasons"][0])
    # FAILURE shape (pdm3 bulldoze): slide 105mm, no lift
    xyf = np.zeros((n, 2)); xyf[60:111, 0] = np.linspace(0, 0.105, 51); xyf[111:, 0] = 0.105
    bzf = np.full(n, 0.464)
    r = check_pick_sim(bzf, xyf, tip, lg, fps)
    assert not r["success"], "bulldoze fixture wrongly passed"
    print("selftest BULLDOZE fixture -> correctly FAILED:", r["reasons"][0])
    # FAILURE shape: closed on air above the book (no lift, no slide)
    r = check_pick_sim(bzf, np.zeros((n, 2)), tip, lg, fps)
    assert not r["success"]
    print("selftest AIR-GRASP fixture -> correctly FAILED:", r["reasons"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=METRICS)
    ap.add_argument("--task", default=None, help="task_id -> 走 check_task 家族分发; 缺省按 pick/book 判")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        _selftest_families()
    if not os.path.isfile(a.metrics):
        if a.selftest:
            return 0
        print(f"(no metrics at {a.metrics})")
        return 3
    try:
        with np.load(a.metrics, allow_pickle=False) as m:
            _dis = {
                k.rsplit("_move", 1)[0]: m[k]
                for k in m.files if k.endswith("_move")
            } if any(k.endswith("_move") for k in m.files) else None
            if a.task:
                r = check_task(a.task, m, distractors=_dis)
            else:
                required = ("book_z", "book_xy", "tip", "lgrip")
                missing = [key for key in required if key not in m.files]
                if missing:
                    print(
                        "corrupt metrics: missing " + ", ".join(missing))
                    return 4
                r = check_pick_sim(
                    m["book_z"], m["book_xy"], m["tip"], m["lgrip"],
                    distractors=_dis, sim_grip=sim_grip_from_metrics(m))
    except (OSError, ValueError, EOFError) as exc:
        print(f"corrupt metrics at {a.metrics}: {type(exc).__name__}: {exc}")
        return 4
    print(f"\n{a.metrics}")
    print("SUCCESS" if r["success"] else "FAILURE")
    for k, v in r["margins"].items():
        print(f"  {k:22s} {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    for s in r["reasons"]:
        print(f"  -> {s}")
    # A valid benchmark FAIL is an outcome, not an execution failure.
    return 0


# bins: (cx, cy, half_x, half_y)。small Task 29/30 的目标桶与干扰桶都必须
# 使用 dual_trash_bin_layout.py 的当前全局双桶坐标；teal28 仅对应 legacy
# small Task 28 初始化器，不能参与当前 small Task 28 抽屉判分。
# 2026-08-05 defect fix: the props are 0.19 x 0.24 m, but both axes used the
# X half-width, so the accepted footprint was a 190 mm SQUARE — 25 mm
# narrower than the bin on each Y side.  An object resting on the bin floor
# 113 mm from centre along Y (inside the authored 120 mm half-depth, and
# visibly inside the composed collision mesh) was reported "not inside any
# bin footprint".  Both half-extents now come from the same authored
# dimensions; the explicit ``bin_half`` argument still overrides both axes.
BINS = {
    "teal": (
        *TEAL_CENTER_XY_M,
        BIN_OUTER_DIMENSIONS_XYZ_M[0] / 2.0,
        BIN_OUTER_DIMENSIONS_XYZ_M[1] / 2.0,
    ),
    "red": (
        *RED_CENTER_XY_M,
        BIN_OUTER_DIMENSIONS_XYZ_M[0] / 2.0,
        BIN_OUTER_DIMENSIONS_XYZ_M[1] / 2.0,
    ),
    "teal28": (-2.26, -1.06, 0.13, 0.13),
    "teal30": (
        *TEAL_CENTER_XY_M,
        BIN_OUTER_DIMENSIONS_XYZ_M[0] / 2.0,
        BIN_OUTER_DIMENSIONS_XYZ_M[1] / 2.0,
    ),
}
# The red and teal props share the audited rigid bottom used by the strict
# task-28 place contract. Moving a bin in x/y does not change this z plane.
BIN_BOTTOM_Z = {
    name: 0.018021 for name in BINS
}
BIN_BODY_PATH = {
    "red": "/World/Assets/TrashBinRed",
    "teal": "/World/Assets/TrashBinTeal",
    "teal28": "/World/Assets/TrashBinTeal",
    "teal30": "/World/Assets/TrashBinTeal",
}

def check_throw_sim(obj_z, obj_xy, fps=FPS, bin_half=None, rim_z=0.30,
                    settle_s=1.0, bins=None, support_z=None,
                    bin_bottom_tol=0.002):
    """Throw-into-trash gate: the object ends INSIDE a bin (xy within a bin footprint,
    z below the rim) and is settled (z/xy drift < 2 mm over the last settle_s).
    bin_half: 显式传入则全局覆盖各桶半宽; 默认 None = 用 BINS 里的每桶实测半宽。
    bins: 本集**在场**桶名单(list)。桶会被 init 按任务挪位, BINS 里其它条目是别的任务的
    桶位=幽灵桶——落在幽灵桶脚下的地板会被误判入桶(t30-r3 实锤假阳性)。None=全部(旧行为)。"""
    import numpy as np
    try:
        obj_z = np.asarray(obj_z, dtype=float)
        obj_xy = np.asarray(obj_xy, dtype=float)
        fps = float(fps)
        settle_s = float(settle_s)
        rim_z = float(rim_z)
        bin_bottom_tol = float(bin_bottom_tol)
        if bin_half is not None:
            bin_half = float(bin_half)
    except (TypeError, ValueError):
        return {
            "success": False,
            "reasons": ["invalid throw final-pose evidence: expected numeric arrays"],
            "margins": {"bin": None},
        }
    if (obj_z.ndim != 1 or obj_xy.shape != (len(obj_z), 2)
            or len(obj_z) < 1
            or not np.isfinite(obj_z).all()
            or not np.isfinite(obj_xy).all()
            or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(settle_s) or settle_s <= 0
            or not np.isfinite(rim_z)
            or not np.isfinite(bin_bottom_tol) or bin_bottom_tol < 0
            or (bin_half is not None
                and (not np.isfinite(bin_half) or bin_half <= 0))):
        return {
            "success": False,
            "reasons": [
                "invalid throw final-pose evidence: shape, finiteness, "
                "fps or settle window"],
            "margins": {"bin": None},
        }
    if support_z is None:
        # Legacy playback has only the tracked rigid-body z. It remains useful
        # as a conservative lower-bound diagnostic, but a strict v1 sidecar
        # must provide audited bbox support_z explicitly.
        support = obj_z
        support_source = "object-origin-legacy-fallback"
    else:
        try:
            support_raw = np.asarray(support_z)
            if support_raw.dtype.kind not in "fiu":
                raise ValueError
            support = np.asarray(support_raw, dtype=float)
        except (TypeError, ValueError):
            support = np.asarray([])
        if support.shape != obj_z.shape or not np.isfinite(support).all():
            return {
                "success": False,
                "reasons": [
                    "invalid throw support evidence: support_z must be one "
                    "finite (T,) array matching obj_z"],
                "margins": {"bin": None},
            }
        support_source = "bbox-support"
    n = max(1, int(settle_s * fps))
    fz = obj_z[-n:]
    fxy_window = obj_xy[-n:]
    fxy = fxy_window[-1]
    which = None
    for name, (bx, by, bhx, bhy) in BINS.items():
        if bins is not None and name not in bins:
            continue
        hx = bin_half if bin_half is not None else bhx
        hy = bin_half if bin_half is not None else bhy
        if abs(fxy[0] - bx) <= hx and abs(fxy[1] - by) <= hy:
            which = name; break
    bottom_z = BIN_BOTTOM_Z.get(which) if which is not None else None
    above_bottom = (
        bottom_z is not None
        and float(support[-1]) >= bottom_z - bin_bottom_tol)
    inside = (
        which is not None
        and float(fz[-1]) < rim_z
        and above_bottom)
    z_drift = float(fz.max() - fz.min())
    xy_drift = float(np.max(np.linalg.norm(fxy_window - fxy, axis=1)))
    settled = z_drift < 0.002 and xy_drift < 0.002
    reasons = []
    if which is None:
        reasons.append(f"final xy ({fxy[0]:.3f},{fxy[1]:.3f}) not inside any bin footprint")
    elif float(fz[-1]) >= rim_z:
        reasons.append(f"final z {float(fz[-1]):.3f} above rim {rim_z}")
    elif not above_bottom:
        reasons.append(
            f"final support z {float(support[-1]):.3f} below audited "
            f"{which} bin bottom {bottom_z:.3f} "
            f"(tolerance {bin_bottom_tol:.3f})")
    if not settled:
        reasons.append(
            f"not settled: z drift {z_drift*1000:.0f} mm, "
            f"xy drift {xy_drift*1000:.0f} mm in last {settle_s}s")
    return {"success": bool(inside and settled), "reasons": reasons,
            "margins": {"bin": which, "final_z": round(float(fz[-1]), 3),
                        "final_support_z": round(float(support[-1]), 3),
                        "support_source": support_source,
                        "bin_bottom_z": (
                            round(float(bottom_z), 6)
                            if bottom_z is not None else None),
                        "destination_body_path": BIN_BODY_PATH.get(which),
                        "final_xy": [round(float(fxy[0]), 3), round(float(fxy[1]), 3)],
                        "settle_z_drift_m": round(z_drift, 5),
                        "settle_xy_drift_m": round(xy_drift, 5)}}


THROW_METRICS_V1_FIELDS = {
    "schema", "task_id", "episode_uid", "run_id", "target_body_path",
    "active_hand", "pad_body_paths",
    "obj_xyz", "support_z", "tip", "grip_joint_closure", "pad_contacts",
    "pad_force_vectors", "contact_force_threshold_n", "contact_available",
    "fps", "sample_time_s",
}
THROW_METRICS_V1_TASK29_DESTINATION_FIELDS = {
    "destination_contract_schema",
    "destination_body_path",
    "destination_collider_path",
    "destination_aabb_min",
    "destination_aabb_max",
    "destination_aabb_center",
    "destination_aabb_size",
    "destination_bottom_z",
    "destination_rim_z",
    "destination_bounds_kind",
    "destination_interior_geometry_asserted",
    "destination_scoring_polygon_asserted",
    "object_aabb_min",
    "object_aabb_max",
}
THROW_METRICS_V1_TASK29_PHASE_FIELDS = {
    "task29_execution_schema",
    "task29_phase_ordinal",
    "task29_phase_cup_id",
    "task29_source_start_frame",
    "task29_source_stop_frame_exclusive",
    "task29_replay_sample_count",
    "task29_settle_policy",
    "task29_settle_source_frame",
    "task29_settle_sample_count",
    "source_frame",
}
THROW_METRICS_V1_TASK29_RELEASE_MATERIAL_FIELDS = {
    "task29_release_material_schema",
    "task29_release_material_path",
    "task29_release_material_bound_collision_paths",
    "task29_release_material_static_friction",
    "task29_release_material_dynamic_friction",
    "task29_release_material_restitution",
    "task29_release_material_friction_combine_mode",
    "task29_release_material_compliant_contact",
    "task29_release_material_physics_scene_path",
    "task29_release_material_scene_ccd_required",
    "task29_release_material_target_ccd_enabled",
    "task29_release_material_landing_collision_policy",
    "task29_release_material_binding_scope",
    "task29_release_material_activation_policy",
    "task29_release_material_release_time_rebinding_forbidden",
    "task29_release_material_velocity_override_forbidden",
}
THROW_METRICS_V1_TRUSTED_EXPECTATIONS = (
    "expected_episode_uid",
    "expected_run_id",
    "expected_pad_body_paths",
)
THROW_METRICS_V1_TASK29_TRUSTED_EXPECTATIONS = ()
TASK29_DESTINATION_AABB_TOLERANCE_M = 2e-6
# The upright 89 mm cup physically stalls the 109 mm gripper well before the
# zero-angle reference.  A measured 0.30 closure is the frozen Task29
# object-scale threshold; uninterrupted bilateral contact through manifest
# release_frame remains separately mandatory.
TASK29_GRIP_CLOSED_THRESHOLD = 0.30


def _canonical_throw_pad_paths(active_hand):
    """Return the exact pad bodies allowed for one throw-v1 hand."""
    if active_hand not in ("l", "r"):
        return ()
    return (
        f"/World/Robot/gripper_{active_hand}_outer_link4",
        f"/World/Robot/gripper_{active_hand}_inner_link4",
    )


def validate_throw_metrics_v1(
        m, task_id, target_body_path, *, expected_episode_uid=None,
        expected_run_id=None, expected_pad_body_paths=None,
        expected_destination_contract=None,
        expected_task29_phase0_execution=None,
        expected_task29_release_material_schema=None,
        **_ignored_legacy_identity):
    """Validate one self-described throw-v1 artifact.

    The legacy ``playback_pd`` artifact only contains object pose, tip pose and
    joint closure.  It is intentionally insufficient here: strict publication
    additionally needs two-pad force vectors from the same run.  This validator
    checks the artifact's structure and compares any supplied trusted external
    identities.  Callers must independently require *all* trusted identities
    before promoting a valid sequence to strict or Task-29 partial-v1 evidence.
    """
    try:
        keys = set(m.files if hasattr(m, "files") else m.keys())
    except (AttributeError, TypeError):
        return ["throw metrics must be dict-like"]
    required_fields = set(THROW_METRICS_V1_FIELDS)
    if str(task_id) == "29":
        required_fields.update(
            THROW_METRICS_V1_TASK29_DESTINATION_FIELDS)
        required_fields.update(
            THROW_METRICS_V1_TASK29_PHASE_FIELDS)
        required_fields.update(
            THROW_METRICS_V1_TASK29_RELEASE_MATERIAL_FIELDS)
    missing = sorted(required_fields - keys)
    if missing:
        return ["throw metrics v1 keys missing: " + ", ".join(missing)]
    errors = []

    def text_scalar(key):
        try:
            raw = np.asarray(m[key])
            if raw.shape != () or raw.dtype.kind != "U":
                raise ValueError
            value = str(raw.item())
            if not _safe_identity_text(value):
                raise ValueError
            return value
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{key} must be one non-empty printable Unicode scalar")
            return ""

    schema = text_scalar("schema")
    metric_task = text_scalar("task_id")
    episode_uid = text_scalar("episode_uid")
    run_id = text_scalar("run_id")
    metric_target = text_scalar("target_body_path")
    active_hand = text_scalar("active_hand")
    if schema and schema != "real2sim.throw_metrics.v1":
        errors.append(f"unexpected throw metrics schema {schema!r}")
    if metric_task and metric_task != str(task_id):
        errors.append(
            f"sidecar task_id {metric_task!r} does not match requested task "
            f"{str(task_id)!r}")
    if metric_target and metric_target != str(target_body_path):
        errors.append(
            f"sidecar target_body_path {metric_target!r} does not match task "
            f"target {str(target_body_path)!r}")
    if episode_uid and not re.fullmatch(r"[A-Za-z0-9._:-]+", episode_uid):
        errors.append("episode_uid contains unsafe characters")
    if run_id and not re.fullmatch(r"[A-Za-z0-9._:-]+", run_id):
        errors.append("run_id contains unsafe characters")
    if active_hand and active_hand not in ("l", "r"):
        errors.append("active_hand must be exactly 'l' or 'r'")
    for name, actual, expected in (
        ("episode_uid", episode_uid, expected_episode_uid),
        ("run_id", run_id, expected_run_id),
    ):
        if expected is not None and actual and actual != str(expected):
            errors.append(
                f"sidecar {name} {actual!r} does not match expected "
                f"{str(expected)!r}")
    try:
        pad_paths_raw = np.asarray(m["pad_body_paths"])
        if pad_paths_raw.dtype.kind != "U" or pad_paths_raw.shape != (2,):
            raise ValueError
        pad_paths = tuple(str(value) for value in pad_paths_raw)
        if not all(_safe_identity_text(path) for path in pad_paths):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        pad_paths = ()
        errors.append(
            "pad_body_paths must contain two printable Unicode paths")
    canonical_pad_paths = _canonical_throw_pad_paths(active_hand)
    if pad_paths and canonical_pad_paths and pad_paths != canonical_pad_paths:
        errors.append(
            f"pad_body_paths {pad_paths!r} do not match active_hand "
            f"{active_hand!r} canonical paths {canonical_pad_paths!r}")
    if expected_pad_body_paths is not None:
        try:
            expected_paths = tuple(
                str(value) for value in expected_pad_body_paths)
        except TypeError:
            expected_paths = ()
        if len(expected_paths) != 2:
            errors.append(
                "expected_pad_body_paths must contain exactly two paths")
        elif pad_paths and pad_paths != expected_paths:
            errors.append(
                f"sidecar pad_body_paths {pad_paths!r} does not match expected "
                f"{expected_paths!r}")

    task29_contract = None
    task29_contract_module = None
    task29_phase_execution = None
    task29_phase_scalar_values = {}
    if str(task_id) == "29":
        import task29_destination_contract as task29_contract_module
        import task29_phase_plan as task29_phase_module
        import task29_release_material as task29_release_material_module

        release_contract = task29_release_material_module.CONTRACT
        release_text = {
            key: text_scalar(key)
            for key in (
                "task29_release_material_schema",
                "task29_release_material_path",
                "task29_release_material_friction_combine_mode",
                "task29_release_material_physics_scene_path",
                "task29_release_material_landing_collision_policy",
                "task29_release_material_binding_scope",
                "task29_release_material_activation_policy",
            )
        }
        expected_release_text = {
            "task29_release_material_schema":
                release_contract.schema,
            "task29_release_material_path":
                release_contract.material_path,
            "task29_release_material_friction_combine_mode":
                release_contract.friction_combine_mode,
            "task29_release_material_physics_scene_path":
                release_contract.physics_scene_path,
            "task29_release_material_landing_collision_policy":
                release_contract.landing_collision_policy,
            "task29_release_material_binding_scope":
                release_contract.binding_scope,
            "task29_release_material_activation_policy":
                release_contract.activation_policy,
        }
        for key, expected_value in expected_release_text.items():
            actual = release_text[key]
            if actual and actual != expected_value:
                errors.append(
                    f"{key} does not match the frozen Task29 release "
                    "material contract")
        for key, expected_value in (
            (
                "task29_release_material_static_friction",
                release_contract.static_friction,
            ),
            (
                "task29_release_material_dynamic_friction",
                release_contract.dynamic_friction,
            ),
            (
                "task29_release_material_restitution",
                release_contract.restitution,
            ),
        ):
            try:
                raw = np.asarray(m[key])
                if raw.shape != () or raw.dtype.kind not in "fiu":
                    raise ValueError
                value = float(raw)
                if (
                    not math.isfinite(value)
                    or not math.isclose(
                        value, expected_value,
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                ):
                    errors.append(
                        f"{key} does not match the frozen Task29 release "
                        "material contract")
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"{key} must be one finite numeric scalar")

        for key, expected_value in (
            (
                "task29_release_material_compliant_contact",
                release_contract.compliant_contact,
            ),
            (
                "task29_release_material_scene_ccd_required",
                release_contract.scene_ccd_required,
            ),
            (
                "task29_release_material_target_ccd_enabled",
                release_contract.target_ccd_enabled,
            ),
            (
                "task29_release_material_release_time_rebinding_forbidden",
                release_contract.release_time_rebinding_forbidden,
            ),
            (
                "task29_release_material_velocity_override_forbidden",
                release_contract.velocity_override_forbidden,
            ),
        ):
            try:
                raw = np.asarray(m[key])
                if (
                    raw.shape != ()
                    or raw.dtype.kind != "b"
                    or bool(raw) is not bool(expected_value)
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"{key} must be the frozen explicit boolean "
                    f"{bool(expected_value)!r}")

        try:
            raw_paths = np.asarray(
                m["task29_release_material_bound_collision_paths"])
            if raw_paths.ndim != 1 or raw_paths.dtype.kind != "U":
                raise ValueError
            release_collision_paths = tuple(
                str(value) for value in raw_paths.tolist())
            if (
                len(release_collision_paths) < 3
                or len(set(release_collision_paths))
                != len(release_collision_paths)
                or not all(
                    _safe_identity_text(path)
                    for path in release_collision_paths
                )
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            release_collision_paths = ()
            errors.append(
                "task29_release_material_bound_collision_paths must "
                "contain unique target collisions followed by two pad "
                "collisions")
        if release_collision_paths:
            target_release_paths = release_collision_paths[:-2]
            pad_release_paths = release_collision_paths[-2:]
            target_prefix = release_contract.target_body_path + "/"
            if not target_release_paths or not all(
                path == release_contract.target_body_path
                or path.startswith(target_prefix)
                for path in target_release_paths
            ):
                errors.append(
                    "Task29 release material target collision paths escape "
                    "the canonical PaperCup subtree")
            expected_release_pad_bodies = (
                pad_paths if len(pad_paths) == 2 else canonical_pad_paths)
            if (
                len(expected_release_pad_bodies) != 2
                or any(
                    path != body_path
                    and not path.startswith(body_path + "/")
                    for path, body_path in zip(
                        pad_release_paths,
                        expected_release_pad_bodies,
                    )
                )
            ):
                errors.append(
                    "Task29 release material pad collision paths do not "
                    "match the active hand's canonical link4 bodies")

        if expected_task29_release_material_schema is not None:
            expected_release_schema = str(
                expected_task29_release_material_schema)
            if expected_release_schema != release_contract.schema:
                errors.append(
                    "expected_task29_release_material_schema does not "
                    "match the frozen Task29 contract")
            elif (
                release_text["task29_release_material_schema"]
                and release_text["task29_release_material_schema"]
                != expected_release_schema
            ):
                errors.append(
                    "sidecar task29_release_material_schema does not match "
                    "the externally expected schema")
        destination_schema = text_scalar(
            "destination_contract_schema")
        destination_body_path = text_scalar(
            "destination_body_path")
        destination_collider_path = text_scalar(
            "destination_collider_path")
        destination_bounds_kind = text_scalar(
            "destination_bounds_kind")
        if (destination_schema
                and destination_schema
                != task29_contract_module.CONTRACT_SCHEMA):
            errors.append(
                "destination_contract_schema does not identify the "
                "Task 29 destination contract")
        if (destination_body_path
                and destination_body_path
                != task29_contract_module.DESTINATION_BODY_PATH):
            errors.append(
                "destination_body_path is not the canonical Task 29 "
                "destination body")
        if (destination_collider_path
                and destination_collider_path
                != task29_contract_module.ENABLED_COLLIDER_PATH):
            errors.append(
                "destination_collider_path is not the canonical enabled "
                "Task 29 collider")
        if (destination_bounds_kind
                and destination_bounds_kind
                != task29_contract_module.BOUNDS_KIND):
            errors.append(
                "destination_bounds_kind is not the enabled-collider "
                "world AABB diagnostic")

        for key in (
                "destination_interior_geometry_asserted",
                "destination_scoring_polygon_asserted"):
            try:
                raw = np.asarray(m[key])
                if raw.shape != () or raw.dtype.kind != "b":
                    raise ValueError
                if bool(raw):
                    errors.append(
                        f"{key} must remain explicit false for Task 29")
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key} must be one explicit boolean")

        if expected_destination_contract is not None:
            if not isinstance(
                    expected_destination_contract,
                    task29_contract_module.Task29DestinationContract):
                errors.append(
                    "expected_destination_contract must be a normalized "
                    "Task29DestinationContract")
            else:
                task29_contract = expected_destination_contract
                if task29_contract is not None:
                    if (
                            task29_contract.schema
                            != task29_contract_module.CONTRACT_SCHEMA
                            or task29_contract.task_id != "29"
                            or task29_contract.body_path
                            != task29_contract_module.DESTINATION_BODY_PATH
                            or task29_contract.collider_path
                            != task29_contract_module.ENABLED_COLLIDER_PATH
                            or task29_contract.geometry_descriptor.bounds_kind
                            != task29_contract_module.BOUNDS_KIND
                            or task29_contract.geometry_descriptor
                            .interior_geometry_asserted is not False
                            or task29_contract.geometry_descriptor
                            .scoring_polygon_asserted is not False):
                        errors.append(
                            "expected_destination_contract is not the "
                            "canonical provisional Task 29 contract")
                    expected_text = {
                        "destination_contract_schema":
                            task29_contract.schema,
                        "destination_body_path":
                            task29_contract.body_path,
                        "destination_collider_path":
                            task29_contract.collider_path,
                        "destination_bounds_kind":
                            task29_contract.geometry_descriptor.bounds_kind,
                    }
                    actual_text = {
                        "destination_contract_schema": destination_schema,
                        "destination_body_path": destination_body_path,
                        "destination_collider_path":
                            destination_collider_path,
                        "destination_bounds_kind": destination_bounds_kind,
                    }
                    for key, expected_value in expected_text.items():
                        if (actual_text[key]
                                and actual_text[key] != expected_value):
                            errors.append(
                                f"sidecar {key} does not match expected "
                                "destination contract")
        try:
            raw_phase_ordinal = np.asarray(
                m["task29_phase_ordinal"])
            if (
                raw_phase_ordinal.shape != ()
                or raw_phase_ordinal.dtype.kind not in "iu"
            ):
                raise ValueError
            phase_ordinal = int(raw_phase_ordinal)
            task29_phase_scalar_values[
                "task29_phase_ordinal"] = phase_ordinal
            frozen_execution = task29_phase_module.phase_plan(
                phase_ordinal)
        except (
                KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(
                "task29_phase_ordinal must select phase 0, 1, or 2")
            phase_ordinal = 0
            frozen_execution = task29_phase_module.phase_plan(0)
        phase_text = {
            key: text_scalar(key)
            for key in (
                "task29_execution_schema",
                "task29_phase_cup_id",
                "task29_settle_policy",
            )
        }
        expected_phase_text = {
            "task29_execution_schema":
                "real2sim.task29_phase_plan.v1",
            "task29_phase_cup_id": frozen_execution.cup_id,
            "task29_settle_policy": task29_phase_module.SETTLE_POLICY,
        }
        for key, expected_value in expected_phase_text.items():
            actual = phase_text[key]
            if (
                key == "task29_execution_schema"
                and actual in {
                    "real2sim.task29_phase0_execution.v1",
                    "real2sim.task29_phase_execution.v1",
                }
            ):
                continue
            if actual and actual != expected_value:
                errors.append(
                    f"{key} does not match the frozen Task29 phase "
                    "execution")

        for key in (
                "task29_source_start_frame",
                "task29_source_stop_frame_exclusive",
                "task29_replay_sample_count",
                "task29_settle_source_frame",
                "task29_settle_sample_count"):
            try:
                raw = np.asarray(m[key])
                if raw.shape != () or raw.dtype.kind not in "iu":
                    raise ValueError
                task29_phase_scalar_values[key] = int(raw)
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"{key} must be one integer scalar")
        expected_phase_scalars = {
            "task29_phase_ordinal": frozen_execution.ordinal,
            "task29_source_start_frame":
                frozen_execution.source_start_frame,
            "task29_source_stop_frame_exclusive":
                frozen_execution.source_stop_frame_exclusive,
            "task29_replay_sample_count":
                frozen_execution.replay_sample_count,
            "task29_settle_source_frame":
                frozen_execution.settle_source_frame,
            "task29_settle_sample_count":
                frozen_execution.settle_sample_count,
        }
        for key, expected_value in expected_phase_scalars.items():
            if (
                key in task29_phase_scalar_values
                and task29_phase_scalar_values[key] != expected_value
            ):
                errors.append(
                    f"{key} does not match the frozen Task29 phase "
                    "execution")

        if expected_task29_phase0_execution is not None:
            task29_phase_execution = expected_task29_phase0_execution
            projection = (
                getattr(
                    task29_phase_execution,
                    "ordinal",
                    getattr(task29_phase_execution, "phase_ordinal", None),
                ),
                getattr(
                    task29_phase_execution,
                    "cup_id",
                    getattr(task29_phase_execution, "phase_cup_id", None),
                ),
                getattr(task29_phase_execution,
                        "source_start_frame", None),
                getattr(task29_phase_execution,
                        "source_stop_frame_exclusive", None),
                getattr(task29_phase_execution, "close_frame", None),
                getattr(task29_phase_execution,
                        "strong_close_frame", None),
                getattr(task29_phase_execution, "release_frame", None),
                getattr(task29_phase_execution,
                        "fully_open_frame", None),
            )
            expected_projection = (
                frozen_execution.ordinal,
                frozen_execution.cup_id,
                frozen_execution.source_start_frame,
                frozen_execution.source_stop_frame_exclusive,
                frozen_execution.close_frame,
                frozen_execution.strong_close_frame,
                frozen_execution.release_frame,
                frozen_execution.fully_open_frame,
            )
            if projection != expected_projection:
                errors.append(
                    "expected Task29 phase plan does not match the "
                    "sidecar phase ordinal")

    def array(key):
        try:
            return np.asarray(m[key])
        except (TypeError, ValueError):
            errors.append(f"{key} cannot be converted to an array")
            return np.asarray([])

    obj = array("obj_xyz")
    total = len(obj) if obj.ndim else 0
    shaped = {
        "obj_xyz": (obj, (total, 3)),
        "support_z": (array("support_z"), (total,)),
        "tip": (array("tip"), (total, 3)),
        "grip_joint_closure": (
            array("grip_joint_closure"), (total,)),
        "pad_contacts": (array("pad_contacts"), (total, 2)),
        "pad_force_vectors": (
            array("pad_force_vectors"), (total, 2, 3)),
        "sample_time_s": (array("sample_time_s"), (total,)),
    }
    if total < 2:
        errors.append("obj_xyz must contain at least two samples")
    for key, (raw, expected_shape) in shaped.items():
        if raw.shape != expected_shape:
            errors.append(f"{key} shape {raw.shape} != {expected_shape}")
        allowed_kinds = "buif" if key == "pad_contacts" else "fiu"
        if raw.dtype.kind not in allowed_kinds:
            errors.append(f"{key} must use a real numeric dtype")
        elif not np.isfinite(raw).all():
            errors.append(f"{key} contains NaN or inf")

    if str(task_id) == "29":
        import task29_phase_plan as task29_phase_module

        frozen_execution = task29_phase_module.phase_plan(
            task29_phase_scalar_values.get("task29_phase_ordinal", 0))
        source_frame = array("source_frame")
        expected_source_frame = frozen_execution.source_frames()
        if source_frame.shape != (total,):
            errors.append(
                f"source_frame shape {source_frame.shape} != ({total},)")
        if source_frame.dtype.kind not in "iu":
            errors.append("source_frame must use an integer dtype")
        elif source_frame.shape == (total,):
            source_frame_i64 = source_frame
            if total != frozen_execution.total_sample_count:
                errors.append(
                    "Task29 total sample count does not match phase replay "
                    "plus fixed settle tail")
            if not np.array_equal(
                    source_frame_i64, expected_source_frame):
                errors.append(
                    "source_frame does not match the frozen phase replay "
                    "and terminal settle mapping")
            if np.any(
                    (source_frame_i64
                     < frozen_execution.source_start_frame)
                    | (source_frame_i64
                       >= frozen_execution
                       .source_stop_frame_exclusive)):
                errors.append(
                    "source_frame escapes the frozen Task29 phase range")
            settle_count = frozen_execution.settle_sample_count
            if (
                total < settle_count
                or not np.all(
                    source_frame_i64[-settle_count:]
                    == frozen_execution.settle_source_frame)
            ):
                errors.append(
                    "Task29 terminal settle window is not wholly bound to "
                    "the frozen phase terminal frame")

        destination_arrays = {
            "destination_aabb_min": array("destination_aabb_min"),
            "destination_aabb_max": array("destination_aabb_max"),
            "destination_aabb_center": array(
                "destination_aabb_center"),
            "destination_aabb_size": array("destination_aabb_size"),
        }
        for key, raw in destination_arrays.items():
            if raw.shape != (3,):
                errors.append(f"{key} shape {raw.shape} != (3,)")
            if raw.dtype.kind not in "fiu":
                errors.append(f"{key} must use a real numeric dtype")
            elif not np.isfinite(raw).all():
                errors.append(f"{key} contains NaN or inf")

        def finite_destination_scalar(key):
            try:
                raw = np.asarray(m[key])
                if raw.shape != () or raw.dtype.kind not in "fiu":
                    raise ValueError
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError
                return value
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"{key} must be one finite real scalar")
                return float("nan")

        destination_bottom_z = finite_destination_scalar(
            "destination_bottom_z")
        destination_rim_z = finite_destination_scalar(
            "destination_rim_z")
        destination_min = destination_arrays[
            "destination_aabb_min"]
        destination_max = destination_arrays[
            "destination_aabb_max"]
        destination_center = destination_arrays[
            "destination_aabb_center"]
        destination_size = destination_arrays[
            "destination_aabb_size"]
        destination_arrays_valid = all(
            raw.shape == (3,)
            and raw.dtype.kind in "fiu"
            and np.isfinite(raw).all()
            for raw in destination_arrays.values()
        )
        if destination_arrays_valid:
            if np.any(destination_min >= destination_max):
                errors.append(
                    "destination enabled-collider world AABB has no "
                    "positive width")
            recomputed_center = (
                destination_min.astype(float)
                + destination_max.astype(float)) / 2.0
            recomputed_size = (
                destination_max.astype(float)
                - destination_min.astype(float))
            if not np.allclose(
                    destination_center, recomputed_center,
                    atol=TASK29_DESTINATION_AABB_TOLERANCE_M, rtol=0):
                errors.append(
                    "destination_aabb_center is inconsistent with "
                    "destination_aabb_min/max")
            if not np.allclose(
                    destination_size, recomputed_size,
                    atol=TASK29_DESTINATION_AABB_TOLERANCE_M, rtol=0):
                errors.append(
                    "destination_aabb_size is inconsistent with "
                    "destination_aabb_min/max")
            if (np.isfinite(destination_bottom_z)
                    and not np.isclose(
                        destination_bottom_z,
                        float(destination_min[2]),
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0)):
                errors.append(
                    "destination_bottom_z is inconsistent with "
                    "destination_aabb_min")
            if (np.isfinite(destination_rim_z)
                    and not np.isclose(
                        destination_rim_z,
                        float(destination_max[2]),
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0)):
                errors.append(
                    "destination_rim_z is inconsistent with "
                    "destination_aabb_max")

        object_aabb_min = array("object_aabb_min")
        object_aabb_max = array("object_aabb_max")
        for key, raw in (
                ("object_aabb_min", object_aabb_min),
                ("object_aabb_max", object_aabb_max)):
            if raw.shape != (total, 3):
                errors.append(
                    f"{key} shape {raw.shape} != ({total}, 3)")
            if raw.dtype.kind not in "fiu":
                errors.append(f"{key} must use a real numeric dtype")
            elif not np.isfinite(raw).all():
                errors.append(f"{key} contains NaN or inf")
        object_aabbs_valid = all(
            raw.shape == (total, 3)
            and raw.dtype.kind in "fiu"
            and np.isfinite(raw).all()
            for raw in (object_aabb_min, object_aabb_max)
        )
        if object_aabbs_valid:
            if np.any(object_aabb_min >= object_aabb_max):
                errors.append(
                    "object live world AABB has no positive width")
            support = shaped["support_z"][0]
            if (support.shape == (total,)
                    and support.dtype.kind in "fiu"
                    and np.isfinite(support).all()
                    and not np.allclose(
                        support, object_aabb_min[:, 2],
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0)):
                errors.append(
                    "support_z is inconsistent with object_aabb_min[:,2]")

        if task29_contract is not None and destination_arrays_valid:
            expected_aabb = task29_contract.world_aabb
            expected_arrays = {
                "destination_aabb_min": expected_aabb.minimum,
                "destination_aabb_max": expected_aabb.maximum,
                "destination_aabb_center": expected_aabb.center,
                "destination_aabb_size": expected_aabb.size,
            }
            for key, expected_value in expected_arrays.items():
                if not np.allclose(
                        destination_arrays[key],
                        np.asarray(expected_value, dtype=float),
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0):
                    errors.append(
                        f"sidecar {key} does not match expected "
                        "destination contract")
            if (np.isfinite(destination_bottom_z)
                    and not np.isclose(
                        destination_bottom_z,
                        expected_aabb.bottom_z,
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0)):
                errors.append(
                    "sidecar destination_bottom_z does not match expected "
                    "destination contract")
            if (np.isfinite(destination_rim_z)
                    and not np.isclose(
                        destination_rim_z,
                        expected_aabb.rim_z,
                        atol=TASK29_DESTINATION_AABB_TOLERANCE_M,
                        rtol=0)):
                errors.append(
                    "sidecar destination_rim_z does not match expected "
                    "destination contract")

    try:
        available = np.asarray(m["contact_available"])
        if available.shape != () or available.dtype.kind != "b":
            raise ValueError
        if not bool(available):
            errors.append(
                "contact_available is false; raw two-pad contact is required")
    except (TypeError, ValueError):
        errors.append("contact_available must be one explicit boolean")

    def finite_positive_scalar(key):
        try:
            raw = np.asarray(m[key])
            if raw.shape != () or raw.dtype.kind not in "fiu":
                raise ValueError
            value = float(raw)
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            return value
        except (TypeError, ValueError):
            errors.append(f"{key} must be one finite positive real scalar")
            return float("nan")

    fps = finite_positive_scalar("fps")
    force_threshold = finite_positive_scalar("contact_force_threshold_n")
    grip = shaped["grip_joint_closure"][0]
    contacts = shaped["pad_contacts"][0]
    forces = shaped["pad_force_vectors"][0]
    times = shaped["sample_time_s"][0]
    if (grip.shape == (total,) and grip.dtype.kind in "fiu"
            and np.isfinite(grip).all()
            and np.any((grip < 0) | (grip > 1))):
        errors.append("grip_joint_closure must be normalized to [0,1]")
    if (contacts.shape == (total, 2) and contacts.dtype.kind in "buif"
            and np.isfinite(contacts).all()
            and np.any((contacts < 0) | (contacts > 1))):
        errors.append("pad_contacts must be contact confidence in [0,1]")
    if (forces.shape == (total, 2, 3)
            and contacts.shape == (total, 2)
            and forces.dtype.kind in "fiu"
            and contacts.dtype.kind in "buif"
            and np.isfinite(forces).all()
            and np.isfinite(contacts).all()
            and np.isfinite(force_threshold)):
        expected_contacts = (
            np.linalg.norm(forces, axis=2) >= force_threshold).astype(float)
        if not np.array_equal(contacts.astype(float), expected_contacts):
            errors.append(
                "pad_contacts is inconsistent with raw force vectors/contact threshold")
    if (times.shape == (total,) and times.dtype.kind in "fiu"
            and np.isfinite(times).all() and total >= 2):
        delta = np.diff(times.astype(float))
        if not np.isclose(float(times[0]), 0.0, atol=1e-9, rtol=0):
            errors.append("sample_time_s must start at zero")
        if np.any(delta <= 0):
            errors.append("sample_time_s must be strictly increasing")
        elif np.isfinite(fps) and not np.allclose(
                delta, 1.0 / fps, atol=1e-6, rtol=1e-4):
            errors.append("sample_time_s cadence does not match fps")
    return errors


def check_task29_outer_aabb_containment_diagnostic(
        destination_aabb_min=None, destination_aabb_max=None,
        destination_bottom_z=None, destination_rim_z=None,
        object_aabb_min=None, object_aabb_max=None, fps=FPS,
        settle_s=1.0, bottom_tol=0.002, rim_clearance=0.002,
        settle_drift_tol=0.002, destination_body_path=None,
        destination_collider_path=None,
        **_ignored_legacy_identity):
    """Conservative Task 29 terminal diagnostic over live world AABBs.

    This predicate deliberately does *not* claim that the destination collider
    AABB is the bin interior or a benchmark scoring polygon.  It only asks
    whether the final live object AABB is horizontally contained by the live
    enabled-collider world AABB, remains above its lower bound, lies below its
    upper/rim bound, and has settled.  Consequently, even a passing,
    identity-bound result remains ``single_cup_partial`` and provisional.
    """
    diagnostic_name = (
        "task29_conservative_outer_aabb_containment_diagnostic")

    def optional_text(value):
        try:
            raw = np.asarray(value)
            if raw.shape == () and raw.dtype.kind == "U":
                return str(raw.item())
        except (TypeError, ValueError):
            pass
        return None

    out = {
        "success": False,
        "reasons": [],
        "margins": {
            "bin": None,
            "terminal_condition": diagnostic_name,
            "destination_geometry_semantics":
                "enabled-collider-world-aabb-not-interior-or-scoring-polygon",
            "destination_interior_geometry_asserted": False,
            "destination_scoring_polygon_asserted": False,
            "destination_body_path": optional_text(
                destination_body_path),
            "destination_collider_path": optional_text(
                destination_collider_path),
        },
    }
    try:
        destination_min = np.asarray(
            destination_aabb_min, dtype=float)
        destination_max = np.asarray(
            destination_aabb_max, dtype=float)
        object_min = np.asarray(object_aabb_min, dtype=float)
        object_max = np.asarray(object_aabb_max, dtype=float)
        bottom_z = float(np.asarray(destination_bottom_z))
        rim_z = float(np.asarray(destination_rim_z))
        fps = float(np.asarray(fps))
        settle_s = float(np.asarray(settle_s))
        bottom_tol = float(np.asarray(bottom_tol))
        rim_clearance = float(np.asarray(rim_clearance))
        settle_drift_tol = float(np.asarray(settle_drift_tol))
    except (TypeError, ValueError):
        out["reasons"].append(
            f"{diagnostic_name} unavailable: expected numeric live AABBs "
            "and thresholds")
        return out
    total = len(object_min) if object_min.ndim else 0
    if (
            destination_min.shape != (3,)
            or destination_max.shape != (3,)
            or object_min.shape != (total, 3)
            or object_max.shape != (total, 3)
            or total < 1
            or not all(np.isfinite(value).all() for value in (
                destination_min, destination_max,
                object_min, object_max))
            or not np.isfinite((
                bottom_z, rim_z, fps, settle_s, bottom_tol,
                rim_clearance, settle_drift_tol)).all()
            or np.any(destination_min >= destination_max)
            or np.any(object_min >= object_max)
            or fps <= 0
            or settle_s <= 0
            or bottom_tol < 0
            or rim_clearance < 0
            or settle_drift_tol < 0):
        out["reasons"].append(
            f"{diagnostic_name} unavailable: invalid live AABB shape, "
            "finiteness, ordering, or thresholds")
        return out

    settle_n = max(1, int(np.ceil(settle_s * fps)))
    settle_min = object_min[-settle_n:]
    settle_max = object_max[-settle_n:]
    settle_center = (settle_min + settle_max) / 2.0
    final_min = settle_min[-1]
    final_max = settle_max[-1]
    final_center = settle_center[-1]

    horizontal_contained = bool(
        np.all(final_min[:2] >= destination_min[:2])
        and np.all(final_max[:2] <= destination_max[:2]))
    support_above_bottom = bool(
        final_min[2] >= bottom_z - bottom_tol)
    object_top_below_rim = bool(
        final_max[2] <= rim_z - rim_clearance)
    object_center_below_rim = bool(
        final_center[2] <= rim_z - rim_clearance)
    center_xy_drift = float(np.max(np.linalg.norm(
        settle_center[:, :2] - final_center[:2], axis=1)))
    center_z_drift = float(
        np.max(np.abs(settle_center[:, 2] - final_center[2])))
    corner_drift = float(max(
        np.max(np.linalg.norm(
            settle_min - final_min, axis=1)),
        np.max(np.linalg.norm(
            settle_max - final_max, axis=1)),
    ))
    settled = bool(
        center_xy_drift <= settle_drift_tol
        and center_z_drift <= settle_drift_tol
        and corner_drift <= settle_drift_tol)
    out["margins"].update({
        "bin": "teal",
        "final_object_aabb_min":
            [round(float(value), 6) for value in final_min],
        "final_object_aabb_max":
            [round(float(value), 6) for value in final_max],
        "final_object_aabb_center":
            [round(float(value), 6) for value in final_center],
        "destination_aabb_min":
            [round(float(value), 6) for value in destination_min],
        "destination_aabb_max":
            [round(float(value), 6) for value in destination_max],
        "horizontal_outer_aabb_contained": horizontal_contained,
        "object_support_above_destination_bottom":
            support_above_bottom,
        "object_top_below_destination_rim":
            object_top_below_rim,
        "object_center_below_destination_rim":
            object_center_below_rim,
        "destination_bottom_z": round(bottom_z, 6),
        "destination_rim_z": round(rim_z, 6),
        "settle_center_xy_drift_m": round(center_xy_drift, 6),
        "settle_center_z_drift_m": round(center_z_drift, 6),
        "settle_aabb_corner_drift_m": round(corner_drift, 6),
        "settle_drift_tolerance_m": settle_drift_tol,
    })
    if not horizontal_contained:
        out["reasons"].append(
            "final live object AABB is not horizontally contained by the "
            "destination enabled-collider outer world AABB")
    if not support_above_bottom:
        out["reasons"].append(
            f"final object support {final_min[2]:.3f} is below "
            f"destination bottom {bottom_z:.3f} minus tolerance "
            f"{bottom_tol:.3f}")
    if not object_top_below_rim:
        out["reasons"].append(
            f"final object top {final_max[2]:.3f} does not clear below "
            f"destination rim {rim_z:.3f} by {rim_clearance:.3f}")
    if not object_center_below_rim:
        out["reasons"].append(
            f"final object center {final_center[2]:.3f} does not clear below "
            f"destination rim {rim_z:.3f} by {rim_clearance:.3f}")
    if not settled:
        out["reasons"].append(
            "live object AABB is not settled: "
            f"xy center drift {center_xy_drift*1000:.1f} mm, "
            f"z center drift {center_z_drift*1000:.1f} mm, "
            f"corner drift {corner_drift*1000:.1f} mm")
    out["success"] = bool(
        horizontal_contained
        and support_above_bottom
        and object_top_below_rim
        and object_center_below_rim
        and settled)
    if out["success"]:
        out["reasons"].append(
            f"{diagnostic_name} passed; this does not assert bin interior "
            "geometry or a scoring polygon")
    return out


def check_grab_carry_release(
        obj_xyz=None, tip=None, grip_joint_closure=None, pad_contacts=None,
        pad_force_vectors=None, contact_available=None,
        contact_force_threshold_n=None, fps=FPS, grab_s=0.10,
        hold_s=0.25, release_s=0.20, grip_closed=0.80, grip_open=0.20,
        contact_thresh=0.5, min_carry=0.040, attach_std=0.030,
        min_release_separation=0.030, opposition_cos_max=0.5,
        opposition_min_fraction=0.8):
    """Fail-closed physical sequence gate shared by trash-to-bin tasks.

    A strict sequence has an open/contact-free pre-grasp, a sustained closed
    bilateral-force hold, attached object travel, and a later sustained
    open/contact-free release whose hand/object relative pose separates.
    """
    out = {
        "success": False,
        "reasons": [],
        "margins": {"sequence_evidence_available": False},
    }
    try:
        available = np.asarray(contact_available)
        if (available.shape != () or available.dtype.kind != "b"
                or not bool(available)):
            raise ValueError
    except (TypeError, ValueError):
        out["reasons"].append(
            "missing raw contact evidence: contact_available must be explicit true")
        return out
    required = {
        "obj_xyz": obj_xyz,
        "tip": tip,
        "grip_joint_closure": grip_joint_closure,
        "pad_contacts": pad_contacts,
        "pad_force_vectors": pad_force_vectors,
    }
    missing = [key for key, value in required.items() if value is None]
    if contact_force_threshold_n is None:
        missing.append("contact_force_threshold_n")
    if missing:
        out["reasons"].append(
            "missing grab/carry/release evidence: " + ", ".join(missing))
        return out
    try:
        obj = np.asarray(obj_xyz, dtype=float)
        hand = np.asarray(tip, dtype=float)
        grip = np.asarray(grip_joint_closure, dtype=float)
        contacts = np.asarray(pad_contacts, dtype=float)
        forces = np.asarray(pad_force_vectors, dtype=float)
        force_threshold = float(np.asarray(contact_force_threshold_n))
        fps = float(fps)
    except (TypeError, ValueError):
        out["reasons"].append(
            "invalid grab/carry/release evidence: arrays must be numeric")
        return out
    total = len(obj) if obj.ndim else 0
    expected = {
        "obj_xyz": (obj, (total, 3)),
        "tip": (hand, (total, 3)),
        "grip_joint_closure": (grip, (total,)),
        "pad_contacts": (contacts, (total, 2)),
        "pad_force_vectors": (forces, (total, 2, 3)),
    }
    bad_shapes = [
        f"{key} {value.shape} != {shape}"
        for key, (value, shape) in expected.items()
        if value.shape != shape
    ]
    if total < 2 or bad_shapes:
        out["reasons"].append(
            "grab/carry/release evidence shape mismatch: "
            + (", ".join(bad_shapes) if bad_shapes else "trajectory too short"))
        return out
    if (not all(np.isfinite(value).all() for value, _shape in expected.values())
            or not np.isfinite(force_threshold) or force_threshold <= 0
            or not np.isfinite(fps) or fps <= 0):
        out["reasons"].append(
            "grab/carry/release evidence contains non-finite/invalid values")
        return out
    if np.any((grip < 0) | (grip > 1)) or np.any(
            (contacts < 0) | (contacts > 1)):
        out["reasons"].append(
            "grip/contact evidence must be normalized to [0,1]")
        return out
    measured_contacts = (
        np.linalg.norm(forces, axis=2) >= force_threshold).astype(float)
    if not np.array_equal(contacts, measured_contacts):
        out["reasons"].append(
            "pad contacts do not match raw force vectors/contact threshold")
        return out
    out["margins"]["sequence_evidence_available"] = True

    grab_n = max(1, int(np.ceil(float(grab_s) * fps)))
    hold_n = max(1, int(np.ceil(float(hold_s) * fps)))
    release_n = max(1, int(np.ceil(float(release_s) * fps)))
    bilateral = np.all(contacts >= contact_thresh, axis=1)
    contact_free = np.all(contacts < contact_thresh, axis=1)
    open_free = (grip <= grip_open) & contact_free
    hold_mask = (grip >= grip_closed) & bilateral
    open_runs = _true_runs(open_free)
    hold_runs = [
        (start, stop) for start, stop in _true_runs(hold_mask)
        if stop - start >= hold_n
    ]
    best = None
    for start, stop in hold_runs:
        has_grab = any(
            before_stop <= start and before_stop - before_start >= grab_n
            for before_start, before_stop in open_runs
        )
        distance = np.linalg.norm(obj[start:stop] - obj[start], axis=1)
        carry_offset = int(np.argmax(distance))
        carry_index = start + carry_offset
        carry = float(distance[carry_offset])
        relative = hand[start:carry_index + 1] - obj[start:carry_index + 1]
        relative_std = float(np.linalg.norm(np.std(relative, axis=0)))
        hold_forces = forces[start:stop]
        force_norms = np.linalg.norm(hold_forces, axis=2)
        force_cosines = np.sum(
            hold_forces[:, 0] * hold_forces[:, 1], axis=1
        ) / (force_norms[:, 0] * force_norms[:, 1])
        opposition_fraction = float(np.mean(
            force_cosines <= opposition_cos_max))
        candidate = {
            "start": start,
            "stop": stop,
            "carry_index": carry_index,
            "carry": carry,
            "attach_std": relative_std,
            "grab_transition": has_grab,
            "opposition_fraction": opposition_fraction,
        }
        if best is None or candidate["carry"] > best["carry"]:
            best = candidate
    if best is None:
        out["reasons"].append(
            f"no bilateral closed hold for {hold_s:.2f}s")
        return out

    out["margins"].update({
        "grab_transition": best["grab_transition"],
        "bilateral_hold_s": round(float(
            (best["stop"] - best["start"]) / fps), 3),
        "carry_m": round(best["carry"], 4),
        "attach_std_m": round(best["attach_std"], 4),
        "opposition_fraction": round(best["opposition_fraction"], 4),
    })
    if not best["grab_transition"]:
        out["reasons"].append(
            f"no open/contact-free pre-grasp for {grab_s:.2f}s")
    if best["carry"] < min_carry:
        out["reasons"].append(
            f"carry {best['carry']*1000:.0f} mm < {min_carry*1000:.0f} mm")
    if best["attach_std"] > attach_std:
        out["reasons"].append(
            f"object-tip attach std {best['attach_std']*1000:.0f} mm > "
            f"{attach_std*1000:.0f} mm")
    if best["opposition_fraction"] < opposition_min_fraction:
        out["reasons"].append(
            f"opposing pad-force fraction {best['opposition_fraction']:.2f} "
            f"< {opposition_min_fraction:.2f}")

    carried_relative = np.median(
        hand[max(best["start"], best["stop"] - hold_n):best["stop"]]
        - obj[max(best["start"], best["stop"] - hold_n):best["stop"]],
        axis=0,
    )
    release_candidates = [
        (start, stop) for start, stop in open_runs
        if start >= best["stop"] and stop - start >= release_n
    ]
    if not release_candidates:
        out["reasons"].append(
            f"no post-carry open/contact-free release for {release_s:.2f}s")
        return out
    release_start, release_stop = release_candidates[0]
    release_relative = (
        hand[release_start:release_stop] - obj[release_start:release_stop])
    separation = float(np.max(np.linalg.norm(
        release_relative - carried_relative, axis=1)))
    out["margins"].update({
        "release_open_s": round(float(
            (release_stop - release_start) / fps), 3),
        "release_separation_m": round(separation, 4),
    })
    if separation < min_release_separation:
        out["reasons"].append(
            f"release separation {separation*1000:.0f} mm < "
            f"{min_release_separation*1000:.0f} mm")
    out["success"] = not out["reasons"]
    return out


def check_task29_phase_hold_through_release(
        source_frame=None, pad_contacts=None, phase_ordinal=0):
    """Require phase bilateral contact through the last pre-release command.

    The generic sequence gate accepts any sufficiently long carry followed by
    a later open interval.  For Task29 that would misclassify a cup ejected
    during the carry as an intentional release.  The frozen manifest provides
    the exact strong-close and release source frames, so require uninterrupted
    bilateral contact over ``[strong_close, release)``.
    """
    from task29_phase_plan import phase_plan

    try:
        ordinal_array = np.asarray(phase_ordinal)
        if (
            ordinal_array.shape != ()
            or ordinal_array.dtype.kind not in "iu"
        ):
            raise ValueError("ordinal must be one integer scalar")
        execution = phase_plan(int(ordinal_array))
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "success": False,
            "reasons": [
                "Task29 phase hold ordinal is invalid: "
                f"{type(exc).__name__}: {exc}"
            ],
            "margins": {
                "task29_phase_hold_required_start_source_frame": None,
                "task29_phase_hold_required_through_source_frame": None,
                "task29_phase_release_source_frame": None,
                "task29_phase_hold_continuous": False,
                "task29_phase_hold_first_missing_source_frame": None,
            },
        }
    required_start = execution.strong_close_frame
    release_frame = execution.release_frame
    required_through = release_frame - 1
    out = {
        "success": False,
        "reasons": [],
        "margins": {
            "task29_phase_hold_required_start_source_frame":
                required_start,
            "task29_phase_hold_required_through_source_frame":
                required_through,
            "task29_phase_release_source_frame": release_frame,
            "task29_phase_hold_continuous": False,
            "task29_phase_hold_first_missing_source_frame": None,
        },
    }
    try:
        frames = np.asarray(source_frame)
        contacts = np.asarray(pad_contacts)
    except (TypeError, ValueError):
        out["reasons"].append(
            "Task29 phase hold evidence is not array-like")
        return out
    if (
        frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or contacts.shape != (len(frames), 2)
        or contacts.dtype.kind not in "buif"
        or not np.isfinite(contacts).all()
    ):
        out["reasons"].append(
            "Task29 phase hold evidence has invalid frame/contact shape")
        return out
    required_frames = np.arange(
        required_start, release_frame, dtype=np.int64)
    required_indices = np.flatnonzero(
        (frames >= required_start) & (frames < release_frame))
    if (
        required_indices.shape != required_frames.shape
        or not np.array_equal(frames[required_indices], required_frames)
    ):
        out["reasons"].append(
            "Task29 phase hold source-frame coverage is incomplete")
        return out
    bilateral = np.all(contacts[required_indices] >= 0.5, axis=1)
    if not np.all(bilateral):
        first_missing = int(
            required_frames[np.flatnonzero(~bilateral)[0]])
        out["margins"][
            "task29_phase_hold_first_missing_source_frame"
        ] = first_missing
        out["reasons"].append(
            "Task29 bilateral hold did not remain continuous through "
            f"source frame {required_through}; first missing frame "
            f"{first_missing}")
        return out
    out["margins"]["task29_phase_hold_continuous"] = True
    out["success"] = True
    return out


def _true_runs(mask):
    """Return half-open intervals for contiguous True runs in a 1-D mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    edge = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def check_place_sim(obj_z=None, obj_xy=None, zone=None, fps=FPS, settle_s=1.0, z_tol=0.06,
                    tip=None, grip=None, pad_contacts=None, obj_xyz=None,
                    obj_quat=None, support_z=None, bbox_center_xyz=None, release_t=None,
                    contact_available=None, pad_force_vectors=None,
                    contact_force_threshold_n=None,
                    destination_kind="surface", destination_bottom_z=None,
                    hold_s=0.30, release_s=0.25,
                    release_transition_s=1.0,
                    grip_closed=0.80, grip_open=0.20, contact_thresh=0.5,
                    min_carry=0.050, attach_std=0.025, min_release_separation=0.030,
                    surface_lower_tol=0.020, settle_z_tol=0.003,
                    settle_xy_tol=0.005, settle_angle_tol=0.0872665,
                    opposition_cos_max=0.5, opposition_min_fraction=0.8,
                    bin_bottom_tol=0.002, bin_rim_clearance=0.002,
                    max_bilateral_dropout_frames=0):
    """Strict physical place predicate.

    ``grip`` is normalized as 0=open, 1=closed. ``pad_contacts`` must contain
    one value for each pad per frame, shaped ``(T, 2)`` (boolean/contact
    confidence) or ``(T, 2, 3)`` (force vectors).  A final pose without these
    physical signals is deliberately unscorable and fails closed.

    A v3 runtime sidecar also supplies ``pad_force_vectors`` in a common
    target-on-pad world convention.  During the selected hold, the two pad
    forces must oppose one another rather than point together.  That rejects an
    object merely bridging or resting on both fingers while preserving a wide
    60-degree opposition margin for noisy frictional contact.

    ``support_z`` is the object's lowest world-space bbox corner.  For surface
    destinations it gates the calibrated support band.  For a bin it must not
    fall below the audited physical bottom; the bbox center must also clear the
    rim, rejecting both ground misses and objects bridged across the opening.
    The rigid-body origin z remains only a compatibility fallback because asset
    origins are not consistently located at their bottom.

    ``bbox_center_xyz`` is used for the target footprint and final translational
    stability.  Carry still uses the rigid-body origin so its object-tip
    relative transform remains a single physical frame.

    ``max_bilateral_dropout_frames`` defaults to zero.  A task contract may
    bridge a bounded target-contact sensor dropout while the gripper remains
    closed and bilateral evidence exists on both sides.  The bridged samples
    still count as non-opposing in the 80% force-density gate and remain part
    of the object-tip attachment RMS, so this cannot turn a dropped object
    into a carry.

    When ``obj_quat`` is present, final orientation must also remain within
    ``settle_angle_tol``.  This catches a symmetric can spinning in place:
    its support height and center can be constant even though it is not settled.

    The accepted sequence is:
      1. a sustained closed interval with bilateral pad contact;
      2. at least ``min_carry`` object travel that remains at the end of the
         bilateral hold, while object-tip relative pose is stable;
      3. a later sustained open/contact-free interval and persistent final
         separation from the carried relative pose;
      4. an open, contact-free, stable final window inside ``zone`` and either
         on its surface band or vertically inside its audited bin volume.
    """
    out = {"success": False, "reasons": [], "margins": {}}

    missing = []
    if obj_xyz is None and obj_z is None:
        missing.append("obj_xyz/obj_z")
    if obj_xyz is None and obj_xy is None:
        missing.append("obj_xyz/obj_xy")
    if tip is None:
        missing.append("tip")
    if grip is None:
        missing.append("grip")
    if pad_contacts is None:
        missing.append("pad_contacts")
    if missing:
        out["reasons"].append("missing place evidence: " + ", ".join(missing))
        return out
    if contact_available is None:
        out["reasons"].append(
            "missing physical contact evidence: contact_available is absent")
        return out
    try:
        available_raw = np.asarray(contact_available)
        available = (available_raw.shape == ()
                     and available_raw.dtype.kind == "b"
                     and bool(available_raw))
    except (TypeError, ValueError):
        available = False
    if not available:
        out["reasons"].append(
            "missing physical contact evidence: contact_available is not explicit true")
        return out
    try:
        zone_raw = np.asarray(zone)
        if zone_raw.dtype.kind not in "fiu":
            raise ValueError("zone is not a real numeric array")
        zone_values = np.asarray(zone_raw, dtype=float)
        valid_zone = (zone_values.shape == (5,)
                      and np.isfinite(zone_values).all()
                      and zone_values[2] >= 0
                      and zone_values[3] >= 0)
    except (TypeError, ValueError):
        valid_zone = False
    if not valid_zone:
        out["reasons"].append(
            "invalid place zone: expected five finite values with non-negative half extents")
        return out
    if not isinstance(destination_kind, str) or destination_kind not in {
            "surface", "bin"}:
        out["reasons"].append(
            "invalid place destination_kind: expected 'surface' or 'bin'")
        return out
    if destination_kind == "bin":
        try:
            bottom_raw = np.asarray(destination_bottom_z)
            valid_bottom = (
                bottom_raw.shape == ()
                and bottom_raw.dtype.kind in "fiu"
                and np.isfinite(float(bottom_raw))
                and float(bottom_raw) < float(zone_values[4])
            )
        except (TypeError, ValueError):
            valid_bottom = False
        if not valid_bottom:
            out["reasons"].append(
                "invalid bin destination: bottom_z must be one finite scalar below rim_z")
            return out
        destination_bottom_z = float(bottom_raw)

    numeric_evidence = {
        "obj_xyz": obj_xyz,
        "obj_z": obj_z if obj_xyz is None else None,
        "obj_xy": obj_xy if obj_xyz is None else None,
        "tip": tip,
        "grip": grip,
        "obj_quat": obj_quat,
        "support_z": support_z,
        "bbox_center_xyz": bbox_center_xyz,
        "pad_force_vectors": pad_force_vectors,
    }
    for evidence_name, evidence_value in numeric_evidence.items():
        if evidence_value is None:
            continue
        try:
            evidence_kind = np.asarray(evidence_value).dtype.kind
        except (TypeError, ValueError):
            evidence_kind = ""
        if evidence_kind not in "fiu":
            out["reasons"].append(
                f"invalid place evidence: {evidence_name} must be a real numeric array")
            return out
    try:
        if obj_xyz is not None:
            xyz = np.asarray(obj_xyz, dtype=float)
        else:
            z = np.asarray(obj_z, dtype=float).reshape(-1)
            xy = np.asarray(obj_xy, dtype=float)
            xyz = np.column_stack((xy, z))
        tip = np.asarray(tip, dtype=float)
        grip = np.asarray(grip, dtype=float).reshape(-1)
        contacts_raw = np.asarray(pad_contacts)
        force_vectors = (None if pad_force_vectors is None
                         else np.asarray(pad_force_vectors, dtype=float))
        quat = (None if obj_quat is None
                else np.asarray(obj_quat, dtype=float))
        support = (xyz[:, 2] if support_z is None
                   else np.asarray(support_z, dtype=float).reshape(-1))
        placement_xyz = (xyz if bbox_center_xyz is None
                         else np.asarray(bbox_center_xyz, dtype=float))
    except (TypeError, ValueError, IndexError) as exc:
        out["reasons"].append(f"invalid place evidence arrays: {exc}")
        return out

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        out["reasons"].append(f"invalid object trajectory shape {xyz.shape}; expected (T,3)")
        return out
    if tip.ndim != 2 or tip.shape[1] != 3:
        out["reasons"].append(f"invalid tip trajectory shape {tip.shape}; expected (T,3)")
        return out
    if placement_xyz.ndim != 2 or placement_xyz.shape[1] != 3:
        out["reasons"].append(
            f"invalid bbox_center_xyz shape {placement_xyz.shape}; expected (T,3)")
        return out
    if quat is not None and (quat.ndim != 2 or quat.shape[1] != 4):
        out["reasons"].append(
            f"invalid obj_quat shape {quat.shape}; expected (T,4)")
        return out
    if contacts_raw.ndim == 0:
        out["reasons"].append(
            f"invalid pad_contacts shape {contacts_raw.shape}; expected (T,2) or (T,2,3)")
        return out
    if (force_vectors is not None
            and (force_vectors.ndim != 3
                 or force_vectors.shape[1:] != (2, 3))):
        out["reasons"].append(
            f"invalid pad_force_vectors shape {force_vectors.shape}; expected (T,2,3)")
        return out
    total = len(xyz)
    if (len(tip) != total or len(grip) != total or len(contacts_raw) != total
            or len(support) != total or len(placement_xyz) != total
            or (force_vectors is not None and len(force_vectors) != total)
            or (quat is not None and len(quat) != total)):
        out["reasons"].append(
            "place evidence length mismatch: "
            f"object={total}, tip={len(tip)}, grip={len(grip)}, "
            f"pad_contacts={len(contacts_raw)}, support_z={len(support)}, "
            f"bbox_center_xyz={len(placement_xyz)}, "
            f"pad_force_vectors={len(force_vectors) if force_vectors is not None else 'absent'}, "
            f"obj_quat={len(quat) if quat is not None else 'absent'}")
        return out
    if total < 2:
        out["reasons"].append("place evidence too short")
        return out
    if (not np.isfinite(xyz).all() or not np.isfinite(tip).all()
            or not np.isfinite(grip).all() or not np.isfinite(support).all()):
        out["reasons"].append(
            "invalid place evidence: object/tip/grip/support_z contains NaN or inf")
        return out
    if not np.isfinite(placement_xyz).all():
        out["reasons"].append(
            "invalid place evidence: bbox_center_xyz contains NaN or inf")
        return out
    if force_vectors is not None and not np.isfinite(force_vectors).all():
        out["reasons"].append(
            "missing physical contact evidence: pad_force_vectors contains NaN or inf")
        return out
    if quat is not None:
        quat_norm = np.linalg.norm(quat, axis=1)
        if (not np.isfinite(quat).all() or np.any(quat_norm <= 1e-9)
                or not np.allclose(quat_norm, 1.0, atol=2e-3, rtol=0)):
            out["reasons"].append(
                "invalid place evidence: obj_quat must contain finite normalized quaternions")
            return out
    try:
        contacts_float = contacts_raw.astype(float)
    except (TypeError, ValueError):
        out["reasons"].append("invalid pad_contacts: values must be boolean or numeric")
        return out
    if contacts_raw.dtype.kind not in "buif":
        out["reasons"].append("invalid pad_contacts: values must be boolean or numeric")
        return out
    if not np.isfinite(contacts_float).all():
        out["reasons"].append(
            "missing physical contact evidence: pad_contacts contains NaN or inf")
        return out
    if contacts_float.ndim == 2 and contacts_float.shape[1] == 2:
        if np.any((contacts_float < 0) | (contacts_float > 1)):
            out["reasons"].append(
                "invalid pad_contacts: (T,2) contact confidence must be in [0,1]")
            return out
        contact_level = contacts_float
    elif contacts_float.ndim == 3 and contacts_float.shape[1:] == (2, 3):
        contact_level = np.linalg.norm(contacts_float, axis=2)
    else:
        out["reasons"].append(
            f"invalid pad_contacts shape {contacts_float.shape}; expected (T,2) or (T,2,3)")
        return out

    threshold_inputs = (
        fps, settle_s, z_tol, hold_s, release_s, grip_closed, grip_open,
        release_transition_s, contact_thresh, min_carry, attach_std,
        min_release_separation, surface_lower_tol, settle_z_tol,
        settle_xy_tol, settle_angle_tol, opposition_cos_max,
        opposition_min_fraction, bin_bottom_tol, bin_rim_clearance,
    )
    try:
        valid_threshold_types = all(
            np.asarray(value).shape == ()
            and np.asarray(value).dtype.kind in "fiu"
            for value in threshold_inputs)
    except (TypeError, ValueError):
        valid_threshold_types = False
    if not valid_threshold_types:
        out["reasons"].append(
            "invalid place threshold: all thresholds must be scalar real numbers")
        return out
    dropout_raw = np.asarray(max_bilateral_dropout_frames)
    if (
        dropout_raw.shape != ()
        or dropout_raw.dtype.kind not in "iu"
        or dropout_raw.dtype.kind == "b"
        or int(dropout_raw) < 0
    ):
        out["reasons"].append(
            "invalid max_bilateral_dropout_frames: expected a non-negative "
            "integer")
        return out
    max_bilateral_dropout_frames = int(dropout_raw)
    try:
        (fps, settle_s, z_tol, hold_s, release_s, grip_closed, grip_open,
         release_transition_s, contact_thresh, min_carry, attach_std, min_release_separation,
         surface_lower_tol, settle_z_tol, settle_xy_tol,
         settle_angle_tol, opposition_cos_max,
         opposition_min_fraction, bin_bottom_tol, bin_rim_clearance) = map(float, (
             fps, settle_s, z_tol, hold_s, release_s, grip_closed, grip_open,
             release_transition_s, contact_thresh, min_carry, attach_std, min_release_separation,
             surface_lower_tol, settle_z_tol, settle_xy_tol,
             settle_angle_tol, opposition_cos_max,
             opposition_min_fraction, bin_bottom_tol, bin_rim_clearance))
    except (TypeError, ValueError):
        out["reasons"].append("invalid place threshold: all thresholds must be scalar numbers")
        return out
    thresholds = (
        fps, settle_s, z_tol, hold_s, release_s, grip_closed, grip_open,
        release_transition_s,
        contact_thresh, min_carry, attach_std, min_release_separation,
        surface_lower_tol, settle_z_tol, settle_xy_tol, settle_angle_tol,
        opposition_cos_max, opposition_min_fraction,
        bin_bottom_tol, bin_rim_clearance)
    if (not np.isfinite(thresholds).all()
            or fps <= 0 or settle_s <= 0 or hold_s <= 0 or release_s <= 0
            or release_transition_s < 0
            or z_tol < 0 or not 0 <= grip_open < grip_closed <= 1
            or contact_thresh < 0 or min_carry <= 0 or attach_std < 0
            or min_release_separation < 0 or surface_lower_tol < 0
            or settle_z_tol < 0 or settle_xy_tol < 0 or settle_angle_tol < 0
            or bin_bottom_tol < 0 or bin_rim_clearance < 0
            or not -1 <= opposition_cos_max <= 1
            or not 0 < opposition_min_fraction <= 1):
        out["reasons"].append("invalid place threshold values")
        return out
    if force_vectors is not None:
        try:
            force_threshold_raw = np.asarray(contact_force_threshold_n)
            if (force_threshold_raw.shape != ()
                    or force_threshold_raw.dtype.kind not in "fiu"):
                raise ValueError
            force_threshold = float(force_threshold_raw)
            if not np.isfinite(force_threshold) or force_threshold <= 0:
                raise ValueError
        except (TypeError, ValueError):
            out["reasons"].append(
                "invalid contact_force_threshold_n for pad force vectors")
            return out
    else:
        force_threshold = None
    if np.any((grip < 0) | (grip > 1)):
        out["reasons"].append("invalid grip evidence: normalized grip must be in [0,1]")
        return out

    window_products = (
        hold_s * fps, release_s * fps, settle_s * fps,
        release_transition_s * fps,
    )
    if not np.isfinite(window_products).all():
        out["reasons"].append("invalid place threshold values")
        return out
    hold_n = max(2, int(np.ceil(window_products[0])))
    release_n = max(2, int(np.ceil(window_products[1])))
    settle_n = max(2, int(np.ceil(window_products[2])))
    if total < max(hold_n + release_n, settle_n):
        out["reasons"].append(
            f"place evidence too short: {total} frames for hold={hold_n}, "
            f"release={release_n}, settle={settle_n}")
        return out

    both_contact = np.all(contact_level >= contact_thresh, axis=1)
    any_contact = np.any(contact_level >= contact_thresh, axis=1)
    closed = grip >= grip_closed
    opened = grip <= grip_open
    hold_evidence = closed & both_contact
    bridged_dropout_frames = 0
    if max_bilateral_dropout_frames:
        for gap_start, gap_end in _true_runs(~both_contact):
            if (
                gap_start > 0
                and gap_end < total
                and gap_end - gap_start <= max_bilateral_dropout_frames
                and both_contact[gap_start - 1]
                and both_contact[gap_end]
                and np.all(closed[gap_start - 1:gap_end + 1])
            ):
                hold_evidence[gap_start:gap_end] = True
                bridged_dropout_frames += gap_end - gap_start
    if release_t is not None:
        try:
            release_raw = np.asarray(release_t)
            if release_raw.shape != () or release_raw.dtype.kind not in "iu":
                raise ValueError("expected one integer scalar")
            release_command = int(release_raw)
            release_valid = 0 <= release_command <= total
        except (TypeError, ValueError, IndexError):
            release_valid = False
            release_command = float("nan")
        if not release_valid:
            out["reasons"].append(
                f"invalid release_t {release_t!r}; expected an integer frame in [0,{total}]")
            return out
        release_command = int(release_command)
    else:
        release_command = None
    held_runs = [(start, end) for start, end in _true_runs(hold_evidence)
                 if end - start >= hold_n]
    out["margins"]["bridged_bilateral_dropout_frames"] = int(
        bridged_dropout_frames)
    out["margins"]["bilateral_contact_s"] = round(
        max((end - start for start, end in held_runs), default=0) / fps, 3)
    if not held_runs:
        out["reasons"].append(
            "no sustained physical grasp: closed gripper with bilateral pad contact "
            f"never lasted {hold_s:.2f}s")
        return out

    release_runs = [(start, end) for start, end in _true_runs(opened & ~any_contact)
                    if end - start >= release_n]
    max_release_transition = int(np.ceil(window_products[3]))
    final_rel_for_candidate = np.median(
        tip[-settle_n:] - xyz[-settle_n:], axis=0)

    force_opposed = force_cosines = None
    if force_vectors is not None:
        force_norms = np.linalg.norm(force_vectors, axis=2)
        valid_force = np.all(force_norms >= force_threshold, axis=1)
        force_cosines = np.full(total, 1.0, dtype=float)
        if np.any(valid_force):
            valid_vectors = force_vectors[valid_force]
            valid_norms = force_norms[valid_force]
            raw_cosines = np.sum(
                valid_vectors[:, 0] * valid_vectors[:, 1], axis=1
            ) / (valid_norms[:, 0] * valid_norms[:, 1])
            force_cosines[valid_force] = np.clip(raw_cosines, -1.0, 1.0)
        force_opposed = valid_force & (
            force_cosines <= opposition_cos_max)
        opposed_prefix = np.pad(
            np.cumsum(force_opposed, dtype=np.int64), (1, 0))

    def opposition_stats(start, end):
        if force_vectors is None:
            return None, None
        fraction = float(
            (opposed_prefix[end] - opposed_prefix[start]) / (end - start))
        return fraction, float(np.median(force_cosines[start:end]))

    def opposition_subwindows(run_start, run_end, release_start):
        """Yield sustained-pinch windows within one uninterrupted contact run.

        A target may initially bridge both pads and later become a real pinch
        without either scalar contact bit dropping.  Scoring the entire run
        would dilute the valid suffix.  Candidate windows therefore need at
        least ``hold_n`` samples, opposed start/end samples, >=80% opposition
        overall, and the same density in their first and last hold-sized
        windows.  The endpoint is also kept within the allowed release
        transition.  These boundary checks prevent bridge-only motion before
        the first real pinch sample from being imported into a valid suffix.
        """
        if force_opposed is None:
            yield run_start, run_end
            return
        earliest_end = max(
            run_start + hold_n, release_start - max_release_transition)
        for start in range(run_start, run_end - hold_n + 1):
            if not force_opposed[start]:
                continue
            first_end = start + hold_n
            first_fraction, _ = opposition_stats(start, first_end)
            if first_fraction < opposition_min_fraction:
                continue
            for end in range(max(first_end, earliest_end), run_end + 1):
                if not force_opposed[end - 1]:
                    continue
                fraction, _ = opposition_stats(start, end)
                last_fraction, _ = opposition_stats(end - hold_n, end)
                if (fraction >= opposition_min_fraction
                        and last_fraction >= opposition_min_fraction):
                    yield start, end

    candidates = []
    for run_start, run_end in held_runs:
        later_releases = [run for run in release_runs if run[0] >= run_end]
        if not later_releases:
            continue
        release_start = later_releases[0][0]
        eligible_windows = list(opposition_subwindows(
            run_start, run_end, release_start))
        windows = [
            (start, end, True) for start, end in eligible_windows
        ]
        # Keep the full scalar-contact run as a diagnostic fallback.  Without
        # this, an all-same-direction bridge would be misreported as having no
        # release rather than failing the opposition gate.  A full-run fallback
        # that did not pass the boundary-density rule is never score-eligible,
        # even if its aggregate fraction happens to reach 80%.
        if (run_start, run_end) not in eligible_windows:
            windows.append((run_start, run_end, False))
        for held_start, held_end, window_eligible in windows:
            travel = float(np.max(np.linalg.norm(
                xyz[held_start:held_end] - xyz[held_start], axis=1)))
            carry_end = float(np.linalg.norm(
                xyz[held_end - 1] - xyz[held_start]))
            # Rank by motion that remains at the end of the physical hold.  A
            # large excursion that returns to the grasp start is not a carry.
            effective_carry = min(travel, carry_end)
            opposition_fraction, median_cosine = opposition_stats(
                held_start, held_end)
            opposition_ok = (
                opposition_fraction is None
                or (
                    window_eligible
                    and opposition_fraction >= opposition_min_fraction
                )
            )
            held_rel = tip[held_start:held_end] - xyz[held_start:held_end]
            rel_center = np.median(held_rel, axis=0)
            rel_rms = float(np.sqrt(np.mean(np.sum(
                (held_rel - rel_center) ** 2, axis=1))))
            net_place = float(np.linalg.norm(xyz[-1] - xyz[held_start]))
            release_separation = float(np.linalg.norm(
                final_rel_for_candidate - rel_center))
            release_transition = release_start - held_end
            release_t_ok = (
                release_command is None
                or (
                    held_end <= release_command
                    and 0 <= release_start - release_command
                    <= int(np.ceil(fps))
                )
            )
            core_ok = (
                opposition_ok
                and release_transition <= max_release_transition
                and release_t_ok
                and travel >= min_carry
                and carry_end >= min_carry
                and net_place >= min_carry
                and rel_rms <= attach_std
                and release_separation >= min_release_separation
            )
            candidates.append({
                "core_ok": core_ok,
                "opposition_ok": opposition_ok,
                "effective_carry": effective_carry,
                "travel": travel,
                "carry_end": carry_end,
                "length": held_end - held_start,
                "held_start": held_start,
                "held_end": held_end,
                "release_start": release_start,
                "opposition_fraction": opposition_fraction,
                "median_cosine": median_cosine,
            })
    if not candidates:
        out["reasons"].append(
            "no sustained physical release after grasp: gripper must open and both "
            f"pad contacts must disappear for {release_s:.2f}s")
        return out

    selected = max(candidates, key=lambda row: (
        row["core_ok"], row["opposition_ok"], row["effective_carry"],
        row["travel"], row["length"],
    ))
    carry_m = selected["travel"]
    carry_end_m = selected["carry_end"]
    held_start = selected["held_start"]
    held_end = selected["held_end"]
    release_start = selected["release_start"]
    opposition_fraction = selected["opposition_fraction"]
    median_cosine = selected["median_cosine"]
    opposition_ok = selected["opposition_ok"]
    release_transition = release_start - held_end
    out["margins"]["release_transition_frames"] = int(release_transition)
    if release_transition > max_release_transition:
        out["reasons"].append(
            f"grasp/release sequence disconnected: bilateral hold ended at frame "
            f"{held_end}, full-open/contact-free release began at {release_start} "
            f"(max transition {max_release_transition} frames)")
    if release_command is not None:
        if held_end > release_command:
            out["reasons"].append(
                f"bilateral hold extends past release_t: hold ended at frame "
                f"{held_end}, command frame {release_command}")
        # HOLD_THR (runtime release command) is intentionally looser than
        # grip_open (fully-open score gate), so the physical release can lag
        # but cannot precede the command or lag it by more than one second.
        release_lag = release_start - release_command
        out["margins"]["release_command_frame"] = release_command
        out["margins"]["release_command_lag_frames"] = int(release_lag)
        if release_lag < 0 or release_lag > int(np.ceil(fps)):
            out["reasons"].append(
                f"release evidence inconsistent with release_t: full-open/contact-free "
                f"frame {release_start}, command frame {release_command}")
    held_rel = tip[held_start:held_end] - xyz[held_start:held_end]
    rel_center = np.median(held_rel, axis=0)
    rel_rms = float(np.sqrt(np.mean(np.sum((held_rel - rel_center) ** 2, axis=1))))
    out["margins"].update({
        "carry_mm": round(carry_m * 1000, 1),
        "carry_end_mm": round(carry_end_m * 1000, 1),
        "attach_rel_rms_mm": round(rel_rms * 1000, 1),
        "hold_s": round((held_end - held_start) / fps, 3),
        "release_frame": int(release_start),
    })
    net_place_m = float(np.linalg.norm(xyz[-1] - xyz[held_start]))
    out["margins"]["net_place_mm"] = round(net_place_m * 1000, 1)
    if carry_m < min_carry:
        out["reasons"].append(
            f"no carried motion: object moved {carry_m*1000:.0f} mm during bilateral "
            f"grasp (need {min_carry*1000:.0f} mm)")
    if carry_end_m < min_carry:
        out["reasons"].append(
            f"no retained carry at release: object ended the bilateral hold "
            f"{carry_end_m*1000:.0f} mm from its grasp start "
            f"(need {min_carry*1000:.0f} mm)")
    if net_place_m < min_carry:
        out["reasons"].append(
            f"no net placement: final object displacement {net_place_m*1000:.0f} mm "
            f"from grasp start < {min_carry*1000:.0f} mm")
    if rel_rms > attach_std:
        out["reasons"].append(
            f"object did not track tip during carry: relative RMS {rel_rms*1000:.0f} mm "
            f"> {attach_std*1000:.0f} mm")
    if force_vectors is not None:
        out["margins"]["pad_force_opposition_fraction"] = round(
            opposition_fraction, 3)
        out["margins"]["pad_force_median_cosine"] = round(
            median_cosine, 3)
        if not opposition_ok:
            out["reasons"].append(
                "bilateral contacts do not form an opposing pinch: "
                f"{opposition_fraction:.0%} of hold frames oppose below cosine "
                f"{opposition_cos_max:.2f}, but no release-adjacent window has "
                f"{opposition_min_fraction:.0%} opposition at both boundaries")

    final_rel = np.median(tip[-settle_n:] - xyz[-settle_n:], axis=0)
    release_separation = float(np.linalg.norm(final_rel - rel_center))
    out["margins"]["release_separation_mm"] = round(release_separation * 1000, 1)
    if release_separation < min_release_separation:
        out["reasons"].append(
            f"object remained coupled to tip after opening: relative separation "
            f"{release_separation*1000:.0f} mm < {min_release_separation*1000:.0f} mm")

    cx, cy, hx, hy, surface_or_rim_z = zone_values
    final_xyz = xyz[-1]
    final_placement_xyz = placement_xyz[-1]
    final_support_z = float(support[-1])
    final_window = placement_xyz[-settle_n:]
    origin_z_drift = float(np.ptp(xyz[-settle_n:, 2]))
    support_z_drift = float(np.ptp(support[-settle_n:]))
    # A rotating object can keep its origin almost fixed while its support
    # envelope still changes.  Count either motion as not settled.
    z_drift = max(origin_z_drift, support_z_drift)
    xy_drift = float(np.max(np.linalg.norm(
        final_window[:, :2] - final_window[-1, :2], axis=1)))
    if quat is None:
        angle_drift = None
    else:
        settle_quat = quat[-settle_n:]
        settle_quat = settle_quat / np.linalg.norm(
            settle_quat, axis=1, keepdims=True)
        final_quat = settle_quat[-1]
        quat_dots = np.clip(
            np.abs(settle_quat @ final_quat), 0.0, 1.0)
        angle_drift = float(np.max(2.0 * np.arccos(quat_dots)))
    in_zone = (abs(final_placement_xyz[0] - cx) <= hx and
               abs(final_placement_xyz[1] - cy) <= hy)
    if destination_kind == "surface":
        on_surface = (
            surface_or_rim_z - surface_lower_tol <= final_support_z
            <= surface_or_rim_z + z_tol
        )
        above_bin_bottom = None
        below_bin_rim = None
        destination_height_ok = on_surface
    else:
        on_surface = None
        above_bin_bottom = (
            final_support_z >= destination_bottom_z - bin_bottom_tol)
        below_bin_rim = (
            float(final_placement_xyz[2])
            <= surface_or_rim_z - bin_rim_clearance)
        destination_height_ok = above_bin_bottom and below_bin_rim
    settled = (z_drift <= settle_z_tol and xy_drift <= settle_xy_tol
               and (angle_drift is None or angle_drift <= settle_angle_tol))
    final_open = bool(np.all(opened[-settle_n:]))
    final_contact_free = bool(np.all(~any_contact[-settle_n:]))
    out["margins"].update({
        "destination_kind": destination_kind,
        "final_z": round(final_support_z, 3),
        "final_origin_z": round(float(final_xyz[2]), 3),
        "final_bbox_center_z": round(float(final_placement_xyz[2]), 3),
        "final_xy": [round(float(final_placement_xyz[0]), 3),
                     round(float(final_placement_xyz[1]), 3)],
        "final_origin_xy": [round(float(final_xyz[0]), 3), round(float(final_xyz[1]), 3)],
        "settle_z_drift_mm": round(z_drift * 1000, 1),
        "settle_origin_z_drift_mm": round(origin_z_drift * 1000, 1),
        "settle_support_z_drift_mm": round(support_z_drift * 1000, 1),
        "settle_xy_drift_mm": round(xy_drift * 1000, 1),
        "settle_angle_drift_deg": (
            None if angle_drift is None else round(float(np.degrees(angle_drift)), 2)),
        "final_open": final_open,
        "final_contact_free": final_contact_free,
    })
    if destination_kind == "surface":
        out["margins"]["surface_band"] = [
            round(surface_or_rim_z - surface_lower_tol, 3),
            round(surface_or_rim_z + z_tol, 3),
        ]
    else:
        out["margins"].update({
            "bin_vertical_band": [
                round(destination_bottom_z - bin_bottom_tol, 3),
                round(surface_or_rim_z - bin_rim_clearance, 3),
            ],
            "above_bin_bottom": bool(above_bin_bottom),
            "below_bin_rim": bool(below_bin_rim),
        })
    if not in_zone:
        out["reasons"].append(
            f"final xy ({final_placement_xyz[0]:.3f},{final_placement_xyz[1]:.3f}) outside zone "
            f"({cx}+-{hx},{cy}+-{hy})")
    if destination_kind == "surface" and not destination_height_ok:
        out["reasons"].append(
            f"final support z {final_support_z:.3f} outside surface band "
            f"[{surface_or_rim_z-surface_lower_tol:.3f},"
            f"{surface_or_rim_z+z_tol:.3f}]")
    if destination_kind == "bin" and not above_bin_bottom:
        out["reasons"].append(
            f"final support z {final_support_z:.3f} below bin bottom "
            f"{destination_bottom_z:.3f} (tolerance {bin_bottom_tol:.3f})")
    if destination_kind == "bin" and not below_bin_rim:
        out["reasons"].append(
            f"final bbox center z {final_placement_xyz[2]:.3f} is not below bin rim "
            f"{surface_or_rim_z:.3f} with clearance {bin_rim_clearance:.3f}")
    if not settled:
        out["reasons"].append(
            f"not settled: final {settle_s:.2f}s z drift {z_drift*1000:.1f} mm, "
            f"xy drift {xy_drift*1000:.1f} mm"
            + ("" if angle_drift is None
               else f", angle drift {np.degrees(angle_drift):.1f} deg"))
    if not final_open:
        out["reasons"].append("gripper did not remain open through final settle window")
    if not final_contact_free:
        out["reasons"].append("pad contact returned during final settle window")

    out["success"] = not out["reasons"]
    if out["success"]:
        out["reasons"].append(
            f"physical place: carried {carry_m*1000:.0f} mm with bilateral contact, "
            f"released/separated {release_separation*1000:.0f} mm, "
            f"settled in {destination_kind} destination")
    return out


def check_lean_sim(obj_z, obj_xy, zone, fps=FPS, settle_s=1.0, z_min=0.06, z_max=0.32):
    """Lean gate (t20 书斜靠桌缘): object at rest with xy in the lean zone AND its tracked
    center in a RAISED z band. pd_metrics has no orientation, so the z band is the lean
    discriminator: a book leaning on the table edge holds its center well above the floor
    (~len/2*sin(tilt)); a book flat on the floor is < ~35 mm; one left on the tabletop is
    above z_max. Thresholds provisional until t20 landing diagnostics measure real values."""
    import numpy as np
    obj_z = np.asarray(obj_z); obj_xy = np.asarray(obj_xy)
    cx, cy, hx, hy = zone[:4]
    n = max(1, int(settle_s * fps))
    fz = obj_z[-n:]; fxy = obj_xy[-1]
    in_zone = abs(fxy[0] - cx) <= hx and abs(fxy[1] - cy) <= hy
    leaned = z_min <= float(fz[-1]) <= z_max
    settled = float(fz.max() - fz.min()) < 0.003
    reasons = []
    if not in_zone:
        reasons.append(f"final xy ({fxy[0]:.3f},{fxy[1]:.3f}) outside lean zone ({cx}+-{hx},{cy}+-{hy})")
    if not leaned:
        how = "flat on floor" if float(fz[-1]) < z_min else "not leaning (on tabletop?)"
        reasons.append(f"final z {float(fz[-1]):.3f} outside lean band [{z_min},{z_max}] — {how}")
    if not settled:
        reasons.append(f"not settled: z drift {(fz.max()-fz.min())*1000:.0f} mm")
    return {"success": bool(in_zone and leaned and settled), "reasons": reasons,
            "margins": {"final_z": round(float(fz[-1]), 3),
                        "final_xy": [round(float(fxy[0]), 3), round(float(fxy[1]), 3)]}}


# ---------------------------------------------------------------------------
# 全任务判分单入口: check_task(task_id, metrics) — 家族分发 + 每任务配置。
# metrics = np.load(pd_metrics.npz) (或同键 dict)。家族归属与 init_*.TASKS 同源:
#   pick  -> check_pick_sim  (legacy geometric diagnostic). Contact-gated
#            tasks remain provisional until an identity-bound raw-force scorer
#            validates a task-specific grab/carry/final-held sequence.
#   throw -> correct-bin terminal gate + identity-bound raw-force
#            grab/carry/release sequence (missing capability = provisional only)
#   place -> check_place_sim (destination 来自 init_place.destination_for_task)
#   lean  -> check_lean_sim  (t20 书斜靠桌缘)
# 52 (Newton 桌布) 走独立管线, 不在此入口。
PICK_TASKS = {
    "6", "8", "9", "10", "11", "12", "13", "21", "22", "23", "31", "32",
    "33", "35", "36", "37", "38", "39", "40", "47", "48", "50", "51",
    "53", "54",
    "55", "56", "57", "58", "59", "60", "61", "78", "85", "90", "91", "92",
    "109", "405",
}
# The tasks below have been identity-audited and wired after the strict place
# audit.  Historical playback archives have only target pose, tip pose, and
# gripper state; they have no target-only raw pad forces.  Keep these gates
# task-local so historical pick outputs are neither silently relabelled nor
# promoted to strict evidence.  Task31's legacy Remote and Task33's legacy
# PaperCup PASS rows are lift-only and likewise cannot certify bilateral
# target contact.  In particular, historical Task53/55/60 metrics that tracked
# GlassesCase cannot certify the authoritative Wallet.
CONTACT_GATED_PICK_TASKS = {
    "6": {},
    "8": {},
    "9": {},
    "10": {},
    "11": {},
    "12": {},
    "13": {},
    "21": {},
    "22": {},
    "31": {},
    "32": {},
    "33": {},
    "35": {},
    "36": {},
    "37": {},
    "38": {},
    "39": {},
    "40": {},
    "47": {},
    "48": {},
    "50": {},
    "51": {},
    "53": {},
    "54": {},
    "55": {},
    "56": {},
    "57": {},
    "58": {},
    "59": {},
    "60": {},
    "78": {},
    "90": {},
    "91": {},
    "92": {},
    "85": {},
    "109": {},
    "405": {},
}
# A normalized closure of 0.80 represents a nearly empty omnipicker.  The
# 110.5 mm narrow side of the audited small Task 53 Wallet fills most of the
# measured 125.4 mm open gap and reaches bilateral contact around 0.30--0.35.
# Requiring 0.80 would reward driving the pads through that rigid body.  Keep
# this obstacle-stall threshold task-local and still require raw bilateral,
# opposing target forces plus the full final-held sequence.
PICK_CONTACT_SEQUENCE_OVERRIDES = {
    "53": {
        "grip_open": 0.10,
        "grip_closed": 0.22,
    },
    # The reviewed folded Umbrella is 65.36 mm across its grasping section.
    # It physically stalls both 125 mm-open pads at about 0.68 normalized
    # closure.  The generic 0.80 threshold instead requires a roughly 35 mm
    # gap, compresses/ejects the cylinder, and rewards a false zero-gap
    # command.  Target-only bilateral raw forces, opposing directions, carry,
    # and final hold remain mandatory.
    "54": {
        "grip_closed": 0.65,
    },
    # Same folded Umbrella, picked from the wooden side table (2026-07-31).
    "90": {
        "grip_closed": 0.65,
    },
    # The ~40 mm soft wallet pouch stalls the pads well before the generic
    # 0.80 closure (Task 53 wallet precedent); bilateral raw force, carry and
    # final hold remain mandatory.
    "92": {
        "grip_closed": 0.30,
    },
}
TASK59_PICK_SUPPORT_AUTHORITY = "task59_rgbd_tabletop_v1"
TASK53_PICK_SUPPORT_AUTHORITY = (
    "task53_real_rgb_rounded_cushion_v2")
PICK_CONTACT_V1_REQUIRED_KEYS = (
    "schema",
    "task_id",
    "episode_uid",
    "run_id",
    "seat_riser_active",
    "seat_riser_support_height_m",
    "seat_riser_support_top_m",
    "support_mode",
    "support_authority",
    "target_body_path",
    "active_hand",
    "pad_body_paths",
    "obj_xyz",
    "object_wxyz",
    "gripper_wxyz",
    "tip",
    "grip_joint_closure",
    "pad_contacts",
    "pad_force_vectors",
    "contact_force_threshold_n",
    "grip_closed_reference_rad",
    "grip_open_reference_rad",
    "contact_available",
    "fps",
    "sample_time_s",
)
PICK_CONTACT_V1_TRUSTED_EXPECTATIONS = (
    "expected_episode_uid",
    "expected_run_id",
    "expected_pad_body_paths",
    "expected_contact_force_threshold_n",
    "expected_grip_closed_reference_rad",
    "expected_grip_open_reference_rad",
    "expected_seat_riser_active",
    "expected_seat_riser_support_height_m",
    "expected_seat_riser_support_top_m",
    "expected_support_authority",
)
# bins = 该任务运行时**在场**的桶。t29 明确要求 teal；逐帧 REAL3 复核
# 确认 t30 最终进入 red，teal30 作为相邻干扰桶。旧位条目是幽灵桶
# (脚下地板会假阳性), 必须排除。所有 throw task 都要求 v1 raw-contact
# grab/carry/release evidence；legacy playback_pd 只能给 diagnostic/provisional。
THROW_TASKS = {
    "29": dict(expect_bin="teal", bins=["teal", "red"],
               require_sequence=True),
    "30": dict(expect_bin="red", bins=["red", "teal30"],
               require_sequence=True),
}
LEAN_TASKS = set()   # t20 曾按"斜靠"归此族, 两集结尾抽帧证实是平放→place (2026-07-21)
PLACE_TASKS = {
    "14", "15", "16", "17", "18", "19", "20", "27", "28",
    "41", "42", "43", "44", "45", "46",
    "71", "72", "73", "74", "86", "87", "88", "89",
}   # 71–74 = initially-in-hand objects placed into the shared soft tote


def task_family(task_id):
    tid = str(task_id)
    if tid in LEAN_TASKS: return "lean"
    if tid in THROW_TASKS: return "throw"
    if tid in PLACE_TASKS: return "place"
    if tid in PICK_TASKS: return "pick"
    return None


def _place_destination(tid):
    import importlib, sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    return importlib.import_module("init_place").destination_for_task(str(tid))


def _place_zone(tid):
    return _place_destination(tid)["zone"]


def _place_target_body_path(tid):
    import importlib
    cfg = importlib.import_module("init_place").TASKS[str(tid)]
    return f"{str(cfg['root']).rstrip('/')}/{str(cfg['prim']).lstrip('/')}"


def _throw_target_body_path(tid):
    import importlib, sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    return importlib.import_module("task_registry").track_prim(str(tid))


def _task_registry_entry(tid):
    import importlib, sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    return importlib.import_module("task_registry").REGISTRY[str(tid)]


def _metric(m, key, default=None):
    """Read one key from dict-like or ``np.load`` metrics without truthiness traps."""
    try:
        keys = m.files if hasattr(m, "files") else m.keys()
        return m[key] if key in keys else default
    except (AttributeError, KeyError, TypeError):
        return default


def _safe_identity_text(value) -> bool:
    """Identity text must contain only printable Unicode scalar values."""
    text = str(value)
    return bool(text.strip()) and all(
        not (0xD800 <= ord(char) <= 0xDFFF)
        and ord(char) >= 0x20
        and ord(char) != 0x7F
        for char in text
    )


PLACE_METRICS_V3_FIELDS = {
    "schema", "task_id", "episode_uid", "run_id", "runtime_source_revision",
    "target_body_path", "destination_kind", "destination_body_path",
    "destination_bottom_z", "zone",
    "obj_xyz", "obj_quat", "bbox_center_xyz", "support_z", "obj_xy", "obj_z",
    "tip", "grip", "grip_joint_closure", "pad_contacts", "pad_forces",
    "pad_force_vectors", "contact_force_threshold_n",
    "hold_threshold", "fps", "sample_time_s", "contact_available",
    "release_t", "rendered_frames",
}


def validate_place_metrics_v3(m, *, semantic_only=False):
    """Return structural/identity errors for a v3 place sidecar.

    Physical success remains the job of :func:`check_place_sim`; this validates
    that every signal belongs to one self-described, uniformly sampled run.
    NaN contact samples are structurally allowed so an unavailable sensor is
    reported as a capability failure by the strict scorer rather than hidden as
    a corrupt file.
    """
    try:
        keys = set(m.files if hasattr(m, "files") else m.keys())
    except (AttributeError, TypeError):
        return ["place metrics must be dict-like"]
    required_fields = set(PLACE_METRICS_V3_FIELDS)
    if semantic_only:
        required_fields.difference_update({"run_id", "runtime_source_revision"})
    missing = sorted(required_fields - keys)
    if missing:
        return ["place metrics v3 keys missing: " + ", ".join(missing)]
    errors = []

    def scalar(key, *, nonempty=False, text=False):
        try:
            raw = np.asarray(m[key])
            values = raw.reshape(-1)
            if values.size != 1:
                raise ValueError("not scalar")
            if text and raw.dtype.kind != "U":
                raise ValueError("not unicode text")
            value = values[0]
            if text and not _safe_identity_text(value):
                raise ValueError("unsafe unicode text")
            if nonempty and not str(value).strip():
                raise ValueError("empty")
            return value
        except Exception:
            errors.append(f"{key} must be a{' non-empty' if nonempty else ''} scalar")
            return None

    schema = scalar("schema", nonempty=True, text=True)
    scalar("task_id", nonempty=True, text=True)
    scalar("episode_uid", nonempty=True, text=True)
    if not semantic_only:
        scalar("run_id", nonempty=True, text=True)
        scalar("runtime_source_revision", nonempty=True, text=True)
    scalar("target_body_path", nonempty=True, text=True)
    destination_kind = scalar(
        "destination_kind", nonempty=True, text=True)
    scalar("destination_body_path", nonempty=True, text=True)
    destination_bottom_z = scalar("destination_bottom_z")
    if schema is not None and str(schema) != "real2sim.place_metrics.v3":
        errors.append(f"unexpected place metrics schema {str(schema)!r}")
    if (destination_kind is not None
            and str(destination_kind) not in {"surface", "bin"}):
        errors.append(
            "destination_kind must be exactly 'surface' or 'bin'")
    try:
        bottom_raw = np.asarray(m["destination_bottom_z"])
        if (bottom_raw.shape != ()
                or bottom_raw.dtype.kind not in "fiu"
                or not np.isfinite(float(destination_bottom_z))):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("destination_bottom_z must be one finite real scalar")

    def array(key):
        try:
            return np.asarray(m[key])
        except (TypeError, ValueError):
            errors.append(f"{key} cannot be converted to an array")
            return np.asarray([])

    raw_xyz = array("obj_xyz")
    total = len(raw_xyz) if raw_xyz.ndim else 0
    xyz = (np.asarray(raw_xyz, dtype=float)
           if raw_xyz.dtype.kind in "fiu" else np.empty((0, 3)))
    shaped = {
        "obj_xyz": (raw_xyz, (total, 3)),
        "obj_quat": (array("obj_quat"), (total, 4)),
        "bbox_center_xyz": (array("bbox_center_xyz"), (total, 3)),
        "support_z": (array("support_z"), (total,)),
        "obj_xy": (array("obj_xy"), (total, 2)),
        "obj_z": (array("obj_z"), (total,)),
        "tip": (array("tip"), (total, 3)),
        "grip": (array("grip"), (total,)),
        "grip_joint_closure": (array("grip_joint_closure"), (total,)),
        "pad_contacts": (array("pad_contacts"), (total, 2)),
        "pad_forces": (array("pad_forces"), (total, 2)),
        "pad_force_vectors": (array("pad_force_vectors"), (total, 2, 3)),
        "sample_time_s": (array("sample_time_s"), (total,)),
    }
    if total < 2:
        errors.append("obj_xyz must contain at least two samples")
    for key, (array, expected) in shaped.items():
        if array.shape != expected:
            errors.append(f"{key} shape {array.shape} != {expected}")

    real_numeric_keys = (
        "obj_xyz", "obj_quat", "bbox_center_xyz", "support_z",
        "obj_xy", "obj_z", "tip", "grip", "grip_joint_closure",
        "sample_time_s",
    )
    numeric_arrays = {}
    for key in real_numeric_keys:
        try:
            raw = np.asarray(m[key])
            numeric_arrays[key] = raw
            if raw.dtype.kind not in "fiu":
                errors.append(f"{key} must use a real numeric dtype")
            elif not np.isfinite(raw).all():
                errors.append(f"{key} contains NaN or inf")
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    for key in ("pad_forces", "pad_force_vectors"):
        try:
            raw = np.asarray(m[key])
            if raw.dtype.kind not in "fiu":
                errors.append(f"{key} must use a real numeric dtype")
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    try:
        grip = numeric_arrays.get("grip", np.asarray([]))
        if grip.shape == (total,) and np.any((grip < 0) | (grip > 1)):
            errors.append("grip must be normalized to [0,1]")
    except (TypeError, ValueError):
        pass
    try:
        closure = numeric_arrays.get(
            "grip_joint_closure", np.asarray([]))
        if closure.shape == (total,) and np.any(
                (closure < 0) | (closure > 1)):
            errors.append("grip_joint_closure must be normalized to [0,1]")
    except (TypeError, ValueError):
        pass
    try:
        contacts = np.asarray(m["pad_contacts"])
        if contacts.dtype.kind not in "buif":
            errors.append("pad_contacts must be boolean or numeric")
        elif contacts.shape == (total, 2):
            contacts_float = contacts.astype(float)
            if np.isfinite(contacts_float).all() and np.any(
                    (contacts_float < 0) | (contacts_float > 1)):
                errors.append("pad_contacts must be contact confidence in [0,1]")
    except (TypeError, ValueError):
        errors.append("pad_contacts must be boolean or numeric")

    if (raw_xyz.dtype.kind in "fiu"
            and xyz.shape == (total, 3) and total):
        try:
            obj_xy_array = np.asarray(m["obj_xy"])
            obj_z_array = np.asarray(m["obj_z"])
            if (obj_xy_array.dtype.kind in "fiu"
                    and not np.allclose(obj_xy_array, xyz[:, :2],
                                        atol=1e-9, rtol=0)):
                errors.append("obj_xy is inconsistent with obj_xyz")
            if (obj_z_array.dtype.kind in "fiu"
                    and not np.allclose(obj_z_array, xyz[:, 2],
                                        atol=1e-9, rtol=0)):
                errors.append("obj_z is inconsistent with obj_xyz")
        except (TypeError, ValueError):
            pass
    try:
        quats = np.asarray(m["obj_quat"])
        if (quats.dtype.kind in "fiu" and quats.shape == (total, 4)
                and not np.allclose(
                    np.linalg.norm(quats, axis=1), 1.0,
                    atol=2e-3, rtol=0)):
            errors.append("obj_quat is not normalized")
    except (TypeError, ValueError):
        pass

    try:
        fps_raw = np.asarray(m["fps"])
        if fps_raw.dtype.kind not in "iuf":
            raise ValueError
        fps_values = np.asarray(fps_raw, dtype=float).reshape(-1)
        fps = float(fps_values[0]) if fps_values.size == 1 else float("nan")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError
    except (TypeError, ValueError, IndexError):
        fps = float("nan")
        errors.append("fps must be one finite positive scalar")
    try:
        force_threshold_raw = np.asarray(m["contact_force_threshold_n"])
        if (force_threshold_raw.shape != ()
                or force_threshold_raw.dtype.kind not in "fiu"):
            raise ValueError
        force_threshold = float(force_threshold_raw)
        if not np.isfinite(force_threshold) or force_threshold <= 0:
            raise ValueError
    except (TypeError, ValueError, IndexError):
        force_threshold = float("nan")
        errors.append(
            "contact_force_threshold_n must be one finite positive scalar")
    try:
        hold_threshold_raw = np.asarray(m["hold_threshold"])
        if (hold_threshold_raw.shape != ()
                or hold_threshold_raw.dtype.kind not in "fiu"):
            raise ValueError
        hold_threshold = float(hold_threshold_raw)
        if not np.isfinite(hold_threshold) or not 0 < hold_threshold < 1:
            raise ValueError
    except (TypeError, ValueError, IndexError):
        hold_threshold = float("nan")
        errors.append("hold_threshold must be one finite scalar in (0,1)")
    try:
        times = np.asarray(m["sample_time_s"])
        if (times.dtype.kind in "fiu"
                and times.shape == (total,) and total >= 2):
            delta = np.diff(times)
            if not np.isfinite(delta).all() or np.any(delta <= 0):
                errors.append("sample_time_s must be finite and strictly increasing")
            elif not np.isclose(times[0], 0.0, atol=1e-9, rtol=0):
                errors.append("sample_time_s must start at zero")
            elif np.isfinite(fps) and not np.allclose(delta, 1.0 / fps, atol=1e-6, rtol=1e-4):
                errors.append("sample_time_s cadence does not match fps")
    except (TypeError, ValueError):
        pass

    try:
        zone_raw = np.asarray(m["zone"])
        if zone_raw.dtype.kind not in "fiu":
            raise ValueError
        zone = np.asarray(zone_raw, dtype=float).reshape(-1)
        if (zone.shape != (5,) or not np.isfinite(zone).all()
                or (zone.shape == (5,) and np.any(zone[2:4] < 0))):
            errors.append(
                "zone must contain five finite values with non-negative half extents")
    except (TypeError, ValueError):
        errors.append("zone must be numeric")
    try:
        available = np.asarray(m["contact_available"])
        if available.shape != () or available.dtype.kind != "b":
            raise ValueError
        contact_available = bool(available)
    except (TypeError, ValueError, IndexError):
        contact_available = False
        errors.append("contact_available must be one explicit boolean")
    if contact_available:
        try:
            contacts_raw = np.asarray(m["pad_contacts"])
            forces_raw = np.asarray(m["pad_forces"])
            force_vectors_raw = np.asarray(m["pad_force_vectors"])
            if (contacts_raw.dtype.kind not in "buif"
                    or forces_raw.dtype.kind not in "uif"
                    or force_vectors_raw.dtype.kind not in "uif"):
                raise ValueError("contact evidence has a non-real dtype")
            contacts = np.asarray(contacts_raw, dtype=float)
            forces = np.asarray(forces_raw, dtype=float)
            force_vectors = np.asarray(force_vectors_raw, dtype=float)
            if not np.isfinite(contacts).all():
                errors.append("pad_contacts contains NaN or inf while contact_available is true")
            if not np.isfinite(forces).all():
                errors.append("pad_forces contains NaN or inf while contact_available is true")
            if not np.isfinite(force_vectors).all():
                errors.append(
                    "pad_force_vectors contains NaN or inf while contact_available is true")
            if (forces.shape == (total, 2)
                    and force_vectors.shape == (total, 2, 3)
                    and np.isfinite(forces).all()
                    and np.isfinite(force_vectors).all()
                    and not np.allclose(
                        forces, np.linalg.norm(force_vectors, axis=2),
                        atol=1e-9, rtol=1e-7)):
                errors.append(
                    "pad_forces is inconsistent with pad_force_vectors")
            if (contacts.shape == (total, 2)
                    and forces.shape == (total, 2)
                    and np.isfinite(force_threshold)
                    and np.isfinite(contacts).all()
                    and np.isfinite(forces).all()
                    and not np.array_equal(
                        contacts, (forces >= force_threshold).astype(float))):
                errors.append(
                    "pad_contacts is inconsistent with pad_forces/contact threshold")
        except (TypeError, ValueError):
            errors.append(
                "pad contact/force evidence must be numeric while contact_available is true")
    try:
        rendered_raw = np.asarray(m["rendered_frames"])
        release_raw = np.asarray(m["release_t"])
        if rendered_raw.dtype.kind not in "iu" or release_raw.dtype.kind not in "iu":
            raise ValueError("expected integer values")
        rendered_values = rendered_raw.reshape(-1)
        release_values = release_raw.reshape(-1)
        if rendered_values.size != 1 or release_values.size != 1:
            raise ValueError("expected scalar values")
        rendered = float(rendered_values[0])
        release = float(release_values[0])
        valid_frames = (np.isfinite(rendered) and rendered.is_integer() and rendered > 0)
        valid_release = (np.isfinite(release) and release.is_integer()
                         and 0 <= release <= rendered)
        if not valid_frames:
            errors.append("rendered_frames must be a positive integer")
        elif rendered > total:
            errors.append("rendered_frames cannot exceed trajectory sample count")
        if not valid_release:
            errors.append("release_t must be an integer in [0, rendered_frames]")
        if (valid_frames and valid_release and rendered <= total
                and np.isfinite(hold_threshold)):
            grip_values = np.asarray(m["grip"])
            if (grip_values.dtype.kind in "fiu"
                    and grip_values.shape == (total,)):
                rendered_i, release_i = int(rendered), int(release)
                opens = np.flatnonzero(
                    grip_values[:rendered_i] < hold_threshold)
                expected_release = (
                    int(opens[0]) if opens.size else rendered_i)
                if release_i != expected_release:
                    errors.append(
                        f"release_t {release_i} does not match the first grip "
                        f"sample below hold_threshold ({expected_release})")
                if (total > rendered_i
                        and not np.array_equal(
                            grip_values[rendered_i:],
                            np.full(
                                total - rendered_i,
                                grip_values[rendered_i - 1],
                                dtype=grip_values.dtype))):
                    errors.append(
                        "post-render grip samples must repeat the final rendered command")
    except (TypeError, ValueError, IndexError):
        errors.append("release_t/rendered_frames must be scalar numbers")
    return errors


def _canonical_pick_pad_paths(active_hand):
    """Return the only pad-body identity accepted by pick-contact v1."""
    if active_hand not in ("l", "r"):
        return ()
    return (
        f"/World/Robot/gripper_{active_hand}_outer_link4",
        f"/World/Robot/gripper_{active_hand}_inner_link4",
    )


def _relative_quaternion_wxyz(reference_wxyz, object_wxyz):
    """Return ``conjugate(reference) * object`` for aligned unit arrays."""
    reference = np.asarray(reference_wxyz, dtype=float)
    obj = np.asarray(object_wxyz, dtype=float)
    rw, rx, ry, rz = np.moveaxis(reference, -1, 0)
    ow, ox, oy, oz = np.moveaxis(obj, -1, 0)
    return np.stack((
        rw * ow + rx * ox + ry * oy + rz * oz,
        rw * ox - rx * ow - ry * oz + rz * oy,
        rw * oy + rx * oz - ry * ow - rz * ox,
        rw * oz - rx * oy + ry * ox - rz * ow,
    ), axis=-1)


def _vectors_in_reference_frame(reference_wxyz, vectors_xyz):
    """Rotate world vectors into each unit quaternion's local frame."""
    quaternion = np.asarray(reference_wxyz, dtype=float)
    vectors = np.asarray(vectors_xyz, dtype=float)
    if (
        quaternion.ndim != 2
        or quaternion.shape[1] != 4
        or vectors.shape != (len(quaternion), 3)
    ):
        raise ValueError(
            "reference-frame transform requires aligned (T,4)/(T,3) arrays")
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack((
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ), axis=-1).reshape(-1, 3, 3)
    # Quaternions map local vectors into world.  Apply the transpose so a
    # rigid object/gripper pair remains constant while the wrist turns.
    return np.einsum("tji,tj->ti", rotation, vectors)


def _quaternion_spread_deg(wxyz):
    """Maximum sign-invariant angular distance from the first quaternion."""
    quaternion = np.asarray(wxyz, dtype=float)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or len(quaternion) < 1:
        raise ValueError("quaternion spread requires one non-empty (T,4) array")
    normalized = quaternion / np.linalg.norm(
        quaternion, axis=1, keepdims=True)
    dots = np.clip(
        np.abs(normalized @ normalized[0]),
        0.0,
        1.0,
    )
    return float(np.degrees(2.0 * np.max(np.arccos(dots))))


def validate_pick_metrics_v1(
        m, task_id, target_body_path, *, expected_episode_uid=None,
        expected_run_id=None, expected_pad_body_paths=None,
        expected_contact_force_threshold_n=None,
        expected_grip_closed_reference_rad=None,
        expected_grip_open_reference_rad=None,
        expected_seat_riser_active=None,
        expected_seat_riser_support_height_m=None,
        expected_seat_riser_support_top_m=None,
        expected_support_authority=None,
        **_ignored_legacy_identity):
    """Validate one self-described pick-contact artifact.

    Structural validity is deliberately separate from publication authority.
    Optional ``expected_*`` values are compared when supplied, but callers
    must independently require all of them before treating a passing sequence
    as publishable.  This lets offline inspection report useful v1 sequence
    diagnostics without allowing an NPZ to certify its own identity.
    """
    try:
        keys = set(m.files if hasattr(m, "files") else m.keys())
    except (AttributeError, TypeError):
        return ["pick contact metrics must be dict-like"]
    missing = sorted(set(PICK_CONTACT_V1_REQUIRED_KEYS) - keys)
    if missing:
        return ["pick contact v1 keys missing: " + ", ".join(missing)]
    errors = []

    def text_scalar(key):
        try:
            raw = np.asarray(m[key])
            if raw.shape != () or raw.dtype.kind != "U":
                raise ValueError
            value = str(raw.item())
            if not _safe_identity_text(value):
                raise ValueError
            return value
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{key} must be one non-empty printable Unicode scalar")
            return ""

    schema = text_scalar("schema")
    metric_task = text_scalar("task_id")
    episode_uid = text_scalar("episode_uid")
    run_id = text_scalar("run_id")
    support_mode = text_scalar("support_mode")
    support_authority = text_scalar("support_authority")
    metric_target = text_scalar("target_body_path")
    active_hand = text_scalar("active_hand")

    if schema and schema != "real2sim.pick_contact.v1":
        errors.append(f"unexpected pick contact schema {schema!r}")
    if metric_task and metric_task != str(task_id):
        errors.append(
            f"sidecar task_id {metric_task!r} does not match requested task "
            f"{str(task_id)!r}")
    if metric_target and metric_target != str(target_body_path):
        errors.append(
            f"sidecar target_body_path {metric_target!r} does not match task "
            f"target {str(target_body_path)!r}")
    for key, value in (("episode_uid", episode_uid), ("run_id", run_id)):
        if value and not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
            errors.append(f"{key} contains unsafe characters")
    if active_hand and active_hand not in ("l", "r"):
        errors.append("active_hand must be exactly 'l' or 'r'")
    if support_mode and support_mode not in (
            "native_surface", "seat_riser", "visible_cushion"):
        errors.append(
            "support_mode must be native_surface, seat_riser, or "
            "visible_cushion")
    if support_authority and not re.fullmatch(
            r"[a-z0-9][a-z0-9._:-]{0,127}", support_authority):
        errors.append(
            "support_authority must be one safe lower-case authority id")

    for name, actual, expected in (
        ("episode_uid", episode_uid, expected_episode_uid),
        ("run_id", run_id, expected_run_id),
    ):
        if expected is not None and actual and actual != str(expected):
            errors.append(
                f"sidecar {name} {actual!r} does not match expected "
                f"{str(expected)!r}")
    if (
        expected_support_authority is not None
        and support_authority
        and support_authority != str(expected_support_authority)
    ):
        errors.append(
            f"sidecar support_authority {support_authority!r} does not "
            f"match expected {str(expected_support_authority)!r}")

    try:
        pad_paths_raw = np.asarray(m["pad_body_paths"])
        if pad_paths_raw.dtype.kind != "U" or pad_paths_raw.shape != (2,):
            raise ValueError
        pad_paths = tuple(str(value) for value in pad_paths_raw)
        if not all(_safe_identity_text(path) for path in pad_paths):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        pad_paths = ()
        errors.append(
            "pad_body_paths must contain two printable Unicode paths")
    canonical_pad_paths = _canonical_pick_pad_paths(active_hand)
    if pad_paths and canonical_pad_paths and pad_paths != canonical_pad_paths:
        errors.append(
            f"pad_body_paths {pad_paths!r} do not match active_hand "
            f"{active_hand!r} canonical paths {canonical_pad_paths!r}")
    if expected_pad_body_paths is not None:
        try:
            expected_paths = tuple(
                str(value) for value in expected_pad_body_paths)
        except TypeError:
            expected_paths = ()
        if len(expected_paths) != 2:
            errors.append(
                "expected_pad_body_paths must contain exactly two paths")
        elif pad_paths and pad_paths != expected_paths:
            errors.append(
                f"sidecar pad_body_paths {pad_paths!r} does not match expected "
                f"{expected_paths!r}")
    if "hand" in keys:
        try:
            legacy_hand_raw = np.asarray(m["hand"])
            if (legacy_hand_raw.shape != ()
                    or legacy_hand_raw.dtype.kind != "U"):
                raise ValueError
            legacy_hand = str(legacy_hand_raw.item()).lower()
            if legacy_hand not in ("l", "r") or (
                    active_hand and legacy_hand != active_hand):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(
                "legacy hand must be one Unicode l/r scalar matching "
                "active_hand")

    def array(key):
        try:
            return np.asarray(m[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key} cannot be converted to an array")
            return np.asarray([])

    obj = array("obj_xyz")
    total = len(obj) if obj.ndim else 0
    shaped = {
        "obj_xyz": (obj, (total, 3), "fiu"),
        "object_wxyz": (
            array("object_wxyz"), (total, 4), "fiu"),
        "gripper_wxyz": (
            array("gripper_wxyz"), (total, 4), "fiu"),
        "tip": (array("tip"), (total, 3), "fiu"),
        "grip_joint_closure": (
            array("grip_joint_closure"), (total,), "fiu"),
        "pad_contacts": (array("pad_contacts"), (total, 2), "buif"),
        "pad_force_vectors": (
            array("pad_force_vectors"), (total, 2, 3), "fiu"),
        "sample_time_s": (array("sample_time_s"), (total,), "fiu"),
    }
    if total < 2:
        errors.append("obj_xyz must contain at least two samples")
    for key, (raw, expected_shape, allowed_kinds) in shaped.items():
        if raw.shape != expected_shape:
            errors.append(f"{key} shape {raw.shape} != {expected_shape}")
        if raw.dtype.kind not in allowed_kinds:
            errors.append(f"{key} must use a real numeric dtype")
        elif not np.isfinite(raw).all():
            errors.append(f"{key} contains NaN or inf")
    for key in ("object_wxyz", "gripper_wxyz"):
        quaternion = shaped[key][0]
        if (
            quaternion.shape == (total, 4)
            and quaternion.dtype.kind in "fiu"
            and np.isfinite(quaternion).all()
            and not np.allclose(
                np.linalg.norm(
                    np.asarray(quaternion, dtype=float), axis=1),
                1.0,
                atol=1e-4,
                rtol=0.0,
            )
        ):
            errors.append(f"{key} must contain unit wxyz quaternions")

    try:
        available = np.asarray(m["contact_available"])
        if available.shape != () or available.dtype.kind != "b":
            raise ValueError
        if not bool(available):
            errors.append(
                "contact_available is false; raw two-pad contact is required")
    except (KeyError, TypeError, ValueError):
        errors.append("contact_available must be one explicit boolean")

    def finite_positive_scalar(key):
        try:
            raw = np.asarray(m[key])
            if raw.shape != () or raw.dtype.kind not in "fiu":
                raise ValueError
            value = float(raw)
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            return value
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{key} must be one finite positive real scalar")
            return float("nan")

    fps = finite_positive_scalar("fps")
    force_threshold = finite_positive_scalar("contact_force_threshold_n")
    try:
        grip_closed_reference = np.asarray(
            m["grip_closed_reference_rad"])
        if (
            grip_closed_reference.shape != ()
            or grip_closed_reference.dtype.kind not in "fiu"
            or not np.isfinite(float(grip_closed_reference))
        ):
            raise ValueError
        grip_closed_reference = float(grip_closed_reference)
    except (KeyError, TypeError, ValueError):
        errors.append(
            "grip_closed_reference_rad must be one finite real scalar")
        grip_closed_reference = float("nan")
    try:
        grip_open_reference = np.asarray(
            m["grip_open_reference_rad"])
        if (
            grip_open_reference.shape != ()
            or grip_open_reference.dtype.kind not in "fiu"
            or not np.isfinite(float(grip_open_reference))
        ):
            raise ValueError
        grip_open_reference = float(grip_open_reference)
    except (KeyError, TypeError, ValueError):
        errors.append(
            "grip_open_reference_rad must be one finite real scalar")
        grip_open_reference = float("nan")
    if (
        np.isfinite(grip_closed_reference)
        and np.isfinite(grip_open_reference)
        and grip_open_reference <= grip_closed_reference
    ):
        errors.append(
            "grip references must satisfy open > closed")
    try:
        seat_active_raw = np.asarray(m["seat_riser_active"])
        if seat_active_raw.shape != () or seat_active_raw.dtype.kind != "b":
            raise ValueError
        seat_riser_active = bool(seat_active_raw)
    except (KeyError, TypeError, ValueError):
        errors.append(
            "seat_riser_active must be one explicit boolean")
        seat_riser_active = None
    try:
        seat_height_raw = np.asarray(
            m["seat_riser_support_height_m"])
        if (
            seat_height_raw.shape != ()
            or seat_height_raw.dtype.kind not in "fiu"
        ):
            raise ValueError
        seat_riser_support_height_m = float(seat_height_raw)
        if (
            not np.isfinite(seat_riser_support_height_m)
            or seat_riser_support_height_m < 0.0
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append(
            "seat_riser_support_height_m must be one finite non-negative "
            "real scalar")
        seat_riser_support_height_m = float("nan")
    try:
        seat_top_raw = np.asarray(
            m["seat_riser_support_top_m"])
        if (
            seat_top_raw.shape != ()
            or seat_top_raw.dtype.kind not in "fiu"
        ):
            raise ValueError
        seat_riser_support_top_m = float(seat_top_raw)
        if (
            not np.isfinite(seat_riser_support_top_m)
            or seat_riser_support_top_m < 0.0
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append(
            "seat_riser_support_top_m must be one finite non-negative "
            "real scalar")
        seat_riser_support_top_m = float("nan")
    if (
        seat_riser_active is not None
        and np.isfinite(seat_riser_support_height_m)
        and np.isfinite(seat_riser_support_top_m)
    ):
        expected_support_mode = (
            "visible_cushion"
            if str(task_id) == "53" and not seat_riser_active
            else (
                "seat_riser"
                if seat_riser_active else "native_surface"
            )
        )
        if support_mode and support_mode != expected_support_mode:
            errors.append(
                "support_mode disagrees with seat_riser_active")
        if (
            seat_riser_active
            != (seat_riser_support_height_m > 1e-12)
        ):
            errors.append(
                "SeatRiser active state and support height disagree")
        if seat_riser_active != (seat_riser_support_top_m > 1e-12):
            errors.append(
                "SeatRiser active state and support top disagree")
        if str(task_id) == "53":
            if seat_riser_active:
                errors.append(
                    "Task53 visible cushion forbids an active SeatRiser")
            if support_authority != TASK53_PICK_SUPPORT_AUTHORITY:
                errors.append(
                    "Task53 support_authority is not the pinned visible "
                    "cushion authority")
        elif str(task_id) == "59":
            expected_authority = (
                TASK59_PICK_SUPPORT_AUTHORITY
                if seat_riser_active else "native_surface"
            )
            if support_authority != expected_authority:
                errors.append(
                    "Task59 support_authority is not calibrated for the "
                    "active/native support state")
        elif support_authority != "scene_provenance_only":
            errors.append(
                "non-Task53/59 support authority must be "
                "scene_provenance_only")
    if (
        expected_seat_riser_active is not None
        and (
            not isinstance(expected_seat_riser_active, (bool, np.bool_))
            or seat_riser_active is None
            or seat_riser_active != bool(expected_seat_riser_active)
        )
    ):
        errors.append(
            "sidecar seat_riser_active does not match expected "
            f"{expected_seat_riser_active!r}")
    if expected_seat_riser_support_height_m is not None:
        try:
            expected_seat_height = float(
                expected_seat_riser_support_height_m)
        except (TypeError, ValueError):
            expected_seat_height = float("nan")
        if (
            not np.isfinite(expected_seat_height)
            or not np.isfinite(seat_riser_support_height_m)
            or not np.isclose(
                seat_riser_support_height_m,
                expected_seat_height,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            errors.append(
                "sidecar seat_riser_support_height_m "
                f"{seat_riser_support_height_m!r} does not match expected "
                f"{expected_seat_height!r}")
    if expected_seat_riser_support_top_m is not None:
        try:
            expected_seat_top = float(
                expected_seat_riser_support_top_m)
        except (TypeError, ValueError):
            expected_seat_top = float("nan")
        if (
            not np.isfinite(expected_seat_top)
            or not np.isfinite(seat_riser_support_top_m)
            or not np.isclose(
                seat_riser_support_top_m,
                expected_seat_top,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            errors.append(
                "sidecar seat_riser_support_top_m "
                f"{seat_riser_support_top_m!r} does not match expected "
                f"{expected_seat_top!r}")

    for name, actual, expected in (
        (
            "contact_force_threshold_n",
            force_threshold,
            expected_contact_force_threshold_n,
        ),
        (
            "grip_closed_reference_rad",
            grip_closed_reference,
            expected_grip_closed_reference_rad,
        ),
        (
            "grip_open_reference_rad",
            grip_open_reference,
            expected_grip_open_reference_rad,
        ),
    ):
        if expected is None:
            continue
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            errors.append(
                f"expected_{name} must be one finite real scalar")
            continue
        if (
            not np.isfinite(expected_value)
            or not np.isfinite(actual)
            or not np.isclose(
                actual, expected_value, atol=1e-12, rtol=0.0)
        ):
            errors.append(
                f"sidecar {name} {actual!r} does not match expected "
                f"{expected_value!r}")
    grip = shaped["grip_joint_closure"][0]
    contacts = shaped["pad_contacts"][0]
    forces = shaped["pad_force_vectors"][0]
    times = shaped["sample_time_s"][0]
    if (grip.shape == (total,) and grip.dtype.kind in "fiu"
            and np.isfinite(grip).all()
            and np.any((grip < 0) | (grip > 1))):
        errors.append("grip_joint_closure must be normalized to [0,1]")
    if (contacts.shape == (total, 2)
            and contacts.dtype.kind in "buif"
            and np.isfinite(contacts).all()
            and np.any((contacts < 0) | (contacts > 1))):
        errors.append("pad_contacts must be normalized to [0,1]")
    if (forces.shape == (total, 2, 3)
            and contacts.shape == (total, 2)
            and forces.dtype.kind in "fiu"
            and contacts.dtype.kind in "buif"
            and np.isfinite(forces).all()
            and np.isfinite(contacts).all()
            and np.isfinite(force_threshold)):
        expected_contacts = (
            np.linalg.norm(forces, axis=2) >= force_threshold).astype(float)
        if not np.array_equal(
                contacts.astype(float), expected_contacts):
            errors.append(
                "pad_contacts is inconsistent with raw force vectors/"
                "contact threshold")
    if (times.shape == (total,) and times.dtype.kind in "fiu"
            and np.isfinite(times).all() and total >= 2):
        delta = np.diff(times.astype(float))
        if not np.isclose(float(times[0]), 0.0, atol=1e-9, rtol=0):
            errors.append("sample_time_s must start at zero")
        if np.any(delta <= 0):
            errors.append("sample_time_s must be strictly increasing")
        elif np.isfinite(fps) and not np.allclose(
                delta, 1.0 / fps, atol=1e-6, rtol=1e-4):
            errors.append("sample_time_s cadence does not match fps")
    for alias, expected_values, expected_shape in (
        ("book_z", obj[:, 2] if obj.shape == (total, 3) else None,
         (total,)),
        ("book_xy", obj[:, :2] if obj.shape == (total, 3) else None,
         (total, 2)),
    ):
        if alias not in keys or expected_values is None:
            continue
        try:
            raw_alias = np.asarray(m[alias])
            if (raw_alias.dtype.kind not in "fiu"
                    or raw_alias.shape != expected_shape
                    or not np.isfinite(raw_alias).all()
                    or not np.array_equal(
                        np.asarray(raw_alias, dtype=float),
                        np.asarray(expected_values, dtype=float))):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{alias} must exactly agree with obj_xyz")
    return errors


def check_grab_carry_final_held(
        obj_xyz=None, object_wxyz=None, gripper_wxyz=None, tip=None,
        grip_joint_closure=None, pad_contacts=None,
        pad_force_vectors=None, contact_available=None,
        contact_force_threshold_n=None, fps=FPS, pregrasp_s=0.10,
        hold_s=0.25, final_hold_s=1.0, grip_closed=0.80,
        grip_open=0.20, contact_thresh=0.5, min_lift=0.040,
        attach_std=0.030, attach_drift=0.040, bulldoze_xy=0.080,
        opposition_cos_max=0.5, opposition_min_fraction=0.8,
        final_relative_orientation_drift_deg=20.0):
    """Score an open-pregrasp -> physical carry -> final-held pick sequence.

    Unlike :func:`check_grab_carry_release`, this contract has no release
    phase.  The same measured-closed, target-bilateral run must contain the
    lift and remain present through the final one-second held window.
    """
    out = {
        "success": False,
        "reasons": [],
        "margins": {"sequence_evidence_available": False},
    }
    try:
        available = np.asarray(contact_available)
        if (available.shape != () or available.dtype.kind != "b"
                or not bool(available)):
            raise ValueError
    except (TypeError, ValueError):
        out["reasons"].append(
            "missing raw contact evidence: contact_available must be explicit true")
        return out
    required = {
        "obj_xyz": obj_xyz,
        "object_wxyz": object_wxyz,
        "gripper_wxyz": gripper_wxyz,
        "tip": tip,
        "grip_joint_closure": grip_joint_closure,
        "pad_contacts": pad_contacts,
        "pad_force_vectors": pad_force_vectors,
    }
    missing = [key for key, value in required.items() if value is None]
    if contact_force_threshold_n is None:
        missing.append("contact_force_threshold_n")
    if missing:
        out["reasons"].append(
            "missing pick contact evidence: " + ", ".join(missing))
        return out
    raw_values = {}
    try:
        for key, value in required.items():
            raw = np.asarray(value)
            allowed = "buif" if key == "pad_contacts" else "fiu"
            if raw.dtype.kind not in allowed:
                raise ValueError(f"{key} must use a real numeric dtype")
            raw_values[key] = np.asarray(raw, dtype=float)
        force_threshold = float(np.asarray(contact_force_threshold_n))
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        out["reasons"].append(
            f"invalid pick contact evidence: {exc}")
        return out
    obj = raw_values["obj_xyz"]
    hand = raw_values["tip"]
    grip = raw_values["grip_joint_closure"]
    contacts = raw_values["pad_contacts"]
    forces = raw_values["pad_force_vectors"]
    total = len(obj) if obj.ndim else 0
    expected = {
        "obj_xyz": (obj, (total, 3)),
        "object_wxyz": (
            raw_values["object_wxyz"], (total, 4)),
        "gripper_wxyz": (
            raw_values["gripper_wxyz"], (total, 4)),
        "tip": (hand, (total, 3)),
        "grip_joint_closure": (grip, (total,)),
        "pad_contacts": (contacts, (total, 2)),
        "pad_force_vectors": (forces, (total, 2, 3)),
    }
    bad_shapes = [
        f"{key} {value.shape} != {shape}"
        for key, (value, shape) in expected.items()
        if value.shape != shape
    ]
    if total < 2 or bad_shapes:
        out["reasons"].append(
            "pick contact evidence shape mismatch: "
            + (", ".join(bad_shapes) if bad_shapes else "trajectory too short"))
        return out
    thresholds = {
        "fps": fps,
        "force_threshold": force_threshold,
        "pregrasp_s": pregrasp_s,
        "hold_s": hold_s,
        "final_hold_s": final_hold_s,
        "grip_closed": grip_closed,
        "grip_open": grip_open,
        "contact_thresh": contact_thresh,
        "min_lift": min_lift,
        "attach_std": attach_std,
        "attach_drift": attach_drift,
        "bulldoze_xy": bulldoze_xy,
        "opposition_cos_max": opposition_cos_max,
        "opposition_min_fraction": opposition_min_fraction,
        "final_relative_orientation_drift_deg": (
            final_relative_orientation_drift_deg),
    }
    try:
        thresholds = {
            key: float(np.asarray(value))
            for key, value in thresholds.items()
            if np.asarray(value).shape == ()
            and np.asarray(value).dtype.kind in "fiu"
        }
    except (TypeError, ValueError):
        thresholds = {}
    if (len(thresholds) != 15
            or not np.isfinite(tuple(thresholds.values())).all()
            or thresholds["fps"] <= 0
            or thresholds["force_threshold"] <= 0
            or thresholds["pregrasp_s"] <= 0
            or thresholds["hold_s"] <= 0
            or thresholds["final_hold_s"] <= 0
            or thresholds["min_lift"] < 0
            or thresholds["attach_std"] < 0
            or thresholds["attach_drift"] < 0
            or thresholds["bulldoze_xy"] < 0
            or not 0 <= thresholds["grip_open"] < thresholds["grip_closed"] <= 1
            or not 0 <= thresholds["contact_thresh"] <= 1
            or not -1 <= thresholds["opposition_cos_max"] <= 1
            or not 0 <= thresholds["opposition_min_fraction"] <= 1):
        out["reasons"].append(
            "invalid pick contact thresholds")
        return out
    if thresholds["final_relative_orientation_drift_deg"] < 0:
        out["reasons"].append(
            "invalid pick contact thresholds")
        return out
    fps = thresholds["fps"]
    force_threshold = thresholds["force_threshold"]
    if (not all(np.isfinite(value).all()
                for value, _shape in expected.values())
            or np.any((grip < 0) | (grip > 1))
            or np.any((contacts < 0) | (contacts > 1))):
        out["reasons"].append(
            "pick contact evidence contains non-finite or non-normalized values")
        return out
    object_quaternion = raw_values["object_wxyz"]
    gripper_quaternion = raw_values["gripper_wxyz"]
    if (
        not np.allclose(
            np.linalg.norm(object_quaternion, axis=1),
            1.0,
            atol=1e-4,
            rtol=0.0,
        )
        or not np.allclose(
            np.linalg.norm(gripper_quaternion, axis=1),
            1.0,
            atol=1e-4,
            rtol=0.0,
        )
    ):
        out["reasons"].append(
            "pick contact evidence contains non-unit orientations")
        return out
    measured_contacts = (
        np.linalg.norm(forces, axis=2) >= force_threshold).astype(float)
    if not np.array_equal(contacts, measured_contacts):
        out["reasons"].append(
            "pad contacts do not match raw force vectors/contact threshold")
        return out
    out["margins"]["sequence_evidence_available"] = True

    pregrasp_n = max(
        2, int(np.ceil(thresholds["pregrasp_s"] * fps)))
    hold_n = max(2, int(np.ceil(thresholds["hold_s"] * fps)))
    final_n = max(
        2, int(np.ceil(thresholds["final_hold_s"] * fps)))
    bilateral = np.all(contacts >= thresholds["contact_thresh"], axis=1)
    contact_free = np.all(contacts < thresholds["contact_thresh"], axis=1)
    open_free = (
        (grip <= thresholds["grip_open"]) & contact_free)
    closed_bilateral = (
        (grip >= thresholds["grip_closed"]) & bilateral)
    open_runs = [
        (start, stop) for start, stop in _true_runs(open_free)
        if stop - start >= pregrasp_n
    ]
    terminal_runs = [
        (start, stop) for start, stop in _true_runs(closed_bilateral)
        if stop == total and stop - start >= max(hold_n, final_n)
    ]
    out["margins"].update({
        "required_pregrasp_s": round(thresholds["pregrasp_s"], 3),
        "required_final_held_s": round(thresholds["final_hold_s"], 3),
        "longest_open_free_s": round(
            max((stop - start for start, stop in open_runs), default=0)
            / fps, 3),
        "longest_closed_bilateral_s": round(
            max((stop - start for start, stop in _true_runs(
                closed_bilateral)), default=0) / fps, 3),
    })
    if not terminal_runs:
        out["reasons"].append(
            f"no final {thresholds['final_hold_s']:.2f}s measured-closed "
            "bilateral target hold")
        return out
    hold_start, hold_stop = terminal_runs[-1]
    terminal_bilateral_runs = [
        (start, stop) for start, stop in _true_runs(bilateral)
        if stop == total and start <= hold_start
    ]
    if not terminal_bilateral_runs:
        out["reasons"].append(
            "final measured-closed hold has no raw bilateral contact onset")
        return out
    contact_start, _contact_stop = terminal_bilateral_runs[-1]
    preceding_open = [
        (start, stop) for start, stop in open_runs if stop <= contact_start
    ]
    if not preceding_open:
        out["reasons"].append(
            f"no open/contact-free pre-grasp for "
            f"{thresholds['pregrasp_s']:.2f}s before the final hold")
        return out
    pre_start, pre_stop = preceding_open[-1]
    baseline_z = float(np.median(obj[pre_start:pre_stop, 2]))
    pre_slide = float(np.max(np.linalg.norm(
        obj[:contact_start + 1, :2] - obj[0, :2], axis=1)))
    hold_lift = obj[hold_start:hold_stop, 2] - baseline_z
    peak_offset = int(np.argmax(hold_lift))
    peak_index = hold_start + peak_offset
    peak_lift = float(hold_lift[peak_offset])
    hold_start_lift = float(hold_lift[0])
    contact_start_lift = float(
        obj[contact_start, 2] - baseline_z)
    contact_bound_lift = float(
        obj[peak_index, 2] - obj[contact_start, 2])
    carry_relative = _vectors_in_reference_frame(
        gripper_quaternion[hold_start:hold_stop],
        obj[hold_start:hold_stop] - hand[hold_start:hold_stop],
    )
    carry_attach_std = float(np.linalg.norm(
        np.std(carry_relative, axis=0)))
    carry_attach_drift = float(np.max(np.linalg.norm(
        carry_relative - carry_relative[0], axis=1)))
    carry_relative_quaternion = _relative_quaternion_wxyz(
        gripper_quaternion[hold_start:hold_stop],
        object_quaternion[hold_start:hold_stop],
    )
    carry_relative_orientation_drift = _quaternion_spread_deg(
        carry_relative_quaternion)

    hold_forces = forces[hold_start:hold_stop]
    hold_norms = np.linalg.norm(hold_forces, axis=2)
    hold_cosines = np.sum(
        hold_forces[:, 0] * hold_forces[:, 1], axis=1
    ) / (hold_norms[:, 0] * hold_norms[:, 1])
    opposition_fraction = float(np.mean(
        hold_cosines <= thresholds["opposition_cos_max"]))
    hold_start_opposed = bool(
        hold_cosines[0] <= thresholds["opposition_cos_max"])
    hold_end_opposed = bool(
        hold_cosines[-1] <= thresholds["opposition_cos_max"])

    final_slice = slice(total - final_n, total)
    final_lift = obj[final_slice, 2] - baseline_z
    final_contact_bound_lift = (
        obj[final_slice, 2] - obj[contact_start, 2])
    final_relative = _vectors_in_reference_frame(
        gripper_quaternion[final_slice],
        obj[final_slice] - hand[final_slice],
    )
    final_relative_quaternion = _relative_quaternion_wxyz(
        gripper_quaternion[final_slice],
        object_quaternion[final_slice],
    )
    final_relative_orientation_drift = _quaternion_spread_deg(
        final_relative_quaternion)
    final_attach_std = float(np.linalg.norm(
        np.std(final_relative, axis=0)))
    final_attach_drift = float(np.max(np.linalg.norm(
        final_relative - final_relative[0], axis=1)))
    final_forces = forces[final_slice]
    final_norms = np.linalg.norm(final_forces, axis=2)
    final_cosines = np.sum(
        final_forces[:, 0] * final_forces[:, 1], axis=1
    ) / (final_norms[:, 0] * final_norms[:, 1])
    final_opposition_fraction = float(np.mean(
        final_cosines <= thresholds["opposition_cos_max"]))
    out["margins"].update({
        "pregrasp_start": int(pre_start),
        "pregrasp_stop": int(pre_stop),
        "contact_start": int(contact_start),
        "hold_start": int(hold_start),
        "bilateral_hold_s": round(
            (hold_stop - hold_start) / fps, 3),
        "pre_grasp_slide_mm": round(pre_slide * 1000, 3),
        "contact_start_lift_mm": round(
            contact_start_lift * 1000, 3),
        "hold_start_lift_mm": round(
            hold_start_lift * 1000, 3),
        "max_lift_mm": round(peak_lift * 1000, 3),
        "max_contact_bound_lift_mm": round(
            contact_bound_lift * 1000, 3),
        "attach_std_mm": round(carry_attach_std * 1000, 3),
        "attach_drift_mm": round(carry_attach_drift * 1000, 3),
        "attach_frame": "gripper_local",
        "relative_orientation_drift_deg": round(
            carry_relative_orientation_drift, 4),
        "opposition_fraction": round(opposition_fraction, 4),
        "hold_start_opposed": hold_start_opposed,
        "hold_end_opposed": hold_end_opposed,
        "final_held_s": round(final_n / fps, 3),
        "final_min_lift_mm": round(
            float(np.min(final_lift)) * 1000, 3),
        "final_min_contact_bound_lift_mm": round(
            float(np.min(final_contact_bound_lift)) * 1000, 3),
        "final_attach_std_mm": round(final_attach_std * 1000, 3),
        "final_attach_drift_mm": round(
            final_attach_drift * 1000, 3),
        "final_relative_orientation_drift_deg": round(
            final_relative_orientation_drift, 4),
        "max_final_relative_orientation_drift_deg": round(
            thresholds[
                "final_relative_orientation_drift_deg"], 4),
        "final_opposition_fraction": round(
            final_opposition_fraction, 4),
        "final_pad_contact_fractions": [
            round(float(np.mean(contacts[final_slice, pad]
                                >= thresholds["contact_thresh"])), 4)
            for pad in range(2)
        ],
        "final_pad_force_min_n": [
            round(float(np.min(final_norms[:, pad])), 6)
            for pad in range(2)
        ],
    })

    if pre_slide > thresholds["bulldoze_xy"]:
        out["reasons"].append(
            f"bulldozed before grasp: {pre_slide*1000:.0f} mm > "
            f"{thresholds['bulldoze_xy']*1000:.0f} mm")
    if contact_start_lift >= thresholds["min_lift"]:
        out["reasons"].append(
            "bilateral target contact starts after the object was already "
            "lifted; no contact-bound lift/carry")
    if peak_lift < thresholds["min_lift"]:
        out["reasons"].append(
            f"lift {peak_lift*1000:.0f} mm < "
            f"{thresholds['min_lift']*1000:.0f} mm")
    if contact_bound_lift < thresholds["min_lift"]:
        out["reasons"].append(
            f"contact-bound lift {contact_bound_lift*1000:.0f} mm < "
            f"{thresholds['min_lift']*1000:.0f} mm")
    if carry_attach_std > thresholds["attach_std"]:
        out["reasons"].append(
            f"object-tip attach std {carry_attach_std*1000:.0f} mm > "
            f"{thresholds['attach_std']*1000:.0f} mm")
    if carry_attach_drift > thresholds["attach_drift"]:
        out["reasons"].append(
            f"object-tip attach drift {carry_attach_drift*1000:.0f} mm > "
            f"{thresholds['attach_drift']*1000:.0f} mm")
    if (
        carry_relative_orientation_drift
        > thresholds["final_relative_orientation_drift_deg"]
    ):
        out["reasons"].append(
            "object-gripper relative orientation drift "
            f"{carry_relative_orientation_drift:.1f} deg > "
            f"{thresholds['final_relative_orientation_drift_deg']:.1f} deg")
    if opposition_fraction < thresholds["opposition_min_fraction"]:
        out["reasons"].append(
            f"opposing pad-force fraction {opposition_fraction:.2f} < "
            f"{thresholds['opposition_min_fraction']:.2f}")
    if not hold_start_opposed:
        out["reasons"].append(
            "grasp-start pad forces are not opposing")
    if not hold_end_opposed:
        out["reasons"].append(
            "final pad forces are not opposing")
    if float(np.min(final_lift)) < thresholds["min_lift"]:
        out["reasons"].append(
            f"final held lift {float(np.min(final_lift))*1000:.0f} mm < "
            f"{thresholds['min_lift']*1000:.0f} mm")
    if (float(np.min(final_contact_bound_lift))
            < thresholds["min_lift"]):
        out["reasons"].append(
            f"final contact-bound lift "
            f"{float(np.min(final_contact_bound_lift))*1000:.0f} mm < "
            f"{thresholds['min_lift']*1000:.0f} mm")
    if final_attach_std > thresholds["attach_std"]:
        out["reasons"].append(
            f"final object-tip attach std {final_attach_std*1000:.0f} mm > "
            f"{thresholds['attach_std']*1000:.0f} mm")
    if final_attach_drift > thresholds["attach_drift"]:
        out["reasons"].append(
            f"final object-tip attach drift "
            f"{final_attach_drift*1000:.0f} mm > "
            f"{thresholds['attach_drift']*1000:.0f} mm")
    if (
        final_relative_orientation_drift
        > thresholds["final_relative_orientation_drift_deg"]
    ):
        out["reasons"].append(
            "final object-gripper relative orientation drift "
            f"{final_relative_orientation_drift:.1f} deg > "
            f"{thresholds['final_relative_orientation_drift_deg']:.1f} deg")
    if (final_opposition_fraction
            < thresholds["opposition_min_fraction"]):
        out["reasons"].append(
            f"final opposing pad-force fraction "
            f"{final_opposition_fraction:.2f} < "
            f"{thresholds['opposition_min_fraction']:.2f}")
    out["success"] = not out["reasons"]
    return out


def sim_grip_from_metrics(m):
    """sim 手指真实闭合信号 (audit BLOCKER: 不用录制命令当抓取门控)。
    sim_state 列 7=左手指 15=右手指, hand 键选边。metrics 缺 sim_state 时返回 None。"""
    try:
        files = m.files if hasattr(m, "files") else list(m.keys())
        if "sim_state" not in files or m["sim_state"].shape[0] == 0:
            return None
        hand = str(m["hand"]) if "hand" in files else "l"
        return m["sim_state"][:, 7 if hand.lower().startswith("l") else 15]
    except Exception:
        return None


def _score_or_withhold_pick_contact(
        task_id, metrics, diagnostic, *, expected_episode_uid=None,
        expected_run_id=None, expected_pad_body_paths=None,
        expected_contact_force_threshold_n=None,
        expected_grip_closed_reference_rad=None,
        expected_grip_open_reference_rad=None,
        expected_seat_riser_active=None,
        expected_seat_riser_support_height_m=None,
        expected_seat_riser_support_top_m=None,
        expected_support_authority=None,
        sequence_overrides=None, **_ignored_legacy_identity):
    """Apply the identity-bound v1 scorer or preserve a legacy diagnostic.

    A structurally valid v1 artifact may be semantically inspected while
    unbound, but it cannot produce strict success until every trusted
    ``expected_*`` value comes from outside that artifact.
    """
    registry_entry = _task_registry_entry(str(task_id))
    cfg = {
        "schema": registry_entry["required_evidence_contract"],
        "target_body_path": _throw_target_body_path(str(task_id)),
    }
    try:
        keys = set(
            metrics.files
            if hasattr(metrics, "files")
            else metrics.keys()
        )
    except (AttributeError, TypeError):
        keys = set()
    missing = [
        key for key in PICK_CONTACT_V1_REQUIRED_KEYS
        if key not in keys
    ]

    def scalar_text(key):
        try:
            value = np.asarray(metrics[key])
            return str(value.reshape(-1)[0]) if value.size == 1 else ""
        except (KeyError, TypeError, ValueError, IndexError):
            return ""

    identity_errors = []
    identity_keys = {"schema", "task_id", "target_body_path"}
    for key, expected in (
        ("schema", cfg["schema"]),
        ("task_id", str(task_id)),
        ("target_body_path", cfg["target_body_path"]),
    ):
        if key in keys and scalar_text(key) != expected:
            identity_errors.append(
                f"{key}={scalar_text(key)!r}, expected {expected!r}")

    trusted_values = {
        "expected_episode_uid": expected_episode_uid,
        "expected_run_id": expected_run_id,
        "expected_pad_body_paths": expected_pad_body_paths,
        "expected_contact_force_threshold_n": (
            expected_contact_force_threshold_n),
        "expected_grip_closed_reference_rad": (
            expected_grip_closed_reference_rad),
        "expected_grip_open_reference_rad": (
            expected_grip_open_reference_rad),
        "expected_seat_riser_active": expected_seat_riser_active,
        "expected_seat_riser_support_height_m": (
            expected_seat_riser_support_height_m),
        "expected_seat_riser_support_top_m": (
            expected_seat_riser_support_top_m),
        "expected_support_authority": expected_support_authority,
    }
    missing_trusted_identity = [
        key for key in PICK_CONTACT_V1_TRUSTED_EXPECTATIONS
        if trusted_values[key] is None
    ]
    contract_errors = []
    sequence = None
    if not missing:
        contract_errors = validate_pick_metrics_v1(
            metrics,
            str(task_id),
            cfg["target_body_path"],
            expected_episode_uid=expected_episode_uid,
            expected_run_id=expected_run_id,
            expected_pad_body_paths=expected_pad_body_paths,
            expected_contact_force_threshold_n=(
                expected_contact_force_threshold_n),
            expected_grip_closed_reference_rad=(
                expected_grip_closed_reference_rad),
            expected_grip_open_reference_rad=(
                expected_grip_open_reference_rad),
            expected_seat_riser_active=expected_seat_riser_active,
            expected_seat_riser_support_height_m=(
                expected_seat_riser_support_height_m),
            expected_seat_riser_support_top_m=(
                expected_seat_riser_support_top_m),
            expected_support_authority=expected_support_authority,
        )
        if not contract_errors:
            metric_fps = float(np.asarray(
                _metric(metrics, "fps", FPS)))
            allowed_overrides = {
                "pregrasp_s", "hold_s", "final_hold_s",
                "grip_closed", "grip_open", "contact_thresh",
                "min_lift", "attach_std", "attach_drift",
                "bulldoze_xy", "opposition_cos_max",
                "opposition_min_fraction",
                "final_relative_orientation_drift_deg",
            }
            score_kw = {
                key: value
                for key, value in (sequence_overrides or {}).items()
                if key in allowed_overrides
            }
            sequence = check_grab_carry_final_held(
                obj_xyz=_metric(metrics, "obj_xyz"),
                object_wxyz=_metric(metrics, "object_wxyz"),
                gripper_wxyz=_metric(metrics, "gripper_wxyz"),
                tip=_metric(metrics, "tip"),
                grip_joint_closure=_metric(
                    metrics, "grip_joint_closure"),
                pad_contacts=_metric(metrics, "pad_contacts"),
                pad_force_vectors=_metric(
                    metrics, "pad_force_vectors"),
                contact_available=_metric(
                    metrics, "contact_available"),
                contact_force_threshold_n=_metric(
                    metrics, "contact_force_threshold_n"),
                fps=metric_fps,
                **score_kw,
            )
            diagnostic["margins"].update(sequence["margins"])

    # A present invalid v1 artifact is not trustworthy evidence for the
    # requested run.  A legacy artifact with no v1 header remains usable at
    # the same diagnostic-only level as before.
    artifact_invalid = bool(identity_errors or contract_errors)
    legacy_diagnostic_success = bool(diagnostic.get("success"))
    diagnostic_success = (
        legacy_diagnostic_success and not artifact_invalid)
    trusted_identity_bound = (
        not missing and not contract_errors
        and not missing_trusted_identity)
    sequence_pass = bool(sequence and sequence["success"])
    strict_success = bool(
        trusted_identity_bound and sequence_pass)

    if strict_success:
        assessment = "strict-pass"
        evidence_level = "pick_contact_v1"
        outcome = "pass"
        provisional = False
        reasons = list(sequence["reasons"])
    elif trusted_identity_bound and sequence is not None:
        assessment = "strict-fail"
        evidence_level = "pick_contact_v1"
        outcome = "fail"
        provisional = False
        reasons = list(sequence["reasons"])
    elif artifact_invalid:
        assessment = "diagnostic-fail"
        evidence_level = "invalid_pick_contact_v1"
        outcome = "fail"
        provisional = True
        diagnostic_success = False
        reasons = list(diagnostic["reasons"])
        rejection = "; ".join(identity_errors + contract_errors)
        if identity_errors:
            reasons.append(
                "pick contact identity rejected: " + rejection)
        else:
            reasons.append(
                "pick contact v1 rejected: " + rejection)
    elif not missing and sequence is not None:
        # The sequence can be useful for diagnosis, but self-described
        # Self-described episode/run fields are not publication authority.
        diagnostic_success = bool(
            diagnostic_success or sequence_pass)
        assessment = (
            "diagnostic-pass/provisional"
            if diagnostic_success else "diagnostic-fail")
        evidence_level = "pick_contact_v1_unbound"
        outcome = (
            "diagnostic-pass" if diagnostic_success else "fail")
        provisional = True
        reasons = list(diagnostic["reasons"])
        if not sequence_pass:
            reasons.extend(sequence["reasons"])
        reasons.append(
            "strict PASS withheld: pick_contact.v1 lacks trusted external "
            "identity: " + ", ".join(missing_trusted_identity))
    else:
        assessment = (
            "diagnostic-pass/provisional"
            if diagnostic_success else "diagnostic-fail")
        evidence_level = "legacy_diagnostic"
        outcome = (
            "diagnostic-pass" if diagnostic_success else "fail")
        provisional = True
        reasons = list(diagnostic["reasons"])
        detail = (
            "identity mismatch: " + "; ".join(identity_errors)
            if identity_errors
            else "missing " + ", ".join(missing)
        )
        reasons.append(
            "strict PASS withheld: task "
            f"{task_id} requires identity-bound target-only bilateral raw "
            "opposing pad-force evidence for open pregrasp -> lift/carry -> "
            f"final held (no release); {detail}")

    diagnostic["success"] = strict_success
    diagnostic["diagnostic_success"] = diagnostic_success
    diagnostic["provisional"] = provisional
    diagnostic["assessment"] = assessment
    diagnostic["scope"] = "full_task"
    diagnostic["evidence_level"] = evidence_level
    diagnostic["outcome"] = outcome
    diagnostic["publishable"] = strict_success
    diagnostic["reasons"] = reasons
    diagnostic["margins"].update({
        "assessment": assessment,
        "scope": diagnostic["scope"],
        "evidence_level": evidence_level,
        "outcome": outcome,
        "publishable": strict_success,
        "diagnostic_success": diagnostic_success,
        "required_contact_schema": cfg["schema"],
        "required_target_body_path": cfg["target_body_path"],
        "contact_v1_fields_present": not missing,
        "contact_v1_identity_header_present": identity_keys <= keys,
        "contact_v1_identity_header_matches": (
            identity_keys <= keys and not identity_errors),
        "contact_sequence_scored": sequence is not None,
        "strict_contact_evidence_scored": (
            trusted_identity_bound and sequence is not None),
        "trusted_identity_bound": trusted_identity_bound,
        "missing_trusted_identity": missing_trusted_identity,
        "missing_contact_evidence": missing,
        "contact_identity_errors": identity_errors + contract_errors,
        "contact_contract_errors": contract_errors,
    })
    return diagnostic


def check_task(
        task_id, m, distractors=None, *, distractor_audit=None,
        expected_distractor_prims=None, expected_episode_uid=None,
        expected_run_id=None, expected_pad_body_paths=None,
        expected_destination_contract=None,
        expected_task29_phase0_execution=None,
        expected_task29_release_material_schema=None,
        expected_contact_force_threshold_n=None,
        expected_grip_closed_reference_rad=None,
        expected_grip_open_reference_rad=None,
        expected_place_zone=None,
        expected_seat_riser_active=None,
        expected_seat_riser_support_height_m=None,
        expected_seat_riser_support_top_m=None,
        expected_support_authority=None,
        semantic_only=False,
        **overrides):
    """Dispatch to the family checker with this task's calibrated config.
    Returns the checker's dict(success, reasons, margins) + margins['family'/'task']."""
    tid = str(task_id)
    fam = task_family(tid)
    if fam is None:
        return {"success": False, "reasons": [f"task {tid} not in registry (52=Newton管线, 其余缺资产/数据)"],
                "margins": {"task": tid, "family": None}}
    if fam == "pick":
        kw = thresholds_for_task(tid)
        kw.update({
            key: value for key, value in overrides.items()
            if key in kw
        })
        legacy_pick = {
            key: _metric(m, key)
            for key in ("book_z", "book_xy", "tip", "lgrip")
        }
        missing_legacy = [
            key for key, value in legacy_pick.items()
            if value is None
        ]
        if missing_legacy and tid in CONTACT_GATED_PICK_TASKS:
            res = {
                "success": False,
                "reasons": [
                    "missing legacy pick diagnostic evidence: "
                    + ", ".join(missing_legacy)
                ],
                "margins": {},
            }
        else:
            res = check_pick_sim(
                legacy_pick["book_z"],
                legacy_pick["book_xy"],
                legacy_pick["tip"],
                legacy_pick["lgrip"],
                distractors=distractors,
                sim_grip=sim_grip_from_metrics(m),
                **kw,
            )
        if tid in CONTACT_GATED_PICK_TASKS:
            res = _score_or_withhold_pick_contact(
                tid,
                m,
                res,
                expected_episode_uid=expected_episode_uid,
                expected_run_id=expected_run_id,
                expected_pad_body_paths=expected_pad_body_paths,
                expected_contact_force_threshold_n=(
                    expected_contact_force_threshold_n),
                expected_grip_closed_reference_rad=(
                    expected_grip_closed_reference_rad),
                expected_grip_open_reference_rad=(
                    expected_grip_open_reference_rad),
                expected_seat_riser_active=expected_seat_riser_active,
                expected_seat_riser_support_height_m=(
                    expected_seat_riser_support_height_m),
                expected_seat_riser_support_top_m=(
                    expected_seat_riser_support_top_m),
                expected_support_authority=expected_support_authority,
                sequence_overrides={
                    **kw,
                    **PICK_CONTACT_SEQUENCE_OVERRIDES.get(tid, {}),
                },
            )
    elif fam == "throw":
        cfg = THROW_TASKS[tid]
        obj_xyz = _metric(m, "obj_xyz")
        obj_z = _metric(m, "obj_z", _metric(m, "book_z"))
        obj_xy = _metric(m, "obj_xy", _metric(m, "book_xy"))
        if obj_xyz is not None:
            try:
                xyz = np.asarray(obj_xyz)
                if xyz.ndim == 2 and xyz.shape[1] == 3:
                    if obj_z is None:
                        obj_z = xyz[:, 2]
                    if obj_xy is None:
                        obj_xy = xyz[:, :2]
            except (TypeError, ValueError):
                pass
        if tid == "29":
            task29_override_names = {
                "settle_s",
                "bottom_tol",
                "rim_clearance",
                "settle_drift_tol",
            }
            task29_overrides = {
                key: value for key, value in overrides.items()
                if key in task29_override_names
            }
            res = check_task29_outer_aabb_containment_diagnostic(
                destination_aabb_min=_metric(
                    m, "destination_aabb_min"),
                destination_aabb_max=_metric(
                    m, "destination_aabb_max"),
                destination_bottom_z=_metric(
                    m, "destination_bottom_z"),
                destination_rim_z=_metric(
                    m, "destination_rim_z"),
                object_aabb_min=_metric(m, "object_aabb_min"),
                object_aabb_max=_metric(m, "object_aabb_max"),
                fps=_metric(m, "fps", FPS),
                destination_body_path=_metric(
                    m, "destination_body_path"),
                destination_collider_path=_metric(
                    m, "destination_collider_path"),
                **task29_overrides,
            )
        else:
            kw = dict(
                bins=cfg.get("bins"),
                support_z=_metric(m, "support_z"),
            )
            kw.update({
                key: value for key, value in overrides.items()
                if key in {
                    "fps", "bin_half", "rim_z", "settle_s",
                    "bin_bottom_tol",
                }
            })
            if obj_z is None or obj_xy is None:
                res = {
                    "success": False,
                    "reasons": [
                        "missing throw final-pose evidence: obj_z/obj_xy"],
                    "margins": {"bin": None},
                }
            else:
                res = check_throw_sim(obj_z, obj_xy, **kw)
            exp = cfg.get("expect_bin")
            if (res["success"] and exp
                    and res["margins"].get("bin") != exp):
                res["success"] = False
                res["reasons"].append(
                    f"wrong bin: {res['margins'].get('bin')} "
                    f"(expect {exp})")
        final_condition_pass = bool(res["success"])
        res["margins"]["final_condition_pass"] = final_condition_pass

        if cfg.get("require_sequence"):
            target_body_path = _throw_target_body_path(tid)
            try:
                metric_keys = set(
                    m.files if hasattr(m, "files") else m.keys())
            except (AttributeError, TypeError):
                metric_keys = set()
            v1_header_present = "schema" in metric_keys
            task_required_fields = set(THROW_METRICS_V1_FIELDS)
            if tid == "29":
                task_required_fields.update(
                    THROW_METRICS_V1_TASK29_DESTINATION_FIELDS)
                task_required_fields.update(
                    THROW_METRICS_V1_TASK29_PHASE_FIELDS)
                task_required_fields.update(
                    THROW_METRICS_V1_TASK29_RELEASE_MATERIAL_FIELDS)
            missing_v1_fields = sorted(
                task_required_fields - metric_keys)
            trusted_values = {
                "expected_episode_uid": expected_episode_uid,
                "expected_run_id": expected_run_id,
                "expected_pad_body_paths": expected_pad_body_paths,
            }
            trusted_expectations = list(
                THROW_METRICS_V1_TRUSTED_EXPECTATIONS)
            if tid == "29":
                trusted_expectations.extend(
                    THROW_METRICS_V1_TASK29_TRUSTED_EXPECTATIONS)
            missing_trusted_identity = [
                key for key in trusted_expectations
                if trusted_values[key] is None
            ]
            contract_errors = []
            sequence = None
            if v1_header_present:
                contract_errors = validate_throw_metrics_v1(
                    m, tid, target_body_path,
                    expected_episode_uid=expected_episode_uid,
                    expected_run_id=expected_run_id,
                    expected_pad_body_paths=expected_pad_body_paths,
                    expected_destination_contract=(
                        expected_destination_contract),
                    expected_task29_phase0_execution=(
                        expected_task29_phase0_execution),
                    expected_task29_release_material_schema=(
                        expected_task29_release_material_schema),
                )
            if v1_header_present and not contract_errors:
                metric_fps = _metric(m, "fps", FPS)
                try:
                    metric_fps = float(np.asarray(metric_fps))
                except (TypeError, ValueError):
                    metric_fps = FPS
                sequence = check_grab_carry_release(
                    obj_xyz=_metric(m, "obj_xyz"),
                    tip=_metric(m, "tip"),
                    grip_joint_closure=_metric(
                        m, "grip_joint_closure"),
                    pad_contacts=_metric(m, "pad_contacts"),
                    pad_force_vectors=_metric(
                        m, "pad_force_vectors"),
                    contact_available=_metric(
                        m, "contact_available"),
                    contact_force_threshold_n=_metric(
                        m, "contact_force_threshold_n"),
                    fps=metric_fps,
                    grip_closed=(
                        TASK29_GRIP_CLOSED_THRESHOLD
                        if tid == "29" else 0.80
                    ),
                )
                if tid == "29":
                    sequence["margins"][
                        "task29_grip_closed_threshold"
                    ] = TASK29_GRIP_CLOSED_THRESHOLD
                    phase_hold = (
                        check_task29_phase_hold_through_release(
                            source_frame=_metric(m, "source_frame"),
                            pad_contacts=_metric(m, "pad_contacts"),
                            phase_ordinal=_metric(
                                m, "task29_phase_ordinal", 0),
                        )
                    )
                    sequence["margins"].update(
                        phase_hold["margins"])
                    if not phase_hold["success"]:
                        sequence["success"] = False
                        sequence["reasons"].extend(
                            phase_hold["reasons"])
                res["margins"].update(sequence["margins"])
            else:
                res["margins"]["sequence_evidence_available"] = False

            sequence_pass = bool(sequence and sequence["success"])
            registry_entry = _task_registry_entry(tid)
            # Task 29 is definitionally a three-cup task while this playback
            # follows exactly one target body.  Preserve that boundary even if
            # unrelated registry edits accidentally weaken required_count.
            required_count = (
                3 if tid == "29"
                else int(registry_entry.get("required_count", 1)))
            tracked_count = 1
            partial_scope = (
                tid == "29" or tracked_count < required_count)
            artifact_invalid = bool(
                v1_header_present and contract_errors)
            trusted_identity_bound = bool(
                v1_header_present and not contract_errors
                and not missing_trusted_identity)
            strict_success = bool(
                trusted_identity_bound
                and final_condition_pass
                and sequence_pass
                and not partial_scope)
            diagnostic_success = bool(
                final_condition_pass and not artifact_invalid)

            if strict_success:
                assessment = "strict-pass"
                evidence_level = "throw_metrics_v1"
                outcome = "pass"
                provisional = False
                reasons = list(res["reasons"])
                reasons.extend(sequence["reasons"])
            elif trusted_identity_bound and sequence is not None:
                evidence_level = "throw_metrics_v1"
                if (partial_scope and final_condition_pass
                        and sequence_pass):
                    assessment = "partial-pass/provisional"
                    outcome = "pass"
                elif partial_scope:
                    assessment = "partial-fail"
                    outcome = "fail"
                else:
                    assessment = "strict-fail"
                    outcome = "fail"
                provisional = partial_scope
                reasons = list(res["reasons"])
                reasons.extend(sequence["reasons"])
            elif artifact_invalid:
                assessment = "diagnostic-fail"
                evidence_level = "invalid_throw_metrics_v1"
                outcome = "fail"
                provisional = True
                diagnostic_success = False
                reasons = list(res["reasons"])
                reasons.extend(contract_errors)
                reasons.append(
                    "throw metrics v1 rejected: "
                    + "; ".join(contract_errors))
            elif v1_header_present and sequence is not None:
                assessment = (
                    "diagnostic-pass/provisional"
                    if diagnostic_success else "diagnostic-fail")
                evidence_level = "throw_metrics_v1_unbound"
                outcome = (
                    "diagnostic-pass" if diagnostic_success else "fail")
                provisional = True
                reasons = list(res["reasons"])
                if not sequence_pass:
                    reasons.extend(sequence["reasons"])
                reasons.append(
                    "strict PASS withheld: throw_metrics.v1 lacks trusted "
                    "external identity: "
                    + ", ".join(missing_trusted_identity))
            else:
                assessment = (
                    "diagnostic-pass/provisional"
                    if diagnostic_success else "diagnostic-fail")
                evidence_level = "legacy_diagnostic"
                outcome = (
                    "diagnostic-pass" if diagnostic_success else "fail")
                provisional = True
                reasons = list(res["reasons"])
                reasons.append(
                    "strict PASS withheld: task "
                    f"{tid} requires identity-bound target-only bilateral raw "
                    "opposing pad-force evidence for open pregrasp -> "
                    "carry -> release; missing "
                    + ", ".join(missing_v1_fields))

            if partial_scope:
                reasons.append(
                    f"strict PASS withheld: task {tid} requires "
                    f"{required_count} independently evidenced cups; this "
                    f"playback tracks only {tracked_count}")

            strictly_evidenced_count = (
                1 if (
                    trusted_identity_bound
                    and final_condition_pass
                    and sequence_pass
                ) else 0
            )
            scope = registry_entry.get("probe_scope", "full_task")
            res["success"] = strict_success
            res["diagnostic_success"] = diagnostic_success
            res["provisional"] = provisional
            res["assessment"] = assessment
            res["scope"] = scope
            res["evidence_level"] = evidence_level
            res["outcome"] = outcome
            res["publishable"] = strict_success
            res["reasons"] = reasons
            res["margins"].update({
                "assessment": assessment,
                "required_object_count": required_count,
                "tracked_object_count": tracked_count,
                "strictly_evidenced_object_count": strictly_evidenced_count,
                "probe_scope": scope,
                "scope": scope,
                "evidence_level": evidence_level,
                "outcome": outcome,
                "publishable": strict_success,
                "diagnostic_success": diagnostic_success,
                "required_contact_schema": "real2sim.throw_metrics.v1",
                "required_target_body_path": target_body_path,
                "throw_v1_fields_present": not missing_v1_fields,
                "throw_v1_identity_header_present": v1_header_present,
                "throw_sequence_scored": sequence is not None,
                "strict_throw_evidence_scored": (
                    trusted_identity_bound and sequence is not None),
                "task29_destination_contract_bound": (
                    tid == "29"
                    and trusted_identity_bound),
                "task29_phase0_execution_bound": (
                    tid == "29"
                    and trusted_identity_bound
                    and getattr(
                        expected_task29_phase0_execution,
                        "ordinal",
                        getattr(
                            expected_task29_phase0_execution,
                            "phase_ordinal",
                            None,
                        ),
                    ) == 0),
                "task29_phase_execution_bound": (
                    tid == "29"
                    and trusted_identity_bound),
                "task29_release_material_bound": (
                    tid == "29"
                    and trusted_identity_bound),
                "task29_release_material_schema": (
                    expected_task29_release_material_schema
                    if tid == "29" else None),
                "trusted_identity_bound": trusted_identity_bound,
                "missing_trusted_identity": missing_trusted_identity,
                "missing_throw_evidence": missing_v1_fields,
                "throw_contract_errors": contract_errors,
            })
        else:
            res["diagnostic_success"] = final_condition_pass
            res["provisional"] = False
            res["assessment"] = (
                "strict-pass" if final_condition_pass else "strict-fail")
            res["scope"] = "full_task"
            res["evidence_level"] = "terminal_pose"
            res["outcome"] = "pass" if final_condition_pass else "fail"
            res["publishable"] = final_condition_pass
            res["margins"]["assessment"] = res["assessment"]
            res["margins"].update({
                "scope": res["scope"],
                "evidence_level": res["evidence_level"],
                "outcome": res["outcome"],
                "publishable": res["publishable"],
            })
    elif fam == "place":
        expected_destination = _place_destination(tid)
        expected_zone = tuple(expected_destination["zone"])
        if expected_place_zone is not None:
            layout_zone = np.asarray(expected_place_zone, dtype=float)
            if (
                layout_zone.shape != (5,)
                or not np.isfinite(layout_zone).all()
                or np.any(layout_zone[2:4] <= 0.0)
            ):
                raise ValueError(
                    "expected_place_zone must be one finite positive "
                    "5-vector")
            expected_zone = tuple(float(value) for value in layout_zone)
        expected_target = _place_target_body_path(tid)
        # Family dispatch is a publication/resume boundary: unlike direct
        # check_place_sim fixtures, it must never accept an anonymous legacy
        # mapping with hand-authored contact bits.
        contract_errors = validate_place_metrics_v3(m, semantic_only=semantic_only)
        obj_xyz = _metric(m, "obj_xyz")
        obj_z = _metric(m, "obj_z", _metric(m, "book_z"))
        obj_xy = _metric(m, "obj_xy", _metric(m, "book_xy"))
        grip = _metric(m, "grip")
        grip_source = "grip"
        if grip is None:
            grip = sim_grip_from_metrics(m)
            grip_source = "sim_state"
        if grip is None:
            grip = _metric(m, "lgrip")
            grip_source = "command"
        kw = {
            "obj_xyz": obj_xyz,
            "obj_quat": _metric(m, "obj_quat"),
            "support_z": _metric(m, "support_z"),
            "bbox_center_xyz": _metric(m, "bbox_center_xyz"),
            "release_t": _metric(m, "release_t"),
            "tip": _metric(m, "tip"),
            "grip": grip,
            "pad_contacts": _metric(m, "pad_contacts"),
            "pad_force_vectors": _metric(m, "pad_force_vectors"),
            "contact_force_threshold_n": _metric(
                m, "contact_force_threshold_n"),
            "contact_available": _metric(m, "contact_available"),
            "destination_kind": expected_destination["kind"],
            "destination_bottom_z": expected_destination.get("bottom_z"),
        }
        metric_fps = _metric(m, "fps")
        if metric_fps is not None:
            try:
                kw["fps"] = float(np.asarray(metric_fps).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass
        kw.update(overrides)
        # Per-task destination identity is not a caller-tunable threshold.
        # Keep init_place.destination_for_task as the dispatch source of truth.
        kw["destination_kind"] = expected_destination["kind"]
        kw["destination_bottom_z"] = expected_destination.get("bottom_z")
        identity_reasons = list(contract_errors)
        metric_schema = _metric(m, "schema")
        if metric_schema is not None:
            try:
                schema_values = np.asarray(metric_schema).reshape(-1)
                metric_schema_text = (
                    str(schema_values[0]) if schema_values.size == 1 else "")
            except Exception:
                metric_schema_text = ""
            if metric_schema_text != "real2sim.place_metrics.v3":
                identity_reasons.append(
                    f"unexpected place metrics schema {metric_schema_text!r}")
        metric_task = _metric(m, "task_id")
        if metric_task is not None:
            try:
                task_values = np.asarray(metric_task).reshape(-1)
                metric_task_text = str(task_values[0]) if task_values.size == 1 else ""
            except Exception:
                metric_task_text = ""
            if metric_task_text != tid:
                identity_reasons.append(
                    f"sidecar task_id {metric_task_text!r} does not match requested task {tid}")
        metric_target = _metric(m, "target_body_path")
        if metric_target is not None:
            try:
                target_values = np.asarray(metric_target).reshape(-1)
                metric_target_text = (
                    str(target_values[0]) if target_values.size == 1 else "")
            except Exception:
                metric_target_text = ""
            if metric_target_text != expected_target:
                identity_reasons.append(
                    f"sidecar target_body_path {metric_target_text!r} does not "
                    f"match task {tid} target {expected_target!r}")
        for metric_key, expected_value in (
            ("destination_kind", expected_destination["kind"]),
            ("destination_body_path", expected_destination["prim_path"]),
        ):
            metric_value = _metric(m, metric_key)
            try:
                values = np.asarray(metric_value).reshape(-1)
                metric_text = (
                    str(values[0]) if values.size == 1 else "")
            except Exception:
                metric_text = ""
            if metric_text != str(expected_value):
                identity_reasons.append(
                    f"sidecar {metric_key} {metric_text!r} does not match "
                    f"task {tid} destination {str(expected_value)!r}")
        metric_bottom = _metric(m, "destination_bottom_z")
        try:
            raw_bottom = np.asarray(metric_bottom)
            actual_bottom = (
                float(raw_bottom)
                if (raw_bottom.shape == ()
                    and raw_bottom.dtype.kind in "fiu"
                    and np.isfinite(float(raw_bottom)))
                else float("nan")
            )
        except (TypeError, ValueError):
            actual_bottom = float("nan")
        expected_bottom = float(expected_destination["bottom_z"])
        if (not np.isfinite(actual_bottom)
                or not np.isclose(
                    actual_bottom, expected_bottom, atol=1e-9, rtol=0)):
            identity_reasons.append(
                f"sidecar destination_bottom_z {actual_bottom!r} does not "
                f"match task {tid} destination {expected_bottom!r}")
        metric_zone = _metric(m, "zone")
        if metric_zone is not None:
            try:
                actual_zone = np.asarray(metric_zone, dtype=float).reshape(-1)
                zone_matches = (actual_zone.shape == (5,)
                                and np.isfinite(actual_zone).all()
                                and np.allclose(
                                    actual_zone, np.asarray(expected_zone, dtype=float),
                                    atol=1e-9, rtol=0))
            except (TypeError, ValueError):
                zone_matches = False
                actual_zone = np.asarray([])
            if not zone_matches:
                identity_reasons.append(
                    f"sidecar zone {actual_zone.tolist()} does not match task {tid} "
                    f"zone {list(expected_zone)}")
        if identity_reasons:
            res = {"success": False, "reasons": identity_reasons, "margins": {}}
        else:
            res = check_place_sim(obj_z, obj_xy, expected_zone, **kw)
        res["margins"]["grip_source"] = grip_source
    else:  # lean (t20)
        res = check_lean_sim(m["book_z"], m["book_xy"], _place_zone(tid), **overrides)
    res["margins"]["task"] = tid
    res["margins"]["family"] = fam
    if tid in CURRENT_LAYOUT_SUCCESS_TASKS and not semantic_only:
        layout_errors = current_layout_trace_errors(tid, m)
        res["margins"]["current_layout_errors"] = layout_errors
        if layout_errors:
            layout_was_decisive = bool(
                res.get("success") or res.get("diagnostic_success"))
            res["success"] = False
            res["diagnostic_success"] = False
            res["publishable"] = False
            res["provisional"] = True
            if layout_was_decisive:
                res["assessment"] = "layout-contract-fail"
            res["outcome"] = "fail"
            res["reasons"].extend(
                error for error in layout_errors
                if error not in res["reasons"]
            )
        res["margins"]["current_layout_contract"] = (
            current_layout_success_contract(tid))
        res["margins"]["current_layout_gate"] = (
            "enforced_identity_bound_trace")
    audit_errors = distractor_audit_errors(
        distractor_audit,
        task_id=tid,
        episode_uid=expected_episode_uid,
        expected_prims=expected_distractor_prims,
    )
    if distractor_audit is not None or expected_distractor_prims is not None:
        res["margins"]["distractor_audit_errors"] = audit_errors
        res["margins"]["distractor_no_touch_gate"] = (
            "pass" if not audit_errors else "fail")
    if audit_errors:
        res["success"] = False
        res["diagnostic_success"] = False
        res["publishable"] = False
        res["provisional"] = True
        res["assessment"] = "distractor-audit-fail"
        res["outcome"] = "fail"
        res["reasons"].extend(
            error for error in audit_errors
            if error not in res["reasons"]
        )
    return res


def _selftest_families():
    """throw/place/lean fixture regression (pick fixtures live in _selftest)."""
    n = 80
    # THROW: ends inside teal bin below rim, settled
    z = np.concatenate([np.linspace(0.9, 0.15, n - 25), np.full(25, 0.12)])
    xy = np.tile(BINS["teal"][:2], (n, 1))
    r = check_throw_sim(z, xy)
    assert r["success"], f"throw fixture failed: {r}"
    print("selftest THROW in-bin -> PASS:", r["margins"])
    r = check_throw_sim(np.full(n, 0.033), np.tile((-1.5, -0.5), (n, 1)))
    assert not r["success"]
    print("selftest THROW floor-miss -> correctly FAILED")
    # PLACE: bilateral physical carry, release/retract, then rest on table.
    zone = (-2.10, -0.40, 0.34, 0.52, 0.423)
    xyz = np.empty((n, 3))
    xyz[:40] = np.linspace((-1.55, 0.10, 0.72), (-2.20, -0.50, 0.456), 40)
    xyz[40:] = (-2.20, -0.50, 0.456)
    tip = xyz + (0.0, 0.0, 0.08)
    tip[40:] = np.linspace(tip[39], (-2.05, -0.50, 0.70), n - 40)
    grip = np.zeros(n); grip[:40] = 1.0
    contacts = np.zeros((n, 2)); contacts[:40] = 1.0
    r = check_place_sim(obj_xyz=xyz, obj_xy=xyz[:, :2], zone=zone, tip=tip,
                        grip=grip, pad_contacts=contacts, contact_available=True)
    assert r["success"], f"place fixture failed: {r}"
    print("selftest PLACE physical carry/release -> PASS")
    floor_xyz = xyz.copy()
    floor_xyz[40:] = (-2.80, -1.50, 0.033)
    r = check_place_sim(obj_xyz=floor_xyz, obj_xy=floor_xyz[:, :2], zone=zone,
                        tip=tip, grip=grip, pad_contacts=contacts, contact_available=True)
    assert not r["success"]
    print("selftest PLACE dropped-on-floor -> correctly FAILED")
    # LEAN: center held at 0.15 in zone = leaning; 0.02 = flat on floor
    lz = (-2.10, -0.85, 0.50, 0.45, 0.05)
    r = check_lean_sim(np.full(n, 0.15), np.tile((-2.1, -0.9), (n, 1)), lz)
    assert r["success"], f"lean fixture failed: {r}"
    print("selftest LEAN upright -> PASS")
    r = check_lean_sim(np.full(n, 0.02), np.tile((-2.1, -0.9), (n, 1)), lz)
    assert not r["success"]
    print("selftest LEAN flat-on-floor -> correctly FAILED")
    # dispatcher wiring: registry covers every task the runners emit
    fam_counts = {}
    for t in sorted(PICK_TASKS | set(THROW_TASKS) | PLACE_TASKS | LEAN_TASKS, key=lambda s: int(s)):
        fam_counts.setdefault(task_family(t), []).append(t)
    print("selftest REGISTRY:", {k: len(v) for k, v in fam_counts.items()})


if __name__ == "__main__":
    raise SystemExit(main())

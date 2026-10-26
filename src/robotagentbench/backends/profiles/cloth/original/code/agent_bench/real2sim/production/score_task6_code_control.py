#!/usr/bin/env python3
"""Strict state-bound scorer for benchmark Task-6 code-control places."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
TELEOP = REPO / "real2sim" / "teleop"
if str(TELEOP) not in sys.path:
    sys.path.insert(0, str(TELEOP))

from check_success import check_place_sim, validate_place_metrics_v3  # noqa: E402
from task6_code_control_contract import mode  # noqa: E402
from coat_rack_contract import (  # noqa: E402
    BRANCH_RADIUS_M,
    HANG_BRANCH_JUNCTION_XYZ_M,
    HANG_BRANCH_SHAFT_PATH,
    HANG_BRANCH_TIP_CENTER_XYZ_M,
    HANG_BRANCH_TIP_PATH,
)
from task6_drawer_sequence_contract import (  # noqa: E402
    DRAWER_PATH,
    FRONT_WALL_CONTACT_AABB_MARGIN_M,
    MAX_FINAL_ARM_HOME_ERROR_RAD,
    MAX_FRONT_WALL_CONTACT_X_ERROR_M,
    MAX_FINAL_RETURN_ERROR_M,
    MAX_LATERAL_DRIFT_M,
    MAX_RIGHT_PAD_FORCE_COSINE,
    MAX_UNCLAMPED_PULL_MOTION_M,
    MIN_CLAMPED_PULL_MOTION_RATIO,
    MIN_DRAWER_PULL_MOTION_STEP_M,
    MIN_PULL_TRAVEL_M,
    MIN_RIGHT_PAD_CONTACT_FRAMES,
    MIN_RIGHT_PAD_FORCE_N,
    GRIP_OPEN_START_FRAME,
    PULL_START_FRAME,
    PUSH_START_FRAME,
    TASK65_DRAWER_PUSH_START_FRAME,
    TASK65_RIGHT_GRIP_OPEN_START_FRAME,
)


SCHEMA = "real2sim.task6_code_control_score.v1"
STATE_METRICS_TOLERANCE_M = 5e-6
MAX_ONE_FRAME_DISPLACEMENT_M = 0.15
MAX_HANG_TAIL_SPEED_MPS = 0.12
MAX_HANG_TAIL_SPAN_M = 0.04
MIN_HANG_TIP_SEPARATION_M = 0.10
HAT_BRIM_CONTACT_LOCAL_MIN_XYZ_M = np.asarray(
    [-0.060, 1.760, -0.120], dtype=float)
HAT_BRIM_CONTACT_LOCAL_MAX_XYZ_M = np.asarray(
    [0.060, 1.825, -0.060], dtype=float)


def _scalar(values: dict[str, np.ndarray], key: str) -> Any:
    value = np.asarray(values[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar")
    return value.item()


def _load(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _segment_intersects_aabb(
    start: np.ndarray,
    end: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
    """Slab-test one finite segment against one axis-aligned box."""
    direction = end - start
    lower = 0.0
    upper = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= 1.0e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        first = (minimum[axis] - start[axis]) / direction[axis]
        second = (maximum[axis] - start[axis]) / direction[axis]
        axis_min, axis_max = sorted((float(first), float(second)))
        lower = max(lower, axis_min)
        upper = min(upper, axis_max)
        if lower > upper:
            return False
    return True


def _score_hat_hang(
    metrics: dict[str, np.ndarray],
    velocity: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    required = {
        "bbox_center_xyz", "bbox_min_xyz", "bbox_max_xyz", "support_z",
        "tip", "grip", "pad_contacts", "release_t",
        "hat_rack_contact_body_paths",
        "hat_rack_contact_force_vectors",
        "hat_rack_contact_counts",
        "contact_force_threshold_n",
        "object_width", "clamp_slip", "clamp_bilateral_contact",
        "clamp_pad_contact_counts",
        "clamp_pad_contact_local_mean_xyz",
        "seat_long_m", "seat_pinch_m",
        "grasp_pinch_axis", "grasp_upright_axis",
        "grasp_long_axis_sign", "grasp_long_plane_rotation_deg",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError("hat-hang metrics lack " + ", ".join(missing))
    centers = np.asarray(metrics["bbox_center_xyz"], dtype=float)
    minimum = np.asarray(metrics["bbox_min_xyz"], dtype=float)
    maximum = np.asarray(metrics["bbox_max_xyz"], dtype=float)
    support_z = np.asarray(metrics["support_z"], dtype=float)
    tip = np.asarray(metrics["tip"], dtype=float)
    grip = np.asarray(metrics["grip"], dtype=float)
    contacts = np.asarray(metrics["pad_contacts"], dtype=float)
    rack_paths = [
        str(value)
        for value in np.asarray(
            metrics["hat_rack_contact_body_paths"]).tolist()
    ]
    rack_forces = np.asarray(
        metrics["hat_rack_contact_force_vectors"], dtype=float)
    rack_counts = np.asarray(
        metrics["hat_rack_contact_counts"], dtype=np.int64)
    contact_threshold = float(
        _scalar(metrics, "contact_force_threshold_n"))
    object_width = float(_scalar(metrics, "object_width"))
    clamp_slip = float(_scalar(metrics, "clamp_slip"))
    clamp_bilateral = bool(
        _scalar(metrics, "clamp_bilateral_contact"))
    clamp_contact_counts = np.asarray(
        metrics["clamp_pad_contact_counts"], dtype=np.int64)
    clamp_contact_local = np.asarray(
        metrics["clamp_pad_contact_local_mean_xyz"], dtype=float)
    seat_long = float(_scalar(metrics, "seat_long_m"))
    seat_pinch = float(_scalar(metrics, "seat_pinch_m"))
    grasp_pinch_axis = int(_scalar(metrics, "grasp_pinch_axis"))
    grasp_upright_axis = int(_scalar(metrics, "grasp_upright_axis"))
    grasp_long_axis_sign = float(
        _scalar(metrics, "grasp_long_axis_sign"))
    grasp_long_plane_rotation_deg = float(
        _scalar(metrics, "grasp_long_plane_rotation_deg"))
    frames = len(centers)
    if not (
        centers.shape == minimum.shape == maximum.shape == (frames, 3)
        and support_z.shape == grip.shape == (frames,)
        and tip.shape == (frames, 3)
        and contacts.shape == (frames, 2)
        and rack_paths == [
            HANG_BRANCH_SHAFT_PATH,
            HANG_BRANCH_TIP_PATH,
        ]
        and rack_forces.shape == (frames, 2, 3)
        and rack_counts.shape == (frames, 2)
        and velocity.shape == (frames, 3)
        and np.isfinite(centers).all()
        and np.isfinite(minimum).all()
        and np.isfinite(maximum).all()
        and np.isfinite(support_z).all()
        and np.isfinite(tip).all()
        and np.isfinite(grip).all()
        and np.isfinite(contacts).all()
        and np.isfinite(rack_forces).all()
        and np.all(rack_counts >= 0)
        and np.isfinite(contact_threshold)
        and contact_threshold > 0.0
        and np.isfinite(object_width)
        and np.isfinite(clamp_slip)
        and clamp_contact_counts.shape == (2,)
        and np.all(clamp_contact_counts > 0)
        and clamp_contact_local.shape == (2, 3)
        and np.isfinite(clamp_contact_local).all()
        and np.isfinite(seat_long)
        and np.isfinite(seat_pinch)
        and np.isfinite(grasp_long_axis_sign)
        and np.isfinite(grasp_long_plane_rotation_deg)
    ):
        raise ValueError("hat-hang evidence has invalid shape or non-finite values")
    release = int(_scalar(metrics, "release_t"))
    if not 3 <= release < frames - 3:
        raise ValueError(f"invalid hat release frame {release}")
    tail = min(20, frames - release)
    tail_slice = slice(frames - tail, frames)
    branch_start = np.asarray(HANG_BRANCH_JUNCTION_XYZ_M, dtype=float)
    branch_end = np.asarray(HANG_BRANCH_TIP_CENTER_XYZ_M, dtype=float)
    overlap = np.asarray([
        _segment_intersects_aabb(
            branch_start,
            branch_end,
            minimum[index] - BRANCH_RADIUS_M,
            maximum[index] + BRANCH_RADIUS_M,
        )
        for index in range(frames)
    ], dtype=bool)
    bilateral = np.all(contacts >= 0.5, axis=1)
    rack_force_contact = np.linalg.norm(
        rack_forces, axis=2) >= contact_threshold
    rack_physical_contact = np.any(
        rack_force_contact | (rack_counts > 0), axis=1)
    brim_contact_patch = bool(
        np.all(
            clamp_contact_local
            >= HAT_BRIM_CONTACT_LOCAL_MIN_XYZ_M[None, :]
        )
        and np.all(
            clamp_contact_local
            <= HAT_BRIM_CONTACT_LOCAL_MAX_XYZ_M[None, :]
        )
    )
    reviewed_brim_slice = bool(
        grasp_pinch_axis == 2
        and grasp_upright_axis == 1
        and np.isclose(object_width, 0.020, atol=0.002, rtol=0.0)
        and np.isclose(seat_long, 0.063, atol=0.003, rtol=0.0)
        and np.isclose(seat_pinch, 0.113, atol=0.003, rtol=0.0)
        and grasp_long_axis_sign == 1.0
        and np.isclose(
            grasp_long_plane_rotation_deg, 45.0,
            atol=1.0e-6, rtol=0.0)
    )
    tail_speed = float(np.max(np.linalg.norm(
        velocity[tail_slice], axis=1)))
    tail_span = float(np.max(np.linalg.norm(
        centers[tail_slice] - centers[-1], axis=1)))
    tail_tip_separation = float(np.min(np.linalg.norm(
        tip[tail_slice] - centers[tail_slice], axis=1)))
    gates = {
        "physical_bilateral_brim_clamp": bool(
            clamp_bilateral
            and brim_contact_patch
            and reviewed_brim_slice),
        "held_before_release": bool(np.count_nonzero(
            bilateral[:release]) >= 2),
        "physical_open_release": bool(
            np.max(grip[: max(1, release)]) >= 0.80
            and np.max(grip[tail_slice]) <= 0.10),
        "released_from_pad_contact": bool(
            not np.any(bilateral[tail_slice])),
        "branch_intersects_hat_collision_envelope": bool(
            np.all(overlap[tail_slice])),
        "physical_hat_rack_contact_after_release": bool(
            np.count_nonzero(
                rack_physical_contact[release:]) >= 2),
        "physical_hat_rack_contact_in_tail": bool(
            np.count_nonzero(
                rack_physical_contact[tail_slice]) >= 1),
        "hangs_above_floor": bool(np.min(support_z[tail_slice]) >= 0.65),
        "hand_separated_after_release": bool(
            tail_tip_separation >= MIN_HANG_TIP_SEPARATION_M),
        "stable_hang_tail_speed": bool(
            tail_speed <= MAX_HANG_TAIL_SPEED_MPS),
        "stable_hang_tail_span": bool(
            tail_span <= MAX_HANG_TAIL_SPAN_M),
    }
    return {
        "success": all(gates.values()),
        "gates": gates,
        "reasons": sorted(key for key, passed in gates.items() if not passed),
        "margins": {
            "tail_frames": tail,
            "tail_duration_s": tail / fps,
            # BaseballCap's authored rigid-body origin is about 1.84 m from
            # its collision-envelope center.  Rotation while the dynamic brim
            # settles therefore produces a large root-origin Z delta even
            # when the physical hat stays at the pads.  Preserve it only as a
            # diagnostic; the body-local brim contacts, pre-release bilateral
            # carry trace and no-teleport gate are the valid authorities.
            "legacy_root_origin_vertical_delta_not_scored_m": clamp_slip,
            "initial_clamp_contact_counts": clamp_contact_counts,
            "initial_clamp_contact_local_mean_xyz_m": (
                clamp_contact_local),
            "reviewed_brim_slice": reviewed_brim_slice,
            "brim_contact_patch": brim_contact_patch,
            "tail_max_speed_mps": tail_speed,
            "tail_span_m": tail_span,
            "tail_min_tip_separation_m": tail_tip_separation,
            "tail_min_support_z_m": float(np.min(support_z[tail_slice])),
            "tail_branch_overlap_frames": int(np.count_nonzero(
                overlap[tail_slice])),
            "post_release_hat_rack_contact_frames": int(
                np.count_nonzero(rack_physical_contact[release:])),
            "tail_hat_rack_contact_frames": int(
                np.count_nonzero(rack_physical_contact[tail_slice])),
            "tail_hat_rack_max_force_n": float(np.max(
                np.linalg.norm(rack_forces[tail_slice], axis=2),
                initial=0.0,
            )),
        },
    }


def _score_drawer_containment(
    state: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    selected: Any,
) -> dict[str, Any]:
    """Check the whole recorded collision envelope against the moving drawer.

    Interior XY is the producer-measured bottom-mesh footprint; its lower Z
    is the bottom's top face and upper Z is the drawer lip.  These are existing
    rollout observations, not the generic zone or a newly invented task box.
    They do not establish agreement with the rendered mesh or detailed wall
    clearance.  Boundary violations up to 10 mm therefore remain unverified;
    this affects failure classification only and never enlarges the pass box.
    """
    required = {
        "bbox_min_xyz", "bbox_max_xyz", "bbox_center_xyz", "support_z",
        "obj_xyz", "obj_quat", "task28_drawer_xyz", "task28_drawer_pos0_xyz",
        "task28_drawer_interior_min0_xyz_m",
        "task28_drawer_interior_max0_xyz_m",
        "task6_destination_aabb_min_xyz_m",
        "task6_destination_aabb_max_xyz_m", "destination_collision_lod_path",
        "destination_bottom_z", "fps",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError("drawer containment evidence lacks " + ", ".join(missing))
    for key in ("prop_xyz_m", "prop_wxyz", "rigid_body_paths", "frame_index"):
        if key not in state:
            raise ValueError("drawer containment state lacks " + key)
    frames = len(state["frame_index"])
    fps = float(_scalar(metrics, "fps"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("drawer containment fps is invalid")
    tail_frames = int(np.ceil(fps))  # Same 1 s settle window as check_place_sim.
    if frames < tail_frames:
        raise ValueError("drawer containment lacks a complete final settle window")

    def array(key: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(metrics[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("drawer containment invalid " + key)
        return value

    minimum = array("bbox_min_xyz", (frames, 3))
    maximum = array("bbox_max_xyz", (frames, 3))
    center = array("bbox_center_xyz", (frames, 3))
    support = array("support_z", (frames,))
    drawer = array("task28_drawer_xyz", (frames, 3))
    origin = array("task28_drawer_pos0_xyz", (3,))
    interior_min = array("task28_drawer_interior_min0_xyz_m", (3,))
    interior_max = array("task28_drawer_interior_max0_xyz_m", (3,))
    outer_min = array("task28_drawer_outer_min0_xyz_m" if "target_collision_points_body_local_m" in metrics else "task6_destination_aabb_min_xyz_m", (3,))
    outer_max = array("task28_drawer_outer_max0_xyz_m" if "target_collision_points_body_local_m" in metrics else "task6_destination_aabb_max_xyz_m", (3,))
    xyz = array("obj_xyz", (frames, 3))
    quat = array("obj_quat", (frames, 4))
    tol = STATE_METRICS_TOLERANCE_M
    if (
        np.any(minimum >= maximum)
        or np.any(interior_min >= interior_max)
        or np.any(outer_min >= outer_max)
        or np.any(interior_min < outer_min - tol)
        or np.any(interior_max > outer_max + tol)
        or not np.allclose(center, (minimum + maximum) / 2, atol=tol, rtol=0)
        or not np.allclose(support, minimum[:, 2], atol=tol, rtol=0)
        or not np.allclose(origin, drawer[0], atol=tol, rtol=0)
        or abs(interior_min[2] - float(_scalar(metrics, "task28_drawer_bottom_surface0_z_m" if "target_collision_points_body_local_m" in metrics else "destination_bottom_z"))) > tol
        or abs(interior_max[2] - outer_max[2]) > tol
        or not str(_scalar(metrics, "destination_collision_lod_path")).startswith(
            selected.destination_path + "/")
    ):
        raise ValueError("drawer containment geometry observations are inconsistent")
    paths = [str(p) for p in np.asarray(state["rigid_body_paths"]).tolist()]
    if paths.count(selected.target_path) != 1 or paths.count(selected.destination_path) != 1:
        raise ValueError("drawer containment target/destination binding is not unique")
    positions = np.asarray(state["prop_xyz_m"], dtype=float)
    quaternions = np.asarray(state["prop_wxyz"], dtype=float)
    if (
        positions.shape != (frames, len(paths), 3)
        or quaternions.shape != (frames, len(paths), 4)
    ):
        raise ValueError("drawer containment state pose shape is invalid")
    ti, di = paths.index(selected.target_path), paths.index(selected.destination_path)
    target_q, drawer_q = quaternions[:, ti], quaternions[:, di]
    if (
        not np.isfinite(target_q).all() or not np.isfinite(drawer_q).all()
        or not np.allclose(np.linalg.norm(target_q, axis=1), 1, atol=2e-3, rtol=0)
        or not np.allclose(np.linalg.norm(drawer_q, axis=1), 1, atol=2e-3, rtol=0)
        or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=2e-3, rtol=0)
        or not np.allclose(positions[:, ti], xyz, atol=tol, rtol=0)
        or not np.allclose(positions[:, di], drawer, atol=tol, rtol=0)
        or np.max(np.minimum(
            np.max(np.abs(target_q - quat), axis=1),
            np.max(np.abs(target_q + quat), axis=1))) > tol
    ):
        raise ValueError("drawer containment pose disagrees with full_state")
    # Transform the entire world AABB into the recorded initial basket
    # frame. This conservatively encloses the target when the free basket
    # rotates; it never enlarges the measured interior to make a pass.
    def rotation_matrices(q):
        q=q/np.linalg.norm(q,axis=1)[:,None]
        w,x,y,z=q.T
        return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                         2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                         2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(-1,3,3)
    rotations=rotation_matrices(drawer_q)
    corners=np.stack([np.where(np.asarray(bits),maximum,minimum)
                      for bits in __import__('itertools').product((0,1),repeat=3)],axis=1)
    if "target_collision_points_body_local_m" in metrics:
        vertices=np.asarray(metrics["target_collision_points_body_local_m"],dtype=float)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
            raise ValueError("invalid captured target collision vertices")
        corners=np.einsum('tij,kj->tki',rotation_matrices(target_q),vertices)+xyz[:,None,:]
        if not np.allclose(corners.min(axis=1),minimum,atol=tol,rtol=0) or not np.allclose(corners.max(axis=1),maximum,atol=tol,rtol=0):
            raise ValueError("captured collision vertices disagree with per-frame envelope")
    initial_frame=np.einsum('ij,tkj->tki',rotations[0],
        np.einsum('tji,tkj->tki',rotations,corners-drawer[:,None,:]))+origin
    margins=np.concatenate((initial_frame.min(axis=1)-interior_min,
                            interior_max-initial_frame.max(axis=1)),axis=1)
    displacement=drawer-origin
    moving_min=interior_min+displacement
    moving_max=interior_max+displacement
    tail = margins[-tail_frames:]
    # Preserve existing bottom contact tolerance and rim clearance; never use
    # a visual/proxy uncertainty allowance to expand lateral containment.
    required_clearance = np.asarray([0., 0., -0.002, 0., 0., 0.002])
    minimum_margins = np.min(tail, axis=0)
    side_names = ["xmin", "ymin", "bottom", "xmax", "ymax", "rim"]
    failed_sides = [name for name, value, required_value in
                    zip(side_names, minimum_margins, required_clearance)
                    if value < required_value - tol]
    maximum_outside = max(0., -float(np.min(tail)))
    if not failed_sides:
        status, reasons = "passed", []
    elif maximum_outside > 0.010:
        status = "failed"
        reasons = ["target collision envelope substantially outside measured drawer interior"]
    else:
        status = "unverified"
        reasons = ["drawer boundary clearance is unverified: measured collision envelope "
                   "crosses or lacks clearance at the recorded floor-footprint/rim; "
                   "render-mesh and wall-proxy agreement was not captured"]
    return {
        "status": status,
        "success": status == "passed",
        "reasons": reasons,
        "failed_sides": failed_sides,
        "margins": {
            "tail_start_frame": frames - tail_frames,
            "tail_end_frame_exclusive": frames,
            "side_order": side_names,
            "minimum_tail_signed_clearance_m": minimum_margins.tolist(),
            "required_clearance_m": required_clearance.tolist(),
            "maximum_tail_outside_m": maximum_outside,
            "final_target_min_xyz_m": minimum[-1].tolist(),
            "final_target_max_xyz_m": maximum[-1].tolist(),
            "final_drawer_interior_min_xyz_m": moving_min[-1].tolist(),
            "final_drawer_interior_max_xyz_m": moving_max[-1].tolist(),
            "basis": "captured_collision_vertices_in_recorded_initial_basket_frame" if "target_collision_points_body_local_m" in metrics else "conservative_aabb_in_recorded_initial_basket_frame",
            "render_mesh_collision_agreement": "unverified_not_captured",
            "detailed_wall_clearance": "unverified_not_captured",
        },
    }


def _score_drawer_sequence(
    state: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    destination_result: dict[str, Any],
    *,
    object_label: str,
    semantic_only: bool = False,
) -> dict[str, Any]:
    """Require the physical right-hand pull/place/push sequence."""
    if "drawer_contacts_json" in state:
        from real2sim.production.drawer_recovery_validation import evaluate_drawer_recovery
        recovery = evaluate_drawer_recovery(state, "6.5" if object_label == "bill" else "6.15")
        drawer = np.asarray(metrics["task28_drawer_xyz"], dtype=float)
        release = int(destination_result.get("margins", {}).get("release_frame", len(drawer)))
        pull = drawer[:,0] - drawer[0,0]
        gates = dict(recovery["gates"])
        gates.update({f"{object_label}_destination_and_settle":bool(destination_result.get("success")),
                      "basket_open_at_object_release":bool(0 <= release < len(drawer) and pull[release] >= MIN_PULL_TRAVEL_M)})
        return dict(success=all(gates.values()),gates=gates,
                    reasons=[k for k,v in gates.items() if not v] + list(destination_result.get("reasons", [])),
                    margins={"drawer_recovery":recovery,"destination":destination_result.get("margins", {})})
    required_metrics = {
        "task28_drawer_xyz", "release_t",
        "task6_destination_aabb_min_xyz_m",
        "task6_destination_aabb_max_xyz_m",
    }
    required_state = {
        "contact_target_paths", "target_pad_contact_force_vector_n",
        "target_pad_contact_count", "target_pad_contact_position_mean_m",
        "command_arm_rad", "command_grip_rad",
    }
    missing = sorted(
        (required_metrics - set(metrics))
        | (required_state - set(state))
    )
    if missing:
        raise ValueError(
            "drawer-sequence evidence lacks " + ", ".join(missing))
    drawer = np.asarray(metrics["task28_drawer_xyz"], dtype=float)
    frames = int(len(state["frame_index"]))
    if (
        drawer.shape != (frames, 3)
        or not np.isfinite(drawer).all()
    ):
        raise ValueError("drawer trace is invalid")
    target_paths = [
        str(value)
        for value in np.asarray(state["contact_target_paths"]).tolist()
    ]
    if target_paths != [DRAWER_PATH]:
        raise ValueError(
            "drawer contact target path does not match the contract")
    force = np.asarray(
        state["target_pad_contact_force_vector_n"], dtype=float)
    count = np.asarray(
        state["target_pad_contact_count"], dtype=np.int64)
    contact_position = np.asarray(
        state["target_pad_contact_position_mean_m"], dtype=float)
    arm = np.asarray(state["command_arm_rad"], dtype=float)
    grip = np.asarray(state["command_grip_rad"], dtype=float)
    if (
        force.shape != (frames, 1, 2, 2, 3)
        or count.shape != (frames, 1, 2, 2)
        or contact_position.shape != (frames, 1, 2, 2, 3)
        or arm.shape != (frames, 14)
        or grip.shape != (frames, 2)
        or not np.isfinite(force).all()
        or not np.isfinite(contact_position).all()
        or not np.isfinite(arm).all()
        or np.any(count < 0)
        or not np.isfinite(grip).all()
    ):
        raise ValueError("drawer hand/contact evidence is invalid")

    displacement = drawer - drawer[0]
    pull = displacement[:, 0]
    peak_frame = int(np.argmax(pull))
    peak_pull = float(pull[peak_frame])
    release_command = int(_scalar(metrics, "release_t"))
    try:
        release = int(destination_result["margins"]["release_frame"])
    except (KeyError, TypeError, ValueError, OverflowError):
        release = frames
    pre_release_peak = float(
        np.max(pull[:release + 1])
        if 0 <= release < frames
        else float("-inf")
    )
    right_force = np.linalg.norm(
        force[:, 0, 1, :, :], axis=2)
    right_force_vectors = force[:, 0, 1, :, :]
    right_force_cosine = np.sum(
        right_force_vectors[:, 0, :] * right_force_vectors[:, 1, :],
        axis=1,
    ) / np.maximum(
        right_force[:, 0] * right_force[:, 1], 1.0e-12,
    )
    initial_aabb_min = np.asarray(
        metrics["task6_destination_aabb_min_xyz_m"], dtype=float)
    initial_aabb_max = np.asarray(
        metrics["task6_destination_aabb_max_xyz_m"], dtype=float)
    if (
        initial_aabb_min.shape != (3,)
        or initial_aabb_max.shape != (3,)
        or not np.isfinite(initial_aabb_min).all()
        or not np.isfinite(initial_aabb_max).all()
        or np.any(initial_aabb_min >= initial_aabb_max)
    ):
        raise ValueError("drawer initial contact AABB is invalid")
    right_contact_position = contact_position[:, 0, 1, :, :]
    moving_front_x = initial_aabb_max[0] + displacement[:, 0]
    on_front_x = np.all(
        np.abs(right_contact_position[:, :, 0] - moving_front_x[:, None])
        <= MAX_FRONT_WALL_CONTACT_X_ERROR_M,
        axis=1,
    )
    within_front_yz = np.all(
        (
            right_contact_position[:, :, 1:]
            >= initial_aabb_min[None, None, 1:]
            - FRONT_WALL_CONTACT_AABB_MARGIN_M
        )
        & (
            right_contact_position[:, :, 1:]
            <= initial_aabb_max[None, None, 1:]
            + FRONT_WALL_CONTACT_AABB_MARGIN_M
        ),
        axis=(1, 2),
    )
    right_bilateral = (
        np.all(right_force >= MIN_RIGHT_PAD_FORCE_N, axis=1)
        & np.all(count[:, 0, 1, :] > 0, axis=1)
        & (right_force_cosine <= MAX_RIGHT_PAD_FORCE_COSINE)
        & on_front_x
        & within_front_yz
    )
    push_start_frame = (
        TASK65_DRAWER_PUSH_START_FRAME
        if object_label == "bill"
        else PUSH_START_FRAME
    )
    pull_window = slice(PULL_START_FRAME, push_start_frame)
    contact_frames = int(np.count_nonzero(
        right_bilateral[pull_window]))
    positive_pull_step = np.maximum(
        np.diff(pull, prepend=pull[0]), 0.0)
    pull_motion = np.zeros(frames, dtype=bool)
    pull_motion[pull_window] = (
        positive_pull_step[pull_window]
        > MIN_DRAWER_PULL_MOTION_STEP_M
    )
    pull_motion_frames = int(np.count_nonzero(pull_motion))
    clamped_pull_motion_frames = int(np.count_nonzero(
        pull_motion & right_bilateral))
    clamped_pull_motion_ratio = (
        clamped_pull_motion_frames / pull_motion_frames
        if pull_motion_frames else 0.0
    )
    unclamped_pull_motion = float(np.sum(
        positive_pull_step[pull_motion & ~right_bilateral]))
    push_end_frame = (
        TASK65_RIGHT_GRIP_OPEN_START_FRAME
        if object_label == "bill"
        else GRIP_OPEN_START_FRAME
    )
    negative_push_step = np.maximum(
        -np.diff(pull, prepend=pull[0]), 0.0)
    push_motion = np.zeros(frames, dtype=bool)
    push_window = slice(push_start_frame, push_end_frame)
    push_motion[push_window] = (
        negative_push_step[push_window]
        > MIN_DRAWER_PULL_MOTION_STEP_M
    )
    push_motion_frames = int(np.count_nonzero(push_motion))
    clamped_push_motion_frames = int(np.count_nonzero(
        push_motion & right_bilateral))
    clamped_push_motion_ratio = (
        clamped_push_motion_frames / push_motion_frames
        if push_motion_frames else 0.0
    )
    unclamped_push_motion = float(np.sum(
        negative_push_step[push_motion & ~right_bilateral]))
    final_error = float(abs(pull[-1]))
    lateral_drift = float(np.max(np.abs(displacement[:, 1:])))
    final_tail_span = float(np.max(np.linalg.norm(
        drawer[-20:] - drawer[-1], axis=1)))
    final_arm_home_error = float(np.max(np.abs(
        arm[-20:, :] - arm[0, :][None, :])))
    gates = {
        f"{object_label}_destination_and_settle": bool(
            destination_result.get("success")),
        f"drawer_pulled_before_{object_label}_release": bool(
            pre_release_peak >= MIN_PULL_TRAVEL_M),
        f"drawer_open_while_{object_label}_released": bool(
            0 <= release < frames
            and pull[release] >= MIN_PULL_TRAVEL_M),
        "right_hand_physically_clamped_drawer": bool(
            contact_frames >= MIN_RIGHT_PAD_CONTACT_FRAMES
            and clamped_pull_motion_ratio >= MIN_CLAMPED_PULL_MOTION_RATIO
            and unclamped_pull_motion <= MAX_UNCLAMPED_PULL_MOTION_M),
        "right_hand_physically_pushed_drawer": bool(
            push_motion_frames > 0
            and clamped_push_motion_ratio >= MIN_CLAMPED_PULL_MOTION_RATIO
            and unclamped_push_motion <= MAX_UNCLAMPED_PULL_MOTION_M),
        "right_gripper_closed_for_pull": bool(
            float(np.max(grip[PULL_START_FRAME:push_start_frame, 1]))
            <= 0.15),
        "right_gripper_released_after_push": bool(
            float(np.min(grip[-10:, 1])) >= 0.45),
        "drawer_pushed_back_almost_closed": bool(
            final_error <= MAX_FINAL_RETURN_ERROR_M),
        "drawer_remained_on_rail": bool(
            lateral_drift <= MAX_LATERAL_DRIFT_M),
        "drawer_stable_after_return": bool(final_tail_span <= 0.012),
        "both_arms_returned_to_frame_zero_standard_pose": bool(
            object_label != "bill"
            or final_arm_home_error <= MAX_FINAL_ARM_HOME_ERROR_RAD),
    }
    post_task_cleanup_gates = {
        "right_hand_physically_pushed_drawer",
        "right_gripper_released_after_push",
        "drawer_pushed_back_almost_closed",
        "drawer_stable_after_return",
        "both_arms_returned_to_frame_zero_standard_pose",
    }
    scored_gates = {
        key: passed for key, passed in gates.items()
        if not semantic_only or key not in post_task_cleanup_gates
    }
    reasons = sorted(
        key for key, passed in scored_gates.items() if not passed
    )
    if not destination_result.get("success"):
        reasons.extend(
            f"{object_label}_destination: {reason}"
            for reason in destination_result.get("reasons", ())
        )
    return {
        "success": all(scored_gates.values()),
        "gates": gates,
        "scored_gates": scored_gates,
        "semantic_only": semantic_only,
        "reasons": reasons,
        "margins": {
            "peak_pull_m": peak_pull,
            "peak_pull_frame": peak_frame,
            "pre_release_peak_pull_m": pre_release_peak,
            "release_frame": release,
            "release_command_frame": release_command,
            "opening_at_release_m": (
                float(pull[release]) if 0 <= release < frames else None),
            "right_bilateral_drawer_contact_frames": contact_frames,
            "drawer_positive_pull_motion_frames": pull_motion_frames,
            "clamped_drawer_positive_pull_motion_frames": (
                clamped_pull_motion_frames),
            "clamped_drawer_positive_pull_motion_ratio": (
                clamped_pull_motion_ratio),
            "unclamped_drawer_positive_pull_motion_m": (
                unclamped_pull_motion),
            "drawer_negative_push_motion_frames": push_motion_frames,
            "clamped_drawer_negative_push_motion_frames": (
                clamped_push_motion_frames),
            "clamped_drawer_negative_push_motion_ratio": (
                clamped_push_motion_ratio),
            "unclamped_drawer_negative_push_motion_m": (
                unclamped_push_motion),
            "minimum_right_pad_force_n": float(np.min(
                right_force[right_bilateral], initial=float("inf"))),
            "maximum_right_pad_force_cosine": float(np.max(
                right_force_cosine[right_bilateral],
                initial=float("-inf"),
            )),
            "maximum_front_wall_contact_x_error_m": float(np.max(
                np.abs(
                    right_contact_position[right_bilateral, :, 0]
                    - moving_front_x[right_bilateral, None]
                ),
                initial=0.0,
            )),
            "final_return_error_m": final_error,
            "maximum_lateral_or_vertical_drift_m": lateral_drift,
            "final_tail_span_m": final_tail_span,
            "final_arm_home_max_error_rad": final_arm_home_error,
            f"{object_label}_destination": _json_safe(
                destination_result.get("margins", {})),
        },
    }


def score(
    benchmark_subtask: str,
    state_path: Path | str,
    metrics_path: Path | str,
    *,
    expected_robot_base_delta_xy_m: Sequence[float] | None = None,
    expected_destination_zone: Sequence[float] | None = None,
    expected_furniture_aabb_min_xyz_m: Sequence[float] | None = None,
    expected_furniture_aabb_max_xyz_m: Sequence[float] | None = None,
    semantic_only: bool = False,
) -> dict[str, Any]:
    # Retained as a no-op compatibility argument for existing callers.  Robot
    # initial pose is task-owned evidence, not a shared success gate.
    _ = expected_robot_base_delta_xy_m
    selected = mode(benchmark_subtask)
    effective_zone = np.asarray(
        selected.destination_zone
        if expected_destination_zone is None
        else expected_destination_zone,
        dtype=float,
    )
    if (
        effective_zone.shape != (5,)
        or not np.isfinite(effective_zone).all()
        or np.any(effective_zone[2:4] <= 0.0)
    ):
        raise ValueError(
            "expected_destination_zone must be one finite positive-half-"
            "extent 5-vector")
    if (
        (expected_furniture_aabb_min_xyz_m is None)
        != (expected_furniture_aabb_max_xyz_m is None)
    ):
        raise ValueError(
            "expected furniture AABB minimum and maximum must be supplied "
            "together")
    effective_furniture_min = np.asarray(
        selected.furniture_aabb_min_xyz_m
        if expected_furniture_aabb_min_xyz_m is None
        else expected_furniture_aabb_min_xyz_m,
        dtype=float,
    )
    effective_furniture_max = np.asarray(
        selected.furniture_aabb_max_xyz_m
        if expected_furniture_aabb_max_xyz_m is None
        else expected_furniture_aabb_max_xyz_m,
        dtype=float,
    )
    if (
        effective_furniture_min.shape != (3,)
        or effective_furniture_max.shape != (3,)
        or not np.isfinite(effective_furniture_min).all()
        or not np.isfinite(effective_furniture_max).all()
        or np.any(effective_furniture_min >= effective_furniture_max)
    ):
        raise ValueError("expected furniture AABB must be one finite 3D box")
    state = _load(state_path)
    metrics = _load(metrics_path)
    from real2sim.production.native_hat_hang_metrics import (
        SCHEMA as NATIVE_HAT_SCHEMA, score_native,
    )
    if str(np.asarray(metrics.get("schema", "")).item()) == NATIVE_HAT_SCHEMA:
        if selected.benchmark_subtask != "6.7":
            raise ValueError("native hat metrics are only valid for Sheet 6.7")
        # The native exporter currently measures the canonical rack. Reject
        # a transformed layout until its branch frame is recorded as well.
        if not (
            np.allclose(effective_furniture_min, selected.furniture_aabb_min_xyz_m, atol=1e-9, rtol=0)
            and np.allclose(effective_furniture_max, selected.furniture_aabb_max_xyz_m, atol=1e-9, rtol=0)
        ):
            raise ValueError("native hat metrics currently require the canonical rack layout")
        return score_native(state_path, metrics, effective_furniture_min, effective_furniture_max)
    errors = (
        validate_place_metrics_v3(metrics, semantic_only=True)
        if semantic_only
        else validate_place_metrics_v3(metrics)
    )
    if errors:
        raise ValueError("invalid place metrics: " + "; ".join(errors))
    required_state = {
        "task_id", "target_prim", "frame_index", "fps",
        "rigid_body_paths", "prop_xyz_m", "prop_velocity_mps_radps",
    }
    missing = sorted(required_state - set(state))
    if missing:
        raise ValueError("full state lacks " + ", ".join(missing))

    expected = {
        "task_id": selected.runtime_task_id,
        "target_body_path": selected.target_path,
        "destination_kind": selected.destination_kind,
        "destination_body_path": selected.destination_path,
    }
    identity_errors = []
    for key, value in expected.items():
        actual = str(_scalar(metrics, key))
        if actual != value:
            identity_errors.append(f"metrics {key} {actual!r} != {value!r}")
    metric_zone = np.asarray(metrics["zone"], dtype=float)
    if (
        metric_zone.shape != (5,)
        or not np.allclose(
            metric_zone, effective_zone, atol=1e-9, rtol=0.0)
    ):
        identity_errors.append("metrics zone does not match contract")
    if not np.isclose(
        float(_scalar(metrics, "destination_bottom_z")),
        selected.destination_bottom_z,
        atol=1e-9,
        rtol=0.0,
    ):
        identity_errors.append(
            "metrics destination_bottom_z does not match contract")

    state_identity = {
        "task_id": str(_scalar(state, "task_id")),
        "target_prim": str(_scalar(state, "target_prim")),
    }
    for key, expected_key in (
        ("task_id", "task_id"),
        ("target_prim", "target_body_path"),
    ):
        if state_identity[key] != expected[expected_key]:
            identity_errors.append(
                f"state {key} does not match benchmark contract")

    # Real recordings are optional provenance, never a benchmark runtime gate.
    # When both artifacts record a run identity, they must still agree.
    if "episode_uid" in state and "episode_uid" in metrics:
        if str(_scalar(state, "episode_uid")) != str(_scalar(metrics, "episode_uid")):
            identity_errors.append("state/metrics episode provenance differs")

    expected_family = {
        "6.3": "small_table",
        "6.5": "coffee_table",
        "6.7": "coat_rack",
        "6.9": "coffee_table",
        "6.10": "tv_table",
        "6.15": "coffee_table",
    }[selected.benchmark_subtask]
    layout_required = {
        "current_scene_layout_schema",
        "current_scene_layout_family",
        "current_furniture_body_path",
        "current_furniture_aabb_min_xyz_m",
        "current_furniture_aabb_max_xyz_m",
        "current_furniture_collision_path_count",
        "current_furniture_visible",
    }
    if not layout_required.issubset(metrics):
        identity_errors.append("current furniture trace is missing")
    else:
        expected_layout_text = {
            "current_scene_layout_schema": (
                "real2sim.current_scene_layout_trace.v1"),
            "current_scene_layout_family": expected_family,
            "current_furniture_body_path": selected.furniture_path,
        }
        for key, expected_value in expected_layout_text.items():
            if str(_scalar(metrics, key)) != expected_value:
                identity_errors.append(f"{key} does not match contract")
        actual_min = np.asarray(
            metrics["current_furniture_aabb_min_xyz_m"], dtype=float)
        actual_max = np.asarray(
            metrics["current_furniture_aabb_max_xyz_m"], dtype=float)
        if (
            actual_min.shape != (3,)
            or actual_max.shape != (3,)
            or not np.allclose(
                actual_min, effective_furniture_min,
                atol=0.002, rtol=0.0)
            or not np.allclose(
                actual_max, effective_furniture_max,
                atol=0.002, rtol=0.0)
        ):
            identity_errors.append("live furniture AABB changed")
        collision_count = np.asarray(
            metrics["current_furniture_collision_path_count"])
        visible = np.asarray(metrics["current_furniture_visible"])
        if (
            collision_count.shape != ()
            or collision_count.dtype.kind not in "iu"
            or int(collision_count) < 1
            or visible.shape != ()
            or visible.dtype.kind != "b"
            or not bool(visible)
        ):
            identity_errors.append(
                "live furniture is not visible and collision-backed")
    if selected.destination_aabb_min_xyz_m is not None:
        destination_fields = {
            "task6_destination_body_path",
            "task6_destination_aabb_min_xyz_m",
            "task6_destination_aabb_max_xyz_m",
            "task6_destination_collision_path_count",
            "task6_destination_visible",
        }
        if not destination_fields.issubset(metrics):
            identity_errors.append("live drawer destination trace is missing")
        else:
            actual_destination_min = np.asarray(
                metrics["task6_destination_aabb_min_xyz_m"], dtype=float)
            actual_destination_max = np.asarray(
                metrics["task6_destination_aabb_max_xyz_m"], dtype=float)
            if (
                str(_scalar(metrics, "task6_destination_body_path"))
                != selected.destination_path
                or actual_destination_min.shape != (3,)
                or actual_destination_max.shape != (3,)
                or not np.allclose(
                    actual_destination_min,
                    selected.destination_aabb_min_xyz_m,
                    atol=0.003,
                    rtol=0.0,
                )
                or not np.allclose(
                    actual_destination_max,
                    selected.destination_aabb_max_xyz_m,
                    atol=0.003,
                    rtol=0.0,
                )
                or int(np.asarray(
                    metrics["task6_destination_collision_path_count"]
                )) < 1
                or not bool(np.asarray(
                    metrics["task6_destination_visible"]))
            ):
                identity_errors.append(
                    "live drawer destination geometry changed")

    frames = int(len(state["frame_index"]))
    metric_frames = int(_scalar(metrics, "rendered_frames"))
    state_fps = float(_scalar(state, "fps"))
    metric_fps = float(_scalar(metrics, "fps"))
    if metric_frames != frames or not np.isclose(
        state_fps, metric_fps, atol=1e-9, rtol=0.0
    ):
        identity_errors.append("state/metrics frame clock differs")

    paths = [str(value) for value in state["rigid_body_paths"].tolist()]
    if paths.count(selected.target_path) != 1:
        raise ValueError("target is not exactly one captured rigid body")
    target_index = paths.index(selected.target_path)
    destination_xyz = None
    if selected.destination_aabb_min_xyz_m is not None:
        if paths.count(selected.destination_path) != 1:
            identity_errors.append(
                "drawer is not exactly one captured rigid body")
        else:
            destination_index = paths.index(selected.destination_path)
            destination_xyz = np.asarray(
                state["prop_xyz_m"][:, destination_index, :], dtype=float)
            if (
                destination_xyz.shape != (frames, 3)
                or not np.isfinite(destination_xyz).all()
            ):
                identity_errors.append(
                    "drawer state trace is invalid")
    drawer_state_delta = 0.0
    if selected.benchmark_subtask in {"6.5", "6.15"}:
        metric_drawer_xyz = np.asarray(
            metrics.get("task28_drawer_xyz", []), dtype=float)
        drawer_state_delta = (
            float(np.max(np.abs(destination_xyz - metric_drawer_xyz)))
            if destination_xyz is not None
            and destination_xyz.shape == metric_drawer_xyz.shape == (frames, 3)
            and np.isfinite(destination_xyz).all()
            and np.isfinite(metric_drawer_xyz).all()
            else float("inf")
        )
    xyz = np.asarray(state["prop_xyz_m"][:, target_index, :], dtype=float)
    metric_xyz = np.asarray(metrics["obj_xyz"], dtype=float)
    velocity = np.asarray(
        state["prop_velocity_mps_radps"][:, target_index, :3], dtype=float)
    finite = bool(
        xyz.shape == metric_xyz.shape == (frames, 3)
        and velocity.shape == (frames, 3)
        and np.isfinite(xyz).all()
        and np.isfinite(metric_xyz).all()
        and np.isfinite(velocity).all()
    )
    state_delta = (
        float(np.max(np.abs(xyz - metric_xyz))) if finite else float("inf"))
    # BaseballCap's authored rigid-body root is about 1.84 m from its compact
    # collision geometry.  Normal rotation around the physical hat therefore
    # makes the arbitrary root translation move nearly a metre in one frame;
    # the collision-envelope center remains continuous.  Use that recorded
    # physical center only for the no-teleport gate while retaining and
    # cross-checking the complete root pose in full_state/place_metrics.
    displacement_basis = "rigid_body_root"
    displacement_xyz = xyz
    if selected.benchmark_subtask == "6.7":
        displacement_basis = "collision_bbox_center"
        displacement_xyz = np.asarray(
            metrics.get("bbox_center_xyz", np.empty((0, 3))), dtype=float)
        finite = bool(
            finite
            and displacement_xyz.shape == (frames, 3)
            and np.isfinite(displacement_xyz).all()
        )
    max_step = (
        float(np.linalg.norm(
            np.diff(displacement_xyz, axis=0), axis=1
        ).max(initial=0.0))
        if finite else float("inf")
    )

    if selected.benchmark_subtask == "6.7":
        posture_fields = {
            "command_waist_rad_m", "command_head_rad",
        }
        if not posture_fields.issubset(state):
            identity_errors.append("hat robot posture trace is missing")
        else:
            commanded_waist = np.asarray(
                state["command_waist_rad_m"], dtype=float)
            commanded_head = np.asarray(
                state["command_head_rad"], dtype=float)
            if (
                commanded_waist.shape != (frames, 2)
                or commanded_head.shape != (frames, 2)
                or not np.isfinite(commanded_waist).all()
                or not np.isfinite(commanded_head).all()
                or not np.allclose(
                    commanded_waist[:, 0], 0.0,
                    atol=1e-6, rtol=0.0)
                or not np.allclose(
                    commanded_waist[:, 1], 0.30,
                    atol=1e-6, rtol=0.0)
                or not np.allclose(
                    commanded_head[:, 0], 0.0,
                    atol=1e-6, rtol=0.0)
                or not np.allclose(
                    commanded_head[:, 1], np.deg2rad(25.0),
                    atol=1e-6, rtol=0.0)
            ):
                identity_errors.append(
                    "hat robot posture is not head=25deg waist=0deg "
                    "lift=0.30m")

    destination_strict = (
        {}
        if selected.benchmark_subtask == "6.7"
        else check_place_sim(
            obj_xyz=metrics["obj_xyz"],
            obj_quat=metrics["obj_quat"],
            support_z=metrics["support_z"],
            bbox_center_xyz=metrics["bbox_center_xyz"],
            release_t=metrics["release_t"],
            tip=metrics["tip"],
            grip=metrics["grip"],
            pad_contacts=metrics["pad_contacts"],
            pad_force_vectors=metrics["pad_force_vectors"],
            contact_force_threshold_n=metrics["contact_force_threshold_n"],
            contact_available=metrics["contact_available"],
            destination_kind=selected.destination_kind,
            destination_bottom_z=selected.destination_bottom_z,
            zone=effective_zone,
            fps=metric_fps,
            max_bilateral_dropout_frames=(
                selected.max_bilateral_contact_dropout_frames),
            attach_std=selected.attach_rms_tolerance_m,
        )
    )
    if selected.benchmark_subtask == "6.7":
        strict = _score_hat_hang(metrics, velocity, metric_fps)
    elif selected.benchmark_subtask in {"6.5", "6.15"}:
        strict = _score_drawer_sequence(
            state,
            metrics,
            destination_strict,
            object_label=(
                "bill" if selected.benchmark_subtask == "6.5" else "wallet"
            ),
            semantic_only=semantic_only,
        )
    else:
        strict = destination_strict
    gates = {
        "benchmark_identity": not identity_errors,
        "finite_target_state": finite,
        "state_matches_place_metrics": (
            state_delta <= STATE_METRICS_TOLERANCE_M),
        "drawer_state_matches_place_metrics": (
            drawer_state_delta <= STATE_METRICS_TOLERANCE_M),
        "no_single_frame_teleport": (
            max_step <= MAX_ONE_FRAME_DISPLACEMENT_M),
        "strict_grasp_carry_release_destination_settle": bool(
            strict.get("success")),
    }
    drawer_containment = None
    if selected.benchmark_subtask in {"6.5", "6.15"}:
        drawer_containment = _score_drawer_containment(state, metrics, selected)
        if drawer_containment["status"] == "unverified" and all(gates.values()):
            raise ValueError(drawer_containment["reasons"][0] + ": " + json.dumps(
                drawer_containment["margins"], sort_keys=True))
        gates["full_target_contained_in_recorded_drawer"] = bool(
            drawer_containment["success"])
    failed = sorted(key for key, passed in gates.items() if not passed)
    return {
        "schema": SCHEMA,
        "ok": not failed,
        "benchmark_subtask": selected.benchmark_subtask,
        "runtime_task_id": selected.runtime_task_id,
        "real_data_reference_label": (
            f"legacy recording label {selected.real_data_reference_id}"
        ),
        "failed_gates": failed,
        "identity_errors": identity_errors,
        "gates": gates,
        "drawer_containment": drawer_containment,
        "strict_place_reasons": [
            str(value) for value in strict.get("reasons", ())] + (
                drawer_containment["reasons"] if drawer_containment else []),
        "strict_place_margins": _json_safe(strict.get("margins", {})),
        "metrics": {
            "frames": frames,
            "fps": state_fps,
            "state_metrics_max_delta_m": state_delta,
            "drawer_state_metrics_max_delta_m": drawer_state_delta,
            "maximum_one_frame_displacement_m": max_step,
            "maximum_one_frame_displacement_basis": displacement_basis,
            "robot_initial_pose_gate": "not_enforced_task_owned_evidence",
            "layout_identity_basis": (
                "catalog_transformed"
                if expected_destination_zone is not None
                or expected_furniture_aabb_min_xyz_m is not None
                else "canonical_task_contract"
            ),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtask", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = score(args.subtask, args.state, args.metrics)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise SystemExit(0 if result["ok"] else 2)
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, sort_keys=True))
        raise SystemExit(2)

"""Portable support-pivot math, extracted from audited L1 utensil_pose_geometry.

Only mathematical sampling is retained; no old scene loading or runtime imports.
Source SHA256: 3ff9873e242a3614697a49da368c8f06a5d628dfbca00812b23bbebbf167c225.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
TOOLS=("knife_0","fork_1","fork_2")
SCHEMA="short_tasks.prepare_cutlery.support_pivot_geometry.v1"

def _align_axis(top, direction, angle_deg: float) -> np.ndarray:
    top = np.asarray(top, float); top /= np.linalg.norm(top)
    direction = np.asarray(direction, float); direction /= np.linalg.norm(direction)
    desired = math.cos(math.radians(angle_deg)) * top + math.sin(math.radians(angle_deg)) * direction
    desired /= np.linalg.norm(desired)
    axis = np.cross(top, desired)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(axis / norm * math.acos(np.clip(top @ desired, -1.0, 1.0))).as_matrix()


def _wrap_degrees(value: float) -> float:
    """Return a stable representative in [-180, 180)."""
    return float((value + 180.0) % 360.0 - 180.0)


def _draw_parameters(name: str, rng, policy: dict) -> tuple[float, float, float, dict | None]:
    region = policy.get("reachable_region")
    if region is None:
        # Preserve the released sampler's calls and order byte-for-byte in this branch.
        return (
            float(rng.uniform(*policy["tilt_magnitude_range_deg"])),
            float(rng.uniform(*policy["lean_azimuth_range_deg"])),
            float(rng.uniform(*policy["long_axis_yaw_range_deg"])),
            None,
        )
    if region.get("schema") not in {"native_screened_pose_neighborhood.v1", "ziche.meal_cutlery50.accepted_template_neighborhood.v1"}:
        raise ValueError(f"unsupported reachable_region schema {region.get('schema')!r}")
    try:
        anchor = region["tools"][name]
        halfwidths = region["halfwidths"]
        tilt_halfwidth = float(halfwidths["tilt_deg"])
        azimuth_halfwidth = float(halfwidths["lean_azimuth_deg"])
        yaw_halfwidth = float(halfwidths["long_axis_yaw_deg"])
        xy_halfwidth = float(halfwidths["xy_m"])
        anchor_tilt = float(anchor["tilt_magnitude_deg"])
        anchor_azimuth = float(anchor["lean_azimuth_holder_deg"])
        anchor_yaw = float(anchor["long_axis_yaw_deg"])
        anchor_xy = np.asarray(anchor["sampled_root_xy_shift_holder_m"], float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid reachable_region anchor for {name}: {exc}") from exc
    if anchor_xy.shape != (2,) or not np.all(np.isfinite(anchor_xy)):
        raise ValueError(f"invalid reachable_region XY anchor for {name}")
    widths = np.array([tilt_halfwidth, azimuth_halfwidth, yaw_halfwidth, xy_halfwidth])
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError(f"reachable_region halfwidths must be finite and nonnegative for {name}")

    tilt_limits = [max(0.0, anchor_tilt - tilt_halfwidth),
                   min(30.0, anchor_tilt + tilt_halfwidth)]
    if tilt_limits[0] > tilt_limits[1]:
        raise ValueError(f"reachable_region tilt neighborhood misses [0, 30] degrees for {name}")
    azimuth_limits = [anchor_azimuth - azimuth_halfwidth, anchor_azimuth + azimuth_halfwidth]
    yaw_limits = [anchor_yaw - yaw_halfwidth, anchor_yaw + yaw_halfwidth]
    angle = float(rng.uniform(*tilt_limits))
    azimuth = _wrap_degrees(float(rng.uniform(*azimuth_limits)))
    yaw_deg = _wrap_degrees(float(rng.uniform(*yaw_limits)))
    return angle, azimuth, yaw_deg, {
        "schema": region["schema"],
        "anchor_id": region.get("anchor_id"),
        "environment_signature": region.get("environment_signature"),
        "source_job_sha256": region.get("source_job_sha256"),
        "anchor": {
            "tilt_magnitude_deg": anchor_tilt,
            "lean_azimuth_holder_deg": anchor_azimuth,
            "long_axis_yaw_deg": anchor_yaw,
            "sampled_root_xy_shift_holder_m": anchor_xy.tolist(),
        },
        "halfwidths": {
            "tilt_deg": tilt_halfwidth,
            "lean_azimuth_deg": azimuth_halfwidth,
            "long_axis_yaw_deg": yaw_halfwidth,
            "xy_m": xy_halfwidth,
        },
        "actual_limits": {
            "tilt_magnitude_deg": tilt_limits,
            "lean_azimuth_holder_deg_unwrapped": azimuth_limits,
            "long_axis_yaw_deg_unwrapped": yaw_limits,
            "sampled_root_xy_shift_holder_m": None,
        },
        "sampled_offsets_from_anchor": {
            "tilt_deg": angle - anchor_tilt,
            "lean_azimuth_deg": _wrap_degrees(azimuth - anchor_azimuth),
            "long_axis_yaw_deg": _wrap_degrees(yaw_deg - anchor_yaw),
            "xy_m": None,
        },
    }


def _draw_one(points, root_bound_corners, root_relative, slot, rng, policy: dict,
              name: str | None = None) -> dict | None:
    if name is None and policy.get("reachable_region") is not None:
        raise ValueError("tool name is required for reachable_region sampling")
    angle, azimuth, yaw_deg, reachable_audit = _draw_parameters(name or "", rng, policy)
    direction = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth)), 0.0])
    top = root_relative[:3, :3] @ np.array([0.0, -1.0, 0.0]); top /= np.linalg.norm(top)
    yaw = Rotation.from_rotvec(top * math.radians(yaw_deg)).as_matrix()
    rotation = _align_axis(top, direction, angle) @ yaw
    source_min_z = float(points[:, 2].min())
    support = points[points[:, 2] <= source_min_z + 0.0005]
    pivot = support.mean(0)
    rotated = (points - pivot) @ rotation.T + pivot
    dz = source_min_z - float(rotated[:, 2].min())
    rotated[:, 2] += dz
    margin = float(policy["minimum_below_rim_wall_margin_m"])
    lo, hi = slot["inner_lo_xy_m"], slot["inner_hi_xy_m"]
    affine_without_xy = np.eye(4); affine_without_xy[:3, :3] = rotation
    affine_without_xy[:3, 3] = pivot - rotation @ pivot + np.array([0.0, 0.0, dz])
    candidate_without_xy = affine_without_xy @ root_relative
    proxy = (root_bound_corners @ candidate_without_xy[:3, :3].T
             + candidate_without_xy[:3, 3])
    bbox_lo = np.minimum(rotated[:, :2].min(0), proxy[:, :2].min(0))
    bbox_hi = np.maximum(rotated[:, :2].max(0), proxy[:, :2].max(0))
    shift_lo, shift_hi = lo + margin - bbox_lo, hi - margin - bbox_hi
    if reachable_audit is not None:
        anchor_xy = np.asarray(
            reachable_audit["anchor"]["sampled_root_xy_shift_holder_m"], float)
        xy_halfwidth = float(reachable_audit["halfwidths"]["xy_m"])
        shift_lo = np.maximum(shift_lo, anchor_xy - xy_halfwidth)
        shift_hi = np.minimum(shift_hi, anchor_xy + xy_halfwidth)
    if np.any(shift_lo > shift_hi):
        return None
    shift = rng.uniform(shift_lo, shift_hi)
    if reachable_audit is not None:
        reachable_audit["actual_limits"]["sampled_root_xy_shift_holder_m"] = [
            shift_lo.tolist(), shift_hi.tolist()]
        reachable_audit["sampled_offsets_from_anchor"]["xy_m"] = (shift - anchor_xy).tolist()
    moved = rotated.copy(); moved[:, :2] += shift
    below = moved[:, 2] <= slot["rim_z_m"] + 1e-9
    subset = moved[below]
    margins = np.array([subset[:, 0].min() - lo[0], hi[0] - subset[:, 0].max(),
                        subset[:, 1].min() - lo[1], hi[1] - subset[:, 1].max()])
    all_margins = np.array([moved[:, 0].min() - lo[0], hi[0] - moved[:, 0].max(),
                            moved[:, 1].min() - lo[1], hi[1] - moved[:, 1].max()])
    affine = affine_without_xy.copy(); affine[:2, 3] += shift
    candidate = affine @ root_relative
    proxy[:, :2] += shift
    proxy_margins = np.array([proxy[:, 0].min() - lo[0], hi[0] - proxy[:, 0].max(),
                              proxy[:, 1].min() - lo[1], hi[1] - proxy[:, 1].max()])
    result = {
        "root_relative": candidate.tolist(),
        "audit": {
            "tilt_magnitude_deg": angle,
            "lean_azimuth_holder_deg": azimuth,
            "lean_direction_holder_local": direction.tolist(),
            "long_axis_yaw_deg": yaw_deg,
            "root_xy_feasible_interval_holder_m": [shift_lo.tolist(), shift_hi.tolist()],
            "sampled_root_xy_shift_holder_m": shift.tolist(),
            "support_pivot_holder_local_m": pivot.tolist(),
            "support_patch_vertex_count": int(len(support)),
            "source_min_z_m": source_min_z,
            "candidate_min_z_m": float(moved[:, 2].min()),
            "support_height_delta_m": float(moved[:, 2].min() - source_min_z),
            "root_translation_delta_holder_local_m": (candidate[:3, 3] - root_relative[:3, 3]).tolist(),
            "minimum_below_rim_wall_margin_m": float(margins.min()),
            "minimum_all_vertex_wall_margin_m": float(all_margins.min()),
            "minimum_authored_root_bbox_proxy_wall_margin_m": float(proxy_margins.min()),
            "pass_below_rim_inside_expected_cavity": bool(margins.min() >= margin - 1e-9),
            "pass_current_generator_all_vertex_aabb": bool(
                all_margins.min() >= margin - 1e-9 and proxy_margins.min() >= margin - 1e-9),
        },
    }
    if reachable_audit is not None:
        result["audit"]["reachable_region"] = reachable_audit
    return result


def sample_geometry(cache: dict, policy: dict) -> dict:
    """Sample all tools from an attempt-fixed independent RNG, rejecting only geometry failures."""
    rng = np.random.default_rng(int(policy["pose_seed"]))
    tools = {}
    for slot_index, name in enumerate(TOOLS):
        selected = None
        for draw_index in range(int(policy["maximum_geometry_draws_per_tool"])):
            selected = _draw_one(cache["points"][name], cache["root_bound_corners"][name], cache["roots"][name],
                                 cache["cavities"][slot_index], rng, policy, name)
            if selected is not None:
                selected["audit"]["accepted_draw_index"] = draw_index
                selected["audit"]["rejected_geometry_draw_count"] = draw_index
                break
        if selected is None:
            if policy.get("reachable_region") is not None:
                anchor_id = policy["reachable_region"].get("anchor_id")
                raise RuntimeError(
                    f"no geometry-feasible pose for {name} inside reachable_region "
                    f"anchor {anchor_id!r}; XY neighborhood never intersected the cavity-feasible interval"
                )
            raise RuntimeError(f"no geometry-feasible random pose for {name}")
        tools[name] = selected
    return {
        "schema": SCHEMA,
        "policy_schema": policy["schema"],
        "pose_seed": int(policy["pose_seed"]),
        "canonical_scene_path": cache["scene_path"],
        "canonical_scene_sha256": cache["scene_sha256"],
        "support_minimum_z_preserved": True,
        "tools": tools,
    }

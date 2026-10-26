"""Canonical onboard-camera pose calibration for Real2Sim renders.

The user-approved global visual standard (2026-07-26) is ``HEAD_ADJ=0``:
replay the recorded head-pitch target without an extra task-specific
correction.

The full mount-local +16 degree experiment remains available below for
diagnostics, but it is not the gallery standard: it produced an approximately
71-degree-down view and exposed the robot chassis in the head image.
"""

from __future__ import annotations

import math


HEAD_CAMERA_PATH = "/World/Robot/head_link2/Head_Camera"
HEAD_CAMERA_AUTHORED_ORIENT_WXYZ = (
    6.123233995736766e-17,
    0.7071067811865476,
    -4.329780281177467e-17,
    -0.7071067811865476,
)
DEFAULT_HEAD_CAMERA_REQUESTED_ADJ_DEG = 0.0
DOCUMENTED_HEAD_CAMERA_METHOD = (
    "recorded_head_joint_plus_requested_adjustment_clamped"
)


def _finite_quaternion(values, *, name):
    try:
        quaternion = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric quaternion") from exc
    if len(quaternion) != 4 or not all(
            math.isfinite(value) for value in quaternion):
        raise ValueError(f"{name} must contain four finite values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in quaternion)


def quaternion_multiply_wxyz(left, right):
    """Hamilton product of two normalized-or-normalizable WXYZ quaternions."""
    lw, lx, ly, lz = _finite_quaternion(left, name="left quaternion")
    rw, rx, ry, rz = _finite_quaternion(right, name="right quaternion")
    product = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    return _finite_quaternion(product, name="quaternion product")


def head_camera_orient_wxyz(
        pitch_adjustment_degrees=0.0,
        authored_orient_wxyz=HEAD_CAMERA_AUTHORED_ORIENT_WXYZ):
    """Return mount orientation plus a diagnostic camera-local pitch."""
    adjustment = float(pitch_adjustment_degrees)
    if not math.isfinite(adjustment) or abs(adjustment) > 90.0:
        raise ValueError(
            "head camera pitch adjustment must be finite and within 90 degrees")
    half_angle = math.radians(-adjustment) / 2.0
    local_down_pitch = (
        math.cos(half_angle),
        math.sin(half_angle),
        0.0,
        0.0,
    )
    return quaternion_multiply_wxyz(
        authored_orient_wxyz, local_down_pitch)


def documented_head_joint_target(
        raw_radians,
        lower_limit_radians,
        upper_limit_radians,
        requested_adjustment_degrees=DEFAULT_HEAD_CAMERA_REQUESTED_ADJ_DEG):
    """Return the user-approved, joint-limited head-pitch target.

    Positive head pitch is image-down for this composed robot.  The returned
    dictionary makes the difference between requested and realized adjustment
    explicit so manifests cannot label a five-degree realized correction as a
    full mount-local sixteen-degree rotation.
    """
    raw = float(raw_radians)
    lower = float(lower_limit_radians)
    upper = float(upper_limit_radians)
    requested = float(requested_adjustment_degrees)
    if not all(math.isfinite(value) for value in (
            raw, lower, upper, requested)):
        raise ValueError("head target inputs must be finite")
    if lower > upper:
        raise ValueError("head joint lower limit exceeds upper limit")
    if abs(requested) > 90.0:
        raise ValueError(
            "head camera requested adjustment must be within 90 degrees")
    unconstrained = raw + math.radians(requested)
    target = min(max(unconstrained, lower), upper)
    return {
        "schema": "real2sim.documented_head_camera.v1",
        "method": DOCUMENTED_HEAD_CAMERA_METHOD,
        "raw_head_pitch_degrees": math.degrees(raw),
        "requested_adjustment_degrees": requested,
        "unconstrained_target_degrees": math.degrees(unconstrained),
        "joint_lower_limit_degrees": math.degrees(lower),
        "joint_upper_limit_degrees": math.degrees(upper),
        "target_head_pitch_degrees": math.degrees(target),
        "realized_adjustment_degrees": math.degrees(target - raw),
        "clamped": not math.isclose(
            target, unconstrained, abs_tol=1e-12, rel_tol=0.0),
        "mount_local_adjustment_degrees": 0.0,
    }


def _quaternion_equivalent(left, right, *, atol=1e-8):
    lhs = _finite_quaternion(left, name="left quaternion")
    rhs = _finite_quaternion(right, name="right quaternion")
    direct = max(abs(a - b) for a, b in zip(lhs, rhs))
    antipodal = max(abs(a + b) for a, b in zip(lhs, rhs))
    return min(direct, antipodal) <= atol


def apply_head_camera_pitch_calibration(
        stage,
        pitch_adjustment_degrees=0.0,
        camera_path=HEAD_CAMERA_PATH):
    """Apply and verify the canonical correction on one composed USD stage.

    The authored mount is checked before mutation so an asset change cannot be
    mistaken for the same calibration or have the offset accumulated twice.
    """
    from pxr import Gf

    prim = stage.GetPrimAtPath(camera_path)
    if not prim or not prim.IsValid() or prim.GetTypeName() != "Camera":
        raise RuntimeError(f"missing canonical head camera {camera_path}")
    attribute = prim.GetAttribute("xformOp:orient")
    if not attribute or not attribute.IsValid():
        raise RuntimeError(
            f"head camera has no authored xformOp:orient: {camera_path}")
    authored = attribute.Get()
    if authored is None:
        raise RuntimeError(
            f"head camera orientation has no composed value: {camera_path}")
    imaginary = authored.GetImaginary()
    observed = (
        float(authored.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    if not _quaternion_equivalent(
            observed, HEAD_CAMERA_AUTHORED_ORIENT_WXYZ):
        raise RuntimeError(
            "head camera authored orientation drifted; refusing to stack "
            f"a mount-local calibration on {observed}")

    corrected = head_camera_orient_wxyz(pitch_adjustment_degrees)
    if not attribute.Set(
            Gf.Quatd(corrected[0], Gf.Vec3d(*corrected[1:]))):
        raise RuntimeError("failed to author calibrated head camera orientation")
    readback = attribute.Get()
    readback_imaginary = readback.GetImaginary()
    verified = (
        float(readback.GetReal()),
        float(readback_imaginary[0]),
        float(readback_imaginary[1]),
        float(readback_imaginary[2]),
    )
    if not _quaternion_equivalent(verified, corrected):
        raise RuntimeError(
            "head camera calibrated orientation readback mismatch")
    return {
        "schema": "real2sim.head_camera_mount_pitch.v1",
        "camera_path": camera_path,
        "pitch_adjustment_degrees": float(pitch_adjustment_degrees),
        "authored_orient_wxyz": list(
            _finite_quaternion(
                HEAD_CAMERA_AUTHORED_ORIENT_WXYZ,
                name="canonical authored orientation")),
        "calibrated_orient_wxyz": list(corrected),
    }

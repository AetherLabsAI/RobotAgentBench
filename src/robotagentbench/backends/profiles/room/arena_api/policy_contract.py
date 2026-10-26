"""Arena training/inference contract, using XPolicyLab's per-arm packing order."""
import hashlib
import json
import math

from . import bimanual as b, robot_profiles as rp

BLACKBOX_PROFILE = "arena.g1.joint16.blackbox.v1"
ORDER = b.LEFT_ARM + (b.GRIPPERS[0],) + b.RIGHT_ARM + (b.GRIPPERS[1],)
GRIP_RANGE = math.pi / 4


def _joint_only(profile):
    if profile == rp.MOBILE22:
        raise ValueError('Legacy joint-only policy chunks do not support 22D; use the explicit HTTP base_velocity action')
    return rp.get(profile)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def blackbox_specification(robot_profile=rp.JOINT16):
    """IO agreement only; model identity and training provenance are not verified."""
    p = _joint_only(robot_profile)
    return {"profile_id": BLACKBOX_PROFILE if p.id == rp.JOINT16 else "arena.g1.joint20.blackbox.v1",
            "robot_profile": p.id, "restricted": p.id == rp.JOINT16, "held_commands": list(p.held_commands),
            "joint_units": {n: ("normalized" if n in b.GRIPPERS else u) for n,u in p.units.items()}, "state_order": list(p.order), "action_order": list(p.order),
            "arm_unit": "rad", "arm_action": "absolute_joint_position",
            "gripper_indices": [p.order.index(n) for n in b.GRIPPERS], "gripper_encoding": "normalized_outer_joint_coordinate",
            "gripper_zero_rad": 0., "gripper_one_rad": GRIP_RANGE,
            "gripper_is_metric_aperture": False,
            "state_source": "measured_Isaac_joint_positions",
            "camera_order": ["head", "right", "left"], "camera_encoding": "RGB PNG bytes",
            "image_resolution": [640, 480], "control_dt_s": b.CONTROL_DT,
            "state_shape": [1, len(p.order)], "action_shape": [30, len(p.order)],
            "provenance": "provider_declared_only; model identity and action origin unverified"}


def pack_measured(joints, robot_profile=rp.JOINT16):
    p = _joint_only(robot_profile)
    if set(joints) != set(p.order):
        raise ValueError("Expected exactly the profile named measured joints; no padding or fallback")
    values = []
    for name in p.order:
        q = joints[name]
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q):
            raise ValueError("Invalid measured joint value")
        if name in b.GRIPPERS:
            if not -1e-5 <= q <= GRIP_RANGE + 1e-5:
                raise ValueError("Measured gripper outside declared USD joint range")
            q = min(GRIP_RANGE, max(0., q)) / GRIP_RANGE
        elif p.units[name] == "rad" and abs(q) > math.pi + 1e-5:
            raise ValueError("Measured arm outside declared joint range")
        values.append(float(q))
    return [values]


def validate_chunk(chunk, robot_profile=rp.JOINT16):
    size = len(_joint_only(robot_profile).order)
    if not isinstance(chunk, list) or len(chunk) != 30:
        raise ValueError(f"Expected a 30 x {size} action chunk")
    for row in chunk:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("Expected a 30 x 16 action chunk")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError("Non-finite/non-numeric action")


if __name__ == "__main__":
    profile = blackbox_specification()
    print(json.dumps({"profile": profile, "profile_sha256": digest(profile)}, indent=2))

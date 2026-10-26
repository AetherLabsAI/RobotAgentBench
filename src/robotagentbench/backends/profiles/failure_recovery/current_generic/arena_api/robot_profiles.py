"""Robot-owned action/state contracts. No USD authoring or physical parameter edits."""
from dataclasses import dataclass
import math

from . import bimanual as b

JOINT16 = "g1.joint16.restricted.v1"
JOINT20 = "g1.joint20.v1"
MOBILE22 = "g1_omnipicker_jointpos20_basevw_v1"
DEFAULT = JOINT20
BODY = ("idx01_body_joint1", "idx02_body_joint2")
HEAD = ("idx11_head_joint1", "idx12_head_joint2")
INNER = ("idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1")
FOLLOWERS = dict(zip(INNER, b.GRIPPERS))
GRIPPER_INTERNAL = tuple(
    f"idx{base + suffix:02d}_gripper_{side}_{part}_joint{joint}"
    for side, start in (("l", 30), ("r", 70))
    for part, base in (("inner", start), ("outer", start + 10))
    for suffix, joint in ((1, 1), (2, 3), (3, 4), (9, 0)))
NATIVE_JOINTS = BODY + HEAD + b.LEFT_ARM + b.RIGHT_ARM + GRIPPER_INTERNAL
NATIVE_UNITS = {n: "m" if n == BODY[0] else "rad" for n in NATIVE_JOINTS}
ADAPTER_ID = "g1.native34.paired_gripper.v1"


@dataclass(frozen=True)
class RobotProfile:
    id: str
    joints: tuple
    order: tuple

    @property
    def units(self):
        return {n: NATIVE_UNITS[n] for n in self.joints}

    @property
    def wire_unit(self):
        return "native" if "m" in self.units.values() else "rad"

    @property
    def held_commands(self):
        return tuple(n for n in BODY + HEAD if n not in self.joints)


_PROFILES = {
    JOINT16: RobotProfile(JOINT16, b.JOINTS,
                         b.LEFT_ARM + (b.GRIPPERS[0],) + b.RIGHT_ARM + (b.GRIPPERS[1],)),
    JOINT20: RobotProfile(JOINT20, b.LEFT_ARM + b.RIGHT_ARM + BODY + HEAD + b.GRIPPERS,
                         b.LEFT_ARM + b.RIGHT_ARM + BODY + HEAD + b.GRIPPERS),
}
_PROFILES[MOBILE22] = RobotProfile(MOBILE22, _PROFILES[JOINT20].joints, _PROFILES[JOINT20].order)


def get(profile):
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError("Unknown robot profile; no automatic joint dropping")
    return _PROFILES[profile]


def specification(profile):
    p = get(profile)
    result = dict(robot_profile=p.id, command_joints=list(p.joints), command_units=p.units,
                state_joints=list(NATIVE_JOINTS), state_units=dict(NATIVE_UNITS),
                restricted=p.id == JOINT16, held_commands=list(p.held_commands),
                adapter_id=ADAPTER_ID, gripper_followers=dict(FOLLOWERS),
                follower_rule="inner = -outer; preserve existing Mimic or synchronize existing active drives",
                other_native_targets="preserve explicit initialized drive targets")
    if profile == MOBILE22:
        from .mobile_base import specification as base_specification
        result.update(action_dimensions=22, base_velocity=base_specification(),
                      action_order=list(p.order) + ['base_forward_m_s', 'base_yaw_rate_rad_s'])
    return result


def measured(observation, profile):
    p = get(profile)
    return {n: getattr(observation, "joint_positions_" + unit)[n]
            for n, unit in p.units.items()}


def values(targets, profile, label="targets"):
    p = get(profile)
    if not isinstance(targets, dict) or set(targets) != set(p.joints):
        raise ValueError(f"{label}: expected exactly {len(p.joints)} named {p.id} joints")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in targets.values()):
        raise ValueError("Non-finite/non-numeric " + label)


class NativeAdapter:
    """Expand commands before apply_action; never edit drives, q, dq or physics.

    Limits/types come from the composed runtime robot, not hard-coded substitutes.
    Both profiles use the same pairing. Existing Mimic joints stay passive:
    only active drives receive targets. Other drive targets are held unchanged.
    """
    def __init__(self, profile, names, units, driven, initial_targets, limits, *, mimic_followers=None):
        self.profile = get(profile)
        if len(names) != len(set(names)) or set(names) != set(NATIVE_JOINTS):
            raise ValueError("Expected complete named native34 robot; unknown/missing joints")
        if dict(units) != NATIVE_UNITS:
            raise ValueError("Native robot joint types/units do not match the robot profile")
        if len(driven) != len(set(driven)) or not set(driven) <= set(names):
            raise ValueError("Invalid native driven-joint mapping")
        self.driven = tuple(driven)
        self.held = dict(initial_targets)
        if set(self.held) != set(self.driven) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in self.held.values()):
            raise ValueError("Missing/non-finite initialized native drive targets")
        mimic_followers = dict(mimic_followers or {})
        if set(mimic_followers) - set(INNER):
            raise ValueError("Unknown gripper mimic follower")
        if not set(self.profile.joints) <= set(driven):
            raise ValueError("Profile requires existing command drives")
        self.follower_modes = {}
        for inner in INNER:
            if inner in driven and inner in mimic_followers:
                raise ValueError("Mimic follower must not also be explicitly driven")
            if inner in driven:
                self.follower_modes[inner] = "existing_drive"
            elif inner in mimic_followers:
                if mimic_followers[inner] != FOLLOWERS[inner]:
                    raise ValueError("Wrong mimic reference joint")
                self.follower_modes[inner] = "existing_mimic"
            else:
                raise ValueError("Missing active inner drive or verified Mimic relationship")
        required = set(self.profile.joints) | set(INNER)
        self.limits = {n: tuple(limits[n]) for n in required}
        for n, pair in self.limits.items():
            if (len(pair) != 2 or any(not math.isfinite(v) for v in pair) or pair[0] >= pair[1]):
                raise ValueError("Invalid native USD limit: " + n)

    def expand(self, commands):
        values(commands, self.profile.id)
        targets = dict(commands)
        targets.update({inner: -commands[outer] for inner, outer in FOLLOWERS.items()})
        for n, v in targets.items():
            low, high = self.limits[n]
            if not low <= v <= high:
                raise ValueError("Native target outside USD limit: " + n)
        # Passive mimic coordinates are validated but never driven by this adapter.
        result = {**self.held, **{n:v for n,v in targets.items() if n in self.driven}}
        return result

    def vector(self, commands):
        targets = self.expand(commands)
        return [targets[n] for n in self.driven]


def inspect_mimic_followers(stage):
    """Read-only validation of the existing native G1 coupling, before reset.

    Authored drive attributes without PhysicsDriveAPI are NOT active drives.
    Missing/altered coupling fails; this function never installs or fixes it.
    """
    result = {}
    for inner, outer in FOLLOWERS.items():
        p = stage.GetPrimAtPath("/World/Robot/joints/" + inner)
        if not p:
            raise ValueError("Missing native inner joint: " + inner)
        schemas = p.GetAppliedSchemas()
        mimic = "PhysxMimicJointAPI:rotX" in schemas
        active_drive = "PhysicsDriveAPI:angular" in schemas
        if active_drive and mimic:
            raise ValueError("Both drive and mimic on inner joint: " + inner)
        if not mimic:
            if not active_drive: raise ValueError("Inner joint has no drive or mimic: " + inner)
            continue
        if p.GetAttribute("physics:jointEnabled").Get() is False:
            raise ValueError("Mimic joint is disabled: " + inner)
        stem = "physxMimicJoint:rotX:"
        reference = p.GetRelationship(stem + "referenceJoint").GetTargets()
        if ([str(x) for x in reference] != ["/World/Robot/joints/" + outer] or
                p.GetAttribute(stem + "referenceJointAxis").Get() != "rotX" or
                p.GetAttribute(stem + "gearing").Get() != 1.0 or
                p.GetAttribute(stem + "offset").Get() != 0.0):
            raise ValueError("G1 mimic coupling differs from the declared profile: " + inner)
        result[inner] = outer
    return result

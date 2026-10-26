"""A15: mirror only the two right pad contact offsets from the left pads."""
import math

from .isaac_backend import contact_offset_issue

POLICY = "g1.right_pad_contact_matches_left.v1"
ATTRIBUTE = "physxCollision:contactOffset"
PAIRS = tuple((f"/World/Robot/gripper_l_{part}_link4/collisions/{part}_link4/mesh",
               f"/World/Robot/gripper_r_{part}_link4/collisions/{part}_link4/mesh")
              for part in ("inner", "outer"))


def configuration(stage):
    from pxr import UsdPhysics
    result = {}
    for pair in PAIRS:
        for path in pair:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.IsActive() or not prim.IsLoaded() or prim.IsInstanceProxy()
                    or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True):
                raise ValueError("Expected active finger-pad collider: " + path)
            contact = prim.GetAttribute(ATTRIBUTE).Get()
            rest = prim.GetAttribute("physxCollision:restOffset").Get()
            if contact is None or not math.isfinite(contact):
                raise ValueError("Expected explicit finite finger-pad contact offset: " + path)
            if rest is not None and rest != -math.inf and not math.isfinite(rest):
                raise ValueError("Invalid finger-pad rest offset: " + path)
            result[path] = dict(contact_offset=contact,
                                rest_offset="automatic" if rest is None or rest == -math.inf else rest)
    return result


def apply(stage):
    from pxr import Usd, UsdGeom
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError("Finger-pad contact policy requires meter-scale stage")
    before = configuration(stage)
    # Validate both pairs before writing anything. Do not fix or replace the
    # left reference, or modify a right rest offset to make the copy legal.
    for left, right in PAIRS:
        value = before[left]["contact_offset"]
        for path in (left, right):
            rest = before[path]["rest_offset"]
            if contact_offset_issue(value, None if rest == "automatic" else rest):
                raise ValueError("Left contact offset incompatible with finger-pad rest offset: " + path)
    changes = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for left, right in PAIRS:
            value = before[left]["contact_offset"]
            if before[right]["contact_offset"] != value:
                stage.GetPrimAtPath(right).GetAttribute(ATTRIBUTE).Set(value)
                changes.append(dict(path=right, reference=left, attribute=ATTRIBUTE,
                                    before=before[right]["contact_offset"], after=value))
    return dict(policy_id=POLICY, changes=changes, configuration=configuration(stage),
                reference_pads_modified=False, rest_offsets_modified=False,
                source_assets_modified=False)


def verify(stage, expected):
    if configuration(stage) != expected:
        raise RuntimeError("Finger-pad contact offsets changed after initialization; reset required")

"""Validate an optional House microwave; retain legacy session-only migration.

The approved asset already contains the fixed-base configuration. Its normal
load path is read-only. Absence/explicit deactivation is not a broken asset;
an active placement with a missing or malformed Asset still fails validation.

Keep the existing articulation and passive hinge. A world fixed joint selects
the body as the fixed base; both links remain non-kinematic. No geometry,
collision filter, door drive, joint limit, mass or trajectory is changed.
"""

ROOT = "/World/Assets/DiningTableMicrowave/Asset"
BODY = ROOT + "/body"
DOOR = ROOT + "/door"
HINGE = ROOT + "/door_hinge"
MOUNT = ROOT + "/arena_body_world_mount"
POLICY = "house.microwave_fixed_base.v1"


def _unavailable(stage):
    # Check ancestors first: USD does not compose descendants of inactive prims.
    parts = ROOT.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        path = "/" + "/".join(parts[:i])
        prim = stage.GetPrimAtPath(path)
        if prim and not prim.IsActive():
            return {"presence": "disabled", "at": path}
    placement = ROOT.rsplit("/", 1)[0]
    if not stage.GetPrimAtPath(placement):
        return {"presence": "absent", "at": placement}
    return None


def configuration(stage):
    from pxr import UsdPhysics
    unavailable = _unavailable(stage)
    if unavailable is not None:
        return unavailable
    root = stage.GetPrimAtPath(ROOT)
    if not root:
        raise ValueError("Active microwave placement is missing its Asset")
    result = {"articulation": root.HasAPI(UsdPhysics.ArticulationRootAPI)}
    for path in (BODY, DOOR):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Missing microwave rigid body: " + path)
        api = UsdPhysics.RigidBodyAPI(prim)
        result[path] = {"enabled": api.GetRigidBodyEnabledAttr().Get(),
                        "kinematic": api.GetKinematicEnabledAttr().Get(),
                        "mass": str(UsdPhysics.MassAPI(prim).GetMassAttr().Get())}
    for path in (HINGE, MOUNT):
        prim = stage.GetPrimAtPath(path)
        row = {"exists": bool(prim), "active": bool(prim and prim.IsActive())}
        if prim:
            row.update(type=prim.GetTypeName(), schemas=prim.GetAppliedSchemas(),
                       attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                                   if a.GetName().startswith(("physics:", "drive:", "physxJoint:"))},
                       relationships={r.GetName(): list(map(str, r.GetTargets()))
                                      for r in prim.GetRelationships()})
        result[path] = row
    return result


def verify(stage, expected):
    if configuration(stage) != expected:
        raise RuntimeError("Microwave fixed-base configuration changed during execution")


def apply(stage):
    from pxr import Usd, UsdPhysics
    unavailable = _unavailable(stage)
    if unavailable is not None:
        return {"policy_id": POLICY, "configuration": unavailable,
                "not_applicable": True, "already_applied": False,
                "source_assets_modified": False}
    root = stage.GetPrimAtPath(ROOT)
    if not root or not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("Expected House microwave articulation is missing")
    before = configuration(stage)
    joints = {str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}
    bodies = {str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if bodies != {BODY, DOOR} or joints - {HINGE, MOUNT}:
        raise ValueError("Unreviewed microwave body/joint topology")
    hinge_prim = stage.GetPrimAtPath(HINGE)
    hinge = UsdPhysics.RevoluteJoint(hinge_prim)
    if (not hinge or not hinge_prim.IsActive() or not hinge.GetJointEnabledAttr().Get()
            or hinge.GetExcludeFromArticulationAttr().Get()
            or list(map(str, hinge.GetBody0Rel().GetTargets())) != [BODY]
            or list(map(str, hinge.GetBody1Rel().GetTargets())) != [DOOR]):
        raise ValueError("Expected enabled body-to-door articulated hinge")
    if before[MOUNT]["exists"]:
        mount = UsdPhysics.FixedJoint(stage.GetPrimAtPath(MOUNT))
        if (root.GetCustomDataByKey("arena:microwavePhysicsPolicy") != POLICY
                or not mount or not mount.GetPrim().IsActive()
                or not mount.GetJointEnabledAttr().Get()
                or mount.GetExcludeFromArticulationAttr().Get()
                or list(map(str, mount.GetBody0Rel().GetTargets())) != [BODY]
                or mount.GetBody1Rel().GetTargets()
                or any(not before[p]["enabled"] or before[p]["kinematic"] for p in (BODY, DOOR))):
            raise ValueError("Unknown or partially configured microwave world mount")
        return {"policy_id": POLICY, "configuration": before, "already_applied": True}
    if (not before[BODY]["enabled"] or before[BODY]["kinematic"] is not True
            or not before[DOOR]["enabled"] or before[DOOR]["kinematic"] is not False):
        raise ValueError("Microwave differs from the audited kinematic-body/dynamic-door baseline")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(BODY)).GetKinematicEnabledAttr().Set(False)
        # The world attachment makes this a fixed-base articulation. Its joint
        # frames are ignored by PhysX for root selection, preserving the body's
        # current world pose, including the existing non-uniform geometry scale.
        mount = UsdPhysics.FixedJoint.Define(stage, MOUNT)
        mount.CreateBody0Rel().SetTargets([BODY])
        mount.CreateBody1Rel().SetTargets([])
        mount.CreateJointEnabledAttr(True)
        mount.CreateExcludeFromArticulationAttr(False)
        root.SetCustomDataByKey("arena:microwavePhysicsPolicy", POLICY)
    after = configuration(stage)
    if before[HINGE] != after[HINGE] or before[DOOR] != after[DOOR]:
        raise RuntimeError("Microwave repair unexpectedly changed the hinge or door")
    return {"policy_id": POLICY, "configuration": after, "before": before,
            "already_applied": False, "body_fixed_to_world": True,
            "door_drives_added": False, "collision_filters_changed": False,
            "geometry_changed": False, "source_assets_modified": False}

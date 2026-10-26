"""A09: release only black_box_1/2; existing dynamic black_box_3/4 are read-only.

Session-only, before reset. Keep original mass, collision, material and pose.

Drawer slide (user-approved public scene layer, 2026-09-15): when the scene composes
assets/house_scene_l1/demo_sim/livingroom_drawer_slide.usda, black_box_1 is already a
dynamic drawer on a prismatic slide with an open (convex-decomposition) wall collider and
its own physics materials. That authored state is accepted as-is and recorded; nothing is
written to it here. The policy never authors the slide itself.
"""

POLICY = "house.coffee_box_dynamic.v1"
ROOTS = tuple(f"/World/Assets/CoffeeTable/black_box_{i}" for i in range(1, 5))
TARGETS, REFERENCES = ROOTS[:2], ROOTS[2:]
DRAWER_SLIDE_JOINT = "/World/Assets/CoffeeTable/black_box_1_slide"
DRAWER_SLIDE_MATERIALS = ("DrawerFloorMat", "DrawerWallMat")


def drawer_slide_present(stage):
    """True only for the authored public slide layer: the joint prim targets black_box_1."""
    from pxr import UsdPhysics
    joint = stage.GetPrimAtPath(DRAWER_SLIDE_JOINT)
    if not joint or not joint.IsA(UsdPhysics.PrismaticJoint):
        return False
    targets = [str(t) for t in UsdPhysics.PrismaticJoint(joint).GetBody1Rel().GetTargets()]
    return targets == [ROOTS[0]]


def configuration(stage):
    from pxr import Usd, UsdPhysics
    result = {}
    slide = drawer_slide_present(stage)
    for path in ROOTS:
        drawer = slide and path == ROOTS[0]
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsActive() or not root.IsLoaded() or root.IsInstanceProxy():
            raise ValueError("Expected active, editable coffee box: " + path)
        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Coffee box rigid body missing: " + path)
        body = UsdPhysics.RigidBodyAPI(root)
        if body.GetRigidBodyEnabledAttr().Get() is not True:
            raise ValueError("Coffee box rigid body disabled: " + path)
        # Dynamic velocities/poses are intentionally not frozen by this guard.
        physics = {}
        colliders = 0
        for prim in Usd.PrimRange(root):
            relative = str(prim.GetPath())[len(path):]
            if drawer and prim.GetName() in DRAWER_SLIDE_MATERIALS:
                continue  # slide-layer physics materials are part of the accepted drawer state
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith("physics:") and name not in (
                    "physics:kinematicEnabled", "physics:velocity", "physics:angularVelocity"):
                    physics[relative + "." + name] = str(attr.Get())
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if (UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                        or approximation not in (("convexHull", "convexDecomposition") if drawer else ("convexHull",))):
                    raise ValueError("Unreviewed coffee box collider: " + str(prim.GetPath()))
                colliders += 1
        if colliders != 2:
            raise ValueError("Expected two original coffee box colliders: " + path)
        result[path] = dict(kinematic=body.GetKinematicEnabledAttr().Get(), physics=physics, drawer_slide=drawer)
    return result


def apply(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError("Coffee box policy requires meter-scale stage")
    before = configuration(stage)
    reference = before[REFERENCES[0]]["physics"]
    if any(before[p]["kinematic"] is not False for p in REFERENCES):
        raise ValueError("Existing movable coffee boxes must already be dynamic; refusing to edit them")
    if any(row["physics"] != reference for row in before.values() if not row["drawer_slide"]):
        raise ValueError("Coffee box physical settings differ beyond kinematic mode; review before editing")
    if before[ROOTS[0]]["drawer_slide"] and before[ROOTS[0]]["kinematic"] is not False:
        raise ValueError("Drawer slide layer must author a dynamic black_box_1")
    changes = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in TARGETS:
            if before[path]["kinematic"] is not False:
                UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Set(False)
                changes.append(dict(path=path, attribute="physics:kinematicEnabled", before=True, after=False))
    return dict(policy_id=POLICY, changes=changes, configuration=configuration(stage),
                reference_boxes_modified=False, source_assets_modified=False,
                mass_collision_material_pose_changed=False,
                drawer_slide_layer=before[ROOTS[0]]["drawer_slide"])


def verify(stage, expected):
    if configuration(stage) != expected:
        raise RuntimeError("Coffee box physical identity changed after initialization; reset required")

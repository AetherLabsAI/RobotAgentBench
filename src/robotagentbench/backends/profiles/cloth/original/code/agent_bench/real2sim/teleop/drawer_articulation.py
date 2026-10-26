"""Coffee-table black drawer real-physics articulation (shared T28/T92).

The two collected drawer tasks (small Task 28 bill-in, small Task 92
wallet-out) interact with the SAME native black storage box in the
CoffeeTable's lower compartment: the real robot pinches the slightly
protruding front wall with one gripper, pulls the box out along the shelf,
and the other arm puts the bill in / takes the wallet out.

Which box: **black_box_1**, the box nearest the robot.  Confirmed two ways
on 2026-08-02: the user identified it directly from the recordings, and the
solved robot station (``station_solve.py``, which fits the base pose to the
table corners in the REAL head frame) puts the recorded pull pinch at
world (-1.47, -0.56) — inside black_box_1's y span [-0.630, -0.406] and
5 cm proud of its closed front face.  The earlier black_box_2 reading came
from running that same FK under the WRONG station, which displaced the
pinch ~0.3 m north into the middle box.

Physics: the default basket is a free dynamic rigid body. Gravity and
native basket-to-table collisions remain enabled; the right gripper must
support the basket through pulling, manipulation and reinsertion. Wall
convex decomposition preserves the open interior. No rail or detent holds
the default basket. An explicit legacy option retains the old guided rail
for historical reproduction only.

The emitted code runs in the host ``mesh311`` pxr environment against the
scene file (same execution model as ``init_place.run_mesh_script``).  The
host pxr build has no PhysxSchema, so PhysX-only attributes are written as
raw attributes with hand-listed apiSchemas metadata (established repo
workaround).
"""

COFFEE_TABLE_PATH = "/World/Assets/CoffeeTable"
DRAWER_BODY_PATH = COFFEE_TABLE_PATH + "/black_box_1"
_COLLIDER_SUBPATH = (
    "/box/Geom/Meshes/FAB_converted_model/root/GLTF_SceneRootNode/Box_1")
DRAWER_WALLS_COLLIDER_PATH = (
    DRAWER_BODY_PATH + _COLLIDER_SUBPATH + "/Object_4/Object_0")
DRAWER_BOTTOM_COLLIDER_PATH = (
    DRAWER_BODY_PATH + _COLLIDER_SUBPATH + "/Bottom_0/Object_6/Object_1")
DRAWER_JOINT_PATH = COFFEE_TABLE_PATH + "/black_box_1_slide"
# Full drawer travel from the flush/closed pose.  The box is 0.31 m deep and
# the real pulls stop with roughly a third still supported in the compartment.
DRAWER_MAX_TRAVEL_M = 0.26
# The teleoperator's "slightly open" staging lip, used only when no per-episode
# FK front-face estimate is available.
DRAWER_DEFAULT_GAP_M = 0.04
DRAWER_MIN_GAP_M = 0.005
# Some raw recordings start with the drawer already pulled out.  The current
# benchmark 6.15 contract stages a 20 mm lip and executes the complete
# right-hand pull/place/push sequence.
DRAWER_MAX_GAP_M = 0.26
DRAWER_MASS_KG = 0.3
# A/B-measured on the Task 28 exemplar: 1.0 N / damping 8 pulls 0.160 m,
# while a lighter 0.5 N / damping 4 joint lets the drawer scoot west during
# the closure phase and only nets 0.095 m.  Keep the firmer joint.
DRAWER_JOINT_FRICTION_N = 1.0
DRAWER_LINEAR_DAMPING = 8.0
DRAWER_ANGULAR_DAMPING = 8.0
# Viscous resistance of the rail, N per (m/s).  Sized so an incidental brush
# (fast, ~0.5 m/s) meets tens of newtons while the recorded pull (slow,
# ~0.05 m/s) meets only a few.
DRAWER_RAIL_DAMPING = 120.0
DRAWER_RAIL_MAX_FORCE = 60.0
# Rubber-pad-on-fabric-box pinch: the box front wall gets a high-friction
# physics material so the recorded pull cannot skate along the wall faces.
DRAWER_WALL_FRICTION = 1.0
DRAWER_WALL_MATERIAL_PATH = DRAWER_BODY_PATH + "/DrawerWallMat"
DRAWER_AUTHORED_SCALE = 1.3
DRAWER_WALL_SCALE_MIN = 1.0
DRAWER_WALL_SCALE_MAX = 1.6
# Authored closed-pose local translates inside the CoffeeTable
# (scene_room.usda).  Every init writes drawer poses ABSOLUTELY from these
# canonical values so per-episode gaps can never accumulate drift in the
# shared, in-place-edited scene file.
DRAWER_CLOSED_LOCAL_TRANSLATE = (
    -0.38999152183532715, -0.25856897234916687, 0.20000892877578735)
# black_box_2 was articulated by the retired 2026-08-01 revision that
# mis-identified the slot; reset restores it too.
LEGACY_BOX1_PATH = COFFEE_TABLE_PATH + "/black_box_2"
LEGACY_BOX1_JOINT_PATH = COFFEE_TABLE_PATH + "/black_box_2_slide"
LEGACY_BOX1_CLOSED_LOCAL_TRANSLATE = (
    -0.13000145554542542, -0.2585703134536743, 0.20008280873298645)


def drawer_articulation_code(front_face_x=None, wall_top_z=None,
                             grip_walls=False, fixed=False, free_basket=True):
    """Return the pxr edit block that articulates the drawer.

    ``front_face_x``: desired world X of the drawer's front (outer) face at
    frame 0 — normally the FK pad midpoint of the pull hand's first
    closure; ``None`` keeps the closed pose plus ``DRAWER_DEFAULT_GAP_M``.
    ``wall_top_z``: desired world Z of the drawer wall lip (same FK pinch
    height); ``None`` keeps the authored box height.
    ``grip_walls``: bind the high-friction material to the WALLS too.  The
    wallet-out task needs it so the leaning wallet's top edge cannot slide
    down the wall (slump mode = wall slide + base roll); the bill-in task
    keeps native walls because a grabby wall degrades the pinch-and-pull
    (A/B: 0.160 -> 0.095 m).
    """
    # ``fixed`` remains only as a diagnostic escape hatch.  Production drawer
    # tasks use the dynamic slide and physical pad contact.
    front_expr = "None" if front_face_x is None else repr(float(front_face_x))
    lip_expr = "None" if wall_top_z is None else repr(float(wall_top_z))
    return f"""
# --- articulated coffee-table drawer (shared Task 28 / Task 92 contract) ---
{drawer_reset_code()}
_drawer = s.GetPrimAtPath({DRAWER_BODY_PATH!r})
assert _drawer and _drawer.IsValid(), 'missing coffee-table drawer'
if not _drawer.IsActive(): _drawer.SetActive(True)
# ONE consolidated apiSchemas write.  A prior session may have left either
# the authored prepended list or an explicit list; merge both plus the
# canonical drawer APIs so repeated inits never drop an applied schema.
_authored = _drawer.GetMetadata('apiSchemas')
_schemas = list(_authored.GetAddedOrExplicitItems()) if _authored else []
for _api in ('PhysicsRigidBodyAPI', 'PhysicsMassAPI',
             'PhysxRigidBodyAPI', 'PhysicsFilteredPairsAPI'):
    if _api not in _schemas: _schemas.append(_api)
assert _drawer.SetMetadata(
    'apiSchemas', Sdf.TokenListOp.CreateExplicit(_schemas))
_drawer_body = UsdPhysics.RigidBodyAPI(_drawer)
assert _drawer_body, 'drawer lacks RigidBodyAPI'
assert _drawer_body.CreateRigidBodyEnabledAttr().Set(True)
assert _drawer_body.CreateKinematicEnabledAttr().Set(
    {fixed!r}), 'could not set the drawer body mode'
for _nm in ('physics:velocity', 'physics:angularVelocity'):
    _a = _drawer.GetAttribute(_nm)
    if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))
_drawer.CreateAttribute(
    'physxRigidBody:linearDamping', Sdf.ValueTypeNames.Float).Set(
        {DRAWER_LINEAR_DAMPING!r})
_drawer.CreateAttribute(
    'physxRigidBody:angularDamping', Sdf.ValueTypeNames.Float).Set(
        {DRAWER_ANGULAR_DAMPING!r})
_drawer.CreateAttribute(
    'physics:mass', Sdf.ValueTypeNames.Float).Set({DRAWER_MASS_KG!r})
# open interior: walls must not be a sealed hull
_walls = s.GetPrimAtPath({DRAWER_WALLS_COLLIDER_PATH!r})
assert _walls and _walls.IsValid() and _walls.IsActive(), \\
    'missing drawer walls collider'
assert UsdPhysics.CollisionAPI(_walls).CreateCollisionEnabledAttr().Set(True)
assert UsdPhysics.MeshCollisionAPI(
    _walls).CreateApproximationAttr().Set('convexDecomposition'), \\
    'could not open the drawer wall collision'
# NOTE: a mu=1.0 wall physics material was A/B-tested 2026-08-01 and made
# the pull WORSE (0.160 -> 0.095 m): during closure the grabby wall follows
# pad micro-motion instead of letting the pads seat.  Walls keep the asset's
# native material; any stale override from that experiment is removed.
_stale_wall_material = s.GetPrimAtPath({DRAWER_WALL_MATERIAL_PATH!r})
if _stale_wall_material and _stale_wall_material.IsValid():
    s.RemovePrim(Sdf.Path({DRAWER_WALL_MATERIAL_PATH!r}))
from pxr import UsdShade as _DrawerUsdShade
_wall_binding = _DrawerUsdShade.MaterialBindingAPI(_walls)
_wall_binding.UnbindDirectBinding('physics')
_bottom = s.GetPrimAtPath({DRAWER_BOTTOM_COLLIDER_PATH!r})
assert _bottom and _bottom.IsValid() and _bottom.IsActive(), \\
    'missing drawer bottom collider'
assert UsdPhysics.CollisionAPI(_bottom).CreateCollisionEnabledAttr().Set(True)
# High-friction FLOOR only: pins a leaning wallet's base edge (the smooth
# native floor lets the rounded pillow hull slide into a deep slump).  The
# walls keep their native material — a grabby wall was A/B-measured to
# degrade the pinch-and-pull (0.160 -> 0.095 m).
from pxr import UsdShade as _DrawerFloorShade
_fmat = _DrawerFloorShade.Material.Define(
    s, {DRAWER_BODY_PATH!r} + '/DrawerFloorMat')
_fmp = _fmat.GetPrim()
_fm_authored = _fmp.GetMetadata('apiSchemas')
_fm_schemas = list(
    _fm_authored.GetAddedOrExplicitItems()) if _fm_authored else []
if 'PhysicsMaterialAPI' not in _fm_schemas:
    _fm_schemas.append('PhysicsMaterialAPI')
assert _fmp.SetMetadata(
    'apiSchemas', Sdf.TokenListOp.CreateExplicit(_fm_schemas))
_fmp.CreateAttribute(
    'physics:staticFriction', Sdf.ValueTypeNames.Float).Set(1.2)
_fmp.CreateAttribute(
    'physics:dynamicFriction', Sdf.ValueTypeNames.Float).Set(1.0)
_fmp.CreateAttribute(
    'physics:restitution', Sdf.ValueTypeNames.Float).Set(0.0)
_DrawerFloorShade.MaterialBindingAPI.Apply(_bottom).Bind(
    _fmat, materialPurpose='physics')
# COMPLIANT wall material: the real box is fabric-over-board, not rigid
# plastic.  A rigid thin wall makes the pad pinch a knife edge — round-to-
# round the same closure either grips or ejects the drawer westward.
# Compliant contact (episode-support/cushion precedent) bounds the
# penetration force so the pads conform and the friction pull becomes
# repeatable.  Friction: moderate for the bill task (a grabby wall was
# A/B-measured to degrade the pull); high for the wallet task where the
# leaning wallet must not slide down the wall.
_wcmat = _DrawerFloorShade.Material.Define(
    s, {DRAWER_BODY_PATH!r} + '/DrawerWallMat')
_wcp = _wcmat.GetPrim()
_wc_authored = _wcp.GetMetadata('apiSchemas')
_wc_schemas = list(
    _wc_authored.GetAddedOrExplicitItems()) if _wc_authored else []
for _api in ('PhysicsMaterialAPI', 'PhysxMaterialAPI'):
    if _api not in _wc_schemas: _wc_schemas.append(_api)
assert _wcp.SetMetadata(
    'apiSchemas', Sdf.TokenListOp.CreateExplicit(_wc_schemas))
_wall_mu_static = 1.2 if {grip_walls!r} else 0.7
_wall_mu_dynamic = 1.0 if {grip_walls!r} else 0.6
_wcp.CreateAttribute(
    'physics:staticFriction', Sdf.ValueTypeNames.Float).Set(_wall_mu_static)
_wcp.CreateAttribute(
    'physics:dynamicFriction', Sdf.ValueTypeNames.Float).Set(
        _wall_mu_dynamic)
_wcp.CreateAttribute(
    'physics:restitution', Sdf.ValueTypeNames.Float).Set(0.0)
_wcp.CreateAttribute(
    'physxMaterial:compliantContactStiffness',
    Sdf.ValueTypeNames.Float).Set(5000.0)
_wcp.CreateAttribute(
    'physxMaterial:compliantContactDamping',
    Sdf.ValueTypeNames.Float).Set(300.0)
_DrawerFloorShade.MaterialBindingAPI.Apply(_walls).Bind(
    _wcmat, materialPurpose='physics')
# canonical closed pose, then optional real-height wall scaling, then the
# per-episode initial lip along world +X — every write is absolute.
_dtr = _drawer.GetAttribute('xformOp:translate')
assert _dtr and _dtr.Get() is not None, 'drawer has no translate op'
_dt0 = _dtr.Get()
assert _dtr.Set(type(_dt0)(*{DRAWER_CLOSED_LOCAL_TRANSLATE!r}))
_dsc = _drawer.GetAttribute('xformOp:scale')
assert _dsc and _dsc.Get() is not None, 'drawer has no scale op'
assert _dsc.Set(Gf.Vec3f({DRAWER_AUTHORED_SCALE!r},
                         {DRAWER_AUTHORED_SCALE!r},
                         {DRAWER_AUTHORED_SCALE!r}))
def _drawer_aabb():
    _c = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    _b = _c.ComputeWorldBound(_drawer).ComputeAlignedRange()
    assert not _b.IsEmpty(), 'drawer has empty bounds'
    return _b.GetMin(), _b.GetMax()
_dmin, _dmax = _drawer_aabb()
_closed_front_x = float(_dmax[0])
_closed_bottom_z = float(_dmin[2])
_closed_top_z = float(_dmax[2])
_lip_target = {lip_expr}
_wall_scale = 1.0
if _lip_target is not None:
    _wall_scale = ((float(_lip_target) - _closed_bottom_z)
                   / (_closed_top_z - _closed_bottom_z))
    _wall_scale = min(max(_wall_scale, {DRAWER_WALL_SCALE_MIN!r}),
                      {DRAWER_WALL_SCALE_MAX!r})
    assert _dsc.Set(Gf.Vec3f(
        {DRAWER_AUTHORED_SCALE!r}, {DRAWER_AUTHORED_SCALE!r},
        {DRAWER_AUTHORED_SCALE!r} * _wall_scale))
_front_target = {front_expr}
_drawer_gap_m = (
    {DRAWER_DEFAULT_GAP_M!r} if _front_target is None
    else float(_front_target) - _closed_front_x)
assert {DRAWER_MIN_GAP_M!r} <= _drawer_gap_m <= {DRAWER_MAX_GAP_M!r}, \\
    f'drawer initial gap out of range: {{_drawer_gap_m:.3f}} m'
_smin, _smax = _drawer_aabb()
_parent_xf = UsdGeom.Imageable(
    s.GetPrimAtPath({COFFEE_TABLE_PATH!r})).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
_local_delta = _parent_xf.GetInverse().TransformDir(Gf.Vec3d(
    _drawer_gap_m, 0.0, _closed_bottom_z - float(_smin[2])))
assert _dtr.Set(type(_dt0)(
    {DRAWER_CLOSED_LOCAL_TRANSLATE!r}[0] + _local_delta[0],
    {DRAWER_CLOSED_LOCAL_TRANSLATE!r}[1] + _local_delta[1],
    {DRAWER_CLOSED_LOCAL_TRANSLATE!r}[2] + _local_delta[2]))
_drawer_front_x = _closed_front_x + _drawer_gap_m
# prismatic slide joint: world-frame body0, drawer body1, axis = world X
if not {fixed!r} and not {free_basket!r}:
  _drawer_world = UsdGeom.Imageable(
      _drawer).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
  _drawer_rot = _drawer_world.RemoveScaleShear().ExtractRotationQuat()
  _drawer_pos = _drawer_world.ExtractTranslation()
  _joint = UsdPhysics.PrismaticJoint.Define(s, {DRAWER_JOINT_PATH!r})
  _joint.CreateAxisAttr().Set('X')
  _joint.CreateBody1Rel().SetTargets([Sdf.Path({DRAWER_BODY_PATH!r})])
  _joint.CreateLocalPos0Attr().Set(Gf.Vec3f(
      float(_drawer_pos[0]), float(_drawer_pos[1]), float(_drawer_pos[2])))
  _joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
  _joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
  _qi = _drawer_rot.GetInverse()
  _joint.CreateLocalRot1Attr().Set(Gf.Quatf(
      float(_qi.GetReal()),
      float(_qi.GetImaginary()[0]),
      float(_qi.GetImaginary()[1]),
      float(_qi.GetImaginary()[2])))
  _joint.CreateLowerLimitAttr().Set(float(-_drawer_gap_m))
  _joint.CreateUpperLimitAttr().Set(
      float(max({DRAWER_MAX_TRAVEL_M!r} - _drawer_gap_m, 0.02)))
  _jp = _joint.GetPrim()
  _jauthored = _jp.GetMetadata('apiSchemas')
  _jschemas = list(_jauthored.GetAddedOrExplicitItems()) if _jauthored else []
  for _api in ('PhysxJointAPI',):
      if _api not in _jschemas: _jschemas.append(_api)
  assert _jp.SetMetadata(
      'apiSchemas', Sdf.TokenListOp.CreateExplicit(_jschemas))
  _jp.CreateAttribute(
      'physxJoint:jointFriction', Sdf.ValueTypeNames.Float).Set(
          {DRAWER_JOINT_FRICTION_N!r})
  # Viscous rail resistance.  A loaded fabric box in a wooden compartment does
  # not skate away when an arm brushes it, but it does yield to a deliberate
  # pull -- damping models exactly that asymmetry (it scales with speed, so it
  # resists the fast incidental contact that was shoving the drawer shut and
  # carrying the wallet 153 mm west before the closure, while leaving the slow
  # recorded pull almost unaffected).  A zero-stiffness drive is the standard
  # way to author it: no preferred position, pure velocity opposition.
  _drive = UsdPhysics.DriveAPI.Apply(_jp, 'linear')
  _drive.CreateTypeAttr().Set('force')
  _drive.CreateStiffnessAttr().Set(0.0)
  _drive.CreateDampingAttr().Set({DRAWER_RAIL_DAMPING!r})
  _drive.CreateTargetVelocityAttr().Set(0.0)
  _drive.CreateMaxForceAttr().Set({DRAWER_RAIL_MAX_FORCE!r})
  # guidance is the joint's job: never let compartment hulls fight it
  # (PhysicsFilteredPairsAPI is already in the consolidated schema list)
_drawer.CreateRelationship('physics:filteredPairs').SetTargets(
    [] if {free_basket!r} else [Sdf.Path({COFFEE_TABLE_PATH!r})])
# A removable basket has no rail to carry its weight or prevent tipping.
# Restore contact with the actual triangle-mesh compartment and leave all
# six rigid-body degrees of freedom to the physical solver.
_drawer.CreateAttribute('agentBench:freeBasket', Sdf.ValueTypeNames.Bool).Set({free_basket!r})
print(f'drawer articulated: gap={{_drawer_gap_m:.3f}} m '
      f'front_x={{_drawer_front_x:.3f}} wall_scale={{_wall_scale:.3f}} '
      f'travel=[{{-_drawer_gap_m:.3f}}, '
      f'{{{DRAWER_MAX_TRAVEL_M!r} - _drawer_gap_m:.3f}}]')
# --- end articulated drawer block ---
"""


def drawer_reset_code():
    """Return the pxr block restoring closed, kinematic drawer boxes.

    Non-drawer tasks run this so a prior Task 28/92 episode cannot leave an
    articulated, slightly-open drawer (visible in front-facing coffee-table
    videos and a live joint in their physics) in the shared scene file.  It
    also repairs the retired black_box_1 articulation.
    """
    resets = (
        (DRAWER_BODY_PATH, DRAWER_JOINT_PATH,
         DRAWER_CLOSED_LOCAL_TRANSLATE, DRAWER_WALLS_COLLIDER_PATH),
        (LEGACY_BOX1_PATH, LEGACY_BOX1_JOINT_PATH,
         LEGACY_BOX1_CLOSED_LOCAL_TRANSLATE,
         LEGACY_BOX1_PATH + _COLLIDER_SUBPATH + "/Object_4/Object_0"),
    )
    return f"""
# --- reset coffee-table drawers to the closed kinematic contract pose ---
for _dr_path, _dr_joint, _dr_closed, _dr_walls in {resets!r}:
    _dr = s.GetPrimAtPath(_dr_path)
    if not (_dr and _dr.IsValid()):
        continue
    _dr_jp = s.GetPrimAtPath(_dr_joint)
    if _dr_jp and _dr_jp.IsValid():
        assert s.RemovePrim(Sdf.Path(_dr_joint)), \\
            'could not remove leftover drawer slide joint'
    _dr_body = UsdPhysics.RigidBodyAPI(_dr)
    if _dr_body:
        _dr_body.CreateKinematicEnabledAttr().Set(True)
    for _nm in ('physics:velocity', 'physics:angularVelocity'):
        _a = _dr.GetAttribute(_nm)
        if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))
    _dr_tr = _dr.GetAttribute('xformOp:translate')
    if _dr_tr and _dr_tr.Get() is not None:
        _dr_tr.Set(type(_dr_tr.Get())(*_dr_closed))
    _dr_sc = _dr.GetAttribute('xformOp:scale')
    if _dr_sc and _dr_sc.Get() is not None:
        _dr_sc.Set(Gf.Vec3f({DRAWER_AUTHORED_SCALE!r},
                            {DRAWER_AUTHORED_SCALE!r},
                            {DRAWER_AUTHORED_SCALE!r}))
    _dr_rel = _dr.GetRelationship('physics:filteredPairs')
    if _dr_rel:
        _dr_rel.SetTargets([])
    _dr_w = s.GetPrimAtPath(_dr_walls)
    if _dr_w and _dr_w.IsValid() and _dr_w.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI(
            _dr_w).CreateApproximationAttr().Set('convexHull')
# --- end drawer reset block ---
"""

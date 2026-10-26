"""Per-episode init for the trash-to-bin tasks (legacy 28, canonical 29/30).

Task 28 (green notebook -> bin, 168 eps; proxy=Note green notepad 21.6x14 — Book 30cm bridges the bin mouth): the episode STARTS with the object already held
(grip closed from frame 0). align_grasp's longest-closed-hold logic lands at/near frame 0,
so its fingertip midpoint IS the in-hand pose: place the proxy object (Book) there; the
closed-loop settle pins it between the pads before playback, then the recorded trajectory
carries it to the bin and opens.

Task 29 (three striped paper cups -> teal bin, 179 eps): the complete task clears
three CoffeeTable cups one-by-one.  The legacy entrypoint still sets up one PaperCup as
a PARTIAL/PROVISIONAL probe only.  Both bins use their original equal asset scale and
original side-by-side roots in the CoffeeTable/fireplace passage.  The robot uses the
shared CoffeeTable-pick parking pose in the Sofa/CoffeeTable corridor, facing the table.
The teal bin is the destination and the burgundy/red bin is the distractor.

Task 30 (crumpled paper -> bin, 188 eps): normal pick of the PaperBall prop from the
PaperPlate zone, then the recorded release over the bin. Unified placement convention
(top = gz + BITE_Z, riser bridges if needed).  Pick placement uses the active collision
bound (falling back to active mesh bounds), never a PaperBall-specific radius.

    THROW_TASK=28 conda run -n base python real2sim/teleop/init_throw.py
Emits JSON {ok, target, hand, kin, base}.
"""
import json, math, os, re, subprocess, sys, tempfile

from robot_positions import (
    COFFEE_TABLE_CURRENT_TOP_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    robot_position,
    translate_interval,
    translate_xy,
    translate_xyz,
)
from dual_trash_bin_layout import (
    BIN_OUTER_DIMENSIONS_XYZ_M as TASK29_BIN_OUTER_DIMENSIONS_XYZ_M,
    BIN_SCALE_XYZ as TASK29_BIN_SCALE_XYZ,
    BOTTOM_LOD_FACE_VERTEX_COUNT as TASK29_BOTTOM_LOD_FACE_VERTEX_COUNT,
    BOTTOM_LOD_SCHEMA as TASK29_BOTTOM_LOD_SCHEMA,
    BOTTOM_LOD_THICKNESS_M as TASK29_BOTTOM_LOD_THICKNESS_M,
    DUAL_BIN_LAYOUT as TASK29_BIN_LAYOUT,
    RED_BOTTOM_LOD_PATH as TASK29_RED_BOTTOM_LOD_PATH,
    RED_COLLIDER_PATH as TASK29_RED_COLLIDER_PATH,
    TEAL_BOTTOM_LOD_PATH as TASK29_TEAL_BOTTOM_LOD_PATH,
    TEAL_COLLIDER_PATH as TASK29_TEAL_COLLIDER_PATH,
)
from task29_full_reproduce import (
    TASK29_BIN_OFFSET_X_M,
    TASK29_BIN_OFFSET_Y_M,
    TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M,
    TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
    TASK29_PAPER_PLATE_BOTTOM_M,
    TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M,
    TASK29_PAPER_PLATE_PATH,
    TASK29_PAPER_PLATE_TOP_M,
    TASK29_PAPER_PLATE_TRANSLATE_XYZ_M,
)
from task30_throw_reproduce import (
    TASK30_BASE_XY_M,
    TASK30_CALIBRATED_EPISODE_UID,
    TASK30_GENIE_PAPER_SCALE_XYZ,
    TASK30_TABLE_PLATE_CONTEXT,
)

from machine_paths import (
    CONTAINER_NAME, DEMO_ROOT, RENDER_GPU, isaac_python_command,
)

DEMO = str(DEMO_ROOT)

INSTANCE = os.environ.get("INSTANCE", "")
_RT = INSTANCE or "_MAIN"
SCENE = f"{DEMO}/scene_room{INSTANCE}.usda"
CAM_PREVIEW = f"{DEMO}/cam_preview{INSTANCE}"

_TABLE_PICK = robot_position("table_pick")
# check_success scores small Task 30 against the reviewed release-mapped
# station (task30_throw_reproduce.TASK30_BASE_XY_M), not the shared
# table_pick parking pose.  Defaulting the initializer to that same constant
# closes the init-vs-scoring station split that rejected every queue render
# of task 30 at the trace-identity gate.  FIXED_BASE still overrides for
# calibration, and the argv-dispatch path (no THROW_TASK env) keeps the
# shared station.
_DEFAULT_BASE_XY = (
    TASK30_BASE_XY_M
    if os.environ.get("THROW_TASK", "") == "30"
    else _TABLE_PICK.xy_m
)
BASE = tuple(map(float, os.environ.get(
    "FIXED_BASE",
    f"{_DEFAULT_BASE_XY[0]},{_DEFAULT_BASE_XY[1]}",
).split(",")))
QUAT = (0.0, 0.0, 0.0, 1.0)  # Sofa/CoffeeTable corridor, facing west across table

TASK29_TEAL_BAG_COLLIDER_PATH = (
    "/World/Assets/TrashBinTeal/geo/Trash_Bag/Trash_Bag_002"
)
def validate_task29_bin_layout(layout):
    """Validate the serialisable, fail-closed Task29 bin reset contract."""
    if not isinstance(layout, dict) or set(layout) != {
            "destination", "distractor"}:
        raise ValueError(
            "Task29 bin layout must contain destination and distractor")
    expected_names = {
        "destination": (
            "teal",
            "fixed_fireplace_side_destination"),
        "distractor": (
            "red",
            "fixed_adjacent_robot_side_distractor"),
    }
    expected_collider_states = {
        "destination": (
            (TASK29_TEAL_COLLIDER_PATH, True),
            (TASK29_TEAL_BAG_COLLIDER_PATH, True),
            (TASK29_TEAL_BOTTOM_LOD_PATH, True),
        ),
        "distractor": (
            (TASK29_RED_COLLIDER_PATH, True),
            (
                "/World/Assets/TrashBinRed/geo/Trash_Bag/"
                "Trash_Bag_002",
                True,
            ),
            (TASK29_RED_BOTTOM_LOD_PATH, True),
        ),
    }
    for slot, (expected_name, expected_role) in expected_names.items():
        spec = layout[slot]
        if not isinstance(spec, dict):
            raise ValueError(f"Task29 {slot} bin spec must be a mapping")
        expected_fields = {
            "name",
            "role",
            "body_path",
            "translate",
            "orient_wxyz",
            "scale",
            "outer_dimensions_xyz_m",
            "calibrated_center_xy",
            "center_tolerance_m",
            "scoring_collider_path",
            "colliders",
        }
        expected_fields.add("bottom_collision_lod")
        if set(spec) != expected_fields:
            raise ValueError(
                f"Task29 {slot} bin fields are not canonical")
        if spec.get("name") != expected_name or spec.get("role") != expected_role:
            raise ValueError(f"Task29 {slot} bin identity is not canonical")
        body_path = spec.get("body_path")
        if not isinstance(body_path, str) or not body_path.startswith(
                "/World/Assets/TrashBin"):
            raise ValueError(f"Task29 {slot} body_path is invalid")
        for field, length in (
                ("translate", 3), ("orient_wxyz", 4), ("scale", 3),
                ("outer_dimensions_xyz_m", 3),
                ("calibrated_center_xy", 2)):
            values = spec.get(field)
            if not isinstance(values, (tuple, list)) or len(values) != length:
                raise ValueError(
                    f"Task29 {slot} {field} must contain {length} values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"Task29 {slot} {field} must be finite")
        if not all(float(value) > 0.0 for value in spec["scale"]):
            raise ValueError(f"Task29 {slot} scale must be positive")
        if tuple(float(value) for value in spec[
                "outer_dimensions_xyz_m"]) != (
                    TASK29_BIN_OUTER_DIMENSIONS_XYZ_M):
            raise ValueError(
                f"Task29 {slot} outer dimensions are not canonical")
        tolerance = float(spec.get("center_tolerance_m", float("nan")))
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 0.02:
            raise ValueError(
                f"Task29 {slot} center tolerance must be in (0, 0.02]")
        colliders = spec.get("colliders")
        if not isinstance(colliders, (tuple, list)) or len(colliders) < 1:
            raise ValueError(f"Task29 {slot} colliders must be non-empty")
        collider_paths = []
        prefix = body_path.rstrip("/") + "/"
        for collider in colliders:
            if (not isinstance(collider, (tuple, list))
                    or len(collider) != 2
                    or type(collider[1]) is not bool):
                raise ValueError(
                    f"Task29 {slot} collider entry must be (path, bool)")
            collider_path, enabled = collider
            if not isinstance(collider_path, str) or not collider_path.startswith(
                    prefix):
                raise ValueError(
                    f"Task29 {slot} collider must descend from its body")
            collider_paths.append(collider_path)
        if len(set(collider_paths)) != len(collider_paths):
            raise ValueError(f"Task29 {slot} collider paths must be unique")
        if tuple(tuple(item) for item in colliders) != (
                expected_collider_states[slot]):
            raise ValueError(
                f"Task29 {slot} collider state is not canonical")
        scoring_collider_path = spec.get("scoring_collider_path")
        if (
            scoring_collider_path != expected_collider_states[slot][0][0]
            or dict(colliders).get(scoring_collider_path)
            is not True
        ):
            raise ValueError(
                f"Task29 {slot} reference collider is not canonical")
        lod = spec.get("bottom_collision_lod")
        expected_lod = {
            "schema": TASK29_BOTTOM_LOD_SCHEMA,
            "path": (
                TASK29_TEAL_BOTTOM_LOD_PATH
                if slot == "destination"
                else TASK29_RED_BOTTOM_LOD_PATH
            ),
            "source_collider_path": (
                TASK29_TEAL_COLLIDER_PATH
                if slot == "destination"
                else TASK29_RED_COLLIDER_PATH
            ),
            "source_face_vertex_count":
                TASK29_BOTTOM_LOD_FACE_VERTEX_COUNT,
            "thickness_m": TASK29_BOTTOM_LOD_THICKNESS_M,
        }
        if lod != expected_lod:
            raise ValueError(
                f"Task29 {slot} bottom collision LOD is not canonical")
    destination_center = layout["destination"]["calibrated_center_xy"]
    distractor_center = layout["distractor"]["calibrated_center_xy"]
    if (
        not math.isclose(
            float(distractor_center[0]) - float(destination_center[0]),
            TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            float(distractor_center[1]),
            float(destination_center[1]),
            abs_tol=1e-12,
            rel_tol=0.0,
        )
    ):
        raise ValueError(
            "Task29 red distractor must be adjacent on teal's robot side")
    return layout


_TASK29_BOTTOM_LOD_RUNTIME = r"""
def _ensure_task29_bottom_lod(stage, root, spec):
    lod_spec = spec.get('bottom_collision_lod')
    if lod_spec is None:
        return None

    source_path = lod_spec['source_collider_path']
    lod_path = lod_spec['path']
    source = stage.GetPrimAtPath(source_path)
    if not source or not source.IsValid() or not source.IsActive():
        raise RuntimeError(
            'missing/inactive Task29 bottom LOD source ' + source_path)
    if source.GetTypeName() != 'Mesh':
        raise RuntimeError(
            'Task29 bottom LOD source is not a mesh: ' + source_path)
    source_mesh = UsdGeom.Mesh(source)
    if not source.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(
            'Task29 bottom LOD source lacks CollisionAPI: ' + source_path)
    if not source.HasAPI(UsdPhysics.MeshCollisionAPI):
        raise RuntimeError(
            'Task29 bottom LOD source lacks MeshCollisionAPI: ' + source_path)
    source_approximation = (
        UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get())
    if str(source_approximation) != 'none':
        raise RuntimeError(
            'Task29 bottom LOD source approximation must be none, found '
            + repr(source_approximation))
    source_double_sided = source_mesh.CreateDoubleSidedAttr()
    if not source_double_sided.Set(True):
        raise RuntimeError(
            'could not make Task29 visible destination double-sided')
    if source_double_sided.Get() is not True:
        raise RuntimeError(
            'Task29 visible destination did not remain double-sided')

    parent = source.GetParent()
    if lod_path.rsplit('/', 1)[0] != str(parent.GetPath()):
        raise RuntimeError(
            'Task29 bottom collision LOD must share the source parent')
    cur = parent
    while cur and cur.IsValid():
        if cur.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                'Task29 bottom collision LOD parent chain is not static: '
                + str(cur.GetPath()))
        if cur == root:
            break
        cur = cur.GetParent()
    if cur != root:
        raise RuntimeError(
            'Task29 bottom collision LOD source escapes destination body')
    cur = root.GetParent()
    while cur and cur.IsValid() and not cur.IsPseudoRoot():
        if cur.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                'Task29 bottom collision LOD ancestor chain is not static: '
                + str(cur.GetPath()))
        cur = cur.GetParent()

    thickness_m = float(lod_spec['thickness_m'])
    if not math.isfinite(thickness_m) or not 0.003 <= thickness_m <= 0.010:
        raise RuntimeError(
            'Task29 bottom collision LOD thickness is implausible')
    face_vertex_count = int(lod_spec['source_face_vertex_count'])
    if face_vertex_count != 16:
        raise RuntimeError(
            'Task29 bottom collision LOD source face must have 16 vertices')

    source_points = list(source_mesh.GetPointsAttr().Get() or [])
    face_counts = [
        int(value)
        for value in (source_mesh.GetFaceVertexCountsAttr().Get() or [])
    ]
    face_indices = [
        int(value)
        for value in (source_mesh.GetFaceVertexIndicesAttr().Get() or [])
    ]
    if (
        not source_points
        or not face_counts
        or sum(face_counts) != len(face_indices)
        or any(index < 0 or index >= len(source_points)
               for index in face_indices)
    ):
        raise RuntimeError(
            'Task29 visible destination mesh topology is invalid')
    sixteen_faces = []
    cursor = 0
    for count in face_counts:
        face = face_indices[cursor:cursor + count]
        cursor += count
        if count == face_vertex_count:
            sixteen_faces.append(face)
    if len(sixteen_faces) != 1:
        raise RuntimeError(
            'Task29 visible destination must contain one unique 16-gon')
    bottom_face = sixteen_faces[0]
    if len(set(bottom_face)) != face_vertex_count:
        raise RuntimeError(
            'Task29 visible destination 16-gon repeats vertices')

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_to_world = xform_cache.GetLocalToWorldTransform(source)
    top_world = [
        source_to_world.Transform(Gf.Vec3d(*source_points[index]))
        for index in bottom_face
    ]
    source_bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
        useExtentsHint=False)
    source_range = source_bounds_cache.ComputeWorldBound(
        source).ComputeAlignedRange()
    if source_range.IsEmpty():
        raise RuntimeError(
            'Task29 visible destination collider bound is empty')
    source_bottom_z = float(source_range.GetMin()[2])
    if (
        max(float(point[2]) for point in top_world)
        - min(float(point[2]) for point in top_world) > 0.0001
        or max(abs(float(point[2]) - source_bottom_z)
               for point in top_world) > 0.0001
    ):
        raise RuntimeError(
            'Task29 unique 16-gon is no longer the planar collider bottom')

    world_to_parent = xform_cache.GetLocalToWorldTransform(
        parent).GetInverse()
    top_local = [
        world_to_parent.Transform(point) for point in top_world
    ]
    bottom_local = [
        world_to_parent.Transform(
            point + Gf.Vec3d(0.0, 0.0, -thickness_m))
        for point in top_world
    ]
    lod_points = [
        Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
        for point in (top_local + bottom_local)
    ]
    count = len(top_local)
    top_face_indices = list(reversed(range(count)))
    bottom_face_indices = list(range(count, 2 * count))
    side_faces = [
        [
            index,
            (index + 1) % count,
            count + ((index + 1) % count),
            count + index,
        ]
        for index in range(count)
    ]

    # The shared initialized scene may contain a LOD authored by an earlier
    # attempt.  Reusing it would preserve unknown PhysX APIs, material
    # bindings, relationships, or custom attributes.  Remove the complete
    # prim and rebuild the small whitelist below on every initialization.
    old_lod = stage.GetPrimAtPath(lod_path)
    if old_lod and old_lod.IsValid():
        if not stage.RemovePrim(old_lod.GetPath()):
            raise RuntimeError(
                'could not remove stale Task29 bottom collision LOD')
        if stage.GetPrimAtPath(lod_path).IsValid():
            raise RuntimeError(
                'stale Task29 bottom collision LOD still composes')
    lod_mesh = UsdGeom.Mesh.Define(stage, lod_path)
    lod_prim = lod_mesh.GetPrim()
    if not lod_prim or not lod_prim.IsValid():
        raise RuntimeError(
            'could not define Task29 bottom collision LOD')
    if not lod_prim.IsActive():
        lod_prim.SetActive(True)
    UsdGeom.Xformable(lod_prim).ClearXformOpOrder()
    for api_schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
        if lod_prim.HasAPI(api_schema) and not lod_prim.RemoveAPI(api_schema):
            raise RuntimeError(
                'could not keep Task29 bottom collision LOD static')
    lod_mesh.CreatePointsAttr(lod_points)
    lod_mesh.CreateFaceVertexCountsAttr([count, count] + [4] * count)
    lod_mesh.CreateFaceVertexIndicesAttr(
        top_face_indices
        + bottom_face_indices
        + [value for face in side_faces for value in face])
    lod_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lod_mesh.CreateDoubleSidedAttr(False)
    lod_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    lod_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    lod_mesh.CreateExtentAttr(
        UsdGeom.PointBased.ComputeExtent(lod_points))
    lod_collision = UsdPhysics.CollisionAPI.Apply(lod_prim)
    lod_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(lod_prim)
    if not lod_collision.CreateCollisionEnabledAttr().Set(True):
        raise RuntimeError(
            'could not enable Task29 bottom collision LOD')
    if not lod_mesh_collision.CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull):
        raise RuntimeError(
            'could not set Task29 bottom collision LOD convexHull')
    if (
        lod_collision.GetCollisionEnabledAttr().Get() is not True
        or str(lod_mesh_collision.GetApproximationAttr().Get())
            != 'convexHull'
        or lod_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        raise RuntimeError(
            'Task29 bottom collision LOD state readback failed')

    actual_points = list(lod_mesh.GetPointsAttr().Get() or [])
    actual_counts = [
        int(value)
        for value in (lod_mesh.GetFaceVertexCountsAttr().Get() or [])
    ]
    actual_indices = [
        int(value)
        for value in (lod_mesh.GetFaceVertexIndicesAttr().Get() or [])
    ]
    expected_counts = [count, count] + [4] * count
    expected_indices = (
        top_face_indices
        + bottom_face_indices
        + [value for face in side_faces for value in face]
    )
    if (
        len(actual_points) != 2 * count
        or actual_counts != expected_counts
        or actual_indices != expected_indices
    ):
        raise RuntimeError(
            'Task29 bottom collision LOD topology readback failed')
    lod_to_world = UsdGeom.XformCache(
        Usd.TimeCode.Default()).GetLocalToWorldTransform(lod_prim)
    actual_world = [
        lod_to_world.Transform(Gf.Vec3d(*point))
        for point in actual_points
    ]
    actual_top_world = actual_world[:count]
    actual_bottom_world = actual_world[count:]
    for index, (source_point, top_point, bottom_point) in enumerate(zip(
            top_world, actual_top_world, actual_bottom_world)):
        if max(abs(float(source_point[axis]) - float(top_point[axis]))
               for axis in range(3)) > 0.000002:
            raise RuntimeError(
                'Task29 bottom collision LOD top escaped source at vertex '
                + str(index))
        if (
            max(abs(float(top_point[axis]) - float(bottom_point[axis]))
                for axis in range(2)) > 0.000002
            or abs(
                (float(top_point[2]) - float(bottom_point[2]))
                - thickness_m
            ) > 0.000002
        ):
            raise RuntimeError(
                'Task29 bottom collision LOD extrusion readback failed')

    def _rounded_points(points):
        return [
            [round(float(point[axis]), 9) for axis in range(3)]
            for point in points
        ]

    return dict(
        schema=lod_spec['schema'],
        path=lod_path,
        source_collider_path=source_path,
        source_face_vertex_count=count,
        top_face_world_xyz=_rounded_points(actual_top_world),
        bottom_face_world_xyz=_rounded_points(actual_bottom_world),
        thickness_m=round(thickness_m, 9),
        extrusion_direction='world_negative_z',
        topology='closed_extruded_polygon_v1',
        approximation='convexHull',
        purpose='guide',
        visibility='invisible',
        double_sided=False,
        collision_enabled=True,
        rigid_body=False,
        enabled_from_frame=0,
        source_double_sided=True,
        source_approximation='none',
    )
"""


THROW_TABLE_PLATE_CONTEXT = {
    "coffee_table_translate_xyz": TASK29_COFFEE_TABLE_ROOT_TRANSLATE_XYZ_M,
    "paper_plate_path": TASK29_PAPER_PLATE_PATH,
    "paper_plate_translate_xyz": TASK29_PAPER_PLATE_TRANSLATE_XYZ_M,
    "paper_plate_expected_size_xyz_m": (
        TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M),
    "paper_plate_bottom_m": TASK29_PAPER_PLATE_BOTTOM_M,
    "paper_plate_top_m": TASK29_PAPER_PLATE_TOP_M,
}


def resolve_table_plate_context(
        task_id, episode_uid, context, plate_xy_override=""):
    """Resolve support geometry without silently crossing episode identities.

    The reviewed small Task 30 plate pose belongs to exactly one calibrated
    recording.  A different recording must provide its reviewed plate XY
    explicitly; otherwise replaying it against the canonical plate can shift
    the resting paper by about 10 cm while still looking superficially valid.
    """
    tid = str(task_id)
    uid = str(episode_uid or "").strip()
    override = str(plate_xy_override or "").strip()
    if context is None:
        if override:
            raise ValueError(
                "THROW_PLATE_XY was provided but this task has no plate")
        return None, {
            "mode": "not_applicable",
            "episode_uid": uid or None,
        }

    if tid == "30" and not override:
        if uid != TASK30_CALIBRATED_EPISODE_UID:
            identity = uid or "<missing>"
            raise ValueError(
                "small Task 30 plate pose is episode-bound: "
                f"EPISODE_UID={identity} is not the calibrated episode; "
                "provide a reviewed THROW_PLATE_XY")
        binding_mode = "calibrated_episode_default"
    elif tid == "30":
        if not uid:
            raise ValueError(
                "small Task 30 explicit THROW_PLATE_XY requires EPISODE_UID")
        binding_mode = "explicit_episode_override"
    else:
        binding_mode = "task_default"

    resolved = dict(context)
    if override:
        try:
            values = tuple(float(value) for value in override.split(","))
        except ValueError as exc:
            raise ValueError(
                "THROW_PLATE_XY must be two finite numbers") from exc
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("THROW_PLATE_XY must be two finite numbers")
        px, py = values
        old = resolved["paper_plate_translate_xyz"]
        resolved["paper_plate_translate_xyz"] = (px, py, old[2])
        half = [
            value / 2.0
            for value in resolved["paper_plate_expected_size_xyz_m"]
        ]
        resolved["paper_plate_aabb_min_xyz"] = [
            px - half[0], py - half[1], resolved["paper_plate_bottom_m"]]
        resolved["paper_plate_aabb_max_xyz"] = [
            px + half[0], py + half[1], resolved["paper_plate_top_m"]]
    elif resolved not in (
            THROW_TABLE_PLATE_CONTEXT, TASK30_TABLE_PLATE_CONTEXT):
        raise ValueError("throw table/plate context is not canonical")

    return resolved, {
        "mode": binding_mode,
        "episode_uid": uid or None,
        "plate_xy_m": list(resolved["paper_plate_translate_xyz"][:2]),
        "explicit_override": bool(override),
    }


TASKS = {
    "28": dict(prim="Note", mode="inhand",
               band=dict(x=(-2.60, -1.40), y=(-1.60, 0.15), z=(0.25, 1.05))),
    "29": dict(prim="PaperCup", mode="pick",
               surface_z=COFFEE_TABLE_CURRENT_TOP_M,
               required_count=3, probe_scope="single_cup_partial",
               bin_layout=TASK29_BIN_LAYOUT,
               table_plate_context=THROW_TABLE_PLATE_CONTEXT,
               # The paper_cup asset's long/cup axis is local +Z.  A prior
               # pick run can leave this shared scene prim at an arbitrary
               # rotateXYZ (the observed stale value -90,-90,0 laid the
               # 130-mm cup sideways).  The real Task29 videos show all
               # three cups upright, so reset the complete orientation
               # deterministically before measuring collision bounds.
               orientation_mode="upright_local_z",
               rotate_xyz=(0.0, 0.0, 0.0),
               band=dict(
                   x=translate_interval(
                       (-2.50, -1.60),
                       COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0],
                   ),
                   y=translate_interval(
                       (-1.10, 0.15),
                       COFFEE_TABLE_LAYOUT_SHIFT_XY_M[1],
                   ),
                   z=(0.35, 0.90),
               )),
    # The old small Task 30 release cluster targeted a task-local bin pose.
    # Current furniture policy forbids that override: small Task 30 consumes
    # the same canonical basket/teal/red row as small Task 29.  Its robot
    # station and relative trajectory must be recalibrated around this layout.
    "30": dict(
        prim="PaperBall",
        mode="pick",
        root_scale_xyz=TASK30_GENIE_PAPER_SCALE_XYZ,
        # PaperBall begins on the floral PaperPlate, not buried in the table.
        surface_z=TASK29_PAPER_PLATE_TOP_M,
        table_plate_context=TASK30_TABLE_PLATE_CONTEXT,
        bin_layout=TASK29_BIN_LAYOUT,
        band=dict(
            x=translate_interval(
                (-2.50, -1.60),
                COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0],
            ),
            y=translate_interval(
                (-1.10, 0.15),
                COFFEE_TABLE_LAYOUT_SHIFT_XY_M[1],
            ),
            z=(0.35, 0.90),
        ),
    ),
}


def validate_paper_cup_upright_size(size):
    """Fail closed unless a composed PaperCup bound is upright and plausible.

    The canonical asset measures about 89 x 89 x 130 mm when local +Z is
    vertical.  Broad tolerances allow normal authored-scale drift while
    rejecting the stale sideways 89 x 130 x 89 mm pose that is wider than the
    live gripper opening.
    """
    try:
        values = tuple(float(value) for value in size)
    except (TypeError, ValueError):
        raise ValueError("PaperCup upright size must contain numeric values")
    if len(values) != 3:
        raise ValueError("PaperCup upright size must contain exactly 3 values")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("PaperCup upright size must be finite and positive")
    sx, sy, sz = values
    if not (0.070 <= sx <= 0.110 and 0.070 <= sy <= 0.110):
        raise ValueError(
            "PaperCup upright horizontal span outside 70-110 mm: "
            f"size={values!r}")
    if not 0.110 <= sz <= 0.150:
        raise ValueError(
            "PaperCup upright vertical span outside 110-150 mm: "
            f"size={values!r}")
    if sz < 1.20 * max(sx, sy):
        raise ValueError(
            "PaperCup long axis is not vertical: "
            f"size={values!r}")
    return values


def solve_pick_placement(anchor, bounds_min, bounds_max, *, bite_z=0.012,
                         surface_z=COFFEE_TABLE_CURRENT_TOP_M,
                         max_riser_height=0.20):
    """Solve a pick prop's world-space placement from its measured active bound.

    ``anchor`` is the fingertip midpoint.  The measured bound's top is aligned
    to ``anchor.z + bite_z`` unless that would put the object through the table;
    the returned translation delta preserves arbitrary asset origins.  The
    riser decision matches the runtime placement policy.
    """
    anchor = tuple(anchor)
    bounds_min = tuple(bounds_min)
    bounds_max = tuple(bounds_max)
    if len(anchor) != 3 or len(bounds_min) != 3 or len(bounds_max) != 3:
        raise ValueError("anchor and bounds must contain exactly three values")
    bite_z = float(bite_z)
    surface_z = float(surface_z)
    max_riser_height = float(max_riser_height)
    values = (
        anchor + bounds_min + bounds_max
        + (bite_z, surface_z, max_riser_height)
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("anchor and bounds must be finite")
    lo = tuple(float(value) for value in bounds_min)
    hi = tuple(float(value) for value in bounds_max)
    size = tuple(hi[index] - lo[index] for index in range(3))
    if not all(value > 1e-6 for value in size):
        raise ValueError(f"active object bound is empty or degenerate: size={size!r}")

    gx, gy, gz = (float(value) for value in anchor)
    support = surface_z + 0.004
    bottom = gz + bite_z - size[2]
    riser_base = min(surface_z, bottom - 0.001)
    riser_height = bottom - riser_base
    if not 0.02 <= max_riser_height <= 0.50:
        raise ValueError("max riser height must be in [0.02, 0.50] m")
    if riser_height >= max_riser_height and bottom > support + 0.003:
        raise ValueError(
            f"required riser {riser_height:.3f} m exceeds safe placement range")
    use_riser = (
        0.008 <= riser_height < max_riser_height
        and bottom > support + 0.003
    )
    if not use_riser:
        bottom = max(bottom, support)
        riser_height = 0.0

    center_x = (lo[0] + hi[0]) / 2.0
    center_y = (lo[1] + hi[1]) / 2.0
    return {
        "size": size,
        "bottom": bottom,
        "translation_delta": (gx - center_x, gy - center_y, bottom - lo[2]),
        "use_riser": use_riser,
        "riser_base": riser_base,
        "riser_height": riser_height,
    }


def sh(c, t=None):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)


def main():
    tid = os.environ.get("THROW_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if tid not in TASKS:
        print(json.dumps({"ok": False, "reason": f"unknown THROW_TASK {tid}"}))
        return 2
    cfg = TASKS[tid]
    table_plate_binding = None
    tp = f"/World/Assets/{cfg['prim']}"
    task29_phase_text = os.environ.get("TASK29_PHASE_ORDINAL")
    if tid == "29":
        task29_phase_text = (
            "0" if task29_phase_text is None else task29_phase_text)
        if not re.fullmatch(r"[0-2]", task29_phase_text):
            print(json.dumps({
                "ok": False,
                "reason": (
                    "TASK29_PHASE_ORDINAL must be exactly 0, 1, or 2"),
            }))
            return 2
        task29_phase_ordinal = int(task29_phase_text)
    else:
        if task29_phase_text is not None:
            print(json.dumps({
                "ok": False,
                "reason": (
                    "TASK29_PHASE_ORDINAL is authorized only for Task29"),
            }))
            return 2
        task29_phase_ordinal = None

    prealign_bin_reset = ""
    if tid == "29":
        try:
            validate_task29_bin_layout(cfg["bin_layout"])
        except (TypeError, ValueError) as exc:
            print(json.dumps({
                "ok": False,
                "reason": f"invalid Task29 bin layout: {exc}",
            }))
            return 2
        prealign_bin_reset = f"""
# Reset the independently calibrated bin geometry before align_grasp starts
# Isaac physics.  Placement repeats this reset and emits the full descriptor,
# guarding both physical phases against shared-scene residue.
{_TASK29_BOTTOM_LOD_RUNTIME}
_layout = {cfg['bin_layout']!r}
for _slot in ('destination', 'distractor'):
    _spec = _layout[_slot]
    _root = s.GetPrimAtPath(_spec['body_path'])
    if not _root or not _root.IsValid():
        raise RuntimeError(
            'missing pre-align Task29 bin ' + _spec['body_path'])
    if not _root.IsActive():
        _root.SetActive(True)
    _ops = UsdGeom.Xformable(_root).GetOrderedXformOps()
    if [op.GetOpType() for op in _ops] != [
            UsdGeom.XformOp.TypeTranslate,
            UsdGeom.XformOp.TypeOrient,
            UsdGeom.XformOp.TypeScale]:
        raise RuntimeError(
            'non-canonical pre-align Task29 root ops: '
            + _spec['body_path'])
    _tx, _ty, _tz = _spec['translate']
    _qw, _qx, _qy, _qz = _spec['orient_wxyz']
    _sx, _sy, _sz = _spec['scale']
    _ops[0].Set(Gf.Vec3d(_tx, _ty, _tz))
    _ops[1].Set(Gf.Quatf(_qw, _qx, _qy, _qz))
    _ops[2].Set(Gf.Vec3f(_sx, _sy, _sz))

    _lod_descriptor = _ensure_task29_bottom_lod(s, _root, _spec)
    _expected = dict(_spec['colliders'])
    for _path, _enabled in _expected.items():
        _collider = s.GetPrimAtPath(_path)
        if not _collider or not _collider.IsValid():
            raise RuntimeError('missing pre-align collider ' + _path)
        _chain = []
        _cur = _collider
        while _cur and _cur.IsValid() and _cur != _root:
            _chain.append(_cur)
            _cur = _cur.GetParent()
        if _cur != _root:
            raise RuntimeError('pre-align collider escapes body: ' + _path)
        for _prim in reversed(_chain):
            if not _prim.IsActive():
                _prim.SetActive(True)
        if not _collider.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(
                'pre-align prim lacks CollisionAPI: ' + _path)
        UsdPhysics.CollisionAPI(
            _collider).CreateCollisionEnabledAttr().Set(bool(_enabled))
    _actual = sorted(
        str(_prim.GetPath())
        for _prim in Usd.PrimRange.AllPrims(_root)
        if _prim.HasAPI(UsdPhysics.CollisionAPI))
    if _actual != sorted(_expected):
        raise RuntimeError(
            'pre-align Task29 collision structure changed: '
            + _spec['body_path'])

    _enabled_paths = sorted(
        _path for _path, _enabled in _expected.items() if _enabled)
    _expected_enabled_count = 3 if _lod_descriptor is not None else 2
    if len(_enabled_paths) != _expected_enabled_count:
        raise RuntimeError(
            'pre-align Task29 bin enabled collider count changed')
    _scoring_path = _spec['scoring_collider_path']
    if _scoring_path not in _enabled_paths:
        raise RuntimeError(
            'pre-align Task29 scoring collider is not enabled')
    _cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
        useExtentsHint=False)
    _range = _cache.ComputeWorldBound(
        s.GetPrimAtPath(_scoring_path)).ComputeAlignedRange()
    if _range.IsEmpty():
        raise RuntimeError('pre-align Task29 collider bound is empty')
    _mn, _mx = _range.GetMin(), _range.GetMax()
    _values = [float(_value)
               for _value in tuple(_mn) + tuple(_mx)]
    if not all(math.isfinite(_value) for _value in _values):
        raise RuntimeError('pre-align Task29 collider bound is non-finite')
    _center = [(_values[_axis] + _values[_axis + 3]) / 2.0
               for _axis in range(3)]
    _seed_x, _seed_y = _spec['calibrated_center_xy']
    if math.hypot(
            _center[0] - _seed_x,
            _center[1] - _seed_y) > _spec['center_tolerance_m']:
        raise RuntimeError(
            'pre-align Task29 collider centre misses RGB-D seed')
"""

    park = f"""import math
from pxr import Usd, UsdGeom, UsdPhysics, Gf
s = Usd.Stage.Open('{SCENE}')
{prealign_bin_reset}
rob = s.GetPrimAtPath('/World/Robot')
for op in UsdGeom.Xformable(rob).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d({BASE[0]}, {BASE[1]}, 0))
rob.GetAttribute('xformOp:orient').Set(Gf.Quatf({QUAT[0]}, {QUAT[1]}, {QUAT[2]}, {QUAT[3]}))
s.GetRootLayer().Save(); print('parked')
"""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="init_throw_park_",
        suffix=".py", delete=False,
    ) as stream:
        stream.write(park)
        park_script = stream.name
    try:
        from machine_paths import MESH311_PYTHON
        parked_result = subprocess.run([str(MESH311_PYTHON), park_script],
            capture_output=True, text=True, timeout=120, check=False)
        parked = parked_result.stdout
        if parked_result.returncode:
            print(parked_result.stderr, file=sys.stderr)
    finally:
        os.unlink(park_script)
    if "parked" not in parked:
        print(json.dumps({"ok": False, "reason": "park failed"}))
        return 3

    kin_override = os.environ.get("THROW_KIN_OVERRIDE")
    if kin_override:
        gx, gy, gz = map(float, kin_override.split(","))
        hand = os.environ.get("THROW_HAND_OVERRIDE", "L")
    else:
        sh(f"docker exec {CONTAINER_NAME} bash -lc \"pkill -9 -f 'runtag{_RT}' 2>/dev/null\"")
        sh(f"docker exec {CONTAINER_NAME} bash -lc 'rm -f /geniesim/main/demo_sim/cam_preview{INSTANCE}/align.txt'; sleep 2")
        # Relay episodes (right picks at the table, left throws from the
        # chest-height handoff) have two closed holds; the default
        # longest-hold rule lands on the wrong hand's hold, whose anchor is
        # un-stageable.  ALIGN_HAND forwards the caller's per-episode hand
        # choice to align_grasp, which validates the value itself.
        align_hand = os.environ.get("ALIGN_HAND", "").strip().lower()
        if align_hand not in ("", "l", "r"):
            print(json.dumps({
                "ok": False,
                "reason": "ALIGN_HAND must be exactly l or r when provided",
            }))
            return 2
        # Multi-object cleanup takes (three cups) hold several closed runs;
        # the default longest-run rule lands on a chest-height carry whose
        # anchor cannot be staged.  ALIGN_RUN_ORDINAL selects the n-th run
        # by start frame, so 0 is the first table pick.
        run_ordinal = os.environ.get("ALIGN_RUN_ORDINAL", "").strip()
        if run_ordinal and not re.fullmatch(r"[0-9]+", run_ordinal):
            print(json.dumps({
                "ok": False,
                "reason": "ALIGN_RUN_ORDINAL must be a non-negative integer",
            }))
            return 2
        # align_grasp walks the trajectory frame by frame, so the wall cost
        # scales with episode length.  The multi-object cleanup takes run
        # ~1000 frames (50+ s of recording) and overran the original 900 s
        # budget, which surfaced as a bogus "no grasp crossing" (the script
        # was killed before writing align.txt).
        align_environment = {
            "CUDA_VISIBLE_DEVICES": str(RENDER_GPU),
            "INSTANCE": INSTANCE,
        }
        if task29_phase_ordinal is not None:
            align_environment["TASK29_PHASE_ORDINAL"] = str(task29_phase_ordinal)
        if align_hand:
            align_environment["ALIGN_HAND"] = align_hand
        if run_ordinal:
            align_environment["ALIGN_RUN_ORDINAL"] = run_ordinal
        align_command = ["timeout", "2400", "docker", "exec"]
        for key, value in sorted(align_environment.items()):
            align_command.extend(["--env", f"{key}={value}"])
        align_command.extend([
            CONTAINER_NAME,
            *isaac_python_command("/geniesim/main/demo_sim/usdz/align_grasp.py"),
        ])
        subprocess.run(
            align_command, capture_output=True, text=True, timeout=2460)
        f = f"{CAM_PREVIEW}/align.txt"
        txt = open(f).read() if os.path.isfile(f) else ""
        m = re.search(r"midpoint world: \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)", txt)
        if not m or "NO grasp" in txt:
            print(json.dumps({"ok": False, "reason": "no grasp crossing"}))
            return 4
        gx, gy, gz = map(float, m.groups())
        hand = (re.search(r"hand (\w)", txt) or [None, "?"])[1]
    b = cfg["band"]
    if not (b["x"][0] < gx < b["x"][1] and b["y"][0] < gy < b["y"][1] and b["z"][0] < gz < b["z"][1]):
        print(json.dumps({"ok": False, "reason": f"kin anchor off band: ({gx:.2f},{gy:.2f},{gz:.2f})",
                          "hand": hand}))
        return 5

    if cfg["mode"] == "inhand":
        # object centre AT the fingertip midpoint; settle pins it between the closed pads.
        # t28 recording had the bin where TableclothBasket now sits (landing cluster
        # -2.29,-0.97): move TrashBinTeal to the RECORDED spot, basket steps aside
        # (secondary-object位置对齐录制 — the user's accuracy requirement).
        place = f"""from pxr import Usd, UsdGeom, Sdf, Gf
s = Usd.Stage.Open('{SCENE}')
def _sett(path, x, y):
    q = s.GetPrimAtPath(path)
    if q and q.IsValid():
        for op in UsdGeom.Xformable(q).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get(); op.Set(Gf.Vec3d(x, y, t[2])); return
_sett('/World/Assets/TableclothBasket', -2.24, -2.20)
_sett('/World/Assets/TrashBinTeal', -2.26, -1.06)
p = s.GetPrimAtPath('{tp}')
assert p and p.IsValid(), 'missing {tp}'
if not p.IsActive(): p.SetActive(True)
p.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(False)
for _nm in ('physics:velocity', 'physics:angularVelocity'):
    _a = p.GetAttribute(_nm)
    if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))
import math
c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
lo=[1e9]*3; hi=[-1e9]*3
for pr in Usd.PrimRange(p):
    if pr.GetTypeName()!='Mesh' or not pr.IsActive(): continue
    bb=c.ComputeWorldBound(pr).ComputeAlignedRange()
    if bb.IsEmpty(): continue
    mn,mx=bb.GetMin(),bb.GetMax()
    if not all(math.isfinite(v) for v in (*mn,*mx)): continue
    for i in range(3):
        lo[i]=min(lo[i],mn[i]); hi[i]=max(hi[i],mx[i])
cur=[(lo[i]+hi[i])/2 for i in range(3)]
dx, dy, dz = {gx}-cur[0], {gy}-cur[1], {gz}-cur[2]
for op in UsdGeom.Xformable(p).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        t = op.Get(); op.Set(Gf.Vec3d(t[0]+dx, t[1]+dy, t[2]+dz))
print('placed')
s.GetRootLayer().Save()
"""
    else:
        try:
            bite_z = float(os.environ.get("BITE_Z", "0.012"))
            max_riser_height = float(
                os.environ.get("THROW_MAX_RISER", "0.20"))
        except ValueError:
            print(json.dumps({
                "ok": False,
                "reason": "BITE_Z and THROW_MAX_RISER must be numeric",
            }))
            return 2
        if not math.isfinite(bite_z) or not math.isfinite(max_riser_height):
            print(json.dumps({
                "ok": False,
                "reason": "BITE_Z and THROW_MAX_RISER must be finite",
            }))
            return 2
        table_plate_context = cfg.get("table_plate_context")
        # The paper rests IN the floral plate in every recording, but the
        # plate is pinned to the reviewed episode's pick point while the
        # object is staged at THIS episode's recorded closure — with the
        # station fixed those diverge by up to 0.4 m, leaving the paper on
        # the bare table and the plate stranded.  THROW_PLATE_XY moves the
        # plate (a task prop, not furniture) under the episode's own grasp
        # so the staged scene matches the footage.
        plate_xy_override = os.environ.get("THROW_PLATE_XY", "").strip()
        try:
            table_plate_context, table_plate_binding = (
                resolve_table_plate_context(
                    tid,
                    os.environ.get("EPISODE_UID", ""),
                    table_plate_context,
                    plate_xy_override,
                )
            )
        except ValueError as exc:
            print(json.dumps({
                "ok": False,
                "reason": str(exc),
            }))
            return 2
        if table_plate_context is None:
            reset_table_plate_context = "_table_plate_context = None"
        else:
            reset_table_plate_context = f"""
_table_plate_spec = {table_plate_context!r}
_table = s.GetPrimAtPath('/World/Assets/CoffeeTable')
if not _table or not _table.IsValid():
    raise RuntimeError('throw scene is missing CoffeeTable')
_table.SetActive(True)
_table_translate_ops = [
    op for op in UsdGeom.Xformable(_table).GetOrderedXformOps()
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
]
if len(_table_translate_ops) != 1:
    raise RuntimeError('CoffeeTable must have one translate op')
_table_translate_ops[0].Set(Gf.Vec3d(
    *_table_plate_spec['coffee_table_translate_xyz']))
_plate = s.GetPrimAtPath(_table_plate_spec['paper_plate_path'])
if not _plate or not _plate.IsValid():
    raise RuntimeError('throw scene is missing PaperPlate')
_plate.SetActive(True)
_plate_ops = UsdGeom.Xformable(_plate).GetOrderedXformOps()
if [op.GetOpType() for op in _plate_ops] != [
    UsdGeom.XformOp.TypeTranslate,
    UsdGeom.XformOp.TypeRotateXYZ,
    UsdGeom.XformOp.TypeScale,
]:
    raise RuntimeError('PaperPlate transform stack changed')
_plate_ops[0].Set(Gf.Vec3d(
    *_table_plate_spec['paper_plate_translate_xyz']))
_plate_ops[1].Set(Gf.Vec3f(0.0, 0.0, 0.0))
_plate_ops[2].Set(Gf.Vec3f(1.0, 1.0, 1.0))
if not _plate.HasAPI(UsdPhysics.RigidBodyAPI):
    raise RuntimeError('PaperPlate distractor/support rigid body is missing')
_plate_rigid = UsdPhysics.RigidBodyAPI(_plate)
_plate_rigid.CreateRigidBodyEnabledAttr().Set(True)
_plate_rigid.CreateKinematicEnabledAttr().Set(True)
_plate_rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
_plate_rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
_plate_colliders = [
    prim for prim in Usd.PrimRange.AllPrims(_plate)
    if prim.HasAPI(UsdPhysics.CollisionAPI)
]
if len(_plate_colliders) != 1:
    raise RuntimeError(
        'PaperPlate must have exactly one authored collision shape')
UsdPhysics.CollisionAPI(
    _plate_colliders[0]).CreateCollisionEnabledAttr().Set(True)
_plate_range = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
    useExtentsHint=False,
).ComputeWorldBound(_plate).ComputeAlignedRange()
if _plate_range.IsEmpty():
    raise RuntimeError('PaperPlate bound is empty')
_plate_min = [float(value) for value in _plate_range.GetMin()]
_plate_max = [float(value) for value in _plate_range.GetMax()]
_plate_size = [
    _plate_max[axis] - _plate_min[axis] for axis in range(3)]
if (
    any(
        abs(actual - expected) > 0.001
        for actual, expected in zip(
            _plate_size,
            _table_plate_spec['paper_plate_expected_size_xyz_m'],
        )
    )
    or abs(
        _plate_min[2] - _table_plate_spec['paper_plate_bottom_m']
    ) > 0.001
    or abs(
        _plate_max[2] - _table_plate_spec['paper_plate_top_m']
    ) > 0.001
):
    raise RuntimeError('PaperPlate support pose/size changed')
_table_plate_context = dict(
    coffee_table_translate_xyz=[
        float(value) for value in _table_translate_ops[0].Get()],
    paper_plate_path=str(_plate.GetPath()),
    paper_plate_translate_xyz=[
        float(value) for value in _plate_ops[0].Get()],
    paper_plate_aabb_min_xyz=_plate_min,
    paper_plate_aabb_max_xyz=_plate_max,
    paper_plate_rigid_body_enabled=True,
    paper_plate_kinematic=True,
    paper_plate_collision_enabled=True,
)
"""
        bin_resets = []
        bx_, by_ = cfg.get("bin_xy", (None, None))
        if bx_ is not None:
            bin_resets.append(
                f"_sett('/World/Assets/TrashBinTeal', {bx_}, {by_})")
        rbx_, rby_ = cfg.get("distractor_bin_xy", (None, None))
        if rbx_ is not None:
            bin_resets.append(
                f"_sett('/World/Assets/TrashBinRed', {rbx_}, {rby_})")
        bin_reset_calls = "\n".join(bin_resets)
        move_bin = "" if not bin_resets else f"""
def _sett(path, x, y):
    q = s.GetPrimAtPath(path)
    if q and q.IsValid():
        for op in UsdGeom.Xformable(q).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get(); op.Set(Gf.Vec3d(x, y, t[2])); return
{bin_reset_calls}
"""
        bin_layout = cfg.get("bin_layout")
        if bin_layout is None:
            reset_task_bins = "_task_bins = None"
        else:
            try:
                validate_task29_bin_layout(bin_layout)
            except (TypeError, ValueError) as exc:
                print(json.dumps({
                    "ok": False,
                    "reason": f"invalid Task29 bin layout: {exc}",
                }))
                return 2
            reset_task_bins = f"""
{_TASK29_BOTTOM_LOD_RUNTIME}
_task29_bin_layout = {bin_layout!r}

def _reset_task29_bin(slot, spec):
    root = s.GetPrimAtPath(spec['body_path'])
    if not root or not root.IsValid():
        raise RuntimeError(
            'missing Task29 ' + slot + ' bin body ' + spec['body_path'])
    if not root.IsActive():
        root.SetActive(True)

    # The calibration is for this exact root transform stack.  Reject added or
    # reordered ops rather than composing a seed onto stale transforms.
    ops = UsdGeom.Xformable(root).GetOrderedXformOps()
    op_types = [op.GetOpType() for op in ops]
    expected_types = [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeOrient,
        UsdGeom.XformOp.TypeScale,
    ]
    if op_types != expected_types:
        raise RuntimeError(
            'Task29 ' + slot
            + ' bin requires translate/orient/scale root ops, found '
            + repr([op.GetOpName() for op in ops]))
    tx, ty, tz = spec['translate']
    qw, qx, qy, qz = spec['orient_wxyz']
    sx, sy, sz = spec['scale']
    ops[0].Set(Gf.Vec3d(tx, ty, tz))
    ops[1].Set(Gf.Quatf(qw, qx, qy, qz))
    ops[2].Set(Gf.Vec3f(sx, sy, sz))
    # Bind the emitted contract to the values actually composed by USD.
    # Vec3f/Quatf quantization (notably scale 1.7) is observable physics state
    # and must match the independent runtime readback byte-for-byte.
    actual_translate = ops[0].Get()
    actual_orient = ops[1].Get()
    actual_scale = ops[2].Get()
    if (
            actual_translate is None
            or actual_orient is None
            or actual_scale is None):
        raise RuntimeError(
            'Task29 ' + slot + ' root transform readback failed')

    bottom_collision_lod = _ensure_task29_bottom_lod(s, root, spec)
    expected_colliders = dict(spec['colliders'])
    for collider_path, enabled in expected_colliders.items():
        collider = s.GetPrimAtPath(collider_path)
        if not collider or not collider.IsValid():
            raise RuntimeError(
                'missing Task29 ' + slot + ' collider ' + collider_path)
        # Reset active on the complete path, not only the leaf, because a stale
        # inactive ancestor otherwise masks the authored collider.
        chain = []
        cur = collider
        while cur and cur.IsValid() and cur != root:
            chain.append(cur)
            cur = cur.GetParent()
        if cur != root:
            raise RuntimeError(
                'Task29 collider escapes body: ' + collider_path)
        for prim in reversed(chain):
            if not prim.IsActive():
                prim.SetActive(True)
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(
                'Task29 expected prim lacks CollisionAPI: ' + collider_path)
        UsdPhysics.CollisionAPI(
            collider).CreateCollisionEnabledAttr().Set(bool(enabled))

    # Include inactive prims in the enumeration.  Any additional CollisionAPI
    # is an uncalibrated change in destination geometry and must fail closed.
    actual_colliders = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange.AllPrims(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if actual_colliders != sorted(expected_colliders):
        raise RuntimeError(
            'Task29 ' + slot + ' collision structure changed: expected '
            + repr(sorted(expected_colliders)) + ', found '
            + repr(actual_colliders))

    enabled_paths = sorted(
        path for path, enabled in expected_colliders.items() if enabled)
    expected_enabled_count = (
        3 if bottom_collision_lod is not None else 2)
    if len(enabled_paths) != expected_enabled_count:
        raise RuntimeError(
            'Task29 ' + slot + ' enabled collider count changed')
    collider_path = spec['scoring_collider_path']
    if slot == 'destination' and collider_path not in enabled_paths:
        raise RuntimeError(
            'Task29 ' + slot + ' scoring collider is not enabled')
    collider = s.GetPrimAtPath(collider_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
        useExtentsHint=False)
    world_range = cache.ComputeWorldBound(collider).ComputeAlignedRange()
    if world_range.IsEmpty():
        raise RuntimeError(
            'Task29 enabled collider has an empty bound: ' + collider_path)
    mn, mx = world_range.GetMin(), world_range.GetMax()
    values = [float(value) for value in tuple(mn) + tuple(mx)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            'Task29 enabled collider has a non-finite bound: '
            + collider_path)
    bound_min = values[:3]
    bound_max = values[3:]
    size = [
        bound_max[axis] - bound_min[axis] for axis in range(3)]
    if not all(value > 1e-6 for value in size):
        raise RuntimeError(
            'Task29 enabled collider has a degenerate bound: '
            + repr(size))
    center = [
        (bound_min[axis] + bound_max[axis]) / 2.0
        for axis in range(3)]
    seed_x, seed_y = spec['calibrated_center_xy']
    center_error_m = math.hypot(center[0] - seed_x, center[1] - seed_y)
    if center_error_m > spec['center_tolerance_m']:
        raise RuntimeError(
            'Task29 ' + slot + ' composed collider centre misses RGB-D seed '
            + 'by ' + format(center_error_m, '.6f') + ' m')

    return dict(
        slot=slot,
        name=spec['name'],
        role=spec['role'],
        body_path=spec['body_path'],
        body_active=True,
        root_transform=dict(
            translate=[round(float(value), 9)
                       for value in actual_translate],
            orient_wxyz=[
                round(float(actual_orient.GetReal()), 9),
                *[round(float(value), 9)
                  for value in actual_orient.GetImaginary()],
            ],
            scale=[round(float(value), 9) for value in actual_scale],
        ),
        collider_path=collider_path,
        collider_states=[
            dict(path=path, active=True, collision_enabled=enabled)
            for path, enabled in sorted(expected_colliders.items())
        ],
        calibrated_center_xy=[
            round(float(seed_x), 6), round(float(seed_y), 6)],
        center_error_m=round(center_error_m, 9),
        bounds=dict(
            minimum=[round(value, 6) for value in bound_min],
            maximum=[round(value, 6) for value in bound_max],
            center=[round(value, 6) for value in center],
            size=[round(value, 6) for value in size],
            bottom_z=round(bound_min[2], 6),
            rim_z=round(bound_max[2], 6),
        ),
        bottom_collision_lod=bottom_collision_lod,
        geometry_descriptor=dict(
            schema='task29.composed_bin_geometry.v2',
            bounds_kind='visible_destination_collider_world_aabb',
            calibration_basis='raw_rgbd_frame0_backprojection_seed',
            rim_definition='collision_aabb_max_z',
            interior_geometry_asserted=False,
            scoring_polygon_asserted=False,
        ),
    )

_task_bins = dict(
    destination=_reset_task29_bin(
        'destination', _task29_bin_layout['destination']),
    distractor=_reset_task29_bin(
        'distractor', _task29_bin_layout['distractor']),
)
"""
        place = f"""import json, math, sys
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from init_throw import solve_pick_placement, validate_paper_cup_upright_size
s = Usd.Stage.Open('{SCENE}')
{reset_table_plate_context}
{move_bin}
{reset_task_bins}
p = s.GetPrimAtPath('{tp}')
assert p and p.IsValid(), 'missing {tp}'
if not p.IsActive(): p.SetActive(True)
p.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(False)
for _nm in ('physics:velocity', 'physics:angularVelocity'):
    _a = p.GetAttribute(_nm)
    if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))

# Small Task 30 uses the exact Genie paper geometry at the current reviewed
# task scale.  Author it before bounds/placement so a stale template cannot
# silently restore the rejected oversized sheet.
root_scale_xyz = {cfg.get('root_scale_xyz')!r}
if root_scale_xyz is not None:
    scale_ops = [
        op for op in UsdGeom.Xformable(p).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(scale_ops) != 1:
        raise RuntimeError(
            'PaperBall requires exactly one ordered scale op, found '
            + str(len(scale_ops)))
    scale_ops[0].Set(Gf.Vec3f(*root_scale_xyz))
    if tuple(float(value) for value in scale_ops[0].Get()) != tuple(
            float(value) for value in root_scale_xyz):
        raise RuntimeError('PaperBall current scale failed readback')
    paper_collider = s.GetPrimAtPath(
        '/World/Assets/PaperBall/Geom/body/visual')
    if (
        not paper_collider
        or not paper_collider.IsValid()
        or not paper_collider.HasAPI(UsdPhysics.CollisionAPI)
    ):
        raise RuntimeError(
            'PaperBall current collision material target is invalid')
    # Remove only the scene-layer override.  The referenced Genie asset has its
    # own physics material, which must remain composed for reliable grasping.
    paper_collider.RemoveProperty('material:binding:physics')
    paper_binding = paper_collider.GetRelationship(
        'material:binding:physics')
    native_material_path = (
        '/World/Assets/PaperBall/Geom/PhysicsMaterial')
    paper_binding_targets = [
        str(path) for path in paper_binding.GetTargets()
    ] if paper_binding else []
    if paper_binding_targets != [native_material_path]:
        raise RuntimeError(
            'PaperBall native physics material binding was not restored: '
            + repr(paper_binding_targets))
    paper_material = s.GetPrimAtPath(native_material_path)
    if (
        not paper_material
        or not paper_material.IsValid()
        or not paper_material.HasAPI(UsdPhysics.MaterialAPI)
    ):
        raise RuntimeError(
            'PaperBall native physics material target is invalid')

# Task29 is an upright-cup task.  Reset the full orientation before computing
# collision bounds so a stale shared-scene pose cannot silently change the
# effective pinch width or the placement height.
orientation_mode = {cfg.get('orientation_mode')!r}
rotate_xyz = {cfg.get('rotate_xyz')!r}
if orientation_mode is not None:
    if orientation_mode != 'upright_local_z' or rotate_xyz is None:
        raise RuntimeError(
            'unsupported canonical throw orientation: '
            + repr((orientation_mode, rotate_xyz)))
    rotate_ops = [
        op for op in UsdGeom.Xformable(p).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
    ]
    if len(rotate_ops) != 1:
        raise RuntimeError(
            'PaperCup requires exactly one ordered rotateXYZ op, found '
            + str(len(rotate_ops)))
    rotate_ops[0].Set(Gf.Vec3f(*rotate_xyz))

# Prefer active physical collision geometry.  If an asset has no authored
# collision API, fall back to its active render meshes; never guess a radius.
_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
    useExtentsHint=False)
def _collision_enabled(pr):
    cur = pr
    found = False
    while cur and cur.IsValid():
        if cur.HasAPI(UsdPhysics.CollisionAPI):
            found = True
            attr = UsdPhysics.CollisionAPI(cur).GetCollisionEnabledAttr()
            if attr and attr.Get() is False:
                return False
        if cur == p:
            break
        cur = cur.GetParent()
    return found

def _active_bounds(collision_only):
    lo = [float('inf')] * 3
    hi = [float('-inf')] * 3
    count = 0
    for pr in Usd.PrimRange(p):
        if not pr.IsActive():
            continue
        boundable = UsdGeom.Boundable(pr)
        if not boundable:
            continue
        if collision_only:
            if not _collision_enabled(pr):
                continue
        elif pr.GetTypeName() != 'Mesh':
            continue
        bb = _cache.ComputeWorldBound(pr).ComputeAlignedRange()
        if bb.IsEmpty():
            continue
        mn, mx = bb.GetMin(), bb.GetMax()
        vals = tuple(mn) + tuple(mx)
        if not all(math.isfinite(float(value)) for value in vals):
            continue
        for axis in range(3):
            lo[axis] = min(lo[axis], float(mn[axis]))
            hi[axis] = max(hi[axis], float(mx[axis]))
        count += 1
    return (lo, hi, count)

lo, hi, bound_count = _active_bounds(True)
bound_source = 'collision'
if not bound_count:
    lo, hi, bound_count = _active_bounds(False)
    bound_source = 'mesh'
if not bound_count:
    raise RuntimeError('PaperCup/PaperBall has no active collision or mesh bound')

if orientation_mode == 'upright_local_z':
    validate_paper_cup_upright_size(
        tuple(hi[axis] - lo[axis] for axis in range(3)))

solution = solve_pick_placement(
    ({gx}, {gy}, {gz}), lo, hi, bite_z={bite_z!r},
    surface_z={cfg.get('surface_z', COFFEE_TABLE_CURRENT_TOP_M)!r},
    max_riser_height={max_riser_height!r})
dx, dy, dz = solution['translation_delta']
rs = s.GetPrimAtPath('/World/Assets/SeatRiser')
if not solution['use_riser']:
    if rs and rs.IsValid() and rs.IsActive(): rs.SetActive(False)
else:
    riser_h = solution['riser_height']
    rbase = solution['riser_base']
    if not rs.IsActive(): rs.SetActive(True)
    for op in UsdGeom.Xformable(rs).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d({gx}, {gy} + 0.055, rbase + riser_h/2))
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            sc = op.Get(); op.Set(Gf.Vec3f(sc[0], sc[1], riser_h))
for op in UsdGeom.Xformable(p).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        old = op.Get()
        op.Set(Gf.Vec3d(old[0] + dx, old[1] + dy, old[2] + dz))
print('placed ' + json.dumps({{
    'bound_source': bound_source,
    'bound_count': bound_count,
    'size': [round(value, 6) for value in solution['size']],
    'bottom': round(solution['bottom'], 6),
    'use_riser': solution['use_riser'],
    'orientation_mode': orientation_mode,
    'rotate_xyz': list(rotate_xyz) if rotate_xyz is not None else None,
    'root_scale_xyz': (
        list(root_scale_xyz) if root_scale_xyz is not None else None),
    'table_plate_context': _table_plate_context,
    'task_bins': _task_bins,
}}))
s.GetRootLayer().Save()
"""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="init_throw_place_",
        suffix=".py", delete=False,
    ) as stream:
        stream.write(place)
        place_script = stream.name
    try:
        from machine_paths import MESH311_PYTHON
        r2 = subprocess.run([str(MESH311_PYTHON), place_script],
            capture_output=True, text=True, timeout=120, check=False)
    finally:
        os.unlink(place_script)
    ok = "placed" in r2.stdout
    out = {"ok": ok, "target": cfg["prim"], "hand": hand, "mode": cfg["mode"],
           "kin": [round(gx, 3), round(gy, 3), round(gz, 3)], "base": list(BASE),
           "required_count": cfg.get("required_count", 1),
           "probe_scope": cfg.get("probe_scope", "full_task")}
    if table_plate_binding is not None:
        out["table_plate_binding"] = table_plate_binding
    if task29_phase_ordinal is not None:
        out["task29_phase_ordinal"] = task29_phase_ordinal
        out["task29_phase_cup_id"] = (
            f"cup{task29_phase_ordinal}")
        out["task29_target_instance_path"] = tp
    placement_line = next(
        (line.split("placed ", 1)[1] for line in r2.stdout.splitlines()
         if line.startswith("placed ")),
        None,
    )
    if placement_line:
        try:
            out["placement"] = json.loads(placement_line)
        except json.JSONDecodeError:
            pass
    if not ok:
        out["reason"] = "place failed: " + (r2.stdout + r2.stderr)[-120:]
    print(json.dumps(out))
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())

"""Generic per-episode init for the new pick tasks (27-60 batch): park base, kinematic FK
anchor, then place the target prim so its bbox bottom rests on the recorded support with the
grasp point intersecting the object body. Task-required direct-contact context is restored from
an external calibration manifest; it must not be replaced by an invisible riser. Honest physics:
the target is dynamic from t=0, velocities zeroed, no freeze.

    PICK_TASK=31 conda run -n base python real2sim/teleop/init_pick.py
Env: PICK_TASK (required), PICK_BITE (m, shift along approach axis into the gripper, default 0),
     INSTANCE, FIXED_BASE override "x,y".

Wallet-only, explicit geometry canary controls:
    PICK_LOCAL_EDGE_OFFSET="x,y"
        Put that target-local XY point, rather than the Wallet origin, at the
        recorded pad midpoint.  The default is unset and remains centered.
    PICK_LOCAL_PINCH_AXIS="x,y,z"
        Request that the measured pad-separation axis align with this
        target-local direction.  Only target yaw is available here: the
        initializer verifies the live 3-D alignment and fails closed when yaw
        cannot satisfy it (notably a local-Z pinch with the current replay).

Task59/GlassesCase-only, explicit real-pose canary:
    PICK_GLASSES_CASE_SIDE_REST="1"
        Stand the case on its long local-Y side, as all three pinned Task59
        recordings show, then align its short local-Z thickness with the
        horizontal pad-separation axis.  Unset remains the historical flat pose.
    PICK_GLASSES_CASE_LONG_AXIS_OFFSET="0.015"
        With side-rest enabled, move the case centre exactly 15 mm along its
        composed local +X long axis.  This is the one bounded rescue for a
        measured long-axis slide; arbitrary values are rejected.

Task53/Wallet-only, explicit real-scene canary:
    PICK_TASK53_REAL_SCENE="1"
        Use the video-reviewed blue-wallet-on-visible-cushion scene.  The
        Wallet XY footprint is scaled from 165x130 mm to 140.25x110.5 mm
        while retaining its 40 mm thickness, and CushionOnBook replaces the
        invisible SeatRiser as the direct kinematic contact support.

These controls never rotate the arm or claim to have changed its wrist pose.
Emits one JSON line: {ok, target, hand, kin:[x,y,z], base:[x,y]}.
"""
import json, math, os, re, subprocess, sys

import group_shift
from pick_provenance import (
    TASK53_CONTACT_SCENE_CALIBRATION_FILENAME,
    task53_contact_scene_calibration_identity,
    task53_episode_calibration,
    task59_support_calibration_identity,
)
from robot_positions import (
    COFFEE_TABLE_CURRENT_BOTTOM_M,
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_CURRENT_SIZE_XYZ_M,
    COFFEE_TABLE_CURRENT_TOP_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    SOFA_CURRENT_SEAT_TOP_M,
    SOFA_INTERACTION_FACE_SHIFT_XY_M,
    TV_CABINET_CURRENT_BOTTOM_M,
    TV_CABINET_CURRENT_CENTER_XY_M,
    TV_CABINET_CURRENT_SIZE_XYZ_M,
    TV_CABINET_CURRENT_TOP_M,
    robot_position,
    translate_interval,
    translate_xy,
)

from machine_paths import CONTAINER_NAME, RENDER_GPU
from machine_paths import DEMO_ROOT as _DEMO_ROOT
DEMO = str(_DEMO_ROOT)
INSTANCE = os.environ.get("INSTANCE", "")
if INSTANCE and not re.fullmatch(r"_[A-Za-z0-9][A-Za-z0-9_-]{0,62}", INSTANCE):
    raise ValueError(
        "INSTANCE must be empty or a safe underscore-prefixed suffix")
_RT = INSTANCE or "_MAIN"
_RT_PATTERN = rf"(^| ){re.escape(f'runtag{_RT}')}( |$)"
SCENE = f"{DEMO}/scene_room{INSTANCE}.usda"
CAM_PREVIEW = f"{DEMO}/cam_preview{INSTANCE}"

# family: base home (x,y), orient quat (w,x,y,z) facing the furniture, surface top z,
# anchor sanity band, approach axis unit vector (bite direction = INTO the gripper = away
# from the robot). Non-sofa bases start as geometric guesses -> calibrated via probe.
FAMILIES = {
    "sofa": dict(base=robot_position("sofa_pick").xy_m,
                 quat=robot_position("sofa_pick").quat_wxyz,
                 surface_z=SOFA_CURRENT_SEAT_TOP_M,
                 band=dict(
                     x=translate_interval(
                         (-1.10, -0.45),
                         SOFA_INTERACTION_FACE_SHIFT_XY_M[0],
                     ),
                     y=(-2.30, 0.15),
                     z=(0.40, 0.85),
                 ),
                 approach=(1, 0)),
    # Shared coffee-table parking pose: in the Sofa--CoffeeTable corridor,
    # facing west across the table and close to the twin bins.  Keep this
    # family-level so every coffee-table pick uses the same room semantics.
    "coffee_table": dict(
        base=robot_position("table_pick").xy_m,
        quat=robot_position("table_pick").quat_wxyz,
        surface_z=COFFEE_TABLE_CURRENT_TOP_M,
                 band=dict(
                     x=translate_interval(
                         (-2.50, -1.70),
                         COFFEE_TABLE_LAYOUT_SHIFT_XY_M[0],
                     ),
                     y=(-0.95, 0.15),
                     z=(0.40, 0.75),
                 ),
                 approach=(-1, 0)),
    # 2026-08-03: base/quat were bare literals here, so the TV cabinet was the
    # one piece of furniture whose station lived outside the station table.
    "tv_table": dict(
                 base=robot_position("tv_cabinet_pick").xy_m,
                 quat=robot_position("tv_cabinet_pick").quat_wxyz,
                 surface_z=TV_CABINET_CURRENT_TOP_M,
                 band=dict(x=(-3.95, -3.45), y=(-1.05, 0.45), z=(0.40, 0.70)),
                 approach=(-1, 0)),
    # 2026-08-03: the station moved into the CoffeeTable/TvTable corridor, so
    # the family approaches the table diagonally from the north-west instead of
    # straight from the north. The axis follows the station's -45 deg yaw; it
    # was task 90's private value and is now what every side-table task uses.
    "small_table": dict(base=robot_position("sidetable_pick").xy_m,
                 quat=robot_position("sidetable_pick").quat_wxyz,
                 # 2026-08-03: the side table is 18 inch (0.4572 m) off the
                 # floor, fixed once in the baseline. This used to read
                 # 0.409 while task 59's contract claimed 0.5201.
                 surface_z=18 * 0.0254,
                 band=dict(x=(-2.65, -1.55), y=(-2.70, -1.30), z=(0.38, 0.70)),
                 approach=(0.7071068, -0.7071068)),
}

# The plausibility bands wrap the task set, not the furniture; they move with
# the group shift exactly like the stations already baked into each family.
if not group_shift.is_identity():
    for _fam in FAMILIES.values():
        _band = _fam["band"]
        _band["x"] = group_shift.apply_interval(_band["x"], "x")
        _band["y"] = group_shift.apply_interval(_band["y"], "y")

# task -> target prim under /World/Assets, sam3 prompt (未来视觉用), family.
# ★ OBJECT IDENTITY corrected 2026-07-22 from the ROBOT's authoritative dlb.db templates
# (dataset/robot_task_names.json). My earlier RGB guesses had glasses<->wallet SWAPPED:
# 53/55/60 are WALLET, 57/59 are GLASSES.  Wallet now has a dedicated asset;
# pre-correction GlassesCase metrics remain wrong-target evidence.
TASKS = {
    "31": dict(
        prim="Remote", prompt="black TV remote control", family="sofa",
        # The rectangular Remote has antipodal equal-width pinch solutions.
        # Bind their tie-break to the checked-in scene template's authored yaw
        # instead of the previous attempt's mutable scene opinion.
        canonical_yaw_seed=99.234764,
        collision_approximation="sdf",
        # Resolve the two equal-width antipodal solutions against the reviewed
        # preferred real episode instead of floating-point extent noise.
        initial_yaw_reference_deg=65.5,
    ),  # sofa pick controller
    "32": dict(
        prim="BaseballCap",
        prompt="black baseball cap/hat",
        family="sofa",
        # The cap's wide brim has antipodal equal-width solutions, but the
        # open-pad approach is not invariant under that 180-degree choice.
        # Bind the current branch that reaches bilateral target contact
        # without high pre-grasp pad friction.
        canonical_yaw_seed=145.0,
    ),  # sofa pick hat
    # 2026-08-03: registered alongside task_registry so the uniform entry
    # point can dispatch them. They inherit the shared sofa station like
    # every other sofa pick; task_objects.json carried a private base for
    # 12/13 that sat 72 mm off it while 11 was already on the shared one.
    "11": dict(prim="Note", prompt="paper note", family="sofa"),
    "12": dict(prim="Book", prompt="hardcover book", family="sofa"),
    "13": dict(prim="Book", prompt="hardcover book", family="sofa"),
    "33": dict(
        prim="PaperCup", prompt="striped paper cup", family="sofa",
        # Task 6.11 uses an upright empty cup on the visible Sofa seat.
        # Its scene pose is independent of an old elevated grasp reference.
        # Producers must supply newly generated native-seat grasp commands.
        native_paper_cup=True,
        native_initial_xy_m=(-0.0426193167064624, -1.1665421842942267),
        canonical_yaw_seed=0.0,
    ),
    "53": dict(prim="Wallet",      prompt="herringbone wallet",       family="sofa"),  # sofa pick wallet
    "54": dict(
        prim="Umbrella", prompt="folded umbrella", family="sofa",
        # All checked-in authored scene variants (v1/v1_1/v1_2) agree on
        # 90 degrees.  Keep the antipodal pinch tie-break independent of a
        # previous attempt's mutable root transform.
        canonical_yaw_seed=90.0,
    ),  # sofa pick umbrella
    "55": dict(
        prim="Wallet",
        prompt="herringbone wallet",
        family="coffee_table",
        # The flat Wallet has equal-width antipodal pinches at about 10 and
        # -170 degrees, but the reviewed soft-pouch collision envelope is
        # intentionally asymmetric.  Choose the -170-degree representative
        # independently of the previous run's mutable scene transform.
        canonical_yaw_seed=180.0,
    ),  # find wallet
    "56": dict(
        prim="Remote", prompt="black TV remote control",
        family="coffee_table", canonical_yaw_seed=99.234764,
        # Use the shared Sofa--CoffeeTable corridor parking pose.  The prior
        # task-specific base made coffee-table tasks disagree about which side
        # of the table the robot occupies.
        contact_supports=("Note", "Wallet", "Book"),
        collision_approximation="convexHull",
        # Head-frame-0 dark-body PCA: real 5.43 deg, sim 6.25 deg from
        # image vertical.  The bounded -0.8 deg correction preserves the
        # narrow-axis grasp while matching the reviewed real initial pose.
        initial_yaw_offset_deg=-0.8,
        initial_yaw_reference_deg=-89.8,
    ),  # find remote on the long coffee table
    "57": dict(
        prim="GlassesCase", prompt="glasses case",
        family="coffee_table", canonical_yaw_seed=0.0,
    ),  # find glasses
    "58": dict(
        prim="Remote", prompt="black TV remote control",
        family="tv_table", canonical_yaw_seed=99.234764,
        collision_approximation="sdf",
        initial_yaw_reference_deg=169.0,
    ),  # find remote by the TV cabinet
    "59": dict(
        prim="GlassesCase", prompt="glasses case",
        family="small_table", canonical_yaw_seed=0.0,
        # The three identity-bound strict probes use the same undirected
        # long-axis grasp at -39, -28.5, and -23 degrees.  Resolve the
        # numerically equal +180-degree representative against their audited
        # robot-facing branch instead of a previously mutated scene yaw.
        initial_yaw_reference_deg=-30.0,
    ),  # find glasses on the small dark table
    "60": dict(prim="Wallet",      prompt="herringbone wallet",       family="small_table"),  # find wallet table
    "61": dict(
        prim="Remote", prompt="black TV remote control",
        family="coffee_table", collision_approximation="sdf",
    ),  # find remote table
    # L2 small Task 109: representative three-camera HDF5 review shows a
    # television remote lifted from the same small wooden side-table support
    # used by the canonical small-table station.  Keep the robot station
    # shared; do not move the table for this task.
    "109": dict(
        prim="Remote",
        prompt="black TV remote control",
        family="small_table",
        canonical_yaw_seed=99.234764,
        collision_approximation="sdf",
    ),
    # Delivered L2 producer alias for Sheet 4.3. Shares the catalog's
    # existing side-table station and changes only the selected target.
    "405": dict(
        prim="GlassesCase", prompt="glasses case",
        family="small_table", canonical_yaw_seed=0.0,
    ),
    "78": dict(
        prim="RedBullPink", prompt="pink slim energy drink can",
        family="coffee_table",
        # A standing cylinder is yaw-invariant for the narrow-span pinch;
        # no canonical seed or antipodal tie-break is needed.
    ),  # empty pink slim can on the coffee table (2026-08-04 batch)
    "90": dict(
        prim="Umbrella", prompt="folded umbrella", family="small_table",
        # Same wooden side table as Task 59 (fireplace).  Keep the Task 54
        # antipodal pinch tie-break; per-episode yaw comes from the align
        # anchor, and the seed only resolves the 180-degree ambiguity.
        canonical_yaw_seed=90.0,
        # 2026-07-31 batch: the robot faces the table from the NW with the
        # fireplace directly behind it (robot--table--fireplace collinear in
        # the real frame 0). This was a private base/quat/approach that
        # disagreed with sidetable_pick by 862 mm; 2026-08-03 the shared
        # station adopted THIS pose rather than the other way round, because
        # the retired one stood beside the CoffeeTable rather than in the
        # corridor and had never produced a passing grasp. The task now
        # inherits the family like every other side-table task.
    ),  # umbrella on the wooden side table (2026-07-31 batch)
    "91": dict(
        prim="GlassesCase", prompt="glasses case",
        family="sofa", canonical_yaw_seed=0.0,
    ),  # find glasses case on the sofa seat (2026-07-31 batch, benchmark 4.1)
    "92": dict(
        prim="Wallet", prompt="herringbone wallet",
        family="coffee_table",
        # 2026-08-03: this pose is now the named coffee_table_drawer_wallet
        # station -- a different drawer from task 28's, hence its own entry.
        # Find the wallet INSIDE the coffee table's black drawer: the right
        # hand pinches the slightly-open drawer front and pulls it out on the
        # shared articulated-drawer physics (drawer_articulation.py), then
        # the left hand lifts the wallet out.  Frame-0 waist pitch is 45 deg
        # (vs 30 deg for the oblique Task 28 bill-in station); the base is the
        # shared front-facing table_pick station.
        drawer_pick=True,
        canonical_yaw_seed=0.0,
        # Station solved from the REAL head frame (station_solve.py): the
        # table's near corner and its two visible edges pin the base to
        # 0.1 px.  A 65 mm shift toward the drawer's left wall was tried and
        # REVERTED: it was inferred from one episode whose grasp anchor sits
        # mid-drawer, but across the batch the anchors already cluster at
        # that wall (y -0.571..-0.631 against a wall at -0.617), and the
        # shift pushed most of them clean outside the box.
        # Task-specific, because the shared coffee_table family station
        # belongs to small Tasks 55/56/57 and must not move.
        base=robot_position("coffee_table_drawer_wallet").xy_m,
        quat=robot_position("coffee_table_drawer_wallet").quat_wxyz,   # yaw 179.763 deg
        # The wallet grasp anchor sits inside the pulled-out drawer: below
        # the tabletop band and possibly east of the table edge.
        band=dict(x=(-1.95, -1.15), y=(-0.80, -0.20), z=(0.18, 0.60)),
    ),  # find wallet in the coffee-table drawer (2026-07-31 batch)
}

PICK_LOCAL_EDGE_OFFSET = "PICK_LOCAL_EDGE_OFFSET"
PICK_LOCAL_PINCH_AXIS = "PICK_LOCAL_PINCH_AXIS"
PICK_INITIAL_YAW_REFERENCE_DEG = "PICK_INITIAL_YAW_REFERENCE_DEG"
# Absolute world yaw for the target's flat pose (degrees, [-180, 180]).
# When set, the placement stages the REAL observed orientation verbatim and
# skips the narrowest-span pinch-axis search — the per-episode vision
# calibration path (SAM3 frame-0 yaw), where fidelity to the recorded scene
# outranks presenting the narrowest width to the jaw.
PICK_TARGET_YAW_WORLD_DEG = "PICK_TARGET_YAW_WORLD_DEG"
EPISODE_UID = "EPISODE_UID"
PICK_GLASSES_CASE_SIDE_REST = "PICK_GLASSES_CASE_SIDE_REST"
PICK_GLASSES_CASE_LONG_AXIS_OFFSET = (
    "PICK_GLASSES_CASE_LONG_AXIS_OFFSET")
PICK_TASK53_REAL_SCENE = "PICK_TASK53_REAL_SCENE"
PICK_TASK55_SOFT_WALLET_COLLISION = "PICK_TASK55_SOFT_WALLET_COLLISION"
_WALLET_LOCAL_XY_HALF_EXTENT_M = (0.0825, 0.065)
_TASK53_WALLET_SCALE_XYZ = (0.78, 1.14, 1.0)
# scene_room.usda is intentionally reused between reproduce runs.  Restore the
# checked-in authored root scale before querying bounds so an episode-specific
# scale cannot leak into the next small task.  These are asset-space unit
# conversions, not tuning knobs: notably Remote is authored in centimetres.
_CANONICAL_ROOT_SCALES = {
    "Remote": (0.009975474, 0.009975474, 0.009975474),
    "BaseballCap": (0.0099270735, 0.0099270735, 0.0099270735),
    "PaperCup": (1.0, 1.0, 1.0),
    "Wallet": (1.0, 1.0, 1.0),
    "Umbrella": (1.0, 1.0, 1.0),
    "GlassesCase": (1.0003073, 1.0003073, 1.0003073),
    "Book": (1.0, 1.0, 1.0),
    "Note": (1.0, 1.0, 1.0),
    "RedBullPink": (1.0, 1.0, 1.0),
}
_TASK53_CUSHION_SUPPORT_MODE = "task53_visible_cushion_v1"
_TASK59_SIDE_REST_MODE = "task59_glasses_case_long_edge_v1"
_TASK55_PAD_AXIS_EPISODE_UID = (
    "7c7a3f0d-d368-4286-9274-e4f8041c9859")
_TASK55_SOFT_WALLET_EPISODE_UIDS = frozenset({
    _TASK55_PAD_AXIS_EPISODE_UID,
    # This audited-l2 episode shows the same filled herringbone pouch, nearly
    # horizontal long edge, coffee-table context and successful right-hand
    # edge lift.  It is the current publication canary because the three newer
    # probes have not yet completed the required three-copy manifest sync.
    "01ed7e25-5dc6-4f5b-bfea-5ceca2d34143",
    # Three-view review confirms the same filled herringbone fabric pouch and
    # right-hand edge-loading interaction in this second small Task 55 probe.
    "f60be28b-c295-4a31-9f07-cc57f4270739",
    # The third preferred probe shows the same target identity, initial
    # table-rest state, and right-hand narrow-axis pickup.
    "6fa97ae0-cf4e-49ca-9fe0-1712e0d83d89",
})
_TASK55_CURRENT_CONTEXT_EPISODE_UIDS = _TASK55_SOFT_WALLET_EPISODE_UIDS
_TASK57_PAD_AXIS_EPISODE_UID = (
    "0138ae6f-3d50-4539-b71c-90a80c961a95")
_TASK59_SUPPORT_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "task59_support_calibration_v1.json",
)
_TASK59_SUPPORT_SCHEMA = "real2sim.task59_support_calibration.v1"
_TASK56_CONTACT_SCENE_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "task56_contact_scene_calibration_v2.json",
)
_TASK56_CONTACT_SCENE_SCHEMA = (
    "real2sim.task56_contact_scene_calibration.v2")
_TASK59_CURRENT_CONTEXT_EPISODE_UID = (
    "d15fc3f0-595d-4df1-b279-fb0ce29a7832")
_TASK57_CURRENT_CONTEXT_EPISODE_UID = (
    "0138ae6f-3d50-4539-b71c-90a80c961a95")
_TASK56_CURRENT_CONTEXT_EPISODE_UIDS = frozenset({
    "003d1033-5f4f-4030-9484-0f2aa53495e6",
    "07ec98c4-9e86-4abe-a301-43ec193068e6",
    "0a5c5dba-3cf7-4f9f-9d18-8117670e37cd",
})


def _finite_vector(raw, *, name, size):
    """Parse one comma-separated finite vector without accepting partial input."""
    values = str(raw).split(",")
    if len(values) != size or any(not value.strip() for value in values):
        raise ValueError(
            f"{name} must contain exactly {size} comma-separated numbers")
    try:
        parsed = tuple(float(value) for value in values)
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain exactly {size} comma-separated numbers") from exc
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must contain only finite numbers")
    return parsed


def wallet_edge_canary_config(environ, target_prim):
    """Return validated opt-in Wallet placement controls.

    The empty/default environment deliberately returns ``(None, None)``.
    Local-Z is not rejected here because a future trajectory may already
    present a vertical pad axis; the live composed pose is the authority.
    """
    edge_raw = str(environ.get(PICK_LOCAL_EDGE_OFFSET, "")).strip()
    axis_raw = str(environ.get(PICK_LOCAL_PINCH_AXIS, "")).strip()
    if not edge_raw and not axis_raw:
        return None, None
    if target_prim != "Wallet":
        raise ValueError(
            f"{PICK_LOCAL_EDGE_OFFSET}/{PICK_LOCAL_PINCH_AXIS} are "
            "Wallet-only canary controls")

    edge = None
    if edge_raw:
        edge = _finite_vector(
            edge_raw, name=PICK_LOCAL_EDGE_OFFSET, size=2)
        if any(
                abs(value) >= limit
                for value, limit in zip(
                    edge, _WALLET_LOCAL_XY_HALF_EXTENT_M)):
            raise ValueError(
                f"{PICK_LOCAL_EDGE_OFFSET} must remain strictly inside "
                "the Wallet's 165x130 mm footprint")

    axis = None
    if axis_raw:
        axis = _finite_vector(
            axis_raw, name=PICK_LOCAL_PINCH_AXIS, size=3)
        norm = math.sqrt(sum(value * value for value in axis))
        if norm < 1e-9:
            raise ValueError(
                f"{PICK_LOCAL_PINCH_AXIS} must be non-zero")
        axis = tuple(value / norm for value in axis)
    return edge, axis


def task53_real_scene_config(environ, task_id, target_prim):
    """Validate the opt-in scene reconstructed from all three real videos."""
    raw = str(environ.get(PICK_TASK53_REAL_SCENE, "")).strip()
    if not raw:
        return False
    if raw != "1":
        raise ValueError(
            f"{PICK_TASK53_REAL_SCENE} must be exactly '1' when set")
    if str(task_id) != "53" or target_prim != "Wallet":
        raise ValueError(
            f"{PICK_TASK53_REAL_SCENE} is Task53/Wallet-only")
    return True


def task59_side_rest_config(environ, task_id, target_prim):
    """Validate the evidence-backed, opt-in Task59 GlassesCase rest pose.

    The switch accepts exactly ``"1"``.  In particular, a supplied ``"0"``
    is rejected instead of being silently interpreted as the default, so
    misspelled or incorrectly forwarded canary configuration fails closed.
    Task57 shares the current proxy asset but has no matching real-video audit
    and is intentionally ineligible.
    """
    raw = str(environ.get(PICK_GLASSES_CASE_SIDE_REST, "")).strip()
    if not raw:
        return False
    if raw != "1":
        raise ValueError(
            f"{PICK_GLASSES_CASE_SIDE_REST} must be exactly '1' when set")
    if str(task_id) != "59" or target_prim != "GlassesCase":
        raise ValueError(
            f"{PICK_GLASSES_CASE_SIDE_REST} is "
            "Task59/GlassesCase-only")
    return True


def task59_long_axis_offset_config(
        environ, task_id, target_prim, side_rest):
    """Return the sole evidence-backed Task59 long-axis rescue offset."""
    raw = str(environ.get(
        PICK_GLASSES_CASE_LONG_AXIS_OFFSET, "")).strip()
    if not raw:
        return 0.0
    if raw != "0.015":
        raise ValueError(
            f"{PICK_GLASSES_CASE_LONG_AXIS_OFFSET} must be exactly "
            "'0.015' when set")
    if (
        str(task_id) != "59"
        or target_prim != "GlassesCase"
        or side_rest is not True
    ):
        raise ValueError(
            f"{PICK_GLASSES_CASE_LONG_AXIS_OFFSET} requires the "
            "Task59/GlassesCase side-rest canary")
    return 0.015


def task59_support_calibration(path=_TASK59_SUPPORT_MANIFEST):
    """Load and validate the reviewed Task59 support calibration values."""
    calibration = task59_support_calibration_identity(path)
    return {
        "manifest_revision": calibration["manifest_revision"],
        "support_mode": calibration["support_authority"],
        "support_top_m": calibration["support_top_m"],
        "authorized_min_m": calibration["authorized_min_m"],
        "authorized_max_m": calibration["authorized_max_m"],
        "uncertainty_m": calibration["uncertainty_m"],
    }


def task56_contact_scene_calibration(
        path=_TASK56_CONTACT_SCENE_MANIFEST, episode_uid=None):
    """Load the exact episode's RGB-D-authorized surface/contact context."""
    if not episode_uid:
        raise ValueError(
            "Task56 contact-scene calibration requires one exact "
            "episode_uid")
    requested_uid = str(episode_uid)
    try:
        manifest = json.loads(
            open(path, encoding="utf-8").read())
        episode = manifest["episodes"][requested_uid]
        support = episode["contact_context"][0]
        surface = episode["scene_surface"]
        normalization = manifest["current_runtime_normalization"]
        active_hand = str(episode["active_hand"]).lower()
        target_initial_pose = episode["target_initial_pose"]
        target_yaw_offset = float(
            target_initial_pose["yaw_offset_deg"])
        target_yaw_reference = float(
            target_initial_pose["yaw_reference_deg"])
        contact_physics = manifest["target_contact_physics"]
        policy = manifest["policy"]
        size = tuple(float(value) for value in support["real_size_m"])
        offset = tuple(
            float(value)
            for value in support["center_offset_from_target_xy_m"])
        source_table_top = float(surface["episode_calibrated_top_m"])
        source_support_bottom = float(support["bottom_m"])
        source_support_top = float(support["top_m"])
        table_top = float(normalization["required_top_m"])
        support_bottom = table_top
        support_top = (
            table_top + source_support_top - source_support_bottom)
        overhang = support.get("robot_facing_overhang_m")
        if overhang is not None:
            overhang = float(overhang)
        yaw = float(support["yaw_deg"])
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError) as exc:
        raise ValueError(
            "Task56 has no reviewed contact-context calibration for "
            f"episode {requested_uid}; cross-episode support-pose reuse is "
            "forbidden") from exc
    expected_supports = {
        "003d1033-5f4f-4030-9484-0f2aa53495e6": "Note",
        "07ec98c4-9e86-4abe-a301-43ec193068e6": "Wallet",
        "0a5c5dba-3cf7-4f9f-9d18-8117670e37cd": "Book",
    }
    if (
            manifest.get("schema") != _TASK56_CONTACT_SCENE_SCHEMA
            or set(manifest.get("episodes", ())) != set(expected_supports)
            or episode.get("source_episode_uuid") != requested_uid
            or active_hand not in ("l", "r")
            or not all(math.isfinite(value) for value in (
                target_yaw_offset, target_yaw_reference))
            or not -180.0 <= target_yaw_offset <= 180.0
            or not -180.0 <= target_yaw_reference <= 180.0
            or surface.get("prim") != "/World/Assets/CoffeeTable"
            or normalization.get("schema")
            != "real2sim.current_coffee_table_normalization.v1"
            or normalization.get("support_prim")
            != "/World/Assets/CoffeeTable"
            or abs(table_top - COFFEE_TABLE_CURRENT_TOP_M) > 1e-12
            or normalization.get(
                "source_episode_depth_planes_are_evidence_only") is not True
            or normalization.get(
                "preserve_support_height_above_table") is not True
            or support.get("prim") != expected_supports.get(requested_uid)
            or support.get("relation") != "supports_target"
            or support.get("required") is not True
            or support.get("kinematic") is not True
            or policy.get("scope_task_ids") != ["56"]
            or policy.get("contact_context_scope") != "per_episode"
            or policy.get("missing_episode_policy") != "fail_closed"
            or policy.get("forbid_cross_episode_pose_reuse") is not True
            or policy.get("forbid_invisible_replacement_support") is not True
            or policy.get("must_not_authorize_camera_mount_edits") is not True
            or contact_physics.get("collision_approximation") != "convexHull"
            or contact_physics.get("visual_mesh_unchanged") is not True
            or contact_physics.get("scope_task_ids") != ["56"]
            or len(size) != 3
            or len(offset) != 2
            or not all(math.isfinite(value) for value in (
                *size, *offset, source_table_top, source_support_bottom,
                source_support_top, table_top, support_bottom, support_top,
                yaw))
            or (
                overhang is not None
                and not math.isfinite(overhang))
            or min(size) <= 0.0
            or (
                overhang is not None
                and not 0.0 <= overhang <= 0.100)
            or abs(source_support_bottom - source_table_top) > 1e-6
            or abs(
                source_support_top
                - source_support_bottom
                - size[2]
            ) > 1e-6
            or abs(support_bottom - table_top) > 1e-12
            or abs(support_top - support_bottom - size[2]) > 1e-6):
        raise ValueError(
            "Task56 contact-scene calibration scope/geometry is inconsistent")
    return {
        "episode_uid": requested_uid,
        "active_hand": active_hand,
        "target_yaw_offset_deg": target_yaw_offset,
        "target_yaw_reference_deg": target_yaw_reference,
        "surface_path": surface["prim"],
        "table_top_m": table_top,
        "source_episode_table_top_m": source_table_top,
        "support_prim": support["prim"],
        "support_size_m": size,
        "support_yaw_deg": yaw,
        "support_center_offset_xy_m": offset,
        "support_bottom_m": support_bottom,
        "support_top_m": support_top,
        "robot_facing_overhang_m": overhang,
        "target_collision_approximation":
            contact_physics["collision_approximation"],
        "relation": support["relation"],
        "manifest_path": os.path.abspath(path),
    }


def sh(c, t=None):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)


def main():
    tid = os.environ.get("PICK_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if tid not in TASKS:
        print(json.dumps({"ok": False, "reason": f"unknown PICK_TASK {tid}"}))
        return 2
    cfg = TASKS[tid]
    episode_uid = str(os.environ.get(EPISODE_UID, "")).strip()
    initial_yaw_offset_deg = float(
        cfg.get("initial_yaw_offset_deg", 0.0))
    initial_yaw_reference_deg = cfg.get("initial_yaw_reference_deg")
    if PICK_INITIAL_YAW_REFERENCE_DEG in os.environ:
        try:
            initial_yaw_reference_deg = float(
                os.environ[PICK_INITIAL_YAW_REFERENCE_DEG])
        except ValueError:
            print(json.dumps({
                "ok": False,
                "reason": (
                    f"{PICK_INITIAL_YAW_REFERENCE_DEG} must be a finite "
                    "angle in [-180, 180]"),
            }))
            return 5
        if (
                not math.isfinite(initial_yaw_reference_deg)
                or not -180.0 <= initial_yaw_reference_deg <= 180.0):
            print(json.dumps({
                "ok": False,
                "reason": (
                    f"{PICK_INITIAL_YAW_REFERENCE_DEG} must be a finite "
                    "angle in [-180, 180]"),
            }))
            return 5
    requested_target_yaw_world_deg = None
    if PICK_TARGET_YAW_WORLD_DEG in os.environ:
        try:
            requested_target_yaw_world_deg = float(
                os.environ[PICK_TARGET_YAW_WORLD_DEG])
        except ValueError:
            requested_target_yaw_world_deg = float("nan")
        if (
                not math.isfinite(requested_target_yaw_world_deg)
                or not -180.0 <= requested_target_yaw_world_deg <= 180.0):
            print(json.dumps({
                "ok": False,
                "reason": (
                    f"{PICK_TARGET_YAW_WORLD_DEG} must be a finite "
                    "angle in [-180, 180]"),
            }))
            return 5
    if cfg.get("no_asset"):
        print(json.dumps({"ok": False, "task": tid, "reason": f"no sim asset for '{cfg['prim']}' "
                          f"(robot task = {cfg.get('prompt')}); needs modeling before reproduce"}))
        return 7
    try:
        edge_offset_local, requested_local_pinch_axis = (
            wallet_edge_canary_config(os.environ, cfg["prim"]))
        task55_soft_wallet_raw = str(os.environ.get(
            PICK_TASK55_SOFT_WALLET_COLLISION, "")).strip()
        if task55_soft_wallet_raw not in ("", "1"):
            raise ValueError(
                f"{PICK_TASK55_SOFT_WALLET_COLLISION} must be exactly '1' "
                "when set")
        task55_soft_wallet = task55_soft_wallet_raw == "1"
        if task55_soft_wallet and (
                tid != "55"
                or cfg["prim"] != "Wallet"
                or episode_uid not in _TASK55_SOFT_WALLET_EPISODE_UIDS):
            raise ValueError(
                f"{PICK_TASK55_SOFT_WALLET_COLLISION} is bound to the "
                "reviewed small Task 55 Wallet episode")
        task53_real_scene = task53_real_scene_config(
            os.environ, tid, cfg["prim"])
        task53_episode = None
        if task53_real_scene:
            if not episode_uid:
                raise ValueError(
                    "EPISODE_UID is required because Task53 visible-cushion "
                    "state is calibrated per episode")
            task53_episode = task53_episode_calibration(
                task53_contact_scene_calibration_identity(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    TASK53_CONTACT_SCENE_CALIBRATION_FILENAME,
                )),
                episode_uid,
            )
            # The pinned episode is the canonical default.  Explicit caller
            # values remain accepted only when they are byte-for-byte the
            # same calibration, as checked below.
            os.environ.setdefault(
                "BITE_Z",
                repr(task53_episode["contact_controls"]["bite_z_m"]),
            )
            os.environ.setdefault(
                "PICK_PAD_AXIS_OFFSET",
                repr(task53_episode[
                    "contact_controls"]["pad_axis_offset_m"]),
            )
        if task53_real_scene and (
                edge_offset_local is not None
                or requested_local_pinch_axis is not None):
            raise ValueError(
                f"{PICK_TASK53_REAL_SCENE} uses the scaled centred Wallet "
                "grasp and cannot be combined with legacy corner controls")
        task59_side_rest = task59_side_rest_config(
            os.environ, tid, cfg["prim"])
        task59_long_axis_offset = task59_long_axis_offset_config(
            os.environ, tid, cfg["prim"], task59_side_rest)
        task59_support = (
            task59_support_calibration()
            if task59_side_rest else None
        )
        if tid == "56" and not episode_uid:
            raise ValueError(
                "EPISODE_UID is required because Task56 contact context "
                "is calibrated per episode")
        task56_contact_scene = (
            task56_contact_scene_calibration(
                episode_uid=episode_uid)
            if tid == "56" else None
        )
        if task56_contact_scene is not None:
            initial_yaw_offset_deg = task56_contact_scene[
                "target_yaw_offset_deg"]
            if PICK_INITIAL_YAW_REFERENCE_DEG not in os.environ:
                initial_yaw_reference_deg = task56_contact_scene[
                    "target_yaw_reference_deg"]
    except ValueError as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}))
        return 5
    _pa = os.environ.get("PICK_ANCHOR", "")
    if requested_local_pinch_axis is not None and _pa:
        print(json.dumps({
            "ok": False,
            "reason": (
                f"{PICK_LOCAL_PINCH_AXIS} needs a live 3-D pad axis; "
                "PICK_ANCHOR refine supplies only PICK_PINCH_AXIS x,y. "
                "Run the bounded non-refine align canary instead."
            ),
        }))
        return 5
    fam = FAMILIES[cfg["family"]]
    root = cfg.get("root", "/World/Assets")
    tp = f"{root}/{cfg['prim']}"
    surface_z = (
        task56_contact_scene["table_top_m"]
        if task56_contact_scene else fam["surface_z"]
    )
    if task56_contact_scene or cfg["family"] == "coffee_table":
        surface_contract = {
            "path": (
                task56_contact_scene["surface_path"]
                if task56_contact_scene
                else "/World/Assets/CoffeeTable"
            ),
            "center_xy_m": COFFEE_TABLE_CURRENT_CENTER_XY_M,
            "size_xyz_m": COFFEE_TABLE_CURRENT_SIZE_XYZ_M,
            "bottom_m": COFFEE_TABLE_CURRENT_BOTTOM_M,
            "label": "CoffeeTable",
        }
    elif cfg["family"] == "tv_table":
        surface_contract = {
            "path": "/World/Assets/TvTable",
            "center_xy_m": TV_CABINET_CURRENT_CENTER_XY_M,
            "size_xyz_m": TV_CABINET_CURRENT_SIZE_XYZ_M,
            "bottom_m": TV_CABINET_CURRENT_BOTTOM_M,
            "label": "TvTable",
        }
    else:
        surface_contract = None
    bite = float(os.environ.get("PICK_BITE", "0"))
    pad_axis_offset = float(
        os.environ.get("PICK_PAD_AXIS_OFFSET", "0"))
    if task53_real_scene:
        _task53_controls = task53_episode["contact_controls"]
        _task53_bite_z = float(os.environ.get("BITE_Z", "0.012"))
        _task53_reviewed_bite_z = _task53_controls["bite_z_m"]
        _task53_reviewed_pad_axis_offset = (
            _task53_controls["pad_axis_offset_m"])
        if (
            abs(_task53_bite_z - _task53_reviewed_bite_z) > 1e-12
            or abs(
                pad_axis_offset
                - _task53_reviewed_pad_axis_offset) > 1e-12
        ):
            print(json.dumps({
                "ok": False,
                "reason": (
                    "Task53 BITE_Z/PICK_PAD_AXIS_OFFSET must match the "
                    "reviewed episode calibration "
                    f"({_task53_reviewed_bite_z}, "
                    f"{_task53_reviewed_pad_axis_offset})"
                ),
            }))
            return 5
    if (
        not math.isfinite(pad_axis_offset)
        or abs(pad_axis_offset) > 0.025
        or (
            pad_axis_offset
            and not (
                (tid == "53" and task53_real_scene)
                or tid == "56"
                or (
                    tid == "55"
                    and task55_soft_wallet
                    and episode_uid in _TASK55_SOFT_WALLET_EPISODE_UIDS
                )
                or (
                    tid == "57"
                    and episode_uid == _TASK57_PAD_AXIS_EPISODE_UID
                )
                or (tid == "59" and task59_side_rest)
            )
        )
    ):
        print(json.dumps({
            "ok": False,
            "reason": (
                "PICK_PAD_AXIS_OFFSET must be finite, within "
                "[-0.025, 0.025], and used only by the reviewed small "
                "Task 53 visible-cushion canary, small Task 56, a reviewed "
                "small Task 55 soft-Wallet episode, the pinned small Task 57 "
                "episode, or the reviewed small Task 59 "
                "side-rest canary"),
        }))
        return 5
    if cfg.get("drawer_pick"):
        # The articulated drawer is furniture and cannot follow a group
        # shift; refuse instead of replaying against a drawer that stayed put.
        group_shift.require_identity("articulated-drawer pick task")
    if "FIXED_BASE" in os.environ:
        bx, by = map(float, os.environ["FIXED_BASE"].split(","))
    else:
        bx, by = cfg.get("base", fam["base"])
    if "FIXED_BASE_YAW_DEG" in os.environ:
        _yaw = math.radians(float(os.environ["FIXED_BASE_YAW_DEG"]))
        qw, qx, qy, qz = (math.cos(_yaw / 2), 0.0, 0.0, math.sin(_yaw / 2))
    else:
        qw, qx, qy, qz = cfg.get("quat", fam["quat"])

    from drawer_articulation import drawer_reset_code as _drawer_reset_code
    park = f"""from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
s = Usd.Stage.Open('{SCENE}')
# A previous articulated-drawer episode (Task 28/92) leaves the black
# storage drawer pulled out, which breaks the shared furniture-contract
# AABB below.  Every pick init first restores the closed kinematic
# drawers; drawer picks re-articulate them in the placement stage.
{_drawer_reset_code()}
rob = s.GetPrimAtPath('/World/Robot')
# The canonical scene is shared across serial renders.  A previous small
# Task 59 run may have authored its episode-bound visible clutter and exact
# tabletop collision group.  Always close that group before initializing a
# new pick; the pinned Task59 context transaction recreates it later in this
# same invocation.  This prevents one task's secondary objects from leaking
# into another task's frame zero.
for stale_path in (
    '/World/Assets/Task55CurrentContext',
    '/World/Assets/Task56CurrentContext',
    '/World/Assets/Task57CurrentContext',
    '/World/Assets/Task59CurrentContext',
    '/World/Assets/Task29Cup0',
    '/World/Assets/Task29Cup1',
    '/World/Assets/Task29Cup2',
):
    stale_context = s.GetPrimAtPath(stale_path)
    if (
            stale_context
            and stale_context.IsValid()
            and stale_context.IsActive()):
        stale_context.SetActive(False)
for op in UsdGeom.Xformable(rob).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d({bx}, {by}, 0))
rob.GetAttribute('xformOp:orient').Set(Gf.Quatf({qw}, {qx}, {qy}, {qz}))
    # Normalize every native-table pick to its one current furniture contract.
    # Episode RGB-D planes remain source evidence for relative object/support
    # geometry; they may not move or resize the shared simulator table.
_surface_contract = {surface_contract!r}
_surface_path = (
    _surface_contract['path'] if _surface_contract is not None else None)
if _surface_path:
    surf = s.GetPrimAtPath(_surface_path)
    assert surf and surf.IsValid(), 'missing task surface ' + _surface_path
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
        useExtentsHint=True)
    box = cache.ComputeWorldBound(surf).ComputeAlignedBox()
    current_center_x = float(box.GetMin()[0] + box.GetMax()[0]) / 2.0
    current_center_y = float(box.GetMin()[1] + box.GetMax()[1]) / 2.0
    current_top = float(box.GetMax()[2])
    dx = _surface_contract['center_xy_m'][0] - current_center_x
    dy = _surface_contract['center_xy_m'][1] - current_center_y
    dz = {surface_z!r} - current_top
    moved = False
    for op in UsdGeom.Xformable(surf).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            op.Set(Gf.Vec3d(t[0] + dx, t[1] + dy, t[2] + dz))
            moved = True
    assert moved, 'task surface has no translate op'
    cache.Clear()
    box = cache.ComputeWorldBound(surf).ComputeAlignedBox()
    actual_min = [float(value) for value in box.GetMin()]
    actual_max = [float(value) for value in box.GetMax()]
    expected_min = [
        _surface_contract['center_xy_m'][0]
        - _surface_contract['size_xyz_m'][0] / 2.0,
        _surface_contract['center_xy_m'][1]
        - _surface_contract['size_xyz_m'][1] / 2.0,
        _surface_contract['bottom_m'],
    ]
    expected_max = [
        expected_min[0] + _surface_contract['size_xyz_m'][0],
        expected_min[1] + _surface_contract['size_xyz_m'][1],
        expected_min[2] + _surface_contract['size_xyz_m'][2],
    ]
    assert all(
        abs(actual - expected) <= 0.002
        for actual, expected in zip(
            actual_min + actual_max, expected_min + expected_max
        )
    ), (
        _surface_contract['label']
        + ' differs from current metric furniture contract: '
        + repr((actual_min, actual_max))
    )
s.GetRootLayer().Save(); print('parked')
"""
    open("/tmp/_ip_park.py", "w").write(park)
    if "parked" not in sh("conda run -n mesh311 python /tmp/_ip_park.py", t=120).stdout:
        print(json.dumps({"ok": False, "reason": "park failed"}))
        return 3

    # PD-REFINE anchor (droop fix): the kinematic FK anchor from align_grasp sits a systematic
    # ~+18 mm (t53: median +17.5, range -20..+55) BELOW where the PD-driven fingertip actually
    # closes (arm-config droop) — so the object placed at the kin anchor is skimmed/bulldozed
    # instead of bitten (t53 pre-grasp slide 50-77 mm, lift <35 mm, hold 0 s). PICK_ANCHOR lets
    # the runner feed back THIS episode's PD-measured close-tip (x,y,z) as the anchor, skipping
    # align_grasp. Same mechanism that doubled the book pipeline's pass rate (refine_episode.py).
    if _pa:
        gx, gy, gz = map(float, _pa.split(","))
        hand = os.environ.get("PICK_HAND", "?")
        try:
            pax, pay = map(
                float, os.environ["PICK_PINCH_AXIS"].split(","))
            pan = (pax * pax + pay * pay) ** 0.5
            if not all(v == v for v in (pax, pay, pan)) or pan < 1e-6:
                raise ValueError
            pax, pay = pax / pan, pay / pan
        except (KeyError, TypeError, ValueError):
            print(json.dumps({
                "ok": False,
                "reason": "refine anchor requires a finite PICK_PINCH_AXIS",
            }))
            return 5
        # PD-measured anchor is trusted (no surface band gate); only sanity-reject non-finite/NaN
        if not all(v == v for v in (gx, gy, gz)):
            print(json.dumps({"ok": False, "reason": f"bad PICK_ANCHOR {_pa!r}"}))
            return 5
        # Edge-offset-only canaries can reuse a refined anchor.  A requested
        # local axis was rejected above because this legacy refine interface
        # carries only the XY projection and cannot certify 3-D alignment.
        pax3, pay3, paz3 = pax, pay, 0.0
    else:
        sh(
            f"docker exec {CONTAINER_NAME} bash -lc "
            f"\"pkill -9 -f '{_RT_PATTERN}' 2>/dev/null\"")
        sh(f"docker exec {CONTAINER_NAME} bash -lc 'rm -f /geniesim/main/demo_sim/cam_preview{INSTANCE}/align.txt'; sleep 2")
        # Drawer picks: the wallet grasp is the LEFT hand's rise; the same
        # align boot additionally FK-reports the RIGHT (pull) hand at its own
        # first rise (drawer front face) and at the wallet-grasp rise (drawer
        # travel by then).
        drawer_pick = bool(cfg.get("drawer_pick"))
        align_hand = (
            task56_contact_scene["active_hand"]
            if task56_contact_scene is not None
            else ("l" if drawer_pick else ""))
        align_extra_reports = ""
        align_run_ordinal_arg = ""
        if drawer_pick:
            import numpy as _np
            _tr = _np.load(f"{DEMO}/playback_traj{INSTANCE}.npz")
            _rises = {}
            _l_rise_count = 0
            for _hd in ("l", "r"):
                _g = _np.asarray(
                    _tr["lgrip" if _hd == "l" else "rgrip"], float) >= 0.5
                _rise = None
                for _i in range(1, len(_g)):
                    if _g[_i] and not _g[_i - 1]:
                        if _rise is None:
                            _rise = _i
                        if _hd == "l":
                            _l_rise_count += 1
                _rises[_hd] = _rise
            # Bimanual recordings close the LEFT hand more than once (e.g.
            # steadying the drawer wall before the wallet grasp).  The
            # wallet grasp is the closure that carries to the end of the
            # episode, i.e. the LAST left run.
            if _l_rise_count > 1:
                align_run_ordinal_arg = (
                    f"ALIGN_RUN_ORDINAL={_l_rise_count - 1} ")
            if _rises["l"] is None:
                print(json.dumps({
                    "ok": False,
                    "reason": "drawer pick needs a wallet-grasp grip rise",
                }))
                return 4
            # Four of the seven 2026-07-31 recordings start with the drawer
            # ALREADY pulled out (no right-hand rise); those get no pull-hand
            # FK reports and fall back to a pre-opened drawer below.
            if _rises["r"] is not None and _rises["r"] < _rises["l"]:
                align_extra_reports = (
                    f"r:{_rises['r']};r:{_rises['l']}")
        align_proc = sh(
            f"docker exec {CONTAINER_NAME} bash -ic "
            f"\"CUDA_VISIBLE_DEVICES={RENDER_GPU} INSTANCE={INSTANCE} "
            f"ALIGN_HAND={align_hand} "
            f"ALIGN_EXTRA_REPORTS='{align_extra_reports}' "
            f"{align_run_ordinal_arg}"
            f"GRASP_MODE={cfg.get('grasp', '')} "
            "exec timeout -k 30s 900s /isaac-sim/python.sh "
            "/geniesim/main/demo_sim/usdz/align_grasp.py "
            f"runtag{_RT}\"",
            t=950,
        )
        f = f"{CAM_PREVIEW}/align.txt"
        txt = open(f).read() if os.path.isfile(f) else ""
        m = re.search(r"midpoint world: \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)", txt)
        if not m or "NO grasp" in txt:
            detail = (align_proc.stdout + align_proc.stderr).strip()[-500:]
            if align_proc.returncode != 0:
                reason = (
                    f"align exited {align_proc.returncode}: "
                    f"{detail or 'no output'}")
            elif not os.path.isfile(f):
                reason = (
                    "align produced no marker: "
                    f"{detail or 'no output'}")
            else:
                reason = "no grasp crossing"
            print(json.dumps({"ok": False, "reason": reason}))
            return 4
        gx, gy, gz = map(float, m.groups())
        hand = (re.search(r"hand (\w)", txt) or [None, "?"])[1]
        tips = re.search(
            r"TIPS outer: \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\) "
            r"inner: \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)",
            txt,
        )
        if not tips:
            print(json.dumps({
                "ok": False,
                "reason": "align marker has no physical pad coordinates",
            }))
            return 4
        tip_values = [float(value) for value in tips.groups()]
        pad_dx = tip_values[3] - tip_values[0]
        pad_dy = tip_values[4] - tip_values[1]
        pad_dz = tip_values[5] - tip_values[2]
        p3n = (pad_dx * pad_dx + pad_dy * pad_dy + pad_dz * pad_dz) ** 0.5
        if not all(v == v for v in (pad_dx, pad_dy, pad_dz, p3n)) or p3n < 1e-6:
            print(json.dumps({
                "ok": False,
                "reason": "align physical 3-D pad axis is degenerate",
            }))
            return 4
        pax3, pay3, paz3 = (
            pad_dx / p3n, pad_dy / p3n, pad_dz / p3n)
        # Preserve the established default XY computation exactly; the
        # separately normalized 3-D vector is used only by an explicit canary.
        pax, pay = pad_dx, pad_dy
        pan = (pax * pax + pay * pay) ** 0.5
        if not all(v == v for v in (pax, pay, pan)) or pan < 1e-6:
            print(json.dumps({
                "ok": False,
                "reason": "align physical pad axis is degenerate",
            }))
            return 4
        pax, pay = pax / pan, pay / pan
        b = dict(fam["band"], **cfg.get("band", {}))
        if not (b["x"][0] < gx < b["x"][1] and b["y"][0] < gy < b["y"][1] and b["z"][0] < gz < b["z"][1]):
            print(json.dumps({"ok": False, "reason": f"kin anchor off surface: ({gx:.2f},{gy:.2f},{gz:.2f})",
                              "hand": hand, "kin": [round(gx,3), round(gy,3), round(gz,3)]}))
            return 5

    ax, ay = cfg.get("approach", fam["approach"])
    tx, ty = gx + ax * bite, gy + ay * bite
    # Drawer picks: the wallet's grasp anchor is where the LEFT hand closes in
    # the already pulled-out drawer.  The wallet must START inside the nearly
    # closed drawer, so shift the anchor back by the drawer's FK-measured
    # travel at the wallet-grasp frame (right pull-hand displacement between
    # its own first close and the wallet close) and clamp into the closed
    # interior footprint.  The drawer itself is articulated by the shared
    # drawer_articulation block appended to the placement script.
    drawer_front_x = None
    drawer_wall_top_z = None
    drawer_travel_m = None
    if bool(cfg.get("drawer_pick")):
        if _pa:
            print(json.dumps({
                "ok": False,
                "reason": "drawer pick is incompatible with PICK_ANCHOR",
            }))
            return 4
        _extras = re.findall(
            r"EXTRA r@([0-9]+) midpoint world: "
            r"\(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\)", txt)
        if len(_extras) >= 2:
            _extras = sorted(
                ((int(fr), float(x), float(y), float(z))
                 for fr, x, y, z in _extras))
            drawer_front_x = _extras[0][1]
            drawer_wall_top_z = _extras[0][3]
            drawer_travel_m = _extras[-1][1] - _extras[0][1]
            # Some recordings STEADY an already-open drawer with the right
            # hand instead of pulling (x displacement ~0, FK noise slightly
            # negative).
            if -0.02 <= drawer_travel_m < 0.0:
                drawer_travel_m = 0.0
            if not 0.0 <= drawer_travel_m <= 0.30:
                print(json.dumps({
                    "ok": False,
                    "reason": (
                        "drawer pick FK travel out of range: "
                        f"{drawer_travel_m:.3f} m"),
                }))
                return 4
        else:
            # Pre-opened drawer recording: no pull-hand FK to read the
            # front face from.  The recordings show the drawer still mostly
            # INSIDE the table at frame 0 -- only a modest lip is proud of
            # the closed face -- so use that fixed lip rather than opening
            # the drawer far enough to centre the wallet in it (which put it
            # ~18 cm out, much further than the real frame 0).  The wallet
            # then sits wherever its own grasp anchor says inside that
            # interior.  Wall height stays native (the FK z is a pad
            # midpoint, not the lip).
            drawer_travel_m = 0.0
            drawer_front_x = -1.550025237 + 0.080
            drawer_wall_top_z = None
        tx -= drawer_travel_m
        # Clamp the STANDING wallet (40 mm thick along Y after the lean
        # pose) fully inside the drawer interior AT FRAME 0; x follows the
        # drawer's initial front face, y comes from the south-wall seat.
        # Keep 5 mm clearance from the west wall — corner-touch spawns get
        # ejected by the penetration solver and fail the settle gate.
        _wallet_half = dict(x=0.022)
        _interior_x = (drawer_front_x - 0.2942, drawer_front_x - 0.0163)
        tx = min(max(tx, _interior_x[0] + _wallet_half["x"] + 0.005),
                 _interior_x[1] - _wallet_half["x"] - 0.005)
    # Convention (book-pipeline calibrated): the FK anchor gz is the FINGERTIP-MIDPOINT at
    # close = the OBJECT TOP surface (fingers wrap from above). So bbox top lands at gz.
    # GRAB_FRAC (default 1.0) tunes it: bottom = gz - GRAB_FRAC*obj_h.
    # On the sofa the real seat is an uncompressed cushion (higher than the rigid sim seat)
    # -> SeatRiser under the object models the true seat height (init_episode's mechanism).
    grab_frac = float(os.environ.get("GRAB_FRAC", "1.0"))
    # Generic per-episode visible-support contract (asset-on-asset initial
    # constructions, allowed 2026-07-31; see
    # episode_support_context_audit_v1.json).  A task opts in by shipping
    # task<ID>_secondary_context_v1.json whose policy declares
    # asset_on_asset_initial_constructions_allowed; tasks with dedicated
    # mechanisms (53/56 etc.) keep their own paths.
    episode_support_items = ()
    _support_contract_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"task{tid}_secondary_context_v1.json")
    if episode_uid and os.path.isfile(_support_contract_path):
        try:
            with open(_support_contract_path, "r", encoding="utf-8") as _fh:
                _support_contract = json.load(_fh)
        except (ValueError, OSError) as exc:
            print(json.dumps({
                "ok": False,
                "reason": f"support contract unreadable: {exc}"}))
            return 5
        if (
            _support_contract.get("schema")
            == f"real2sim.task{tid}_secondary_context.v1"
            and _support_contract.get("task_id") == tid
            and _support_contract.get("policy", {}).get(
                "asset_on_asset_initial_constructions_allowed") is True
        ):
            _support_episode = _support_contract.get(
                "episodes", {}).get(episode_uid)
            if _support_episode:
                episode_support_items = tuple(
                    item
                    for item in _support_episode.get("contact_context", ())
                    if item.get("required") is True
                )
    # A required visible close-contact support replaces SeatRiser.  The exact
    # episode's Note, Wallet, or Book and measured table plane already put the
    # target at the recorded grasp height; adding the generic invisible riser
    # would double-support it.
    use_riser = (
        cfg["family"] not in ("coffee_table", "tv_table")
        and not cfg.get("native_paper_cup")
        and task56_contact_scene is None
        and not task53_real_scene
        and not any(
            item.get("relation") == "supports_target"
            for item in episode_support_items)
    )
    force_native_surface_contact = (
        cfg["family"] in ("coffee_table", "tv_table")
        and task56_contact_scene is None
        and not cfg.get("drawer_pick")
    )
    task59_support_top = (
        task59_support["support_top_m"]
        if task59_support else None
    )
    target_collision_approximation = (
        task56_contact_scene["target_collision_approximation"]
        if task56_contact_scene is not None
        else cfg.get("collision_approximation")
    )
    # all pick-family target prims minus the current one -> deactivated so the episode is clean
    _pick_prims = {t["prim"] for t in TASKS.values()} | {
        "BaseballCap", "PaperCup", "Umbrella", "Remote", "GlassesCase"}
    evict_siblings = sorted(_pick_prims - {cfg["prim"]})
    # Shared articulated coffee-table drawer: drawer picks articulate it with
    # the per-episode FK lip/height; every other pick restores the closed
    # kinematic drawer so leftover state cannot leak into their videos.
    drawer_pick_mode = bool(cfg.get("drawer_pick"))
    if drawer_pick_mode:
        from drawer_articulation import drawer_articulation_code
        # NATIVE wall height (no scaling).  The FK z is the pad MIDPOINT of
        # a gripper hanging over the wall, not the wall lip: scaling the box
        # up to it buried the standing wallet (top 0.371 m) under a 0.376 m
        # rim, so the pads closed on the wall instead of the wallet.  The
        # real frames show the wallet protruding ~3 cm above the rim, which
        # is what the native 0.337 m lip gives.
        pick_drawer_scene_block = drawer_articulation_code(
            front_face_x=drawer_front_x,
            wall_top_z=None,
            grip_walls=True,
            fixed=True)
        _closed_bottom_z = 0.199993230
        _closed_top_z = 0.337311269
        _floor_thickness_m = 0.203662484 - _closed_bottom_z
        _wall_scale = 1.0
        if drawer_wall_top_z is not None:
            _wall_scale = min(max(
                (drawer_wall_top_z - _closed_bottom_z)
                / (_closed_top_z - _closed_bottom_z), 1.0), 1.6)
        # Standing support height.  The wallet is 165 mm long and the box
        # wall 134 mm, so resting it on the bare floor leaves only ~31 mm of
        # edge above the rim -- too short a bite, and the lift torque rolls
        # it out of the pads (max 58 mm then dropped, at every force and
        # friction setting tried).  The real fabric box has a bowed bottom
        # and other contents under the wallet, so its effective support sits
        # higher; DRAWER_PICK_SUPPORT_RAISE_M models that.
        drawer_target_bottom_z = (
            _closed_bottom_z + _floor_thickness_m * _wall_scale + 0.002
            + float(os.environ.get("DRAWER_PICK_SUPPORT_RAISE_M", "0.0")))
        # South inner wall face (bottom slab south edge + wall thickness).
        drawer_lean_south_wall_y = -0.6166
        drawer_lean_floor_z = drawer_target_bottom_z - 0.002
    else:
        from drawer_articulation import drawer_reset_code
        pick_drawer_scene_block = drawer_reset_code()
        drawer_target_bottom_z = None
        drawer_lean_south_wall_y = 0.0
        drawer_lean_floor_z = 0.0
    place = f"""import math
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
s = Usd.Stage.Open('{SCENE}')
p = s.GetPrimAtPath('{tp}')
assert p and p.IsValid(), 'missing prim {tp}'
_edge_offset_local = {
    task53_episode["grasp_local_offset_xy_m"]
    if task53_episode is not None else edge_offset_local
!r}
_requested_local_pinch_axis = {requested_local_pinch_axis!r}
_task53_real_scene = {task53_real_scene!r}
_task53_wallet_scale_xyz = {_TASK53_WALLET_SCALE_XYZ!r}
_task53_wallet_translate_xyz = {
    task53_episode["wallet_translate_xyz_m"]
    if task53_episode is not None else None
!r}
_task53_wallet_yaw_deg = {
    task53_episode["wallet_yaw_deg"]
    if task53_episode is not None else None
!r}
_task53_cushion_translate_xyz = {
    task53_episode["cushion_translate_xyz_m"]
    if task53_episode is not None else None
!r}
_task53_cushion_orient_wxyz = {
    task53_episode["cushion_orient_wxyz"]
    if task53_episode is not None else None
!r}
_task59_side_rest = {task59_side_rest!r}
_task59_long_axis_offset = {task59_long_axis_offset!r}
_task59_support_top = {task59_support_top!r}
_task_contact_support = {task56_contact_scene!r}
_task55_soft_wallet = {task55_soft_wallet!r}
_canonical_root_scales = {_CANONICAL_ROOT_SCALES!r}
_pad_axis_world = ({pax3!r}, {pay3!r}, {paz3!r})

def _restore_canonical_wallet_collision(wallet):
    # scene_room.usda is shared between runs.  The small Task 53 visible-scene
    # mode deliberately replaces the source Wallet collision with a rigid box,
    # but that episode-local child and disabled source collider must never leak
    # into small Task 55 or into the Wallet support used by small Task 56.
    stale_paths = (
        wallet.GetPath().AppendChild('Task53CollisionBox'),
        wallet.GetPath().AppendChild('Task55SoftCollision'),
    )
    for stale_path in stale_paths:
        if s.GetPrimAtPath(stale_path).IsValid():
            s.RemovePrim(stale_path)
    source_collisions = [
        prim for prim in Usd.PrimRange(wallet)
        if (
            prim != wallet
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetPath() not in stale_paths
        )
    ]
    if len(source_collisions) != 1:
        raise RuntimeError(
            'canonical Wallet requires exactly one source collision')
    UsdPhysics.CollisionAPI(
        source_collisions[0]).CreateCollisionEnabledAttr().Set(True)
    texture_shader = s.GetPrimAtPath(
        '/World/Assets/Wallet/Obj/Geom/CanvasMat/Tex')
    texture_file = texture_shader.GetAttribute('inputs:file')
    if (
            not texture_shader
            or not texture_shader.IsValid()
            or not texture_file
            or not texture_file.IsValid()):
        raise RuntimeError('canonical Wallet texture shader is missing')
    texture_file.Set(Sdf.AssetPath(
        './usdz/wallet_tex/wallet_albedo.png'))
    visual_root = wallet.GetChild('Obj')
    if visual_root and visual_root.IsValid():
        visual_xform = UsdGeom.Xformable(visual_root)
        visual_ops = visual_xform.GetOrderedXformOps()
        task55_visual_ops = [
            op for op in visual_ops
            if op.GetOpName() == 'xformOp:transform:task55VisualTilt'
        ]
        if len(task55_visual_ops) > 1:
            raise RuntimeError('Wallet has duplicate Task55 visual tilt ops')
        if task55_visual_ops:
            visual_xform.SetXformOpOrder([
                op for op in visual_ops if op not in task55_visual_ops
            ])
            visual_root.RemoveProperty(task55_visual_ops[0].GetOpName())
    wallet.RemoveProperty('real2sim:collisionGeometryVersion')
    wallet.RemoveProperty('real2sim:visualMaterialVersion')
    return source_collisions[0]

def _author_task55_soft_wallet_collision(wallet, source_collision):
    # The real target is a filled fabric pouch: the near pad loads its low
    # edge, the far fabric edge bulges upward, then both pads clamp it.  A
    # rigid flat convex hull only slides across the table.  This volume-
    # preserving, episode-local deformation envelope keeps the loaded near
    # edge at 40 mm while the far fabric edge folds mildly to 55 mm.  This
    # restores the bounded geometry that preceded the rejected 20/70 mm
    # experiment.
    path = wallet.GetPath().AppendChild('Task55SoftCollision')
    wedge = UsdGeom.Mesh.Define(s, path)
    hx, hy = 0.0825, 0.065
    wedge.CreatePointsAttr([
        Gf.Vec3f(-hx, -hy, 0.000),
        Gf.Vec3f(+hx, -hy, 0.000),
        Gf.Vec3f(+hx, +hy, 0.000),
        Gf.Vec3f(-hx, +hy, 0.000),
        Gf.Vec3f(-hx, -hy, 0.040),
        Gf.Vec3f(+hx, -hy, 0.040),
        Gf.Vec3f(+hx, +hy, 0.055),
        Gf.Vec3f(-hx, +hy, 0.055),
    ])
    wedge.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    wedge.CreateFaceVertexIndicesAttr([
        0, 3, 2, 1,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ])
    wedge.CreateSubdivisionSchemeAttr().Set('none')
    wedge.CreateOrientationAttr().Set('rightHanded')
    wedge.CreateExtentAttr().Set([
        Gf.Vec3f(-hx, -hy, 0.000),
        Gf.Vec3f(+hx, +hy, 0.055),
    ])
    wedge.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision_api = UsdPhysics.CollisionAPI.Apply(wedge.GetPrim())
    collision_api.CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(
        wedge.GetPrim()).CreateApproximationAttr().Set('convexHull')
    binding = wedge.GetPrim().CreateRelationship(
        'material:binding:physics')
    binding.SetTargets([Sdf.Path('/World/PhysicsMaterials/GripMat')])
    binding.SetMetadata('bindMaterialAs', 'strongerThanDescendants')
    UsdPhysics.CollisionAPI(
        source_collision).CreateCollisionEnabledAttr().Set(False)
    texture_shader = s.GetPrimAtPath(
        '/World/Assets/Wallet/Obj/Geom/CanvasMat/Tex')
    texture_file = texture_shader.GetAttribute('inputs:file')
    if not texture_file.Set(Sdf.AssetPath(
            './usdz/wallet_tex/task55_wallet_herringbone_lowfreq_v1.png')):
        raise RuntimeError('Task55 Wallet texture write failed')
    wallet.CreateAttribute(
        'real2sim:visualMaterialVersion',
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set('task55_wallet_herringbone_lowfreq_v1')
    wallet.CreateAttribute(
        'real2sim:collisionGeometryVersion',
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set('task55_soft_wallet_volume_wedge_v10_restored')

if not p.IsActive(): p.SetActive(True)
# Every task shares scene_room.usda, so a prior small Task 53 run can leave
# Wallet's episode-specific (0.78, 1.14, 1) scale authored on the root.
# Restore the asset-specific canonical root scale before any non-Task53 target
# geometry query; otherwise small Task 55 inherits the wrong Wallet dimensions,
# while setting every asset to unit scale makes Remote about 100x too large.
_target_scale_ops = [
    op for op in UsdGeom.Xformable(p).GetOrderedXformOps()
    if op.GetOpType() == UsdGeom.XformOp.TypeScale
]
if len(_target_scale_ops) != 1:
    raise RuntimeError(
        'pick target must have exactly one canonical root scale op')
if not _task53_real_scene:
    _canonical_target_scale = _canonical_root_scales.get(p.GetName())
    if _canonical_target_scale is None:
        raise RuntimeError(
            'missing canonical root scale for pick target ' + p.GetName())
    _target_scale_ops[0].Set(Gf.Vec3f(*_canonical_target_scale))
    if p.GetName() == 'Wallet':
        _wallet_source_collision = _restore_canonical_wallet_collision(p)
        if _task55_soft_wallet:
            _author_task55_soft_wallet_collision(
                p, _wallet_source_collision)
p.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(False)
for _nm in ('physics:velocity', 'physics:angularVelocity'):
    _a = p.GetAttribute(_nm)
    if _a and _a.Get() is not None: _a.Set(Gf.Vec3f(0, 0, 0))
if _task53_real_scene:
    target_scale_ops = [
        op for op in UsdGeom.Xformable(p).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(target_scale_ops) != 1:
        raise RuntimeError(
            'Task53 real-scene Wallet must have exactly one scale op')
    target_scale_ops[0].Set(Gf.Vec3f(*_task53_wallet_scale_xyz))
    source_collisions = [
        prim for prim in Usd.PrimRange(p)
        if (
            prim != p
            and prim.IsActive()
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(
                prim).GetCollisionEnabledAttr().Get() is not False
        )
    ]
    if len(source_collisions) != 1:
        raise RuntimeError(
            'Task53 Wallet box LOD requires exactly one source collision')
    UsdPhysics.CollisionAPI(
        source_collisions[0]).CreateCollisionEnabledAttr().Set(False)
    collision_box_path = p.GetPath().AppendChild('Task53CollisionBox')
    if s.GetPrimAtPath(collision_box_path).IsValid():
        s.RemovePrim(collision_box_path)
    collision_box = UsdGeom.Cube.Define(s, collision_box_path)
    collision_box.CreateSizeAttr(2.0)
    collision_box.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision_box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.02))
    collision_box.AddScaleOp().Set(Gf.Vec3f(0.0825, 0.065, 0.02))
    UsdPhysics.CollisionAPI.Apply(
        collision_box.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    p.CreateAttribute(
        'real2sim:collisionGeometryVersion',
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set('task53_wallet_box_v1')
_target_collision_approximation = {target_collision_approximation!r}
if _target_collision_approximation is not None:
    target_collision_meshes = [
        prim for prim in Usd.PrimRange(p)
        if prim.GetTypeName() == 'Mesh' and prim.IsActive()
    ]
    if len(target_collision_meshes) != 1:
        raise RuntimeError(
            'target collision policy requires exactly one active mesh')
    target_collision_mesh = target_collision_meshes[0]
    target_collision_mesh.CreateAttribute(
        'physics:approximation', Sdf.ValueTypeNames.Token).Set(
            _target_collision_approximation)
    if _target_collision_approximation == 'sdf':
        target_collision_mesh.CreateAttribute(
            'physxSDFMeshCollision:sdfResolution',
            Sdf.ValueTypeNames.Int).Set(256)
# flatten: find the x/y tilt that MINIMIZES bbox z-extent (true lying-flat pose — zeroing
# tilt is NOT enough when the asset's local frame is itself tilted, e.g. Remote 7.4cm
# "thick" at rotXYZ(0,0,yaw)). Grid-search rotX/rotY, keep yaw. rotateXYZ prims only;
# orient prims get yaw-projected (their assets are authored flat).
# bound = manual union of ACTIVE Mesh leaves, extentsHint ignored (converted FBX assets
# carry stale authored extents on ancestors that cover deleted/deactivated siblings)
def _bound(prim):
    import math
    c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    lo = [1e9]*3; hi = [-1e9]*3
    for pr in Usd.PrimRange(prim):
        if pr.GetTypeName() != 'Mesh' or not pr.IsActive():
            continue
        b = c.ComputeWorldBound(pr).ComputeAlignedRange()
        if b.IsEmpty():
            continue
        mn, mx = b.GetMin(), b.GetMax()
        if not all(math.isfinite(v) for v in (*mn, *mx)):
            continue
        for i in range(3):
            lo[i] = min(lo[i], mn[i]); hi[i] = max(hi[i], mx[i])
    return (lo, hi) if lo[0] < 1e8 else (None, None)
def _zext(prim):
    lo, hi = _bound(prim)
    return (hi[2] - lo[2]) if lo else None
def _pinch_extent(prim, ax, ay):
    import math
    lo = 1e9; hi = -1e9
    for pr in Usd.PrimRange(prim):
        if pr.GetTypeName() != 'Mesh' or not pr.IsActive():
            continue
        points = UsdGeom.Mesh(pr).GetPointsAttr().Get()
        if not points:
            continue
        matrix = UsdGeom.Imageable(
            pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in points:
            world = matrix.Transform(point)
            value = float(world[0]) * ax + float(world[1]) * ay
            if not math.isfinite(value):
                raise RuntimeError('non-finite target mesh point')
            lo = min(lo, value); hi = max(hi, value)
    return (hi - lo) if lo < 1e8 else None
rop = None
for op in UsdGeom.Xformable(p).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rop = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        q = op.Get()
        import math
        w, z = float(q.GetReal()), float(q.GetImaginary()[2])
        n = math.hypot(w, z) or 1.0
        op.Set(type(q)(w/n, 0, 0, z/n))
if _requested_local_pinch_axis is not None and rop is None:
    raise RuntimeError(
        'PICK_LOCAL_PINCH_AXIS requires a target rotateXYZ yaw op')
if _task59_side_rest and rop is None:
    raise RuntimeError(
        'PICK_GLASSES_CASE_SIDE_REST requires a target rotateXYZ op')
if rop is not None:
    _canonical_yaw_seed = {cfg.get("canonical_yaw_seed")!r}
    yaw = (
        rop.Get()[2]
        if _canonical_yaw_seed is None
        else float(_canonical_yaw_seed)
    )
    if _task59_side_rest:
        # All three pinned Task59 right-hand videos show the 160x63x37 mm
        # GlassesCase standing on a long side: local Y is vertical and local Z
        # (the 37 mm thickness) is horizontal.  +90 deg about local X chooses
        # local-Y-min as the support face.  Runtime geometry checks below bind
        # this authored-axis assumption and fail closed if the asset changes.
        rop.Set(Gf.Vec3f(90.0, 0.0, yaw))
    else:
        # flip(t57 黑眼镜盒): 平躺搜索以 rotX=180 为中心 → 资产另一面(黑面)朝上平躺;
        # 搜索半径 ±90° 不会翻回正面
        _flip_rx = {180.0 if cfg.get('flip') else 0.0}
        best = (1e9, _flip_rx, 0.0)
        for step, ctr, rng in ((15.0, (_flip_rx, 0.0), 6), (5.0, None, 3), (1.0, None, 3)):
            cx, cy = ctr if ctr is not None else (best[1], best[2])
            for rx in [cx + i*step for i in range(-rng, rng+1)]:
                for ry in [cy + j*step for j in range(-rng, rng+1)]:
                    rop.Set(Gf.Vec3f(rx, ry, yaw))
                    e = _zext(p)
                    if e is not None and 1e-4 < e < 2.0 and e < best[0]:
                        best = (e, rx, ry)
        rop.Set(Gf.Vec3f(best[1], best[2], yaw))
# Match the object's narrowest physical horizontal projection to the recorded
# gripper's physical pad-to-pad axis.  Retaining the asset's arbitrary authored
# yaw made Wallet span about 206 mm across a 129 mm jaw and produced only
# same-direction bridge forces.  This deterministic mesh projection uses no
# pose changes after the episode starts.
pinch_width = None
pinch_yaw = None
pinch_alignment_abs_dot = None
pinch_alignment_error_deg = None
if rop is not None:
    pose = rop.Get()
    rx, ry, yaw0 = float(pose[0]), float(pose[1]), float(pose[2])
    def _yaw_distance(a, b):
        return abs(((a - b + 180.0) % 360.0) - 180.0)
    _requested_world_yaw = {requested_target_yaw_world_deg!r}
    if _requested_local_pinch_axis is None and _requested_world_yaw is not None:
        # Vision-calibrated absolute orientation: stage the REAL frame-0 yaw
        # verbatim.  The narrowest-span search below intentionally does NOT
        # run — reproducing the recorded scene outranks presenting the
        # narrowest width to the jaw, and the eccentric-bite point (edge
        # offset) carries the real pinch-to-body geometry instead.
        pinch_yaw = float(_requested_world_yaw)
        rop.Set(Gf.Vec3f(rx, ry, pinch_yaw))
        pinch_width = _pinch_extent(p, {pax}, {pay})
    elif _requested_local_pinch_axis is None:
        yaw_best = (1e9, 1e9, yaw0)
        for yaw_candidate in [float(value) for value in range(-180, 180, 5)]:
            rop.Set(Gf.Vec3f(rx, ry, yaw_candidate))
            extent = _pinch_extent(p, {pax}, {pay})
            if extent is not None:
                yaw_best = min(
                    yaw_best,
                    (extent, _yaw_distance(yaw_candidate, yaw0), yaw_candidate),
                )
        coarse_yaw = yaw_best[2]
        for step in range(-6, 7):
            yaw_candidate = coarse_yaw + 0.5 * step
            rop.Set(Gf.Vec3f(rx, ry, yaw_candidate))
            extent = _pinch_extent(p, {pax}, {pay})
            if extent is not None:
                yaw_best = min(
                    yaw_best,
                    (extent, _yaw_distance(yaw_candidate, yaw0), yaw_candidate),
                )
        if yaw_best[0] >= 1e8:
            raise RuntimeError('could not measure target pinch projection')
        pinch_width, _, pinch_yaw = yaw_best
        pinch_yaw += {initial_yaw_offset_deg!r}
        _yaw_reference = {initial_yaw_reference_deg!r}
        if _yaw_reference is not None:
            # The Remote has antipodal equal-width pinches, but its button
            # layout is not invariant under 180 degrees.  Resolve the
            # floating-point tie against the audited real frame.
            representatives = (pinch_yaw, pinch_yaw + 180.0)
            pinch_yaw = min(
                representatives,
                key=lambda value: abs(
                    ((value - _yaw_reference + 180.0) % 360.0) - 180.0),
            )
            pinch_yaw = ((pinch_yaw + 180.0) % 360.0) - 180.0
        rop.Set(Gf.Vec3f(rx, ry, pinch_yaw))
        pinch_width = _pinch_extent(p, {pax}, {pay})
        if pinch_width is None:
            raise RuntimeError('could not measure yaw-corrected pinch projection')
    else:
        import math
        pad_axis = [float(value) for value in _pad_axis_world]
        pad_norm = math.sqrt(sum(value * value for value in pad_axis))
        if not math.isfinite(pad_norm) or pad_norm < 1e-8:
            raise RuntimeError(
                'PICK_LOCAL_PINCH_AXIS requires a finite live 3-D pad axis')
        pad_axis = [value / pad_norm for value in pad_axis]

        def _axis_alignment(yaw_candidate):
            rop.Set(Gf.Vec3f(rx, ry, yaw_candidate))
            matrix = UsdGeom.Imageable(
                p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            origin = matrix.Transform(Gf.Vec3d(0, 0, 0))
            endpoint = matrix.Transform(Gf.Vec3d(
                float(_requested_local_pinch_axis[0]),
                float(_requested_local_pinch_axis[1]),
                float(_requested_local_pinch_axis[2]),
            ))
            direction = [
                float(endpoint[i] - origin[i]) for i in range(3)]
            norm = math.sqrt(sum(value * value for value in direction))
            if not math.isfinite(norm) or norm < 1e-8:
                raise RuntimeError(
                    'PICK_LOCAL_PINCH_AXIS target transform is degenerate')
            direction = [value / norm for value in direction]
            return abs(sum(
                direction[i] * pad_axis[i] for i in range(3)))

        alignment_best = (1e9, 1e9, yaw0)
        for yaw_candidate in [
                float(value) for value in range(-180, 180)]:
            dot = _axis_alignment(yaw_candidate)
            alignment_best = min(
                alignment_best,
                (1.0 - dot, _yaw_distance(yaw_candidate, yaw0),
                 yaw_candidate),
            )
        coarse_yaw = alignment_best[2]
        for step in range(-10, 11):
            yaw_candidate = coarse_yaw + 0.1 * step
            dot = _axis_alignment(yaw_candidate)
            alignment_best = min(
                alignment_best,
                (1.0 - dot, _yaw_distance(yaw_candidate, yaw0),
                 yaw_candidate),
            )
        pinch_alignment_abs_dot = max(
            0.0, min(1.0, 1.0 - alignment_best[0]))
        pinch_alignment_error_deg = math.degrees(math.acos(
            pinch_alignment_abs_dot))
        if pinch_alignment_error_deg > 5.0:
            raise RuntimeError(
                'PICK_LOCAL_PINCH_AXIS cannot be met by target yaw: '
                + 'best error ' + f'{{pinch_alignment_error_deg:.2f}} deg; '
                + 'local-z pinch needs a wrist/trajectory pose')
        pinch_yaw = alignment_best[2]
        if _edge_offset_local is not None:
            # Axis matching above is deliberately antipodal: yaw and yaw+180
            # describe the same physical pad line.  The explicit Wallet
            # corner canary is not antipodal, however; one representative
            # puts the requested corner on the robot-facing side and the
            # other puts it behind the object.  Resolve that otherwise
            # unstable tie from the fixed family approach before placement.
            # The runner independently checks the emitted world delta, so a
            # stale/wrong approach still fails closed before playback.
            def _edge_approach_dot_at_yaw(yaw_candidate):
                rop.Set(Gf.Vec3f(rx, ry, yaw_candidate))
                matrix = UsdGeom.Imageable(
                    p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                origin = matrix.Transform(Gf.Vec3d(0, 0, 0))
                edge = matrix.Transform(Gf.Vec3d(
                    float(_edge_offset_local[0]),
                    float(_edge_offset_local[1]),
                    0.0,
                ))
                dx = float(edge[0] - origin[0])
                dy = float(edge[1] - origin[1])
                value = dx * {ax} + dy * {ay}
                if not math.isfinite(value):
                    raise RuntimeError(
                        'PICK_LOCAL_EDGE_OFFSET near-side test is non-finite')
                return value

            if _edge_approach_dot_at_yaw(pinch_yaw) >= 0.0:
                pinch_yaw += 180.0
            if not _edge_approach_dot_at_yaw(pinch_yaw) < 0.0:
                raise RuntimeError(
                    'PICK_LOCAL_EDGE_OFFSET has no near-side yaw representative')
        rop.Set(Gf.Vec3f(rx, ry, pinch_yaw))
        pinch_width = _pinch_extent(p, {pax}, {pay})
        if pinch_width is None:
            raise RuntimeError('could not measure target pinch projection')
if _task53_real_scene:
    pinch_yaw = float(_task53_wallet_yaw_deg)
    rop.Set(Gf.Vec3f(rx, ry, pinch_yaw))
    pinch_width = _pinch_extent(p, {pax}, {pay})
    if pinch_width is None:
        raise RuntimeError(
            'could not measure Task53 RGB-calibrated pinch projection')
side_rest_rotate_xyz = None
side_rest_support_face = None
side_rest_vertical_extent = None
side_rest_long_extent = None
side_rest_axis_alignment_abs_dot = None
side_rest_long_axis_world_delta = None
if _task59_side_rest:
    import math
    pose = rop.Get()
    side_rest_rotate_xyz = tuple(float(value) for value in pose)
    side_rest_vertical_extent = _zext(p)
    side_rest_long_extent = _pinch_extent(p, {-pay}, {pax})
    matrix = UsdGeom.Imageable(
        p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = matrix.Transform(Gf.Vec3d(0, 0, 0))

    def _local_axis_world(x, y, z):
        endpoint = matrix.Transform(Gf.Vec3d(x, y, z))
        vector = [
            float(endpoint[i] - origin[i]) for i in range(3)]
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm < 1e-8:
            raise RuntimeError('Task59 side-rest target transform is degenerate')
        return [value / norm for value in vector]

    local_y_world = _local_axis_world(0, 1, 0)
    local_z_world = _local_axis_world(0, 0, 1)
    local_x_world = _local_axis_world(1, 0, 0)
    local_y_vertical_abs_dot = abs(local_y_world[2])
    side_rest_support_face = (
        'local_y_min' if local_y_world[2] > 0.0 else 'local_y_max')
    side_rest_axis_alignment_abs_dot = abs(
        local_z_world[0] * {pax} + local_z_world[1] * {pay})
    side_rest_long_axis_world_delta = [
        _task59_long_axis_offset * local_x_world[i]
        for i in range(3)]
    if (
            abs(side_rest_long_axis_world_delta[2]) > 1e-6
            or abs(
                math.sqrt(sum(
                    value * value
                    for value in side_rest_long_axis_world_delta))
                - _task59_long_axis_offset
            ) > 1e-6):
        raise RuntimeError(
            'Task59 long-axis offset is not a horizontal composed-local shift')
    if local_y_vertical_abs_dot < 0.995:
        raise RuntimeError(
            'Task59 side-rest local Y is not vertical')
    if side_rest_axis_alignment_abs_dot < 0.995:
        raise RuntimeError(
            'Task59 side-rest local Z is not aligned with the pad axis')
    if not (
            side_rest_vertical_extent is not None
            and 0.050 < side_rest_vertical_extent < 0.075):
        raise RuntimeError(
            'Task59 side-rest expected a 50-75 mm local-Y height')
    if not (
            pinch_width is not None
            and 0.030 < pinch_width < 0.045):
        raise RuntimeError(
            'Task59 side-rest expected a 30-45 mm local-Z pinch width')
    if not (
            side_rest_long_extent is not None
            and 0.140 < side_rest_long_extent < 0.180
            and pinch_width < side_rest_vertical_extent
            < side_rest_long_extent):
        raise RuntimeError(
            'Task59 side-rest expected ordered 37x63x160 mm axes')
mn, mx = _bound(p)
assert mn is not None, 'no active mesh bound for {tp}'
cur = [(mn[i]+mx[i])/2 for i in range(3)]
obj_h = mx[2] - mn[2]
_target_center_x, _target_center_y = {tx}, {ty}
_target_center_x += {pax!r} * {pad_axis_offset!r}
_target_center_y += {pay!r} * {pad_axis_offset!r}
if _task59_long_axis_offset:
    _target_center_x += side_rest_long_axis_world_delta[0]
    _target_center_y += side_rest_long_axis_world_delta[1]
_edge_world_delta = None
if _edge_offset_local is not None:
    import math
    matrix = UsdGeom.Imageable(
        p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local_origin_world = matrix.Transform(Gf.Vec3d(0, 0, 0))
    local_edge_world = matrix.Transform(Gf.Vec3d(
        float(_edge_offset_local[0]),
        float(_edge_offset_local[1]),
        0.0,
    ))
    if math.hypot(
            float(local_origin_world[0]) - cur[0],
            float(local_origin_world[1]) - cur[1]) > 1e-4:
        raise RuntimeError(
            'PICK_LOCAL_EDGE_OFFSET requires Wallet local origin at XY bound centre')
    _edge_world_delta = [
        float(local_edge_world[i] - local_origin_world[i])
        for i in range(3)
    ]
    if abs(_edge_world_delta[2]) > 1e-4:
        raise RuntimeError(
            'PICK_LOCAL_EDGE_OFFSET is not horizontal after target flattening')
    # Put the requested local point at the measured pad midpoint.  Moving the
    # object centre in the opposite direction is what makes this an eccentric
    # edge/corner bite rather than another centred full-width grasp.
    # Preserve any independently reviewed lateral pad-axis correction already
    # applied above.  Re-anchoring to the raw trajectory target here silently
    # cancelled PICK_PAD_AXIS_OFFSET whenever an eccentric Wallet bite was
    # active, so the emitted controls and the authored initial state disagreed.
    _target_center_x -= _edge_world_delta[0]
    _target_center_y -= _edge_world_delta[1]
if _task53_real_scene:
    # The rounded-cushion contract pins one exact Wallet centre reconstructed
    # from the three REAL views.  The FK close point printed by align_grasp is
    # rounded to millimetres, so reconstructing the centre from that display
    # value drifts by several millimetres and breaks the exact support
    # relationship.  Use the calibrated scene centre as the authoring
    # authority; the episode-bound local grasp offset above remains audit
    # evidence for the gripper/Wallet relationship.
    _target_center_x = float(_task53_wallet_translate_xyz[0])
    _target_center_y = float(_task53_wallet_translate_xyz[1])
# Restore required direct-contact scene context before target placement.
# Task56's three preferred real episodes respectively show the Remote resting
# on a Note, Wallet, and Book.  Their exact episode-bound visible assets replace
# the old one-Note-for-all approximation.
contact_support_top = None
contact_support_bounds = None
if _task_contact_support is not None:
    import math
    support_prim = s.GetPrimAtPath(
        '/World/Assets/' + _task_contact_support['support_prim'])
    if not support_prim.IsActive():
        support_prim.SetActive(True)
    if support_prim.GetName() == 'Wallet':
        _restore_canonical_wallet_collision(support_prim)
    _support_scale_ops = [
        op for op in UsdGeom.Xformable(
            support_prim).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(_support_scale_ops) != 1:
        raise RuntimeError(
            'required contact support must have one canonical root scale op')
    _canonical_support_scale = _canonical_root_scales.get(
        support_prim.GetName())
    if _canonical_support_scale is None:
        raise RuntimeError(
            'missing canonical root scale for contact support '
            + support_prim.GetName())
    _support_scale_ops[0].Set(Gf.Vec3f(*_canonical_support_scale))
    support_prim.CreateAttribute(
        'physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(True)
    for _nm in ('physics:velocity', 'physics:angularVelocity'):
        _a = support_prim.GetAttribute(_nm)
        if _a and _a.Get() is not None:
            _a.Set(Gf.Vec3f(0, 0, 0))
    yaw_rad = math.radians(
        _task_contact_support['support_yaw_deg'])
    yaw_quat = Gf.Quatf(
        math.cos(yaw_rad / 2.0), 0.0, 0.0,
        math.sin(yaw_rad / 2.0))
    support_rotation_set = False
    for op in UsdGeom.Xformable(
            support_prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(yaw_quat)
            support_rotation_set = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0.0, 0.0, _task_contact_support[
                'support_yaw_deg']))
            support_rotation_set = True
    if not support_rotation_set:
        raise RuntimeError(
            'required contact support has no supported rotation op')
    smn, smx = _bound(support_prim)
    assert smn is not None, 'required contact support has no mesh bound'
    scur = [(smn[i] + smx[i]) / 2 for i in range(3)]
    soff = _task_contact_support['support_center_offset_xy_m']
    sdx = _target_center_x + soff[0] - scur[0]
    sdy = _target_center_y + soff[1] - scur[1]
    sdz = _task_contact_support['support_bottom_m'] - smn[2]
    for op in UsdGeom.Xformable(
            support_prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            op.Set(Gf.Vec3d(
                t[0] + sdx, t[1] + sdy, t[2] + sdz))
    smn, smx = _bound(support_prim)
    support_size = tuple(float(smx[i] - smn[i]) for i in range(3))
    expected_size = _task_contact_support['support_size_m']
    # Validate the world AABB expected from the true local dimensions at the
    # calibrated yaw.  Comparing a rotated rectangle's world AABB directly to
    # its unrotated real dimensions would be wrong.
    cabs, sabs = abs(math.cos(yaw_rad)), abs(math.sin(yaw_rad))
    expected_world_aabb = (
        cabs * expected_size[0] + sabs * expected_size[1],
        sabs * expected_size[0] + cabs * expected_size[1],
        expected_size[2],
    )
    if any(abs(support_size[i] - expected_world_aabb[i]) > 0.002
           for i in range(3)):
        raise RuntimeError(
            'required contact support size differs from calibration')
    contact_support_top = float(smx[2])
    if abs(
            contact_support_top
            - _task_contact_support['support_top_m']) > 0.001:
        raise RuntimeError(
            'required contact support top differs from calibration')
    contact_support_bounds = (
        tuple(float(value) for value in smn),
        tuple(float(value) for value in smx),
    )
    stale_riser = s.GetPrimAtPath('/World/Assets/SeatRiser')
    if stale_riser and stale_riser.IsValid() and stale_riser.IsActive():
        stale_riser.SetActive(False)

# note-calibrated convention: object TOP sits BITE_Z above the fingertip-close midpoint gz
# (v18 note: riser_drop 0.006 + 1.5cm note = top gz+9mm -> fingers bite INTO the body).
support = (
    contact_support_top
    if contact_support_top is not None
    else {surface_z} + 0.004
)
# unified convention (FK==PD verified within 1.2cm): object TOP = gz + BITE_Z. When the
# real grasp height is above the sim surface (soft seat / taller real table), a static
# riser bridges the gap at the interaction point (background furniture stays put).
# Task59's real tabletop is independently calibrated from all three pinned
# RGB-D episodes.  Its side-rest object must sit on that one constant plane;
# deriving the support from this episode's grasp anchor would leak the answer
# into scene geometry and make cross-episode success incomparable.
bottom = (
    _task59_support_top if _task59_side_rest
    else {gz} + {os.environ.get("BITE_Z", "0.012")} - {grab_frac} * obj_h
)
if _task53_real_scene:
    # The current rounded-cushion calibration pins the Wallet bottom directly.
    # Its visible source mesh is thicker than the collision proxy, so deriving
    # this value from the visible bound would place it 15 mm inside the pillow.
    bottom = float(_task53_wallet_translate_xyz[2])
task53_cushion_top = None
task53_cushion_bounds = None
if _task53_real_scene:
    stale_riser = s.GetPrimAtPath('/World/Assets/SeatRiser')
    if stale_riser and stale_riser.IsValid() and stale_riser.IsActive():
        stale_riser.SetActive(False)
    cushion = s.GetPrimAtPath('/World/Assets/CushionOnBook')
    if not cushion or not cushion.IsValid():
        raise RuntimeError('Task53 visible support cushion is missing')
    if not cushion.IsActive():
        cushion.SetActive(True)
    cushion.CreateAttribute(
        'physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(True)
    cushion.CreateAttribute(
        'physics:rigidBodyEnabled', Sdf.ValueTypeNames.Bool).Set(True)
    for _nm in ('physics:velocity', 'physics:angularVelocity'):
        _a = cushion.GetAttribute(_nm)
        if _a and _a.Get() is not None:
            _a.Set(Gf.Vec3f(0, 0, 0))
    cushion.CreateAttribute(
        'real2sim:geometryVersion',
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set('task53_rounded_pillow_v2')
    cushion_translate = None
    cushion_orient = None
    cushion_scale = None
    for op in UsdGeom.Xformable(cushion).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            cushion_translate = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            cushion_orient = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            cushion_scale = op
    if (
            cushion_translate is None
            or cushion_orient is None
            or cushion_scale is None):
        raise RuntimeError(
            'Task53 visible support cushion has incomplete transform ops')
    cushion_translate.Set(Gf.Vec3d(
        *_task53_cushion_translate_xyz))
    cushion_orient.Set(Gf.Quatf(
        float(_task53_cushion_orient_wxyz[0]),
        float(_task53_cushion_orient_wxyz[1]),
        float(_task53_cushion_orient_wxyz[2]),
        float(_task53_cushion_orient_wxyz[3]),
    ))
    cushion_scale.Set(Gf.Vec3f(1, 1, 1))

    # The source cushion-set member has its own tilted presentation transform,
    # which made the previous SIM secondary object stand like a tent.  Keep
    # the source reference for provenance but deactivate that presentation
    # subtree and author one bounded rounded rectangular pillow whose top is
    # a real visible/colliding surface (never an invisible replacement).
    legacy_cushion_obj = s.GetPrimAtPath(
        '/World/Assets/CushionOnBook/Obj')
    if (
            legacy_cushion_obj
            and legacy_cushion_obj.IsValid()
            and legacy_cushion_obj.IsActive()):
        legacy_cushion_obj.SetActive(False)

    import math

    def _rounded_boundary(hx, hy, radius, segments=8):
        result = []
        corners = (
            (hx - radius, hy - radius, 0.0),
            (-hx + radius, hy - radius, 90.0),
            (-hx + radius, -hy + radius, 180.0),
            (hx - radius, -hy + radius, 270.0),
        )
        for center_x, center_y, start_deg in corners:
            for index in range(segments):
                angle = math.radians(
                    start_deg + 90.0 * index / segments)
                result.append((
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                ))
        return result

    top_ring = _rounded_boundary(0.215, 0.165, 0.035)
    outer_ring = _rounded_boundary(0.230, 0.180, 0.045)
    rings = (
        (top_ring, 0.050),
        (outer_ring, 0.035),
        (outer_ring, -0.035),
        (top_ring, -0.050),
    )
    ring_size = len(top_ring)
    points = [(0.0, 0.0, 0.050)]
    for ring, z_value in rings:
        points.extend(
            (float(x_value), float(y_value), float(z_value))
            for x_value, y_value in ring)
    bottom_center_index = len(points)
    points.append((0.0, 0.0, -0.050))
    face_counts = []
    face_indices = []
    for index in range(ring_size):
        next_index = (index + 1) % ring_size
        face_counts.append(3)
        face_indices.extend((0, 1 + index, 1 + next_index))
    for ring_index in range(len(rings) - 1):
        first = 1 + ring_index * ring_size
        second = first + ring_size
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            face_counts.append(4)
            face_indices.extend((
                first + index,
                second + index,
                second + next_index,
                first + next_index,
            ))
    bottom_start = 1 + (len(rings) - 1) * ring_size
    for index in range(ring_size):
        next_index = (index + 1) % ring_size
        face_counts.append(3)
        face_indices.extend((
            bottom_center_index,
            bottom_start + next_index,
            bottom_start + index,
        ))
    pillow = UsdGeom.Mesh.Define(
        s, '/World/Assets/CushionOnBook/Task53Pillow')
    pillow.CreatePointsAttr(points)
    pillow.CreateFaceVertexCountsAttr(face_counts)
    pillow.CreateFaceVertexIndicesAttr(face_indices)
    pillow.CreateSubdivisionSchemeAttr().Set('none')
    pillow.CreateOrientationAttr().Set('rightHanded')
    pillow.CreateExtentAttr().Set([
        Gf.Vec3f(-0.230, -0.180, -0.050),
        Gf.Vec3f(0.230, 0.180, 0.050),
    ])
    pillow_prim = pillow.GetPrim()
    collision_api = UsdPhysics.CollisionAPI.Apply(pillow_prim)
    collision_api.CreateCollisionEnabledAttr().Set(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(pillow_prim)
    mesh_collision_api.CreateApproximationAttr().Set('convexHull')
    physics_binding = pillow_prim.CreateRelationship(
        'material:binding:physics')
    physics_binding.SetTargets([
        Sdf.Path('/World/PhysicsMaterials/CushionMat')])
    physics_binding.SetMetadata(
        'bindMaterialAs', 'strongerThanDescendants')
    uv_values = [
        Gf.Vec2f(
            float(point[0] / 0.115 + 2.0),
            float(point[1] / 0.090 + 2.0),
        )
        for point in points
    ]
    UsdGeom.PrimvarsAPI(pillow_prim).CreatePrimvar(
        'st0',
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    ).Set(uv_values)
    cushion_box = UsdGeom.Imageable(cushion).ComputeWorldBound(
        0, UsdGeom.Tokens.default_).ComputeAlignedBox()
    cmn, cmx = cushion_box.GetMin(), cushion_box.GetMax()
    if cushion_box.IsEmpty():
        raise RuntimeError(
            'Task53 visible support cushion has no active mesh bound')
    cushion_size = [
        float(cmx[i] - cmn[i]) for i in range(3)]
    if not (
            0.35 <= cushion_size[0] <= 0.60
            and 0.35 <= cushion_size[1] <= 0.60
            and 0.099 <= cushion_size[2] <= 0.101):
        raise RuntimeError(
            'Task53 visible support is not pillow-scale: '
            + str(cushion_size))
    task53_cushion_top = float(cmx[2])
    task53_cushion_bounds = (
        tuple(float(value) for value in cmn),
        tuple(float(value) for value in cmx),
    )
    if (
            abs(task53_cushion_top - bottom) > 1e-6
            or abs(
                float(cushion_translate.Get()[0])
                - _target_center_x) > 1e-6
            or abs(
                float(cushion_translate.Get()[1])
                - _target_center_y) > 1e-6
            or not cmn[0] < _target_center_x < cmx[0]
            or not cmn[1] < _target_center_y < cmx[1]):
        raise RuntimeError(
            'Task53 Wallet is not centred on the visible cushion support: '
            + 'target_center='
            + str((_target_center_x, _target_center_y))
            + ' cushion_translate='
            + str(tuple(float(value) for value in cushion_translate.Get()))
            + ' cushion_bounds='
            + str(task53_cushion_bounds)
            + ' cushion_top=' + str(task53_cushion_top)
            + ' target_bottom=' + str(bottom))
    support = task53_cushion_top
if {use_riser}:
    rbase = min({surface_z}, bottom - 0.001)
    riser_h = bottom - rbase
    r = s.GetPrimAtPath('/World/Assets/SeatRiser')
    if riser_h < 0.008 or bottom <= support + 0.003:
        if r and r.IsValid() and r.IsActive():
            r.SetActive(False)   # 不需要垫: 收起, 免得上集残留挡手指
    elif riser_h < 0.20:
        if not r.IsActive(): r.SetActive(True)
        # edge-overhang trick (book-pipeline calibrated +0.055): shift the riser AWAY from
        # the robot along the approach axis so the grasped edge overhangs free air -- a
        # centered riser blocks the closing fingers (static collider, tipgap pins ~0.09).
        # EDGE_OVERHANG env: thin slabs (remote/gc) need a bigger overhang so the lower
        # finger clears under the slab edge instead of skimming the top (bulldoze).
        _eo = {float(os.environ.get("EDGE_OVERHANG", "0.055"))}
        for op in UsdGeom.Xformable(r).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(
                    _target_center_x + {ax}*_eo,
                    _target_center_y + {ay}*_eo,
                    rbase + riser_h/2))
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                sc = op.Get(); op.Set(Gf.Vec3f(sc[0], sc[1], riser_h))
        support = rbase + riser_h
if {force_native_surface_contact}:
    # The frame-0 state is the object resting on the calibrated, visible
    # native table.  Do not preserve the old grasp-anchor-derived floating
    # height after disabling SeatRiser; playback settle may close the small
    # 4 mm spawn clearance, but it must not be asked to fall from a fake shelf.
    bottom = support
elif {drawer_pick_mode}:
    # Drawer pick: the target starts on the drawer's interior floor, below
    # the tabletop support plane.
    bottom = {drawer_target_bottom_z!r}
else:
    bottom = max(bottom, support)
_episode_support_items = {episode_support_items!r}
for _esi in _episode_support_items:
    _sp = s.GetPrimAtPath('/World/Assets/' + _esi['prim'])
    assert _sp and _sp.IsValid(), 'missing support prim ' + _esi['prim']
    if not _sp.IsActive():
        _sp.SetActive(True)
    _sp.CreateAttribute(
        'physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(
            bool(_esi.get('kinematic', False)))
    _esi_d = float(_esi['body_diameter_m'])
    # center_offset is in the TARGET's yaw frame (x = target long axis):
    # a world-frame offset silently walks off the target when the episode
    # yaw differs from the reviewer's mental frame.
    _esi_off = _esi['center_offset_from_target_xy_m']
    _esi_tyaw = 0.0
    for _op2 in UsdGeom.Xformable(p).GetOrderedXformOps():
        if _op2.GetOpType() == UsdGeom.XformOp.TypeOrient:
            _esi_tq = _op2.Get()
            _esi_ti = _esi_tq.GetImaginary()
            _esi_tyaw = math.atan2(
                2.0 * (_esi_tq.GetReal() * _esi_ti[2]
                       + _esi_ti[0] * _esi_ti[1]),
                1.0 - 2.0 * (_esi_ti[1] * _esi_ti[1]
                             + _esi_ti[2] * _esi_ti[2]))
        elif _op2.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            _esi_tyaw = math.radians(float(_op2.Get()[2]))
    _esi_ca, _esi_sa = math.cos(_esi_tyaw), math.sin(_esi_tyaw)
    _esi_wox = _esi_ca * float(_esi_off[0]) - _esi_sa * float(_esi_off[1])
    _esi_woy = _esi_sa * float(_esi_off[0]) + _esi_ca * float(_esi_off[1])
    if _esi.get('pose') == 'lean':
        # ladder-lean: bottom rim on the table, body resting on the target's
        # flank.  Statically stable rigid contact (rim + flank + friction),
        # matching the real arrangement -- no balancing act needed.
        _esi_l = float(_esi['body_length_m'])
        _esi_t = math.radians(float(_esi.get('tilt_deg', 35.0)))
        _esi_az = math.degrees(math.atan2(-_esi_woy, -_esi_wox))
        _esi_q = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), _esi_az)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(_esi_t))
        ).GetQuat()
        # rim plane: the real table top from the episode contract when given
        # (riser/bite tasks float the target above it -- 'bottom' is the
        # riser top there, and surface_z can sit below the visible table).
        _esi_base = float(_esi.get('rest_plane_z_m', bottom))
        _esi_lean_z = (
            _esi_base + _esi_l / 2.0 * math.cos(_esi_t)
            + _esi_d / 2.0 * math.sin(_esi_t))
    else:
        _esi_rot = _esi.get('lie_rotate_xyz_deg') or (0.0, 0.0, 0.0)
        _esi_q = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), float(_esi_rot[2]))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(_esi_rot[1]))
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), float(_esi_rot[0]))
        ).GetQuat()
        _esi_lean_z = None
    _esi_xf = UsdGeom.Xformable(_sp)
    # the contract speaks WORLD coordinates; the prim may live under a scope
    # with its own transform (e.g. ShelfItems translate) -- convert.
    _esi_p2w = UsdGeom.XformCache().GetLocalToWorldTransform(_sp.GetParent())
    _esi_w2p = _esi_p2w.GetInverse()
    _esi_pq = _esi_p2w.ExtractRotationQuat()
    _esi_lq = _esi_pq.GetInverse() * _esi_q
    for _op in _esi_xf.GetOrderedXformOps():
        if _op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            if _esi_lean_z is not None:
                _esi_z = _esi_lean_z
            elif _esi.get('relation') == 'resting_on_target':
                _esi_z = bottom + obj_h + _esi_d / 2.0
            else:
                _esi_z = {surface_z} + _esi_d / 2.0
            _op.Set(_esi_w2p.Transform(Gf.Vec3d(
                _target_center_x + _esi_wox,
                _target_center_y + _esi_woy,
                _esi_z,
            )))
        elif _op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            if _op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                _op.Set(Gf.Quatf(_esi_lq))
            else:
                _op.Set(_esi_lq)
    if _esi.get('relation') == 'supports_target':
        # target rests ON the support at the grasp anchor; its far end
        # settles to the table under physics (the real draped tilt).
        bottom = max(bottom, {surface_z} + _esi_d)
    if _esi.get('relation') == 'resting_on_target':
        # A rigid cylinder balanced on a rigid cylinder is a point contact and
        # rolls off during settle.  The real support surface is soft (fabric
        # cradles the resting item), so model exactly that: a compliant
        # high-friction contact material on both bodies plus a torsional
        # patch.  The item stays put until the gripper disturbs it -- no
        # welds, no kinematics.
        _esm = UsdShade.Material.Define(
            s, '/World/Assets/EpisodeSupportContactMat')
        _esm_pm = UsdPhysics.MaterialAPI.Apply(_esm.GetPrim())
        _esm_pm.CreateStaticFrictionAttr(4.0)
        _esm_pm.CreateDynamicFrictionAttr(4.0)
        _esm_pm.CreateRestitutionAttr(0.0)
        # host pxr has no PhysxSchema; author the PhysX attrs by name
        _esm.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
        _esm.GetPrim().CreateAttribute(
            'physxMaterial:compliantContactStiffness',
            Sdf.ValueTypeNames.Float).Set(3000.0)
        _esm.GetPrim().CreateAttribute(
            'physxMaterial:compliantContactDamping',
            Sdf.ValueTypeNames.Float).Set(300.0)
        for _esc_root in (p, _sp):
            for _esc in Usd.PrimRange(_esc_root):
                if not _esc.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(
                        _esc).GetCollisionEnabledAttr().Get() is False:
                    continue
                UsdShade.MaterialBindingAPI.Apply(_esc).Bind(
                    _esm, materialPurpose='physics')
                _esc.AddAppliedSchema('PhysxCollisionAPI')
                _esc.CreateAttribute(
                    'physxCollision:torsionalPatchRadius',
                    Sdf.ValueTypeNames.Float).Set(0.006)
                _esc.CreateAttribute(
                    'physxCollision:minTorsionalPatchRadius',
                    Sdf.ValueTypeNames.Float).Set(0.003)
    print('[episode-support] ' + _esi['prim'] + ' '
          + _esi.get('relation', '') + ' d=%.3f' % _esi_d, flush=True)
dz = bottom - mn[2]
dx, dy = _target_center_x - cur[0], _target_center_y - cur[1]
for op in UsdGeom.Xformable(p).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        t = op.Get(); op.Set(Gf.Vec3d(t[0]+dx, t[1]+dy, t[2]+dz))
contact_robot_facing_overhang = None
if _task_contact_support is not None:
    final_target_min, final_target_max = _bound(p)
    if final_target_min is None:
        raise RuntimeError('target has no final mesh bound')
    # Task56 approaches along world -X, so the robot-facing side is +X.  Some
    # reviewed episodes expose a measurable edge overhang; bind it when the
    # calibration declares one.  Other real episodes place the Remote wholly
    # on a larger support and are bound by the full centre offset instead.
    if abs({ax} + 1.0) > 1e-9 or abs({ay}) > 1e-9:
        raise RuntimeError(
            'Task56 contact support expected world -X approach')
    expected_robot_facing_overhang = _task_contact_support[
        'robot_facing_overhang_m']
    if expected_robot_facing_overhang is not None:
        contact_robot_facing_overhang = (
            float(final_target_max[0])
            - float(contact_support_bounds[1][0]))
        if abs(
                contact_robot_facing_overhang
                - expected_robot_facing_overhang) > 0.010:
            raise RuntimeError(
                'target/support robot-facing overhang differs from calibration')
side_rest_settle_z = None
if _task59_side_rest:
    final_mn, final_mx = _bound(p)
    if final_mn is None or abs(float(final_mn[2]) - bottom) > 1e-5:
        raise RuntimeError(
            'Task59 side-rest support face did not settle on its support')
    if abs(float(final_mn[2]) - _task59_support_top) > 1e-6:
        raise RuntimeError(
            'Task59 side-rest support differs from pinned RGB-D calibration')
    final_height = float(final_mx[2] - final_mn[2])
    if abs(final_height - side_rest_vertical_extent) > 1e-5:
        raise RuntimeError(
            'Task59 side-rest height changed during placement')
    side_rest_settle_z = float(final_mn[2])
# Evict unrelated co-props that overlap the target, but never evict a required
# close-contact support from the task scene.
for _n in ('Book', 'Note', 'CushionOnBook'):
    q = s.GetPrimAtPath('/World/Assets/' + _n)
    if not (q and q.IsValid() and q.IsActive()) or '/World/Assets/' + _n == '{tp}':
        continue
    if (
            _task_contact_support is not None
            and _n == _task_contact_support['support_prim']):
        continue
    if _task53_real_scene and _n == 'CushionOnBook':
        continue
    qmn, qmx = _bound(q)
    if qmn is None:
        continue
    if (qmn[0]-0.05 < _target_center_x < qmx[0]+0.05
            and qmn[1]-0.05 < _target_center_y < qmx[1]+0.05):
        for op in UsdGeom.Xformable(q).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get(); op.Set(Gf.Vec3d(t[0], t[1] + 0.45, t[2]))
# ALIGNMENT FIX (2026-07-22): a pick episode shows ONLY its target. The scene keeps every
# pick-family prop (GlassesCase/BaseballCap/PaperCup/Umbrella/Remote) active, so e.g. the glasses
# task rendered a big BaseballCap dome next to the case -> looked like the wrong object. Deactivate
# all sibling pick props so the sim scene matches the real single-object episode.
for _sn in {evict_siblings}:
    if (
            _task_contact_support is not None
            and _sn == _task_contact_support['support_prim']):
        continue
    _q = s.GetPrimAtPath('/World/Assets/' + _sn)
    if _q and _q.IsValid() and _q.IsActive():
        _q.SetActive(False)
_canary_report = ''
if _edge_offset_local is not None:
    _canary_report += (
        ' edge_offset_local=('
        + str(_edge_offset_local[0]) + ',' + str(_edge_offset_local[1])
        + ') edge_world_delta=('
        + ','.join(str(value) for value in _edge_world_delta) + ')')
if _requested_local_pinch_axis is not None:
    _canary_report += (
        ' local_pinch_axis=('
        + ','.join(str(value) for value in _requested_local_pinch_axis)
        + ') alignment_abs_dot=' + str(pinch_alignment_abs_dot)
        + ' alignment_error_deg=' + str(pinch_alignment_error_deg))
if _task59_side_rest:
    _canary_report += (
        ' side_rest_mode={_TASK59_SIDE_REST_MODE}'
        + ' side_rest_rotate_xyz=('
        + ','.join(str(value) for value in side_rest_rotate_xyz) + ')'
        + ' side_rest_support_face=' + side_rest_support_face
        + ' side_rest_vertical_extent='
        + str(side_rest_vertical_extent)
        + ' side_rest_long_extent=' + str(side_rest_long_extent)
        + ' side_rest_axis_alignment_abs_dot='
        + str(side_rest_axis_alignment_abs_dot)
        + ' side_rest_settle_z=' + str(side_rest_settle_z)
        + ' long_axis_offset=' + str(_task59_long_axis_offset)
        + ' long_axis_world_delta=('
        + ','.join(
            str(value) for value in side_rest_long_axis_world_delta)
        + ')')
if _task_contact_support is not None:
    _canary_report += (
        ' contact_support=' + _task_contact_support['support_prim']
        + ' contact_relation=' + _task_contact_support['relation']
        + ' contact_support_top=' + str(contact_support_top)
        + ' contact_robot_facing_overhang='
        + str(contact_robot_facing_overhang)
        + ' target_collision_approximation='
        + _task_contact_support['target_collision_approximation']
        + ' contact_support_bounds=' + str(contact_support_bounds))
if _task53_real_scene:
    _canary_report += (
        ' task53_support_mode={_TASK53_CUSHION_SUPPORT_MODE}'
        + ' task53_wallet_scale=('
        + ','.join(str(value) for value in _task53_wallet_scale_xyz)
        + ') task53_cushion_top=' + str(task53_cushion_top)
        + ' task53_cushion_bounds=' + str(task53_cushion_bounds))
{pick_drawer_scene_block}
if {drawer_pick_mode}:
    # Drawer pose: the wallet STANDS on its long edge in the drawer's LEFT
    # corner (left as the robot sees it = the world -Y wall) and the
    # recorded left closure pinches its exposed top edge (anchor z ~0.36 =
    # floor 0.205 + standing height 0.163).  A flat wallet cannot be caught
    # by that closure.  Override the generic tabletop pose AFTER all generic
    # placement: orientation first, then AABB-based seating.
    # The tilt is only 5 deg toward that wall: the wallet (0.165 m) is
    # TALLER than the drawer wall (0.134 m), so it pivots on the wall's top
    # edge instead of resting on its face, and past ~13.6 deg its centre of
    # mass leaves the 40 mm base and it topples in the pre-frame-zero settle
    # (5 of 7 episodes failed that gate at 12 deg).  The 40 mm thickness
    # stays on world Y, the recorded pinch axis.  rotateXYZ(0,-85,-90) is
    # the XYZ decomposition of Rx(5)*Ry(-90)*Rx(-90).
    _lean_rotated = False
    for op in UsdGeom.Xformable(p).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatf(0.521333804, -0.477714417,
                            -0.477714417, -0.521333804))
            _lean_rotated = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0.0, -85.0, -90.0))
            _lean_rotated = True
    assert _lean_rotated, 'drawer lean pose needs a rotation op'
    _lean_min, _lean_max = _bound(p)
    assert _lean_min is not None, 'drawer lean pose has no bounds'
    _lean_floor_z = {drawer_lean_floor_z!r}
    # Seat on the per-episode grasp anchor -- that anchor is the one direct
    # measurement of where this episode's wallet stood -- but never let it
    # intersect the left inner wall.  Most episodes' anchors already sit at
    # that wall, so this reproduces "leaning in the left corner" without
    # forcing every episode into the same spot.
    _lean_dx = {tx!r} - 0.5 * float(_lean_min[0] + _lean_max[0])
    _lean_dy = {ty!r} - 0.5 * float(_lean_min[1] + _lean_max[1])
    if float(_lean_min[1]) + _lean_dy < {drawer_lean_south_wall_y!r} + 0.002:
        _lean_dy = ({drawer_lean_south_wall_y!r} + 0.002) - float(_lean_min[1])
    # Stand it so its TOP EDGE lands 15 mm above the recorded pad-closure
    # height -- that anchor is the measurement of where the fingers close,
    # so this is what gives them a real 15 mm bite instead of skimming the
    # very tip.  A uniform support height cannot do this: each episode's
    # wallet rests on different contents, and one constant raise helped one
    # episode (74 mm lift) while starving two others.  Never below the
    # drawer floor.
    _lean_dz = ({gz!r} + 0.015) - float(_lean_max[2])
    if float(_lean_min[2]) + _lean_dz < _lean_floor_z + 0.001:
        _lean_dz = (_lean_floor_z + 0.001) - float(_lean_min[2])
    for op in UsdGeom.Xformable(p).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            op.Set(Gf.Vec3d(
                t[0] + _lean_dx, t[1] + _lean_dy, t[2] + _lean_dz))
    # Fabric wallet against rubber pads: the stock asset material is a
    # generic rigid one, which is why the pads reach it (53 mm of lift on
    # one episode) and then let it slide out.  Bind a high-friction
    # physics material to its colliders -- real contact, no attachment.
    _wmat = UsdShade.Material.Define(s, '/World/Assets/Wallet/WalletGripMat')
    _wmp = _wmat.GetPrim()
    _wm_auth = _wmp.GetMetadata('apiSchemas')
    _wm_sch = list(_wm_auth.GetAddedOrExplicitItems()) if _wm_auth else []
    for _api in ('PhysicsMaterialAPI',):
        if _api not in _wm_sch: _wm_sch.append(_api)
    _wmp.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(_wm_sch))
    _wmp.CreateAttribute(
        'physics:staticFriction', Sdf.ValueTypeNames.Float).Set(1.4)
    _wmp.CreateAttribute(
        'physics:dynamicFriction', Sdf.ValueTypeNames.Float).Set(1.2)
    _wmp.CreateAttribute(
        'physics:restitution', Sdf.ValueTypeNames.Float).Set(0.0)
    for _wc in Usd.PrimRange(p):
        if _wc.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(_wc).Bind(
                _wmat, materialPurpose='physics')
    _lean_min, _lean_max = _bound(p)
    print('[drawer-lean] wallet standing bounds: '
          + repr((tuple(float(v) for v in _lean_min),
                  tuple(float(v) for v in _lean_max))))
print(
    'placed pinch_axis=({pax:+.8f},{pay:+.8f}) '
    + 'pinch_yaw=' + str(pinch_yaw)
    + ' pinch_width=' + str(pinch_width)
    + _canary_report)
s.GetRootLayer().Save()
"""
    open("/tmp/_ip_place.py", "w").write(place)
    r2 = sh("conda run -n mesh311 python /tmp/_ip_place.py", t=120)
    ok = "placed" in r2.stdout
    native_cup_support = None
    native_cup_error = ""
    if ok and cfg.get("native_paper_cup"):
        native_xy = cfg["native_initial_xy_m"]
        if os.environ.get("PICK_PAPER_CUP_INITIAL_XY"):
            native_xy = os.environ["PICK_PAPER_CUP_INITIAL_XY"].split(",")
        receipt_directory = os.path.join(os.path.dirname(SCENE), "task33_native_fixture")
        native_result = subprocess.run(
            ["conda", "run", "-n", "mesh311", "python",
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "prepare_native_paper_cup.py"),
             SCENE, receipt_directory, *map(str, native_xy)],
            capture_output=True, text=True, timeout=120,
        )
        if native_result.returncode == 0:
            with open(os.path.join(receipt_directory, "NATIVE_CUP_SUPPORT.json")) as stream:
                native_cup_support = json.load(stream)
        else:
            ok = False
            native_cup_error = "native paper-cup fixture failed: " + (
                native_result.stdout + native_result.stderr)[-1200:]
    task55_current_context = None
    task56_current_context = None
    task59_current_context = None
    task57_current_context = None
    if (
            ok
            and tid == "55"
            and episode_uid in _TASK55_CURRENT_CONTEXT_EPISODE_UIDS):
        context_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "init_task55_context.py",
        )
        context_result = subprocess.run(
            [
                "conda", "run", "-n", "mesh311", "python",
                context_script,
                "--scene", SCENE,
                "--episode-uid", episode_uid,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            task55_current_context = json.loads(
                context_result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            task55_current_context = {
                "ok": False,
                "reason": (
                    "Task55 context initializer returned invalid JSON: "
                    + (context_result.stdout
                       + context_result.stderr).strip()[-400:]
                ),
            }
        if (
                context_result.returncode != 0
                or task55_current_context.get("ok") is not True):
            ok = False
    if (
            ok
            and tid == "56"
            and episode_uid in _TASK56_CURRENT_CONTEXT_EPISODE_UIDS):
        context_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "init_task56_context.py",
        )
        context_result = subprocess.run(
            [
                "conda", "run", "-n", "mesh311", "python",
                context_script,
                "--scene", SCENE,
                "--episode-uid", episode_uid,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            task56_current_context = json.loads(
                context_result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            task56_current_context = {
                "ok": False,
                "reason": (
                    "Task56 context initializer returned invalid JSON: "
                    + (context_result.stdout
                       + context_result.stderr).strip()[-400:]
                ),
            }
        if (
                context_result.returncode != 0
                or task56_current_context.get("ok") is not True):
            ok = False
    if (
            ok
            and tid == "57"
            and episode_uid == _TASK57_CURRENT_CONTEXT_EPISODE_UID):
        context_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "init_task57_context.py",
        )
        context_result = subprocess.run(
            [
                "conda", "run", "-n", "mesh311", "python",
                context_script,
                "--scene", SCENE,
                "--episode-uid", episode_uid,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            task57_current_context = json.loads(
                context_result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            task57_current_context = {
                "ok": False,
                "reason": (
                    "Task57 context initializer returned invalid JSON: "
                    + (context_result.stdout
                       + context_result.stderr).strip()[-400:]
                ),
            }
        if (
                context_result.returncode != 0
                or task57_current_context.get("ok") is not True):
            ok = False
    if (
            ok
            and task59_side_rest
            and episode_uid == _TASK59_CURRENT_CONTEXT_EPISODE_UID):
        context_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "init_task59_context.py",
        )
        context_result = subprocess.run(
            [
                "conda", "run", "-n", "mesh311", "python",
                context_script,
                "--scene", SCENE,
                "--episode-uid", episode_uid,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            task59_current_context = json.loads(
                context_result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            task59_current_context = {
                "ok": False,
                "reason": (
                    "Task59 context initializer returned invalid JSON: "
                    + (context_result.stdout
                       + context_result.stderr).strip()[-400:]
                ),
            }
        if (
                context_result.returncode != 0
                or task59_current_context.get("ok") is not True):
            ok = False
    out = {"ok": ok, "target": cfg["prim"], "hand": hand,
           "kin": [round(gx, 3), round(gy, 3), round(gz, 3)],
           "base": [round(bx, 3), round(by, 3)],
           "base_quat_wxyz": [
               round(value, 8) for value in (qw, qx, qy, qz)],
           "pinch_axis": [round(pax, 8), round(pay, 8)]}
    out["pad_axis_offset_m"] = round(pad_axis_offset, 6)
    if native_cup_support is not None:
        out["native_paper_cup_support"] = native_cup_support
        out["surface_support_mode"] = "visible_native_sofa"
        out["seat_riser_forbidden"] = True
    if task53_real_scene:
        out["task53_support_mode"] = _TASK53_CUSHION_SUPPORT_MODE
        out["task53_wallet_scale_xyz"] = list(
            _TASK53_WALLET_SCALE_XYZ)
    if cfg.get("canonical_yaw_seed") is not None:
        out["canonical_yaw_seed_deg"] = float(
            cfg["canonical_yaw_seed"])
    if initial_yaw_reference_deg is not None:
        out["initial_yaw_reference_deg"] = float(
            initial_yaw_reference_deg)
    if target_collision_approximation is not None:
        out["target_collision_approximation"] = (
            target_collision_approximation)
    if cfg["family"] in ("coffee_table", "tv_table"):
        out["surface_prim"] = surface_contract["path"]
        out["surface_top_m"] = round(surface_z, 9)
        out["surface_support_mode"] = "visible_native_surface"
        out["seat_riser_forbidden"] = True
    if task55_soft_wallet:
        out["target_collision_geometry_version"] = (
            "task55_soft_wallet_volume_wedge_v10_restored")
        out["target_visual_material_version"] = (
            "task55_wallet_herringbone_lowfreq_v1")
        if task55_current_context is not None:
            out["current_secondary_context"] = task55_current_context
    if edge_offset_local is not None:
        out["edge_offset_local_m"] = [
            round(value, 9) for value in edge_offset_local]
    if requested_local_pinch_axis is not None:
        out["requested_local_pinch_axis"] = [
            round(value, 9) for value in requested_local_pinch_axis]
    if task59_side_rest:
        out["requested_side_rest_mode"] = _TASK59_SIDE_REST_MODE
        out["requested_long_axis_offset_m"] = round(
            task59_long_axis_offset, 9)
        if task59_current_context is not None:
            out["current_secondary_context"] = task59_current_context
    if task57_current_context is not None:
        out["current_secondary_context"] = task57_current_context
    if task56_contact_scene is not None:
        out["episode_uid"] = task56_contact_scene["episode_uid"]
        out["surface_prim"] = task56_contact_scene["surface_path"]
        out["surface_top_m"] = round(
            task56_contact_scene["table_top_m"], 9)
        out["contact_context"] = [{
            "prim": task56_contact_scene["support_prim"],
            "relation": task56_contact_scene["relation"],
            "source_episode_uid": task56_contact_scene["episode_uid"],
            "required": True,
            "kinematic": True,
            "support_top_m": round(
                task56_contact_scene["support_top_m"], 9),
            "robot_facing_overhang_m": (
                None
                if task56_contact_scene[
                    "robot_facing_overhang_m"] is None
                else round(
                    task56_contact_scene[
                        "robot_facing_overhang_m"], 9)
            ),
        }]
        if task56_current_context is not None:
            out["current_secondary_context"] = task56_current_context
    pinch_result = re.search(
        r"pinch_yaw=([-+\d.eE]+|None) pinch_width=([-+\d.eE]+|None)",
        r2.stdout,
    )
    if pinch_result:
        if pinch_result.group(1) != "None":
            out["pinch_yaw_deg"] = round(
                float(pinch_result.group(1)), 6)
        if pinch_result.group(2) != "None":
            out["pinch_width_m"] = round(
                float(pinch_result.group(2)), 9)
    if requested_target_yaw_world_deg is not None:
        out["requested_target_yaw_world_deg"] = requested_target_yaw_world_deg
    edge_world_result = re.search(
        r"edge_offset_local=\(([-+\d.eE]+),([-+\d.eE]+)\) "
        r"edge_world_delta=\(([-+\d.eE]+),([-+\d.eE]+),"
        r"([-+\d.eE]+)\)",
        r2.stdout,
    )
    if edge_world_result:
        out["edge_world_delta_m"] = [
            round(float(edge_world_result.group(index)), 9)
            for index in range(3, 6)
        ]
    alignment_result = re.search(
        r"alignment_abs_dot=([-+\d.eE]+) "
        r"alignment_error_deg=([-+\d.eE]+)",
        r2.stdout,
    )
    if alignment_result:
        out["pinch_alignment_abs_dot"] = round(
            float(alignment_result.group(1)), 9)
        out["pinch_alignment_error_deg"] = round(
            float(alignment_result.group(2)), 6)
    task53_scene_result = re.search(
        r"task53_support_mode=([A-Za-z0-9._-]+) "
        r"task53_wallet_scale=\(([-+\d.eE]+),([-+\d.eE]+),"
        r"([-+\d.eE]+)\) task53_cushion_top=([-+\d.eE]+)",
        r2.stdout,
    )
    if task53_scene_result:
        out["task53_support_mode"] = task53_scene_result.group(1)
        out["task53_wallet_scale_xyz"] = [
            round(float(task53_scene_result.group(index)), 9)
            for index in (2, 3, 4)
        ]
        out["task53_cushion_top_m"] = round(
            float(task53_scene_result.group(5)), 9)
    side_rest_result = re.search(
        r"side_rest_mode=([A-Za-z0-9._-]+) "
        r"side_rest_rotate_xyz=\(([-+\d.eE]+),([-+\d.eE]+),"
        r"([-+\d.eE]+)\) "
        r"side_rest_support_face=(local_y_(?:min|max)) "
        r"side_rest_vertical_extent=([-+\d.eE]+) "
        r"side_rest_long_extent=([-+\d.eE]+) "
        r"side_rest_axis_alignment_abs_dot=([-+\d.eE]+) "
        r"side_rest_settle_z=([-+\d.eE]+)",
        r2.stdout,
    )
    side_rest_metadata_ok = bool(
        side_rest_result
        and side_rest_result.group(1) == _TASK59_SIDE_REST_MODE
        and task59_support is not None
        and abs(
            float(side_rest_result.group(9))
            - task59_support["support_top_m"]
        ) <= 1e-6)
    long_axis_result = re.search(
        r"long_axis_offset=([-+\d.eE]+) "
        r"long_axis_world_delta=\(([-+\d.eE]+),([-+\d.eE]+),"
        r"([-+\d.eE]+)\)",
        r2.stdout,
    )
    long_axis_metadata_ok = bool(
        long_axis_result
        and abs(
            float(long_axis_result.group(1))
            - task59_long_axis_offset
        ) <= 1e-9)
    if side_rest_metadata_ok:
        out["side_rest_mode"] = side_rest_result.group(1)
        out["side_rest_rotate_xyz_deg"] = [
            round(float(side_rest_result.group(index)), 6)
            for index in (2, 3, 4)
        ]
        out["side_rest_support_face"] = side_rest_result.group(5)
        out["side_rest_vertical_extent_m"] = round(
            float(side_rest_result.group(6)), 9)
        out["side_rest_long_extent_m"] = round(
            float(side_rest_result.group(7)), 9)
        out["side_rest_axis_alignment_abs_dot"] = round(
            float(side_rest_result.group(8)), 9)
        out["side_rest_settle_z_m"] = round(
            float(side_rest_result.group(9)), 9)
        out["side_rest_support_mode"] = task59_support["support_mode"]
        out["side_rest_support_top_m"] = round(
            task59_support["support_top_m"], 9)
        out["side_rest_support_uncertainty_m"] = round(
            task59_support["uncertainty_m"], 9)
        out["side_rest_support_manifest_revision"] = (
            task59_support["manifest_revision"])
    if long_axis_metadata_ok:
        out["long_axis_offset_m"] = round(
            float(long_axis_result.group(1)), 9)
        out["long_axis_world_delta_m"] = [
            round(float(long_axis_result.group(index)), 9)
            for index in (2, 3, 4)
        ]
    if task59_side_rest and not (
            side_rest_metadata_ok and long_axis_metadata_ok):
        ok = False
        out["ok"] = False
    if task53_real_scene and not (
            task53_scene_result
            and task53_scene_result.group(1)
            == _TASK53_CUSHION_SUPPORT_MODE
            and all(
                abs(
                    float(task53_scene_result.group(index))
                    - _TASK53_WALLET_SCALE_XYZ[index - 2]
                ) <= 1e-9
                for index in (2, 3, 4)
            )):
        ok = False
        out["ok"] = False
    if not ok:
        prefix = (
            "place failed or missing audited Task59 side-rest metadata: "
            if task59_side_rest
            else (
                "place failed or missing audited Task53 real-scene metadata: "
                if task53_real_scene
                else "place failed: "
            )
        )
        out["reason"] = prefix + (r2.stdout + r2.stderr)[-120:]
    if native_cup_error:
        out["reason"] = native_cup_error
    print(json.dumps(out))
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())

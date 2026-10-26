"""Deterministic initial-state specification for task 10.1 (not a reset driver).

Only initialization is defined here. No policy, physics tuning, randomization,
scoring, USD writes, or old-server access occurs on import or plan construction.
The explicit asset-preview command writes a separate USD, never the base scene.
The environment must apply the plan BEFORE the episode, settle, then provide
real measurements to check_initial_state. A plan is not physics acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping
from .parameters import normalize_parameters
from arena_api import beef_asset


TASK_ID = "10.1"
SCHEMA = "arena.microwave_take.initial_state.v1"
INSTRUCTION = "打开微波炉，取出装有牛排盒的盘子，关好炉门，并稳稳拿住盘子。"
ROBOT = "/World/Robot"
MICROWAVE = "/World/Assets/DiningTableMicrowave"
PLATE = "/World/Assets/MicrowaveTakePlate"
PAYLOAD = "/World/Assets/MicrowaveBeefbox"
STAND = "/World/Assets/MicrowavePlateStand"
DOOR_JOINT = MICROWAVE + "/Asset/door_hinge"
# Names are semantic roles, not permission for the VLA to choose an instance.
BINDINGS = {"microwave": MICROWAVE, "robot": ROBOT, "stand": STAND,
            "plate": PLATE, "payload": PAYLOAD}
OTHER_PAYLOADS = ("/World/Assets/MicrowavePayload", "/World/Assets/MicrowaveFoodbox",
                  "/World/Assets/MicrowaveSaladbox")
LEFTOVER_PAYLOAD = "/World/Assets/MicrowaveFoodbox"
LEFTOVER_ASSET_SHA256 = {
    "payload_native_mesh": "7b85e016f1bb11603543ee8f41dda51496c53090300fdf885c7b1c2bcfb41fd6",
    "payload_native_physics": "0659343d305138476b75bbbd512d9ec5571f5de2ae01cf194967b18daf613b51",
    "payload_native_texture": "0ca1381061e4a1422dfd8cdc27eb229443e5700ea411e7718cdc9e2e6ad267e3",
}


def bindings_for(payload_variant="beef_box"):
    variant = normalize_parameters({'payload_variant': payload_variant})['payload_variant']
    return {**BINDINGS, 'payload': PAYLOAD if variant == 'beef_box' else LEFTOVER_PAYLOAD}


def other_payloads_for(payload_variant="beef_box"):
    selected = bindings_for(payload_variant)['payload']
    return tuple(p for p in (PAYLOAD, *OTHER_PAYLOADS) if p != selected)
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
PLATE_V10 = RESOURCE_DIR / "black_plate_325mm_flat_rim_visual_physics_matched_v10.usda"
PLATE_ASSET = RESOURCE_DIR / "black_plate_325mm_filled_recess_v12.usda"
STAND_LAYER = RESOURCE_DIR / "stand_level_override_v1.usda"
PLATE_LAYER = RESOURCE_DIR / "standard_beef_plate_override_v13.usda"
TASK_ASSET_LAYER_SHA256 = "f80d02e9be152d12408651550978f6ae3da0aee5f73b4e7e5165828ce49b6079"
STAND_ALIGNMENT_SHA256 = "d1fd1d6892b15c825bc991221b64d37b6d039ff89b7dc2335539541ac679857a"
STAND_TOP_LOCAL_M = 0.988672992
PLATE_SOURCE_SHA256 = "b52e4ba8f24e557f36457cf7ec1b06fa7048d540d10db53c2bf372846d03668b"
PLATE_SHA256 = "e2d9595695529549f69607dda85fccb4f9768052b981bc480a8f19f188d40746"
CONTACT_SURFACE_FILES = {
    "plate_contact_surface": ("plate_filled_recess_contact_v12.usda",
        "e13ddbf9af93a2435a48eef1b03ddff938e2ed9cbb317a9bbd45a23de6255e5e"),
    "payload_contact_surface": ("../../../../resources/shared_assets/beef_box_standard243_v1/beefbox_contact_standard243_v1.usda",
        beef_asset.CONTACT_SHA256),
}
REQUIRED_ASSET_SHA256 = {"plate": PLATE_SHA256, "stand_alignment": STAND_ALIGNMENT_SHA256,
                       "task_asset_layer": TASK_ASSET_LAYER_SHA256,
                       **{k: v[1] for k, v in CONTACT_SURFACE_FILES.items()}}


def _finite(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Expected a finite real number, not a boolean")
    return float(value)


@dataclass(frozen=True)
class Pose:
    """Rigid Z-up pose in metres/degrees. Roll, pitch and scale are unsupported."""

    xyz_m: tuple[float, float, float]
    yaw_deg: float

    def __post_init__(self):
        if len(self.xyz_m) != 3:
            raise ValueError("xyz_m must have three coordinates")
        object.__setattr__(self, "xyz_m", tuple(_finite(x) for x in self.xyz_m))
        object.__setattr__(self, "yaw_deg", _finite(self.yaw_deg))


def compose(parent: Pose, local: Pose) -> Pose:
    """T_world_child = T_world_parent * T_parent_child; never scale offsets."""
    a = math.radians(parent.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = local.xyz_m
    px, py, pz = parent.xyz_m
    return Pose((px + c*x - s*y, py + s*x + c*y, pz + z),
                parent.yaw_deg + local.yaw_deg)


def relative_to(parent: Pose, world: Pose) -> Pose:
    a = math.radians(parent.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy, dz = (v - p for v, p in zip(world.xyz_m, parent.xyz_m))
    return Pose((c*dx + s*dy, -s*dx + c*dy, dz), world.yaw_deg - parent.yaw_deg)


@dataclass(frozen=True)
class JointTarget:
    value: float
    unit: str

    def __post_init__(self):
        object.__setattr__(self, "value", _finite(self.value))
        if self.unit not in ("rad", "m"):
            raise ValueError("Joint unit must be rad or m")


# Recorded frame zero, not the 140 mm door-handle pregrasp. World transforms
# below are source evidence only; build_initial_state uses the RELATIVE pose.
SOURCE_MICROWAVE = Pose((2.3768601945223806, -0.6229512937654896, 0.92),
                       -85.19136833834688)
SOURCE_ROBOT = Pose((1.506081199906, -0.629556806229, 0.0), 0.0)
ROBOT_IN_MICROWAVE = relative_to(SOURCE_MICROWAVE, SOURCE_ROBOT)
FIXED_MICROWAVE = Pose((2.3878, -0.643, 0.92), -90.0)
LEFT_ARM_RAD = (-1.7944881916046143, 0.0, 2.0309531688690186,
                -1.3114372491836548, -1.031469464302063, 0.6646530628204346,
                1.2486522197723389)
RIGHT_ARM_RAD = (1.0736485719680786, -0.6115309000015259, -0.2806771695613861,
                 1.285379409790039, -0.7304690480232239, -1.481121301651001,
                 0.18747517466545105)

# User-approved stand revision: raise only the three arm tops by 8 mm to the
# existing hub/pad top. The v12 original-size solid bottom sits on that plane; move
# plate AND payload up 8 mm at initialization, with the same 0.2 mm clearance.
# Do not teleport or constrain either body after physics starts.
STAND_WORLD_REFERENCE = Pose((2.5912056170628284, -0.5582333464551501, 0.0), 0.0)
PLATE_WORLD_REFERENCE = Pose((2.5912056170628284, -0.5582333464551501,
                              0.98887299219), 0.0)
STAND_IN_MICROWAVE = relative_to(FIXED_MICROWAVE, STAND_WORLD_REFERENCE)
PLATE_IN_MICROWAVE = relative_to(FIXED_MICROWAVE, PLATE_WORLD_REFERENCE)
# Real tapered underside, not the old box-shaped collider. First geometric
# contact is at 25.87983112 mm relative Z, plus 0.2 mm initialization clearance.
# The box is free to tilt naturally during settle; no adhesion or pose lock.
PAYLOAD_IN_PLATE = Pose((0.0219, 0.0, 0.02607983112), 0.0)
# L1 MicrowaveFoodbox relative placement and native 0.94 scale. Not beef geometry.
LEFTOVER_IN_PLATE = Pose((0.0219, 0.0, 0.0193), 0.0)


def robot_joint_targets() -> dict[str, JointTarget]:
    result = {f"idx{20+i}_arm_l_joint{i}": JointTarget(q, "rad")
              for i, q in enumerate(LEFT_ARM_RAD, 1)}
    result.update({f"idx{60+i}_arm_r_joint{i}": JointTarget(q, "rad")
                   for i, q in enumerate(RIGHT_ARM_RAD, 1)})
    result.update({
        "idx01_body_joint1": JointTarget(0.31978875398635864, "m"),
        "idx02_body_joint2": JointTarget(0.2617988586425781, "rad"),
        "idx11_head_joint1": JointTarget(0.0, "rad"),
        "idx12_head_joint2": JointTarget(0.4363323152065277, "rad"),
        # Legacy command 0 meant OPEN: outer joint = 0.62*(1-command).
        # Current policy 0 means CLOSED; never copy the legacy command directly.
        "idx41_gripper_l_outer_joint1": JointTarget(0.62, "rad"),
        "idx81_gripper_r_outer_joint1": JointTarget(0.62, "rad"),
    })
    return result


@dataclass(frozen=True)
class InitialState:
    microwave: Pose
    placements: Mapping[str, Pose]
    joint_targets: Mapping[str, JointTarget]
    door_angle_deg: float = 0.0
    payload_variant: str = "beef_box"

    def __post_init__(self):
        bindings_for(self.payload_variant)

    @property
    def bindings(self):
        return bindings_for(self.payload_variant)

    @property
    def other_payloads(self):
        return other_payloads_for(self.payload_variant)

    @property
    def required_asset_sha256(self):
        hashes = dict(REQUIRED_ASSET_SHA256)
        if self.payload_variant == 'leftover_box':
            hashes.pop('payload_contact_surface')
            hashes.update(LEFTOVER_ASSET_SHA256)
        return hashes

    def to_dict(self) -> dict:
        result = {
            "schema": SCHEMA, "task_id": TASK_ID, "instruction": INSTRUCTION,
            "status": "candidate_definition_only", "physics_validated": False,
            "runtime_reset_connected": False, "randomization_enabled": False,
            "coordinate_frame": "rigid microwave root, metres, Z-up",
            "bindings": self.bindings,
            "task_asset_layer": str(PLATE_LAYER),
            "required_asset_sha256": self.required_asset_sha256,
            "collision_profile": "plate_filled_recess_v12_payload_standard243_v1_stand_level_v1",
            "shared_beef_asset": beef_asset.REVISION,
            "plate_profile": {"revision": "filled_recess_v12", "central_recess": False,
                              "filled_diameter_mm": 62, "original_outer_contour_preserved": True,
                              "outer_bottom_flat_diameter_mm": 80, "bottom_local_mm": 0},
            "stand_profile": {"revision": "stand_level_v1", "top_local_m": STAND_TOP_LOCAL_M,
                              "arm_top_raise_m": 0.008, "plate_payload_initial_raise_m": 0.008},
            "relative_poses": {
                "robot_in_microwave": asdict(relative_to(self.microwave, self.placements['robot'])),
                "stand_in_microwave": asdict(relative_to(self.microwave, self.placements['stand'])),
                "plate_in_microwave": asdict(relative_to(self.microwave, self.placements['plate'])),
                "payload_in_plate": asdict(relative_to(self.placements['plate'], self.placements['payload']))},
            "world_poses": {key: asdict(value) for key, value in self.placements.items()},
            "joint_targets": {key: asdict(value) for key, value in self.joint_targets.items()},
            "door_joint": DOOR_JOINT, "door_angle_deg": self.door_angle_deg,
            "required_active_prims": [ROBOT, MICROWAVE, PLATE, self.bindings['payload'], STAND],
            "deactivate_if_present": list(self.other_payloads),
            "reset_boundary": "before episode only; then settle and measure; no runtime pose writes",
            "preserve": ["robot and camera collisions", "self collision", "visual geometry",
                         "native gripper mechanisms", "drives and force limits", "mass and friction"],
            "source": {
                "sample": "takev43normal_b05_06", "frame": 0,
                "real_reference_uuid": "38a56d1f-9812-46c2-9a5b-22903cef00b4",
                "trajectory": "action_yaw0_station_grasp_lower10_normalpull_level2p7_v12.npz",
                "role": "embedded reference values, not a runtime dependency",
                "gripper_encoding": "legacy closure=0 -> outer drive 0.62 rad",
                "seating": "visual-surface colliders; foot ring on stand spokes; free box settle; full task reset validation pending"},
        }
        # Preserve the accepted beef reset contract byte-for-byte, including
        # old recordings without parameters. Leftover has a distinct identity.
        if self.payload_variant == 'leftover_box':
            result.update(payload_variant=self.payload_variant,
                instruction=INSTRUCTION.replace('牛排盒', '剩饭盒'),
                collision_profile='plate_filled_recess_v12_native_foodbox_task108_v2_stand_level_v1',
                payload_scale=[0.94, 0.94, 0.94])
            result['source']['payload'] = 'L1 MicrowaveFoodbox / foodbox_burger_v1 / foodbox_task108_physics_v2.usda'
        return result


def build_initial_state(microwave: Pose = FIXED_MICROWAVE, *, payload_variant="beef_box") -> InitialState:
    """Pure deterministic plan. Optional XY/yaw anchor is setup, not sampling.

    Only floor-preserving planar relocation is supported: a new counter height
    requires recalibration rather than lifting the robot base off the floor.
    """
    if not isinstance(microwave, Pose):
        raise TypeError("microwave must be a Pose")
    if abs(microwave.xyz_m[2] - FIXED_MICROWAVE.xyz_m[2]) > 1e-9:
        raise ValueError("Changed counter height requires initial-state recalibration")
    plate = compose(microwave, PLATE_IN_MICROWAVE)
    return InitialState(microwave, {
        "microwave": microwave,
        "robot": compose(microwave, ROBOT_IN_MICROWAVE),
        "stand": compose(microwave, STAND_IN_MICROWAVE),
        "plate": plate,
        "payload": compose(plate, PAYLOAD_IN_PLATE if payload_variant == 'beef_box' else LEFTOVER_IN_PLATE),
    }, robot_joint_targets(), payload_variant=payload_variant)


REQUIRED_FACTS = (
    "settled", "physics_valid", "microwave_fixed", "door_passive_dynamic",
    "plate_dynamic", "payload_dynamic", "collisions_enabled", "self_collision_enabled",
    "plate_supported_by_stand", "payload_supported_by_plate",
    "plate_inside_cavity", "payload_inside_cavity", "other_payloads_inactive",
    "robot_not_holding_objects", "robot_clear_of_environment",
    "no_initial_interpenetration", "no_runtime_assistance", "task_assets_unchanged",
)


def plan_from_stage(stage) -> InitialState:
    """Read reset targets, NOT observations, from the task-owned reset layer.

    Legacy fixed probes have no embedded plan. New relocated probes retain the
    full pre-physics target so acceptance never compares against stale XYZ.
    Joint/door changes are not part of positional randomization.
    """
    raw = stage.GetRootLayer().customLayerData.get('initialPlanJson')
    if raw is None:
        return build_initial_state()
    data = json.loads(raw)
    variant = data.get('payload_variant', 'beef_box')
    if data['schema'] != SCHEMA or data['task_id'] != TASK_ID or data['bindings'] != bindings_for(variant):
        raise ValueError('Wrong embedded initial-state identity')
    if data['joint_targets'] != {n: asdict(q) for n, q in robot_joint_targets().items()} or data['door_angle_deg'] != 0:
        raise ValueError('Pose randomization cannot change robot joints or door state')
    if set(data['world_poses']) != set(BINDINGS):
        raise ValueError('Incomplete embedded placement plan')
    poses = {n: Pose(tuple(p['xyz_m']), p['yaw_deg']) for n, p in data['world_poses'].items()}
    build_initial_state(poses['microwave'])  # same counter height
    if abs(poses['robot'].xyz_m[2]) > 1e-9:
        raise ValueError('Robot must stay on the floor')
    return InitialState(poses['microwave'], poses, robot_joint_targets(), payload_variant=variant)


@dataclass(frozen=True)
class InitialObservation:
    """Privileged, post-settle step-zero data from a trusted environment reader.

    Containment covers complete geometry, not object centres. Support requires
    physical contact/force, not merely similar heights. Unknown facts stay None.
    This data object does not authenticate its producer or collect physics.
    """

    episode_id: str
    source: str
    step_index: int
    task_id: str
    bindings: Mapping[str, str]
    placements: Mapping[str, Pose]
    joint_positions: Mapping[str, JointTarget]
    door_angle_deg: float | None = None
    facts: Mapping[str, bool | None] = field(default_factory=dict)
    asset_sha256: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Tolerances:
    """Provisional reset checks, not task-success or process-score thresholds."""

    position_m: float = 0.005
    yaw_deg: float = 2.0
    angular_joint_rad: float = 0.05
    linear_joint_m: float = 0.003
    closed_door_deg: float = 2.0

    def __post_init__(self):
        for value in asdict(self).values():
            if _finite(value) <= 0:
                raise ValueError("Reset tolerances must be positive and finite")


def check_initial_state(plan: InitialState, observation: InitialObservation | None,
                        tolerances: Tolerances = Tolerances()) -> dict:
    """Check supplied measurements only; missing physical evidence never passes."""
    errors = []
    if observation is None:
        errors.append("missing_post_settle_physics_observation")
    elif not isinstance(observation, InitialObservation):
        errors.append("invalid_observation_type")
    else:
        o = observation
        if not isinstance(o.episode_id, str) or not o.episode_id.strip():
            errors.append("missing_episode_identity")
        if o.source != "isaac_physics":
            errors.append("not_isaac_physics")
        if type(o.step_index) is not int or o.step_index != 0:
            errors.append("not_episode_step_zero")
        if o.task_id != TASK_ID:
            errors.append("wrong_task")
        if not isinstance(o.bindings, Mapping) or dict(o.bindings) != plan.bindings:
            errors.append("wrong_or_missing_instance_bindings")
        placements = o.placements if isinstance(o.placements, Mapping) else {}
        joints = o.joint_positions if isinstance(o.joint_positions, Mapping) else {}
        facts = o.facts if isinstance(o.facts, Mapping) else {}
        if not isinstance(o.asset_sha256, Mapping) or o.asset_sha256.get("plate") != PLATE_SHA256:
            errors.append("missing_or_wrong_plate_asset_version")
        for role, (_, digest) in CONTACT_SURFACE_FILES.items():
            if role not in plan.required_asset_sha256:
                continue
            if not isinstance(o.asset_sha256, Mapping) or o.asset_sha256.get(role) != digest:
                errors.append("missing_or_wrong_collision_surface:" + role)
        if plan.payload_variant == 'leftover_box':
            for role, digest in LEFTOVER_ASSET_SHA256.items():
                if not isinstance(o.asset_sha256, Mapping) or o.asset_sha256.get(role) != digest:
                    errors.append('missing_or_wrong_native_payload_asset:' + role)
        if not isinstance(o.asset_sha256, Mapping) or o.asset_sha256.get("stand_alignment") != STAND_ALIGNMENT_SHA256:
            errors.append("missing_or_wrong_stand_asset_version")
        if not isinstance(o.asset_sha256, Mapping) or o.asset_sha256.get("task_asset_layer") != TASK_ASSET_LAYER_SHA256:
            errors.append("missing_or_wrong_task_asset_layer")
        try:
            if abs(_finite(o.door_angle_deg) - plan.door_angle_deg) > tolerances.closed_door_deg:
                errors.append("door_not_closed")
        except ValueError:
            errors.append("missing_or_invalid_door_angle")
        for role, expected in plan.placements.items():
            measured = placements.get(role)
            if not isinstance(measured, Pose):
                errors.append(f"missing_pose:{role}")
                continue
            delta_yaw = (measured.yaw_deg - expected.yaw_deg + 180) % 360 - 180
            if (math.dist(measured.xyz_m, expected.xyz_m) > tolerances.position_m
                    or abs(delta_yaw) > tolerances.yaw_deg):
                errors.append(f"pose_mismatch:{role}")
        for name, expected in plan.joint_targets.items():
            measured = joints.get(name)
            if not isinstance(measured, JointTarget) or measured.unit != expected.unit:
                errors.append(f"missing_or_wrong_unit_joint:{name}")
                continue
            limit = tolerances.linear_joint_m if expected.unit == "m" else tolerances.angular_joint_rad
            if abs(measured.value - expected.value) > limit:
                errors.append(f"joint_mismatch:{name}")
        for fact in REQUIRED_FACTS:
            if facts.get(fact) is not True:
                errors.append(f"unconfirmed:{fact}")
    return {"schema": SCHEMA, "task_id": TASK_ID,
            "initial_state_checks_passed": not errors, "errors": errors,
            "evidence_collected_by_this_module": False,
            "task_success_evaluated": False}


def verify_plate_asset() -> None:
    """Pin v12, preserved v10 provenance, task layers and actual contact files."""
    beef_asset.verify_files()
    if hashlib.sha256(PLATE_V10.read_bytes()).hexdigest() != PLATE_SOURCE_SHA256:
        raise ValueError("Original v10 source changed")
    if hashlib.sha256(PLATE_ASSET.read_bytes()).hexdigest() != PLATE_SHA256:
        raise ValueError("Task plate is not the approved filled-recess v12")
    if not PLATE_LAYER.is_file():
        raise FileNotFoundError(PLATE_LAYER)
    if hashlib.sha256(PLATE_LAYER.read_bytes()).hexdigest() != TASK_ASSET_LAYER_SHA256:
        raise ValueError("Task asset layer changed")
    if hashlib.sha256(STAND_LAYER.read_bytes()).hexdigest() != STAND_ALIGNMENT_SHA256:
        raise ValueError("Task stand alignment layer changed")
    for role, (filename, digest) in CONTACT_SURFACE_FILES.items():
        if hashlib.sha256((RESOURCE_DIR / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Task collision surface changed: " + role)


def verify_plate_geometry(stage):
    """Reject an old loaded foot-ring mesh even if new files exist on disk."""
    from pxr import Usd, UsdGeom, Gf
    expected = Usd.Stage.Open(str(PLATE_ASSET))
    cache = UsdGeom.XformCache()
    asset = stage.GetPrimAtPath(PLATE + "/Asset")
    if not asset:
        raise ValueError("Missing plate asset")
    for suffix in ("visual", "ContactSurface/surface_0"):
        actual = stage.GetPrimAtPath(PLATE + "/Asset/" + suffix)
        source = expected.GetPrimAtPath("/BlackPlate325mmFlatRimV2/" + suffix)
        if not actual or not actual.IsA(UsdGeom.Mesh):
            raise ValueError("Missing filled-recess plate mesh: " + suffix)
        for name in ("points", "faceVertexCounts", "faceVertexIndices"):
            if actual.GetAttribute(name).Get() != source.GetAttribute(name).Get():
                raise ValueError("Loaded plate is not filled-recess v12: " + suffix)
        local = cache.GetLocalToWorldTransform(actual) * cache.GetLocalToWorldTransform(asset).GetInverse()
        if not Gf.IsClose(local, Gf.Matrix4d(1), 1e-8):
            raise ValueError("Unexpected plate mesh transform")


def verify_stand_geometry(stage) -> dict:
    """Read-only check of actual composed stand, not a claimed file hash."""
    from pxr import Gf, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(STAND)
    if not root:
        raise ValueError("Missing task stand")
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    paths = ["CenterHub", "HubRing"] + [
        f"Spoke{s}/{part}" for s in "ABC" for part in ("Arm", "SupportPad")]
    bounds = {}
    for name in paths:
        prim = stage.GetPrimAtPath(STAND + "/" + name)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("Missing stand collider: " + name)
        if prim.GetAttribute("physics:collisionEnabled").Get() is False or (
                UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"):
            raise ValueError("Stand visual/collision disabled: " + name)
        if prim.IsA(UsdGeom.Cube):
            half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            dimensions = (half, half, half)
        elif prim.IsA(UsdGeom.Cylinder):
            geom = UsdGeom.Cylinder(prim)
            if geom.GetAxisAttr().Get() != "Z":
                raise ValueError("Unexpected stand cylinder axis")
            dimensions = (geom.GetRadiusAttr().Get(), geom.GetRadiusAttr().Get(),
                          geom.GetHeightAttr().Get() / 2)
        else:
            raise ValueError("Unexpected stand geometry: " + name)
        tf = cache.GetLocalToWorldTransform(prim) * inverse
        x, y, z = dimensions
        zs = [tf.Transform(Gf.Vec3d(a, b, c))[2]
              for a in (-x, x) for b in (-y, y) for c in (-z, z)]
        bounds[name] = {"bottom_local_m": min(zs), "top_local_m": max(zs)}
        if abs(max(zs) - STAND_TOP_LOCAL_M) > 1e-7:
            raise ValueError("Stand top is not level: " + name)
        if name.endswith("/Arm") and abs(min(zs) - 0.972672992) > 1e-7:
            raise ValueError("Stand arm bottom changed: " + name)
    return bounds


def compose_asset_preview(base_scene: str | Path, plan: InitialState | None = None):
    """CPU USD composition only; does NOT initialize robot joints or physics.

    Replaces the plate in a new in-memory layer, seats plate/box, and keeps
    the base scene and its robot/microwave physical configuration untouched.
    The existing fixed microwave and stand placement are checked, not moved.
    """
    from pxr import Gf, Usd, UsdGeom

    verify_plate_asset()
    base = Path(base_scene).resolve(strict=True)
    plan = plan or build_initial_state()
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(PLATE_LAYER), str(base)]
    # USD stage metadata is not inherited from sublayers.
    base_stage = Usd.Stage.Open(str(base))
    if abs(UsdGeom.GetStageMetersPerUnit(base_stage) - 1.0) > 1e-9:
        raise ValueError("Task preview requires a metre-scale base scene")
    if UsdGeom.GetStageUpAxis(base_stage) != "Z":
        raise ValueError("Task preview requires a Z-up base scene")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetTimeCodesPerSecond(base_stage.GetTimeCodesPerSecond())
    for role, path in plan.bindings.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetTypeName() != "Xform":
            raise ValueError(f"Missing or wrong base-scene instance: {role}")
    cache = UsdGeom.XformCache()
    for role in ("microwave", "stand"):
        xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(plan.bindings[role]))
        pose = plan.placements[role]
        expected = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), pose.yaw_deg))
        expected.SetTranslateOnly(Gf.Vec3d(*pose.xyz_m))
        if max(abs(xf[i][j] - expected[i][j]) for i in range(4) for j in range(4)) > 1e-6:
            raise ValueError(f"{role} differs from the calibrated fixed anchor")
    for role in ("plate", "payload"):
        prim = stage.GetPrimAtPath(plan.bindings[role])
        prim.SetActive(True)
        pose = plan.placements[role]
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz_m))
        xf.AddRotateZOp().Set(pose.yaw_deg)
        if role == 'payload' and plan.payload_variant == 'leftover_box':
            xf.AddScaleOp().Set(Gf.Vec3f(0.94))
        xf.SetResetXformStack(True)
    for path in plan.other_payloads:
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.SetActive(False)
    if plan.payload_variant == 'leftover_box':
        from .payload_assets import verify_leftover_asset
        verify_leftover_asset(stage)
    stage.GetRootLayer().customLayerData = {
        "arenaTask": TASK_ID, "previewOnly": True, "physicsValidated": False,
        "robotInitialStateApplied": False, "plateSha256": PLATE_SHA256}
    return stage


def write_asset_preview(base_scene: str | Path, output: str | Path) -> Path:
    """Export a separate asset-only preview without overwriting any existing file."""
    import os
    from pxr import Sdf

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.suffix != ".usda":
        raise ValueError("Preview output must end in .usda")
    stage = compose_asset_preview(base_scene)
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(stage.GetRootLayer())
    layer.subLayerPaths = [os.path.relpath(p, output.parent)
                          for p in stage.GetRootLayer().subLayerPaths]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output)):
        raise RuntimeError(f"Could not export {output}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene", type=Path)
    parser.add_argument("--asset-preview", type=Path,
                        help="Asset-only USD; does not apply robot initial pose or run physics")
    args = parser.parse_args()
    if bool(args.base_scene) != bool(args.asset_preview):
        parser.error("--base-scene and --asset-preview must be provided together")
    if args.asset_preview:
        print(write_asset_preview(args.base_scene, args.asset_preview))
    else:
        print(json.dumps(build_initial_state().to_dict(), ensure_ascii=False, indent=2))

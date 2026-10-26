"""Prepare the beef initial scene + native robot reset contract, without physics.

The USD is NOT a standalone accepted grasp: its robot must be initialized by
the existing RobotInitialization hook before settling. No worker is patched.
"""
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .beef_initialization import robot_reset_plan, relative_placement_plan, MICROWAVE_ROOT
from .payload_assets import validate_payload_geometry
from .recorded_checkpoint import load_checkpoint, rigid_matrix

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
ROBOT = "/World/Robot"
PLATE = "/World/Assets/MicrowaveTakePlate"
PAYLOAD = "/World/Assets/MicrowaveBeefbox"


def composed_scene(source):
    """Reuse only the current approved public layers; no copied asset physics."""
    from pxr import Sdf, Usd
    from arena_api import l1_scene
    from evaluation.tasks.microwave_take.initial_state import PLATE_LAYER, verify_plate_asset

    source = Path(source).resolve(strict=True)
    if not source.is_relative_to(ROOT):
        raise ValueError("Source scene must be inside this project")
    verify_plate_asset()
    base = Usd.Stage.Open(str(source))
    if base is None:
        raise ValueError("Cannot load source scene")
    l1_scene.apply(base)  # Existing preparation only, in this memory session.
    layer = Sdf.Layer.CreateAnonymous("beef_initial.usda")
    layer.TransferContent(base.GetSessionLayer())
    layer.subLayerPaths = [str(PLATE_LAYER), *layer.subLayerPaths, str(source)]
    stage = Usd.Stage.Open(layer)
    for key in ("metersPerUnit", "upAxis", "timeCodesPerSecond", "framesPerSecond"):
        stage.SetMetadata(key, base.GetMetadata(key))
    from .door_appearance import apply_opaque_door
    apply_opaque_door(stage)
    return stage


def protected_physics(stage):
    """Include targets and velocities: this authoring step changes none of them."""
    return {str(a.GetPath()): str(a.Get()) for p in stage.Traverse()
            for a in p.GetAttributes() if a.GetName().startswith(
                ("physics:", "physx", "drive:", "state:"))}


def set_world_pose(stage, path, desired):
    from pxr import Gf, UsdGeom
    desired = rigid_matrix(desired)
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive():
        raise ValueError("Missing active pose target: " + path)
    cache = UsdGeom.XformCache()
    # Never erase geometry scale under the guise of setting a rigid pose.
    rigid_matrix(np.asarray(cache.GetLocalToWorldTransform(prim)).T)
    parent = rigid_matrix(np.asarray(cache.GetLocalToWorldTransform(prim.GetParent())).T)
    local = np.linalg.inv(parent) @ desired
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "microwavePutInitial")
    op.Set(Gf.Matrix4d(*[float(v) for v in local.T.flat]))
    xf.SetXformOpOrder([op], resetXformStack=False)


def prepare(source, output, *, replay_case=None):
    from pxr import Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    from evaluation.tasks.microwave_take.initial_state import other_payloads_for

    output = Path(output).resolve()
    if not output.is_relative_to(TASK) or output == TASK:
        raise ValueError("Output must remain inside microwave_put")
    if output.exists():
        raise FileExistsError(output)
    from .replay_cases import checkpoint_for
    checkpoint_dir = checkpoint_for(replay_case)
    cp = load_checkpoint(checkpoint_dir, "beef_box")
    stage = composed_scene(source)
    # Do not hide an unrelated/different instance to make the scene look valid.
    for path in (*other_payloads_for("beef_box"), "/World/Assets/BeefTray", "/World/Assets/MwFoodbox"):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsActive():
            raise ValueError("Another microwave payload is active: " + path)
    payload = stage.GetPrimAtPath(PAYLOAD)
    if not payload:
        raise ValueError("Shared payload binding is missing")
    payload.SetActive(True)
    UsdGeom.Imageable(payload).CreateVisibilityAttr().Set("inherited")
    # User-approved repair: fill the missing relation to the existing shared
    # physics material. Do not author friction values or shared asset files.
    from .payload_material import bind_existing, configuration as material_configuration
    material_binding = bind_existing(stage)
    reset = robot_reset_plan(cp, stage)
    placements, cavity_scale = relative_placement_plan(cp, stage)
    # The existing closed door geometry/hinge is retained, not fitted to tiny
    # recorded solver drift. Refuse a source scene whose door is not closed.
    door = stage.GetPrimAtPath(MICROWAVE_ROOT + "/Asset/door")
    cache = UsdGeom.XformCache()
    mw_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MICROWAVE_ROOT))
    door_relative = cache.GetLocalToWorldTransform(door) * mw_world.GetInverse()
    if not np.allclose(np.asarray(door_relative), np.eye(4), rtol=0, atol=1e-9):
        raise ValueError("Input microwave door is not in the existing closed pose")
    before = protected_physics(stage)
    layers = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            path = Path(layer.realPath).resolve(strict=True)
            if not path.is_relative_to(ROOT):
                raise ValueError("USD dependency outside the project: " + str(path))
            layers[path] = sha256(path.read_bytes()).hexdigest()
    # The plate's existing 0.001 geometry-unit conversion stays on its child.
    plate_body = stage.GetPrimAtPath(PLATE + "/Asset")
    matrix = np.asarray(UsdGeom.Xformable(plate_body).GetLocalTransformation())
    if not np.allclose(matrix, np.diag([.001, .001, .001, 1]), rtol=0, atol=1e-9):
        raise ValueError("Unknown plate body frame; cannot copy recorded rigid pose")
    for role, path in (("robot", ROBOT), ("plate", PLATE), ("payload", PAYLOAD)):
        set_world_pose(stage, path, placements[role])
    geometry = validate_payload_geometry(stage, PAYLOAD, "beef_box")
    if protected_physics(stage) != before:
        raise ValueError("Pose authoring changed physical parameters")
    l1_scene.verify(stage)
    output.mkdir(parents=True)
    scene = output / "initial_scene.usda"
    layer = stage.GetRootLayer()
    export = Sdf.Layer.CreateAnonymous()
    export.TransferContent(layer)
    export.subLayerPaths = [os.path.relpath(p, output) for p in layer.subLayerPaths]
    export.customLayerData = dict(physicalInitialStateAccepted=False,
                                 requiresRobotInitializationHook=True,
                                 sourceEpisode=cp.episode, taskFamily="microwave_put")
    if not export.Export(str(scene)):
        raise RuntimeError("Scene export failed")
    reopened = Usd.Stage.Open(str(scene))
    if not reopened or protected_physics(reopened) != before:
        raise ValueError("Exported scene failed physical-parameter preservation")
    validate_payload_geometry(reopened, PAYLOAD, "beef_box")
    material_configuration(reopened)
    if any(sha256(p.read_bytes()).hexdigest() != h for p, h in layers.items()):
        raise ValueError("Source USD changed during preparation")
    scene_hash = sha256(scene.read_bytes()).hexdigest()
    robot = dict(schema="arena.microwave_put.robot_initialization.v1", scene_sha256=scene_hash,
                 source_episode=cp.episode, positions=reset.positions, hold_targets=reset.hold_targets,
                 velocity_policy="existing public reset zeros restored joint velocities")
    receipt = dict(schema="arena.microwave_put.initial_scene.v1", benchmark_id="9.2",
                   sheet_api_id="27-1", runtime_id=None, payload_variant="beef_box",
                   scene=str(scene), scene_sha256=scene_hash, source_episode=cp.episode,
                   robot_initialization="robot_initialization.json", requires_robot_initialization_hook=True,
                   source_checkpoint_manifest_sha256=sha256((checkpoint_dir / "source_manifest.json").read_bytes()).hexdigest(),
                   world_poses_column_vector={k: placements[k].tolist() for k in ("robot", "plate", "payload")},
                   cavity_geometry_scale_preserved=cavity_scale, payload_geometry=geometry,
                   door="existing_closed_pose_preserved", support_stand="not_moved",
                   protected_physical_parameters_unchanged=True, source_usd_unchanged=True,
                   payload_material_binding=material_binding,
                   input_usd_sha256={str(p): h for p, h in layers.items()},
                   source_dynamic_velocity_restored=False, physics_run=False,
                   initial_state_accepted=False, runtime_registered=False,
                   note="Load USD plus the native reset hook; opening USD alone does not set robot joints.")
    if replay_case is not None:
        receipt['replay_case'] = replay_case
    for name, data in (("robot_initialization.json", robot), ("preparation.json", receipt)):
        with (output / name).open("x") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    return receipt


def load_robot_reset(scene):
    """Read the paired reset for a future WorkerTask.robot_initialization hook."""
    from pxr import Usd
    scene = Path(scene).resolve(strict=True)
    if not scene.is_relative_to(TASK):
        raise ValueError("Reset scene must remain inside microwave_put")
    receipt = json.loads((scene.parent / "preparation.json").read_text())
    recorded = json.loads((scene.parent / "robot_initialization.json").read_text())
    digest = sha256(scene.read_bytes()).hexdigest()
    if (receipt.get("schema") != "arena.microwave_put.initial_scene.v1"
            or receipt.get("payload_variant") != "beef_box"
            or recorded.get("schema") != "arena.microwave_put.robot_initialization.v1"
            or receipt.get("scene_sha256") != digest or recorded.get("scene_sha256") != digest):
        raise ValueError("Scene/reset identity or hash mismatch")
    for name, expected in receipt["input_usd_sha256"].items():
        path = Path(name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Initial scene dependency changed: " + str(path))
    from .replay_cases import checkpoint_for
    checkpoint_dir = checkpoint_for(receipt.get('replay_case'))
    manifest = checkpoint_dir / "source_manifest.json"
    if sha256(manifest.read_bytes()).hexdigest() != receipt.get("source_checkpoint_manifest_sha256"):
        raise ValueError("Source checkpoint manifest changed")
    cp = load_checkpoint(checkpoint_dir, "beef_box")
    if recorded.get("source_episode") != cp.episode or receipt.get("source_episode") != cp.episode:
        raise ValueError("Wrong source checkpoint")
    stage = Usd.Stage.Open(str(scene))
    validate_payload_geometry(stage, PAYLOAD, "beef_box")
    plan = robot_reset_plan(cp, stage)
    if recorded.get("positions") != plan.positions or recorded.get("hold_targets") != plan.hold_targets:
        raise ValueError("Stored reset differs from the named source checkpoint")
    return plan

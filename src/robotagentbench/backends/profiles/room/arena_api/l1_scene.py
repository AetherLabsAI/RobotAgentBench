"""L1 baseline transplant; originals remain immutable, all changes are overlays."""
from pathlib import Path
import math
from arena_api import beef_asset, food_box_assets, coffee_assets, robot_drive_defaults

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/l1_scene"
LAYERS = (coffee_assets.SCENE_LAYER, food_box_assets.SCENE_LAYER, beef_asset.SCENE_LAYER,
          ROOT / "resources/house_scene_l1/approved_microwave.usda",
          ROOT / "resources/l1_scene/baseline_alignment.usda",
          ROOT / "resources/l1_scene/fridge_recovered.usda")
BASE = ROOT / "assets/house_scene_l1/demo_sim/scene_room_baseline.usda"
POLICY = "house.l1_scene_port.v2"
MARKER = "arena:l1ScenePort"
FRIDGE = "/World/Background/Shell/RefrigeratorRecovered"
BASKET = "/World/Assets/TableclothBasket"
DRIVES = {name: values for name, values in robot_drive_defaults.DRIVES.items() if 'gripper_' not in name}


def verify(stage):
    from pxr import UsdGeom, UsdPhysics
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, abs_tol=1e-9):
        raise ValueError("L1 scene must retain metersPerUnit=1")
    if any("pilot" in layer.realPath.lower() for layer in stage.GetUsedLayers()):
        raise ValueError("Legacy pilot scene layers are not a clean L1 baseline")
    for name, (axis, *values) in DRIVES.items():
        prim = stage.GetPrimAtPath("/World/Robot/joints/" + name)
        for field, expected in zip(("stiffness", "damping", "maxForce"), values):
            value = prim.GetAttribute(f"drive:{axis}:physics:{field}").Get()
            if value != expected:
                raise ValueError(f"L1 drive mismatch: {name}.{field}={value}")
    for color in ("Red", "Teal"):
        prim = stage.GetPrimAtPath(f"/World/Assets/TrashBin{color}/geo/Trash_Bag/Trash_Bag_002")
        if not prim or prim.GetAttribute("physics:collisionEnabled").Get() is not True:
            raise ValueError("L1 trash-bag collision missing: " + color)
    if not stage.GetPrimAtPath(BASKET + "/model/Geometry"):
        raise ValueError("L1 basket reference is not loaded")
    for child in ("floor", "wxn", "wxp", "wyn", "wyp"):
        prim = stage.GetPrimAtPath(BASKET + "/" + child)
        if prim and prim.IsActive():
            raise ValueError("Legacy basket proxy still active: " + child)
    group = stage.GetPrimAtPath("/World/CanonicalTableclothBasketCollisionGroup")
    if [str(p) for p in group.GetRelationship("physics:filteredGroups").GetTargets()] != [
            "/World/CanonicalRobotCollisionGroup"]:
        raise ValueError("L1 basket collision group is missing")
    old = stage.GetPrimAtPath("/World/Background/Shell/KitchenRefrigerator")
    if old and old.IsActive():
        raise ValueError("Old and recovered refrigerators must not coexist")
    handle = stage.GetPrimAtPath(FRIDGE + "/right_door/col_handle")
    if not handle or not handle.IsA(UsdGeom.Mesh) or not handle.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Recovered refrigerator handle mesh is missing")
    matrix = UsdGeom.Xformable(handle).GetLocalTransformation()
    if any(not math.isclose(matrix[i][j], float(i == j), abs_tol=1e-8)
           for i in range(4) for j in range(4)):
        raise ValueError("Recovered handle inherited a stale proxy transform")


def apply(stage):
    """Call before physics/reset, after appearance selection. Layout scenes are not reset."""
    from pxr import Sdf, Usd
    world = stage.GetPrimAtPath("/World")
    marker = world.GetAttribute(MARKER).Get() if world else None
    if marker not in (None, POLICY):
        raise ValueError("Unsupported L1 port revision")
    if marker is None:
        for path in LAYERS:
            if not path.is_file():
                raise FileNotFoundError(path)
        session = stage.GetSessionLayer()
        session.subLayerPaths = [str(p) for p in LAYERS] + list(session.subLayerPaths)
        with Usd.EditContext(stage, session):
            stage.GetPrimAtPath("/World").CreateAttribute(MARKER, Sdf.ValueTypeNames.String).Set(POLICY)
    # A prepared scene can carry the old L1 marker yet reference the old beef.
    # Always bind the shared revision, without resetting poses or activation.
    beef = beef_asset.apply(stage)
    food_boxes = food_box_assets.apply(stage)
    coffee = coffee_assets.apply(stage)
    drives = robot_drive_defaults.apply(stage)
    verify(stage)
    return dict(policy_id=POLICY, source_commit="b64c8a19c5e9eee0c93554dbb8c73444bb0c5004",
                prepared_input=marker == POLICY, source_assets_modified=False,
                original_fridge_root=FRIDGE, retained_approved_physics_repairs=True,
                shared_beef_asset=beef, shared_food_box_assets=food_boxes, shared_coffee_assets=coffee,
                shared_robot_drives=drives)

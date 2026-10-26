"""Scene-owner fixture for isolated cloth physics diagnostics, not a room layout.

Only creates static, collidable supports. The task's native scene preparation
creates the single cloth before physics attachment. No robot, grasp constraints,
task scoring or runtime geometry writes are introduced here.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

BODY_ROOT = "/World/Assets/TableclothRuleState"
MESH_PATH = BODY_ROOT + "/Surface"
TABLE_TOP_M = .75
DROP_HEIGHT_M = .025
FIXTURE_SCOPE = "isolated_component_preview"
DIAGNOSTIC_CASES = ("source_support", "edge_drape")


def edge_drape_definition():
    """Explicit geometry-only diagnostic, never a benchmark initial recipe."""
    return dict(schema="openend.short_task.gpu_cloth_geometry_diagnostic.v1",
        case="edge_drape", task_name="collect_tablecloth",
        source="short_tasks.gpu_cloth.fixture.v1", benchmark=False,
        table_size_m=[.24, .9, .05], cloth_size_m=[.42, .62],
        overhang_x_each_side_m=.09, initial_support_area_fraction=.24/.42)


def describe(task_name, *, seed=47, layout_id=None, diagnostic_case="source_support"):
    if task_name not in ("spread_tablecloth", "collect_tablecloth"):
        raise ValueError("The fixture only provides the two document cloth tasks")
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("A valid public scene seed is required")
    if layout_id is not None and (type(layout_id) is not int or not 1 <= layout_id <= 50):
        raise ValueError("Invalid requested layout identifier")
    if diagnostic_case not in DIAGNOSTIC_CASES:
        raise ValueError("Unknown isolated cloth geometry diagnostic")
    if diagnostic_case == "edge_drape" and task_name != "collect_tablecloth":
        raise ValueError("The explicit edge-drape diagnostic uses the collect preview only")
    source_x = 1.15 if task_name == "spread_tablecloth" else 0.
    profile = dict(schema="openend.short_task.gpu_cloth_scene.v1",
        body_root=BODY_ROOT, mesh=MESH_PATH,
        initial_center_world_m=[source_x, 0., TABLE_TOP_M + DROP_HEIGHT_M], initial_yaw_rad=0.,
        table_region=dict(center_world_m=[0., 0., .725], size_m=[1.2, .9, .05],
            prim_path="/World/Assets/CoffeeTable"),
        storage_region=dict(center_world_m=[1.15, 0., .725], size_m=[.85, .85, .05],
            prim_path="/World/Assets/ClothStorage"),
        # This is the actual collidable container envelope, not an invented in-basket success rule.
        basket_region=dict(center_world_m=[-1.15, 0., .825], size_m=[.75, .85, .4],
            prim_path="/World/Assets/ClothBasket"),
        source_role="cloth_storage" if task_name == "spread_tablecloth" else "coffee_table",
        destination_role="coffee_table" if task_name == "spread_tablecloth" else "cloth_basket",
        meters_per_unit=1., up_axis="Z", occupancy=dict(table_clear=True, cloth_clear=True))
    this = Path(__file__).resolve()
    result = dict(schema="openend.short_task.fixture.v1", seed=seed,
        layout_id=layout_id, actual_layout_id=None, layout_scope=FIXTURE_SCOPE,
        initialization_owner="gpu_cloth_scene_owner.v1", scene_generated=True,
        full_room_layout=False, robot_loaded=False, gpu_cloth=profile,
        tasks=[dict(instance_id="gpu_cloth", type="gpu_tablecloth", parameters={})],
        objects=[dict(id="tablecloth", path=BODY_ROOT, role="tablecloth", task_id="gpu_cloth",
            initialization_owner="gpu_cloth_scene_owner.v1"),
            dict(id="fixture_table", path="/World/Assets/CoffeeTable", role="coffee_table"),
            dict(id="fixture_storage", path="/World/Assets/ClothStorage", role="cloth_storage"),
            dict(id="fixture_cloth_basket", path="/World/Assets/ClothBasket", role="cloth_basket")],
        source_hashes={str(this): hashlib.sha256(this.read_bytes()).hexdigest()},
        fixture_basket_interior=dict(center_world_m=[-1.15, 0., .85], size_m=[.7, .8, .35],
            scoring_predicate=False),
        diagnostic_support=dict(top_z_m=TABLE_TOP_M, source_center_xy_m=[source_x, 0.],
            initial_drop_height_m=DROP_HEIGHT_M, geometry_owner="short_tasks.gpu_cloth.fixture.v1"))
    if diagnostic_case == "edge_drape":
        result["geometry_diagnostic"] = edge_drape_definition()
        result["gpu_cloth"]["table_region"]["size_m"] = list(result["geometry_diagnostic"]["table_size_m"])
    return result


def author(stage, manifest):
    """Author fixture geometry in a fresh anonymous stage before simulation."""
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    if not stage.GetEditTarget().GetLayer().anonymous:
        raise ValueError("Fixture construction only permits an anonymous scene layer")
    if stage.GetPrimAtPath("/World"):
        raise ValueError("The isolated fixture must not overwrite an existing world")
    if manifest.get("layout_scope") != FIXTURE_SCOPE or manifest.get("full_room_layout") is not False:
        raise ValueError("The fixture must not claim a complete benchmark room layout")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Assets")
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    apply_physics_profile(stage)

    colliders = []
    def box(path, center, size, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3d(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        api.CreateContactOffsetAttr(.002)
        api.CreateRestOffsetAttr(0.)
        colliders.append(path)
    box("/World/Floor", [0, 0, -.025], [4.5, 2., .05], [.16, .18, .2])
    table = manifest["gpu_cloth"]["table_region"]
    box("/World/Assets/CoffeeTable", table["center_world_m"], table["size_m"], [.43, .27, .14])
    box("/World/Assets/ClothStorage", [1.15, 0, .725], [.85, .85, .05], [.27, .36, .5])
    UsdGeom.Xform.Define(stage, "/World/Assets/ClothBasket")
    basket = "/World/Assets/ClothBasket/"
    box(basket + "Base", [-1.15, 0, .65], [.75, .85, .05], [.48, .42, .29])
    box(basket + "Left", [-1.5125, 0, .85], [.025, .85, .35], [.48, .42, .29])
    box(basket + "Right", [-.7875, 0, .85], [.025, .85, .35], [.48, .42, .29])
    box(basket + "Front", [-1.15, -.4125, .85], [.7, .025, .35], [.48, .42, .29])
    box(basket + "Back", [-1.15, .4125, .85], [.7, .025, .35], [.48, .42, .29])
    result = deepcopy(manifest)
    result["fixture_colliders"] = colliders
    return result


def apply_physics_profile(stage):
    """Reassert GPU configuration after World construction, before reset."""
    from pxr import PhysxSchema
    prim = stage.GetPrimAtPath("/World/PhysicsScene")
    if not prim:
        raise ValueError("Fixture has no physical scene")
    api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    api.CreateEnableGPUDynamicsAttr(True)
    api.CreateBroadphaseTypeAttr("GPU")
    api.CreateSolverTypeAttr("TGS")
    api.CreateTimeStepsPerSecondAttr(240)
    return dict(physics_backend="physx", gpu_dynamics=True, broadphase="GPU", solver="TGS",
        physics_dt_s=1/240, physics_hz=240, lifetime="entire_episode")

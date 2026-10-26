from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/physics_fluid"))
import tablecloth_scene_binding as binding


@pytest.mark.parametrize("task", ["8.7", "8.8", "8.9", "8.10"])
def test_cloth_catalog_station_is_shared_by_physics_and_house(task):
    assert binding.station_xyyaw(task) == pytest.approx([-2.9193, -0.48654358305158935, 0])


def test_unrelated_catalog_task_cannot_use_cloth_adapter():
    with pytest.raises(ValueError, match="tablecloth station"):
        binding.task_environment("7.2")


def test_old_bbox_only_tracks_are_not_current_collision_geometry():
    with pytest.raises(ValueError, match="remeasure"):
        binding.load_scene_collisions({"basket_aabb_min_m": np.zeros(3)})


@pytest.fixture
def stage():
    pytest.importorskip("pxr")
    from pxr import Usd
    return Usd.Stage.Open(str(ROOT / "arena_scene/v1_2/demo_sim/scene_room_baseline.usda"))


def test_current_basket_is_open_mesh_and_riser_not_old_boxes(stage):
    data = binding.export_scene_collisions(stage)
    meshes = binding.load_scene_collisions(data)
    basket = [m for m in meshes if m[0]["group"] == "basket"]
    assert {e["prim_path"] for e, _, _ in basket} == {binding.BASKET_COLLISION_PATH, binding.BASKET_RISER_PATH}
    assert max(len(f) for _, _, f in basket) > 1000
    binding.verify_scene_collisions(stage, data)
    # Same AABB and file size would miss a changed interior surface.
    mesh = next(e for e, _, _ in basket if e["prim_path"] == binding.BASKET_COLLISION_PATH)
    data[f"scene_points_{mesh['key']}"][100, 2] += 0.001
    with pytest.raises(ValueError, match="differs"):
        binding.verify_scene_collisions(stage, data)


def test_cube_winding_survives_mirrored_world_transform(stage):
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(binding.BASKET_RISER_PATH)
    UsdGeom.Xformable(prim).AddScaleOp(opSuffix="testMirror").Set(Gf.Vec3f(-1, 1, 1))
    meshes = binding.load_scene_collisions(binding.export_scene_collisions(stage))
    _, points, faces = next(m for m in meshes if m[0]["prim_path"] == binding.BASKET_RISER_PATH)
    triangles = points[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.sum(normals * (triangles.mean(axis=1) - points.mean(axis=0)), axis=1) > 0)

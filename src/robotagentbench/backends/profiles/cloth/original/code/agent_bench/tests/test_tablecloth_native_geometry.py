"""Contact repair must preserve concavity, geometry, and pre-start state rules."""
import ast
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from experiments.physics_fluid.tablecloth_native_geometry import prepare_static_contact_mesh


def make_scene():
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/World/Table')
    parent.AddTranslateOp().Set((1.2, -2.3, .4))
    parent.AddRotateZOp().Set(37.)
    parent.AddScaleOp().Set((.1, .11, .12))
    mesh = UsdGeom.Mesh.Define(stage, '/World/Table/Mesh')
    # Two separated triangles. Any convexification would fill their gap.
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (3, 0, 0), (4, 0, 0), (4, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 6)
    stage.DefinePrim('/World/Table/Mesh/Looks', 'Scope')
    stage.DefinePrim('/World/Table/Mesh/Looks/Material', 'Material')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    return stage, mesh


def world_vertices(mesh):
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    return np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])


def test_layout_and_triangle_gap_preserved_in_anonymous_stage():
    source, original = make_scene()
    source_text = source.GetRootLayer().ExportToString()
    stage = Usd.Stage.Open(source.Flatten())
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Table/Mesh'))
    points, indices = world_vertices(mesh), list(mesh.GetFaceVertexIndicesAttr().Get())
    report = prepare_static_contact_mesh(stage, str(mesh.GetPath()))
    np.testing.assert_allclose(world_vertices(mesh), points, atol=3e-7, rtol=0)
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == indices
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'none'
    assert source.GetRootLayer().ExportToString() == source_text
    assert report['added_support_bodies'] == 0
    assert len(list(stage.Traverse())) == len(list(source.Traverse()))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(mesh.GetNormalsAttr().Get()), axis=1), 1)


@pytest.mark.parametrize('unsafe', ['dynamic', 'animated', 'descendants'])
def test_unsafe_bake_rejected_without_mutation(unsafe):
    stage, mesh = make_scene()
    if unsafe == 'dynamic':
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Table'))
    elif unsafe == 'animated':
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Table')).GetOrderedXformOps()[0].Set((2, 3, 4), 1.)
    else:
        UsdGeom.Xform.Define(stage, '/World/Table/Mesh/Child')
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        prepare_static_contact_mesh(stage, str(mesh.GetPath()))
    assert stage.GetRootLayer().ExportToString() == before


def test_replay_has_no_runtime_joint_state_setters():
    source = Path(__file__).resolve().parents[1] / 'experiments/physics_fluid/tablecloth_physx_replay.py'
    tree = ast.parse(source.read_text())
    forbidden = {'set_joint_positions', 'set_joint_velocities', 'set_world_pose', 'set_world_poses'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]


@pytest.mark.parametrize('object_name', ['table', 'basket'])
def test_current_house_preserves_every_world_vertex_and_polygon(object_name):
    from real2sim.teleop.tablecloth_basket_contract import BASKET_COLLISION_PATH
    source = Path(__file__).resolve().parents[1] / 'arena_scene/v1_2/demo_sim/scene_room_baseline.usda'
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
    path = '/World/Assets/CoffeeTable/Mesh' if object_name == 'table' else BASKET_COLLISION_PATH
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    before = world_vertices(mesh)
    faces = list(mesh.GetFaceVertexIndicesAttr().Get())
    report = prepare_static_contact_mesh(stage, path)
    np.testing.assert_allclose(world_vertices(mesh), before, atol=3e-7, rtol=0)
    assert faces == list(mesh.GetFaceVertexIndicesAttr().Get())
    assert report['collision_approximation_unchanged']

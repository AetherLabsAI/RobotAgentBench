"""CPU USD structure checks; not a physical settle or native push test."""
import unittest
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import numpy as np
from .build_initial import authorize_dynamic_body, MACHINE_BODY, ALLOWED
from .geometry import static_collision_vertices
from .push_motion import palm_fit
from .geometry import main_case_shapes


class UsdTests(unittest.TestCase):
    def test_open_palm_only_articulates_actor_thumb_not_source(self):
        from .open_palm import OpenPalmHand
        asset = OpenPalmHand()
        points, bones = asset.sample(0.)
        np.testing.assert_array_equal(asset.rest_points, asset.data['frames'][0]['points'])
        for key, matrix in bones.items():
            if not key.startswith('thumb.'):
                np.testing.assert_array_equal(matrix, asset.data['rest_bones'][key])
        groups = asset.collision_groups()
        thumb = np.unique(np.concatenate([ids for k, ids in groups.items() if k.startswith('thumb.')]))
        self.assertLessEqual(points[thumb, 2].max(), points[groups['hand.R'], 2].max() + 1e-12)
        with self.assertRaises(ValueError):
            asset.sample(1.)

    def test_no_hopper_fallback_for_new_palm(self):
        from .open_palm import OpenPalmHand
        from itertools import product
        vertices = np.array(list(product((-.1, .1), (-.1, .1), (.92, 1.22))))
        with self.assertRaisesRegex(ValueError, 'lower case side walls'):
            palm_fit(OpenPalmHand(), vertices, [1., 0., 0.], machine_shapes_world=[vertices])

    def test_main_case_selection_excludes_head_and_hopper(self):
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, MACHINE_BODY).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        for name in ('Left', 'Right', 'Back', 'Head', 'HopperFloor'):
            shape = UsdGeom.Cube.Define(stage, MACHINE_BODY + '/Collision/' + name)
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10 if name in ('Head', 'HopperFloor') else 0))
        shapes = main_case_shapes(stage, MACHINE_BODY)
        self.assertEqual(len(shapes), 2)
        self.assertLess(np.concatenate(shapes)[:, 2].max(), 10)
        shapes_with_back = main_case_shapes(stage, MACHINE_BODY, include_back=True)
        self.assertEqual(len(shapes_with_back), 3)
        self.assertLess(np.concatenate(shapes_with_back)[:, 2].max(), 10)

    def test_exactly_one_physical_property_changed(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, MACHINE_BODY).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(5.)
        shape = UsdGeom.Cube.Define(stage, MACHINE_BODY + '/Collision/Side')
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/Task75/Machine/Joints/Button')
        joint.CreateBody0Rel().SetTargets([MACHINE_BODY])
        joint.CreateLowerLimitAttr(-.005)
        report = authorize_dynamic_body(stage)
        self.assertEqual(report['changed_properties'], [ALLOWED])
        self.assertEqual(prim.GetAttribute('physics:mass').Get(), 5.)
        self.assertEqual([str(p) for p in joint.GetBody0Rel().GetTargets()], [MACHINE_BODY])
        with self.assertRaises(ValueError):
            authorize_dynamic_body(stage)

    def test_existing_scaled_static_counter_is_read_not_modified(self):
        stage = Usd.Stage.CreateInMemory()
        cube = UsdGeom.Cube.Define(stage, '/Counter')
        cube.CreateSizeAttr(2.)
        cube.AddScaleOp().Set(Gf.Vec3f(2., 3., .05))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        before = stage.GetRootLayer().ExportToString()
        points = static_collision_vertices(stage, '/Counter')
        np.testing.assert_allclose(points.min(0), [-2., -3., -.05], atol=1e-8)
        np.testing.assert_allclose(points.max(0), [2., 3., .05], atol=1e-8)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_actual_unchanged_hand_can_present_palm(self):
        from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
        from itertools import product
        asset = AdaptiveHand()
        before, _ = asset.sample(0.)
        vertices = np.array(list(product((-.1, .1), (-.1, .1), (.92, 1.22))))
        protected = np.array([[-.04, -.04, .94], [.04, .04, 1.07]])
        for direction in ([1., 0., 0.], [-1., 0., 0.]):
            fit = palm_fit(asset, vertices, direction, avoid_boxes=[protected])
            self.assertTrue(fit['palm_is_leading_shape'])
            self.assertAlmostEqual(fit['palm_gap_m'], 0.)
            orientation = np.asarray(fit['orientation'])
            np.testing.assert_allclose(orientation.T @ orientation, np.eye(3), atol=1e-14)
            points = before @ orientation.T + fit['wrist_position_m']
            self.assertGreaterEqual(float(points[:, 2].min()), protected[1, 2] - 1e-14)
        np.testing.assert_array_equal(asset.sample(0.)[0], before)

    def test_upper_case_contact_does_not_use_protruding_feet(self):
        from itertools import product
        from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
        asset = AdaptiveHand()
        case = np.array(list(product((-.1, .1), (-.1, .1), (.94, 1.22))))
        feet = np.array(list(product((-.12, .12), (-.12, .12), (.92, .93))))
        all_points = np.concatenate([case, feet])
        protected = np.array([[-.04, -.04, .94], [.04, .04, 1.07]])
        outer = palm_fit(asset, all_points, [1., 0., 0.], avoid_boxes=[protected])
        real = palm_fit(asset, all_points, [1., 0., 0.], avoid_boxes=[protected], machine_shapes_world=[case, feet])
        shift = np.array(real['wrist_position_m']) - outer['wrist_position_m']
        self.assertAlmostEqual(shift[0], .02)
        np.testing.assert_allclose(shift[1:], [0., 0.])
        self.assertEqual(real['side_plane_source'], 'height_clipped_existing_collision_shapes')


if __name__ == '__main__':
    unittest.main()

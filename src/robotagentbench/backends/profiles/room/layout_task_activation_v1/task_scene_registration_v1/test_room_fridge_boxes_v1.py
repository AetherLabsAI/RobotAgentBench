"""CPU USD contract tests; NOT native settling or manipulation tests."""
import json
import math
import unittest
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import initialize_room_fridge_boxes_v1 as init
from complete_coffee_initialization_v1 import fingerprint


class SamplerTests(unittest.TestCase):
    lengths={'beef':.243,'salad':.19246360467077572,'food':.23539205566048624}

    def test_reproducibility(self):
        self.assertEqual(init.sample(42,self.lengths),init.sample(42,self.lengths))
        self.assertNotEqual(init.sample(42,self.lengths),init.sample(43,self.lengths))

    def test_sampling_contract(self):
        combinations=set()
        for seed in range(1000):
            rows=init.sample(seed,self.lengths); combinations.add(tuple(r['slot'] for r in rows))
            self.assertEqual(len({r['slot'] for r in rows}),3)
            for row in rows:
                x,level=init.SLOTS[row['slot']]; pos=row['door_local_position_m']
                self.assertLessEqual(abs(pos[0]-x),.03050001)
                z=.354-self.lengths[row['key']]/2-.005-.010
                self.assertLessEqual(abs(pos[2]-z),.00650001)
                self.assertAlmostEqual(sum(v*v for v in row['door_local_quaternion_wxyz']),1)
                self.assertLessEqual(abs(row['tilt_jitter_deg']),6)
                self.assertLessEqual(abs(row['roll_jitter_deg']),3)
                self.assertGreaterEqual(row['yaw_deg']-init.straight(level),-.05001)
                self.assertLessEqual(row['yaw_deg']-init.straight(level),20.05001)
        self.assertEqual(len(combinations),24)

    def test_requires_seed(self):
        with self.assertRaises(ValueError): init.sample(None,self.lengths)


class StageTests(unittest.TestCase):
    def make(self, translation=(0,0,0), yaw=0):
        s=Usd.Stage.CreateInMemory()
        from arena_api.l1_scene import BASE
        # Real shared material dependency, not a fabricated friction fixture.
        s.DefinePrim('/World/PhysicsMaterials/GripMat','Material').GetReferences().AddReference(str(BASE),'/World/PhysicsMaterials/GripMat')
        p=UsdGeom.Xform.Define(s,init.DOOR)
        m=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw))
        m.SetTranslateOnly(Gf.Vec3d(*translation));p.MakeMatrixXform().Set(m)
        return s

    def test_physics_and_geometry_unchanged(self):
        s=self.make((1,2,3),37); rows=init.apply(s,seed=42)
        for r in rows:
            native=Usd.Stage.Open(r['entry'])
            self.assertEqual(fingerprint(s,r['path']),fingerprint(native,str(native.GetDefaultPrim().GetPath())))
            self.assertFalse(r['native_physics_validated'])
        exported=s.GetRootLayer().ExportToString()
        self.assertNotIn('physics:mass',exported)
        self.assertNotIn('physics:staticFriction',exported)
        self.assertIn('material:binding:physics',exported)
        self.assertNotIn('physx',exported)

    def test_room_frame_and_idempotence(self):
        a=self.make();b=self.make((-3,7,1),93)
        ra=init.apply(a,seed=47);rb=init.apply(b,seed=47)
        for x,y in zip(ra,rb):
            self.assertEqual(x['door_local_position_m'],y['door_local_position_m'])
            self.assertLess(y['door_relative_matrix_error'],1e-8)
        self.assertEqual(rb,init.apply(b,seed=47))
        self.assertEqual(sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in b.Traverse()),3)

    def test_reject_unknown_target(self):
        s=self.make();s.DefinePrim(init.ASSETS['food'][2],'Xform')
        with self.assertRaisesRegex(ValueError,'occupied'): init.apply(s,seed=42)

    def test_missing_door(self):
        with self.assertRaisesRegex(ValueError,'right door'): init.apply(Usd.Stage.CreateInMemory(),seed=42)

    def test_real_room_pose_preserved(self):
        manifests=[init.HERE/'room_demo_v6/composition.json',*sorted((init.HERE/'coffee_initialized_rooms_v1').glob('layout*/composition.json'))]
        self.assertGreaterEqual(len(manifests),3)
        for i,manifest in enumerate(manifests):
            scene=json.loads(manifest.read_text())['scene'];s=Usd.Stage.Open(scene)
            s.SetEditTarget(s.GetSessionLayer())
            old={str(p.GetPath()):str(UsdGeom.Xformable(p).GetLocalTransformation()) for p in s.Traverse() if p.IsA(UsdGeom.Xformable)}
            rows=init.apply(s,seed=42+i)
            new_paths={r['path'] for r in rows}
            for path,xform in old.items():
                if path not in new_paths:self.assertEqual(str(UsdGeom.Xformable(s.GetPrimAtPath(path)).GetLocalTransformation()),xform)


if __name__=='__main__':unittest.main()

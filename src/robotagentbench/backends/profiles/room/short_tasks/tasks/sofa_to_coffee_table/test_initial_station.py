import math
import unittest
from .initial_station import policy, rigid_delta, sample_candidates

IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

class InitialStationTests(unittest.TestCase):
    def sample(self, asset="book_red", seed=42, count=2000, delta=IDENTITY):
        return sample_candidates(seed, delta, dict(asset=asset,source="sofa"),count=count)

    def test_exact_approved_rectangle(self):
        p=policy()
        self.assertEqual(p["front_distance_m"], [.366,.698])
        self.assertEqual(p["right_offset_m"], [-.080,.461])
        self.assertEqual(p["yaw_deviation_deg"], [-5,5])

    def test_all_assets_use_identical_proposals(self):
        expected=self.sample()
        for asset in policy()["assets"]:
            self.assertEqual(expected,self.sample(asset))

    def test_full_rectangle_covered(self):
        rows=self.sample(count=10000)
        for key,lo,hi in [("edge_distance_m",.366,.698),("right_offset_m",-.080,.461),
                          ("yaw_deviation_deg",-5,5)]:
            vs=[r[key] for r in rows]
            self.assertTrue(all(lo <= v <= hi for v in vs))
            self.assertLess(min(vs),lo+(hi-lo)*.01)
            self.assertGreater(max(vs),hi-(hi-lo)*.01)
        # Merged rectangle, not a mixture of the old per-object clusters.
        for near in (False,True):
            for right in (False,True):
                self.assertTrue(any((r["edge_distance_m"] < .532)==near and
                    (r["right_offset_m"] > .1905)==right for r in rows))

    def test_seed_determinism(self):
        self.assertEqual(self.sample(),self.sample())
        self.assertNotEqual(self.sample(),self.sample(seed=43))

    def test_canonical_distance_and_right_sign(self):
        for r in self.sample(count=20):
            x,y,yaw=r["initial_base_xyyaw"]
            self.assertAlmostEqual(x,-.3503563938-r["edge_distance_m"])
            self.assertAlmostEqual(y,-.871-r["right_offset_m"])
            self.assertAlmostEqual(yaw,math.radians(r["yaw_deviation_deg"]))

    def test_translated_rotated_layouts(self):
        reference=self.sample(count=20)
        for deg in (0,45,90,180,270):
            a=math.radians(deg);c=math.cos(a);s=math.sin(a)
            delta=[[c,s,0,0],[-s,c,0,0],[0,0,1,0],[3,-2,0,1]]
            for old,new in zip(reference,self.sample(count=20,delta=delta)):
                x,y,yaw=old["initial_base_xyyaw"]; xx,yy,hh=new["initial_base_xyyaw"]
                self.assertAlmostEqual(xx,x*c-y*s+3)
                self.assertAlmostEqual(yy,x*s+y*c-2)
                self.assertAlmostEqual(math.cos(hh),math.cos(yaw+a))
                self.assertAlmostEqual(math.sin(hh),math.sin(yaw+a))

    def test_reject_invalid_geometry(self):
        for i,j,value in [(0,0,2),(0,0,-1),(3,2,.1),(0,1,.2),(2,2,0),(0,0,float("nan"))]:
            a=[r[:] for r in IDENTITY];a[i][j]=value
            with self.assertRaises(ValueError):rigid_delta(a)
        with self.assertRaises(ValueError):rigid_delta([[1]])

    def test_reject_invalid_selection(self):
        for asset,seed,count,source in [("cap",1,1,"sofa"),("wallet",True,1,"sofa"),
                ("wallet",0,1,"sofa"),("wallet",1,0,"sofa"),("wallet",1,1,"table")]:
            with self.assertRaises(ValueError):
                sample_candidates(seed,IDENTITY,dict(asset=asset,source=source),count=count)

    def test_not_a_physics_certificate(self):
        for r in self.sample(count=2):
            self.assertFalse(r["native_physics_validated"])
            self.assertFalse(r["robot_room_geometry_checked"])
            self.assertNotIn("targets_rad",r)

if __name__ == "__main__":
    unittest.main()

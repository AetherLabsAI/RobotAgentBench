import unittest
from .position_domain import bounds, fits, policy, sample_positions


class PositionDomainTests(unittest.TestCase):
    def test_matches_hat_domain(self):
        a, b = policy()['footprint_bounds_uv_m']
        self.assertAlmostEqual(a[0], -1.835/3+.03)
        self.assertAlmostEqual(b[1], .5214424585367384-.1)

    def test_four_assets_supported(self):
        self.assertEqual(set(policy()['assets']), {'book_red','remote_tv','wallet','glasses_case'})

    def test_each_complete_footprint_fits(self):
        footprint = [[-.12,-.08],[.12,-.08],[.12,.08],[-.12,.08]]
        for asset in policy()['assets']:
            rows = sample_positions(asset, 42, footprint, count=1000)
            self.assertTrue(all(fits(r['footprint_uv']) for r in rows))
            self.assertGreater(max(r['origin_uv_m'][0] for r in rows)-min(r['origin_uv_m'][0] for r in rows), .8)

    def test_seed_repeatable_but_changes(self):
        f = [[0,0],[.1,0],[.1,.1],[0,.1]]
        self.assertEqual(sample_positions('wallet',42,f,count=10), sample_positions('wallet',42,f,count=10))
        self.assertNotEqual(sample_positions('wallet',42,f,count=10), sample_positions('wallet',43,f,count=10))

    def test_offcenter_origin_uses_full_mesh(self):
        f = [[2,3],[2.2,3],[2.2,3.1],[2,3.1]]
        self.assertTrue(all(fits(r['footprint_uv']) for r in sample_positions('glasses_case',1,f,count=50)))

    def test_center_inside_does_not_allow_mesh_outside(self):
        self.assertFalse(fits([[-.8,.1],[.8,.1],[.8,.2],[-.8,.2]]))

    def test_keep_backrest_clearance(self):
        self.assertFalse(fits([[-.1,.4],[.1,.4],[.1,.45],[-.1,.45]]))

    def test_oversize_fails_without_widening(self):
        with self.assertRaises(ValueError):
            sample_positions('book_red',1,[[0,0],[2,0],[2,1],[0,1]],count=1)

    def test_bad_input_rejected(self):
        f = [[0,0],[.1,0],[.1,.1]]
        for asset, seed, count in [('baseball_cap',1,1),('wallet',True,1),('wallet',0,1),('wallet',1,0)]:
            with self.assertRaises(ValueError): sample_positions(asset,seed,f,count=count)
        with self.assertRaises(ValueError): bounds([[0,0],[float('nan'),1],[1,1]])

    def test_no_angle_height_or_robot_override(self):
        r = sample_positions('remote_tv',42,[[0,0],[.1,0],[.1,.1]],count=1)[0]
        self.assertFalse({'yaw','z','joints','robot_position'} & set(r))


if __name__ == '__main__': unittest.main()

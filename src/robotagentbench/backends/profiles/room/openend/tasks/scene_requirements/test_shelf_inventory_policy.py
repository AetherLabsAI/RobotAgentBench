import unittest
from shelf_inventory_policy import compose, verify_placed_inventory


class ShelfInventoryTests(unittest.TestCase):
    def test_required_four_and_background(self):
        required=['coke_c3','gatorade_m','oreo_gf','zbar_t']
        for seed in range(100):
            plan=compose(dict(seed=seed,required_categories=required))
            rows=[o for layer in plan['layers'] for o in layer['objects']]
            self.assertEqual(sum(o['required_instance'] for o in rows),4)
            self.assertTrue(verify_placed_inventory(plan,rows)['inventory_matches_plan'])
            self.assertTrue(all(4<=layer['count']<=10 for layer in plan['layers']))
            self.assertEqual(plan,compose(dict(seed=seed,required_categories=required)))

    def test_seed42_preserves_original_counts(self):
        plan=compose(dict(seed=42,required_categories=['gatorade_m']))
        self.assertEqual([layer['count'] for layer in plan['layers']],[5,6])
        rows=[o for layer in plan['layers'] for o in layer['objects']]
        self.assertEqual(sum(o['required_instance'] for o in rows),1)
        with self.assertRaises(ValueError):verify_placed_inventory(plan,rows[:-1])


if __name__=='__main__':unittest.main()

import copy
from collections import Counter
import unittest

from compose_shelf_inventory import compose, POOLS, verify_placed_inventory


def rows(plan):
    return [o for layer in plan["layers"] for o in layer["objects"]]


class ShelfInventoryTests(unittest.TestCase):
    def test_all_required_over_many_seeds(self):
        raised = False
        for seed in range(500):
            for required in ([], ["coke_c3", "oreo_gf"], list(sum(POOLS.values(), ()))):
                plan = compose({"seed": seed, "required_categories": required})
                self.assertTrue(plan["required_category_coverage_passed"])
                self.assertTrue(set(required) <= {o["product_family"] for o in rows(plan)})
                self.assertEqual(len(rows(plan)), len({o["instance_id"] for o in rows(plan)}))
                for layer in plan["layers"]:
                    n = layer["count"]
                    pool = POOLS[layer["support_id"]]
                    counts = Counter(o["product_family"] for o in layer["objects"])
                    self.assertTrue(4 <= n <= 10)
                    self.assertEqual(n, sum(counts.values()))
                    self.assertEqual(len(counts), min(n, len(pool)))
                    self.assertLessEqual(max(counts[p] for p in pool) - min(counts[p] for p in pool), 1)
                    self.assertEqual(layer["minimum_footprint_gap_m"], .085 if n <= 5 else .055 if n <= 9 else .035)
                    raised |= layer["count_raised_for_required_categories"]
                self.assertTrue(verify_placed_inventory(plan, rows(plan))["inventory_matches_plan"])
        self.assertTrue(raised)

    def test_order_duplicates_and_reproducibility(self):
        a = compose({"seed": 42, "required_categories": ["oreo_gf", "coke_c3"]})
        b = compose({"seed": 42, "required_categories": ["coke_c3", "oreo_gf", "coke_c3"]})
        self.assertEqual(a, b)
        self.assertFalse(a["runtime_ready"])
        self.assertFalse(a["poses_sampled"])

    def test_variety_across_seeds(self):
        results = {tuple(o["product_family"] for o in rows(compose({"seed": seed}))) for seed in range(100)}
        self.assertGreater(len(results), 90)

    def test_bad_requests(self):
        for request in ([], {}, {"seed": True}, {"seed": -1}, {"seed": 1.5},
                        {"seed": 1, "required_categories": "coke_c3"},
                        {"seed": 1, "required_categories": ["unknown"]},
                        {"seed": 1, "required_categories": [None]},
                        {"seed": 1, "surprise": 3}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                compose(request)

    def test_missing_required_rejected(self):
        p = compose({"seed": 5, "required_categories": ["coke_c3", "oreo_gf"]})
        with self.assertRaisesRegex(ValueError, "Required categories missing"):
            verify_placed_inventory(p, [o for o in rows(p) if o["product_family"] != "oreo_gf"])

    def test_dropped_optional_duplicate_or_wrong_tier_rejected(self):
        p = compose({"seed": 5})
        original = rows(p)
        wrong = copy.deepcopy(original)
        wrong[0]["support_id"] = "bad_tier"
        swapped = copy.deepcopy(original)
        swapped[0]["product_family"] = "wrong_product"
        for placed in (original[:-1], original + original[:1], wrong, swapped):
            with self.assertRaises(ValueError):
                verify_placed_inventory(p, placed)


if __name__ == "__main__":
    unittest.main()

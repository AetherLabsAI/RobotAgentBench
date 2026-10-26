"""Regression checks against the corrected room's saved numerical evidence."""
import argparse,json,unittest
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args,remaining=ap.parse_known_args()
receipt=json.loads((args.run/'placement_repair.json').read_text())
audit=json.loads((args.run/'corrected_placement_audit.json').read_text())

class PlacementRegression(unittest.TestCase):
    def test_coffee_matches_original_relative_transforms(self):
        binding=json.loads(Path(receipt['coffee']['binding']).read_text())
        machine=np.array(binding['world']['machine']).reshape(4,4)
        for role in ('cup','bag'):
            source=np.array(binding['world'][role]).reshape(4,4)
            expected=np.linalg.solve(machine.T,source.T).T
            actual=np.array(receipt['coffee']['objects'][role]['source_relative_row_matrix'])
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            self.assertLess(receipt['coffee']['objects'][role]['relative_transform_max_error'],1e-10)
    def test_required_categories_and_random_geometry(self):
        shelf=receipt['shelf'];placed=shelf['placements'];metrics=shelf['metrics']
        self.assertEqual(len({p['product_family'] for p in placed}),14)
        self.assertEqual(metrics['layer_counts'],{'tier3':8,'tier2':6})
        self.assertFalse(metrics['uses_target']);self.assertFalse(metrics['uses_robot'])
        for sid in ('tier2','tier3'):
            rows=[p for p in placed if p['support_id']==sid]
            # Regression for this seed: guard against the old single-depth/no-yaw row.
            self.assertGreater(np.ptp([p['xyz_m'][1] for p in rows]),.1)
            self.assertGreater(np.ptp([p['yaw_deg'] for p in rows]),90)
            m=metrics['spacing_metrics'][sid]
            self.assertGreaterEqual(m['minimum_edge_gap_m'],m['required_edge_gap_m'])
    def test_native_stability_and_scope(self):
        self.assertTrue(audit['passed']);self.assertEqual(audit['frames_checked'],871)
        self.assertEqual(len(audit['objects']),16)
        self.assertFalse(receipt['shared_assets_modified'])
        self.assertFalse(receipt['unrelated_placements_modified'])

unittest.main(argv=['test_room_placement_v2.py']+remaining)

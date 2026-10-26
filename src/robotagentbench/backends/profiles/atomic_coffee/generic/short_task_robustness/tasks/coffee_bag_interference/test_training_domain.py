import json
import unittest
from pathlib import Path
from .training_domain import domain,producer_offset_bounds,contains_origin,maximum_displacement


class TrainingDomainTests(unittest.TestCase):
    def setUp(self):
        self.initial=json.loads(Path(__file__).with_name('preview_scene_002').joinpath('initial.json').read_text())
        self.definition=domain(self.initial)

    def test_bounds_come_from_original_producer(self):
        bounds,source=producer_offset_bounds()
        self.assertEqual(bounds,[[-.05,.05],[-.05,.05]])
        self.assertEqual(len(source['sha256']),64)

    def test_previous_video_endpoint_is_not_training_conforming(self):
        self.assertFalse(contains_origin([2.7232770919799805,-.0268967654556036,.92],self.definition))
        self.assertTrue(contains_origin([2.5826006938903596,-.12088351417664756,.92],self.definition))

    def test_source_start_accepts_sub_10cm_movement_in_training_domain(self):
        from .relocation_profile import sample_destination
        maximum=maximum_displacement([2.5826006938903596,-.12088351417664756],self.definition)
        self.assertLess(maximum,.10)
        self.assertAlmostEqual(maximum,.092,places=3)
        proposal=sample_destination(4126,[2.5826006938903596,-.12088351417664756,.92],self.definition)
        self.assertTrue(0 < proposal['distance_m'] < maximum)
        self.assertTrue(contains_origin(proposal['destination_origin_m'],self.definition))

    def test_destination_sampler_rejects_invalid_start_instead_of_expanding_domain(self):
        from .relocation_profile import sample_destination
        for start in ([9,9,.92],[0,0,float('nan')],[1,2]):
            with self.assertRaises(ValueError):sample_destination(1,start,self.definition)

    def test_domain_is_not_recentered_on_current_bag(self):
        lower,upper=self.definition['origin_bounds_world_xy_m']
        self.assertAlmostEqual(upper[0]-lower[0],.10)
        self.assertAlmostEqual(upper[1]-lower[1],.10)
        center=[(lo+hi)/2 for lo,hi in zip(lower,upper)]
        self.assertEqual(center,self.definition['nominal_bag_origin_m'][:2])


if __name__=='__main__':unittest.main()

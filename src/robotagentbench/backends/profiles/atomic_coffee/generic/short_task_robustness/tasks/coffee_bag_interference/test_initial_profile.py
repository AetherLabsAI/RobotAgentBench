import unittest
from evaluation.tasks.coffee_machine.randomization import sample_pour
from .initial_profile import sample_profile, validate_yaw


class ProfileTests(unittest.TestCase):
    def test_original_sampler_and_draw_order_are_retained(self):
        for seed in (1,7,4126,70005):
            self.assertEqual(sample_profile(seed)['original_proposal'],sample_pour(seed))

    def test_repeatable_and_in_range(self):
        for seed in range(150):
            profile = sample_profile(seed)
            self.assertEqual(profile,sample_profile(seed))
            self.assertGreaterEqual(profile['bag_yaw_deg'],0.)
            self.assertLessEqual(profile['bag_yaw_deg'],90.)
            self.assertFalse(profile['scene_written'])

    def test_boundary_angles_are_allowed(self):
        self.assertEqual(validate_yaw(0),0.)
        self.assertEqual(validate_yaw(90),90.)

    def test_invalid_angles_and_seeds_are_rejected(self):
        for value in (-.001,90.001,float('nan'),float('inf'),True,'0'):
            with self.assertRaises(ValueError): validate_yaw(value)
        for seed in (-1,True,1.5,None):
            with self.assertRaises(ValueError): sample_profile(seed)

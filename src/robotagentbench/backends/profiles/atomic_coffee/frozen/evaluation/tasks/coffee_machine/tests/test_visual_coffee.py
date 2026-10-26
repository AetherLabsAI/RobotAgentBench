import unittest
from evaluation.tasks.coffee_machine.tests.test_process_rules import Run, liquid, FULL, HELD, AWAY

class AnimationRules(unittest.TestCase):
    def test_tilt_rejects_even_when_visual_surface_follows(self):
        r=Run('1-4',liquid=dict(liquid(FULL,initial=FULL),mode='visual_cpu_v1'))
        r.feed(cup=dict(HELD,tilt_deg=16),placement=AWAY)
        self.assertEqual(r.p.failure,'cup_tilt_exceeded')
    def test_upright_hold_and_honest_summary(self):
        r=Run('1-4',liquid=dict(liquid(FULL,initial=FULL),mode='visual_cpu_v1'))
        r.feed(12,cup=HELD,placement=AWAY)
        self.assertTrue(r.p.summary()['success'])
        counts=r.p._final_counts()
        self.assertFalse(counts['physical_liquid'])
        self.assertNotIn('kept_ratio',counts)
        self.assertNotIn('in_cup_particles',counts)

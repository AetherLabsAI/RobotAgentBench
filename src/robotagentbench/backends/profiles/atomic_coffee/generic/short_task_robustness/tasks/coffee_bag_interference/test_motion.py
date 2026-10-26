import json
import unittest
from pathlib import Path
from .motion import RelocationMotion, DURATION_S


class MotionTests(unittest.TestCase):
    def motion(self):
        return RelocationMotion((0.,0.,1.),(0.,0.,1.),(0.,0.,1.),
                                (.02,.03,1.),(.02,.03,1.),(1.02,.03,1.))

    def test_confirmed_duration_matches_definition(self):
        definition=json.loads(Path(__file__).with_name('definition.json').read_text())
        self.assertEqual(DURATION_S,1.5)
        self.assertEqual(definition['relocation']['duration_s'],DURATION_S)

    def test_all_phases_retain_height(self):
        m=self.motion()
        for i in range(151):
            self.assertEqual(m.sample(i/100).wrist_position[2],1.)
        self.assertEqual(m.sample(1.5).phase,'done')

    def test_closed_carry_is_direct_segment(self):
        m=self.motion()
        for fraction in (.31,.4,.5,.6):
            s=m.sample(fraction*DURATION_S)
            self.assertEqual(s.phase,'carry');self.assertEqual(s.closure,1.)
            self.assertAlmostEqual(s.wrist_position[0]*.03,s.wrist_position[1]*.02)
        self.assertEqual(m.sample(.8*DURATION_S).closure,0.)

    def test_appearance_has_no_entry_motion_and_closes_immediately(self):
        m=self.motion()
        self.assertEqual(m.sample(0).wrist_position,m.points[1])
        self.assertEqual(m.sample(.1).wrist_position,m.points[1])
        self.assertGreater(m.sample(.1).closure,0.)

    def test_lift_detours_are_rejected(self):
        with self.assertRaises(ValueError):
            RelocationMotion((1,0,1),(0,0,1),(0,0,1.1),(.02,.03,1.1),(.02,.03,1),(1,.03,1))
        with self.assertRaises(ValueError):self.motion().sample(-.1)
        with self.assertRaises(ValueError):
            RelocationMotion((1,0,1),(0,0,1),(0,0,1),(.02,.03,1),(.02,.03,1),(1,.03,1))


if __name__=='__main__':unittest.main()

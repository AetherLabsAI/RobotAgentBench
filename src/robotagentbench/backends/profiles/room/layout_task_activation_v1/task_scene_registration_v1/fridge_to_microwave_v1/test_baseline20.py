"""Structural regressions for the source-preserving 20 Hz diagnostic adapter."""
import ast
import unittest
from pathlib import Path
from .prepare_baseline20 import adapt


class Baseline20(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).parent/'l1_continuous20_v1/source/replay_code'
        if not root.exists():raise unittest.SkipTest('Run beside the verified source copy on 202')
        cls.source=(root/'run_trial.py').read_text()
        cls.runner,cls.terminal=adapt(cls.source,(root/'terminal_audit.py').read_text())

    def test_helper_unchanged(self):
        def helper(s):return s[s.index('def door_rule():'):s.index('renderers={};video_writers={}')]
        self.assertEqual(helper(self.source),helper(self.runner))

    def test_twelve_steps_one_action(self):
        tree=ast.parse(self.runner)
        advance=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='advance')
        helpers=[n for n in advance.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='helper_tick']
        self.assertEqual(len(helpers),1)
        self.assertEqual(ast.unparse(helpers[0].iter),'range(3)')
        self.assertEqual(ast.unparse(helpers[0].body[1].iter),'range(4)')
        self.assertNotIn('apply_action',ast.unparse(helpers[0]))
        self.assertIn('angle = door_rule()',ast.unparse(helpers[0]))

    def test_same_two_second_terminal_window(self):
        self.assertIn('rows[-40:]',self.terminal)
        self.assertIn('len(tail)==40 and drift<.005',self.terminal)
        self.assertNotIn('120',self.terminal)

    def test_no_recorded_anchor_or_checkpoint(self):
        replay=self.runner[self.runner.index('def replay20():'):]
        self.assertIn('constrain_base(mode)',replay)
        self.assertNotIn('nav_constraint_pose',replay)
        self.assertNotIn('set_world_pose',replay)
        self.assertNotIn('checkpoint=np.load',self.runner)

    def test_clock_and_output(self):
        self.assertIn('timestamp=(np.arange(len(rows))+1)/20',self.runner)
        self.assertIn("'recording_20hz.npz'",self.runner)
        self.assertIn('control_hz=20,accepted=False',self.runner)


if __name__=='__main__':unittest.main()

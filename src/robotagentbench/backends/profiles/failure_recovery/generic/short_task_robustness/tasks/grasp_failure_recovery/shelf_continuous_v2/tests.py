import ast
from pathlib import Path
import unittest
import numpy as np
from .constraint import propose,interval_on_line

class ConstraintTests(unittest.TestCase):
    def test_interval(self):self.assertEqual(interval_on_line([0,1],[1,0],.5,[-1,-1],[1,1]),(-1.,1.))
    def test_no_intersection(self):self.assertIsNone(interval_on_line([0,1],[1,0],2.,[-1,-1],[1,1]))
    def test_50_continuous_reproducible_positions(self):
        tips=np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (.1,.12)])
        points=np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (0,.2)])
        positions=[]
        for seed in range(50):
            a,r=propose(tips,points,points[:,:2],np.eye(3),0,[[-.5,-.5],[.5,.5]],[[-1,-1],[1,1]],np.random.default_rng(seed))
            b,s=propose(tips,points,points[:,:2],np.eye(3),0,[[-.5,-.5],[.5,.5]],[[-1,-1],[1,1]],np.random.default_rng(seed))
            self.assertIsNotNone(a);np.testing.assert_array_equal(a,b);self.assertEqual(r,s)
            self.assertTrue(.02<=r['conservative_3d_envelope_distance_m']<=.05)
            positions.append(tuple(a))
        self.assertEqual(len(set(positions)),50)
    def test_height_budget(self):
        tips=np.array([[0,0,2.],[.01,.01,2.01]]);points=np.array([[0,0,0],[.05,.05,.1]])
        a,r=propose(tips,points,points[:,:2],np.eye(3),0,[[-1,-1],[1,1]],[[-1,-1],[1,1]],np.random.default_rng(4))
        self.assertIsNone(a);self.assertEqual(r['reason'],'height_outside_3d_gap_budget')
    def test_no_optimizer_or_scene_bank_in_startup(self):
        text=(Path(__file__).parent/'generate_v3.py').read_text();tree=ast.parse(text)
        funcs=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(f.endswith('.solve') or f.endswith('least_squares') for f in funcs))
        self.assertNotIn("profile['snapshot']",text)
        self.assertNotIn('bank_cpu.json',text)
        self.assertNotIn('procedural_config.json',text)
        self.assertNotIn("['layouts']",text)
        self.assertNotIn("['snapshot']",text)
    def test_packing_function_matches_source_snapshot(self):
        # Source version hash is recorded remotely; owner calls remain identical.
        root=Path(__file__).parent
        def calls(path):return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Call) and ast.unparse(n.func) in ('SimulationApp','World','world.step','run_public_control')]
        self.assertEqual(calls(root/'owner.py'),calls(root.parent/'closeout_v1/atomic_owner_v2.py'))

if __name__=='__main__':unittest.main()

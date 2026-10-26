import unittest
from types import SimpleNamespace
from .ik_branches import seeds


class NumericalBranches(unittest.TestCase):
    def test_only_numerical_active_arm_seeds_within_actual_limits(self):
        original=dict(a=.1,b=-.4,head=.7,waist=.2,other_arm=1.)
        model=SimpleNamespace(initial=original,names=['a','b'],bounds=dict(a=(-1.,2.),b=(-2.,.5)))
        rows=list(seeds(model))
        self.assertTrue(rows)
        self.assertEqual(model.initial,original)
        self.assertEqual(len(rows),len({tuple(q[n] for n in model.names) for _,q in rows}))
        for _,q in rows:
            for n in model.names:self.assertTrue(model.bounds[n][0]<=q[n]<=model.bounds[n][1])
            for n in ('head','waist','other_arm'):self.assertEqual(q[n],original[n])


if __name__=='__main__':unittest.main()

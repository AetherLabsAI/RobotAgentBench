"""Reader publication ordering is checked without importing the simulator."""
import ast
from pathlib import Path
import unittest

from robotagentbench.backends import sources


class CoffeeReaderPublication(unittest.TestCase):
    def test_both_native_readers_publish_before_the_fresh_usd_comparison(self):
        profiles = Path(sources.__file__).parent / 'profiles'
        readers = (
            profiles / 'atomic_coffee/frozen/evaluation/tasks/coffee_machine/isaac_evidence.py',
            profiles / 'failure_recovery/current_frozen/evaluation/tasks/coffee_machine/isaac_evidence.py',
        )
        for path in readers:
            with self.subTest(reader=path.relative_to(profiles)):
                tree = ast.parse(path.read_text())
                reader = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                              and n.name == 'CoffeeReader')
                initialize = next(n for n in reader.body if isinstance(n, ast.FunctionDef)
                                  and n.name == 'initialize')
                block = next(n for n in initialize.body if isinstance(n, ast.If)
                             and ast.unparse(n.test) == 'self.beans is not None')
                self.assertEqual(ast.unparse(block.body[0]),
                                 'from omni.physx import get_physx_interface')
                self.assertEqual(ast.unparse(block.body[1]),
                                 'get_physx_interface().update_transformations(False, True, False, False)')
                self.assertEqual(ast.unparse(block.body[2]), 'cache = UsdGeom.XformCache()')
                guard = next(n for n in block.body if isinstance(n, ast.If))
                self.assertEqual(ast.unparse(guard.test),
                                 'np.abs(self.beans.positions() - authored).max() > 0.001')
                calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)]
                self.assertFalse(any(n.func.attr in ('step', 'reset', 'set_world_pose',
                                                     'set_world_poses', 'set_velocities')
                                     for n in calls))

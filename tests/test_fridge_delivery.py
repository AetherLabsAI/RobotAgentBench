"""Fridge runtime hooks must ship, including imports resolved only at startup."""
import ast
import hashlib
import unittest

from robotagentbench.backends import root, verify


class FridgeDeliveryTests(unittest.TestCase):
    def test_original_runtime_dependencies_are_in_the_delivery(self):
        folder = root('room') / 'short_tasks/tasks/fridge_to_heated_box'
        expected = {
            'scene.py': '9571e8b0610e29e19b6e296728d321a8c59c5cf41088bab5b53e7b0911c77f2e',
            'port.py': '2d312333d59b16b083e960457f400b5b558b23d68949f85a29d94fd488f82582',
            'evaluator.py': '04bb5ec9bd3316be13ebfe5631060de35b40547012614797fdffeed8bdbc4b66',
            'sources.py': '514820a911cfeae7f88db6faf06d7d1aba432125f90e9aa4d7b27fa0640ee6d2',
            'contact_evidence.py': '39c4c0620f6ab55dc52e5cc37189b0ca416ea75b9ea1bf1c91e690ec9f0ac797',
            'contact_lifecycle.py': 'cef22932c80fa7ad6dbacdc0c003e06dc3d5c30938caf8a1e7421c9f80fd1cb2',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)
        self.assertTrue(verify('room')['verified'])

    def test_task_local_runtime_imports_exist(self):
        folder = root('room') / 'short_tasks/tasks/fridge_to_heated_box'
        for name in ('scene.py', 'port.py', 'evaluator.py', 'contact_evidence.py', 'contact_lifecycle.py'):
            for node in ast.walk(ast.parse((folder / name).read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    self.assertTrue((folder / (node.module.replace('.', '/') + '.py')).is_file(),
                                    (name, node.module))

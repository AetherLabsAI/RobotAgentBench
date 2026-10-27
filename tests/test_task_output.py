"""CPU-only evidence routing against unchanged original task path guards."""
import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.simulation.source_tree import NativeSourceTree
from robotagentbench.simulation.task_output import (
    LIGHTING_MODULE, LIGHTING_ROOT, LIGHTING_TASK, scoring_output,
    RELOCATION_MODULE, RELOCATION_ROOT, RELOCATION_TASK,
)


class TaskOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        bundled = (Path(__file__).resolve().parents[1] / 'src/robotagentbench/backends/profiles/room')
        self.pins = {}
        files = [LIGHTING_ROOT / name for name in ('lighting.py', 'cycle.py')]
        files += [RELOCATION_ROOT / 'native_backend.py']
        for item in files:
            relative = item.as_posix()
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((bundled / relative).read_bytes())
            self.pins[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.sources = NativeSourceTree(self.root, self.pins)
        self.output = self.root / 'short_tasks/agent_probe_v1/cap_v1/model_runs/one/native'
        self.output.mkdir(parents=True)

    def test_original_guard_rejects_generic_path_and_accepts_routed_evidence(self):
        with self.sources.activated():
            module = self.sources.import_module(LIGHTING_MODULE)
            self.assertFalse(module.permitted_audit_output(self.output / 'scoring/pick/lighting.json'))
            result = scoring_output(self.sources, LIGHTING_TASK, self.output)
            self.assertTrue(result.is_relative_to(module.HERE))
            self.assertTrue(module.permitted_audit_output(result / 'pick/lighting.json'))
            self.assertTrue(result.parent.is_dir())
            self.assertFalse(result.exists())  # Original build_session creates it.
            self.assertTrue(self.output.is_dir())
            self.sources.verify()  # No original file, rule, or schedule changed.

    def test_other_tasks_keep_the_original_generic_output(self):
        for task in ('stock_wooden_tray', 'openend', 'long_task_15', 'select_drink'):
            with self.subTest(task=task):
                self.assertEqual(scoring_output(self.sources, task, self.output), self.output / 'scoring')
        self.assertFalse((self.root / LIGHTING_ROOT / 'model_evidence').exists())

    def test_run_identity_is_not_reused(self):
        with self.sources.activated():
            for task in (LIGHTING_TASK, RELOCATION_TASK):
                with self.subTest(task=task):
                    scoring_output(self.sources, task, self.output)
                    with self.assertRaisesRegex(ValueError, 'Fresh evidence'):
                        scoring_output(self.sources, task, self.output)

    def test_original_guard_remains_authoritative(self):
        with self.sources.activated():
            module = self.sources.import_module(LIGHTING_MODULE)
            with patch.object(module, 'permitted_audit_output', return_value=False):
                with self.assertRaisesRegex(ValueError, 'unchanged lighting owner scope'):
                    scoring_output(self.sources, LIGHTING_TASK, self.output)
        self.assertFalse((self.root / LIGHTING_ROOT / 'model_evidence').exists())

    def test_foreign_and_indirect_output_are_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.output, target_is_directory=True)
        with self.sources.activated():
            for task in (LIGHTING_TASK, RELOCATION_TASK):
                for output in (self.root, self.root.parent, alias, self.root / 'missing'):
                    with self.subTest(task=task, output=str(output)):
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            scoring_output(self.sources, task, output)

    def test_task_evidence_parent_cannot_escape_through_a_link(self):
        for task, relative in ((LIGHTING_TASK, LIGHTING_ROOT), (RELOCATION_TASK, RELOCATION_ROOT)):
            (self.root / relative / 'model_evidence').symlink_to(self.output, target_is_directory=True)
            with self.subTest(task=task), self.sources.activated():
                with self.assertRaisesRegex(ValueError, 'Fresh evidence'):
                    scoring_output(self.sources, task, self.output)

    def test_relocation_constructor_keeps_its_original_path_guard(self):
        # Execute the exact original constructor's path assignment and guard
        # AST, not a rewritten predicate. No native dependency is imported.
        path = self.root / RELOCATION_ROOT / 'native_backend.py'
        tree = ast.parse(path.read_bytes())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeBackend')
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        guard_index = next(i for i, n in enumerate(init.body) if isinstance(n, ast.If)
            and any(isinstance(v, ast.Constant) and v.value == 'Human evidence must stay in its task directory'
                for v in ast.walk(n)))
        guard = compile(ast.Module(body=init.body[guard_index - 1:guard_index + 1], type_ignores=[]), str(path), 'exec')
        here_node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(v, ast.Name) and v.id == 'HERE' for v in n.targets))
        namespace = {'Path': Path, '__file__': str(path), 'self': SimpleNamespace()}
        exec(compile(ast.Module(body=[here_node], type_ignores=[]), str(path), 'exec'), namespace)
        with self.sources.activated():
            result = scoring_output(self.sources, RELOCATION_TASK, self.output)
            for bad in (self.output / 'scoring', namespace['HERE'], self.root.parent):
                namespace['output'] = bad
                with self.subTest(output=str(bad)), self.assertRaisesRegex(ValueError, 'Human evidence'):
                    exec(guard, namespace)
            namespace['output'] = result
            exec(guard, namespace)
            self.assertEqual(namespace['self'].output, result)
            self.assertTrue(result.parent.is_dir())
            self.assertFalse(result.exists())
            self.assertTrue(self.output.is_dir())
            self.sources.verify()

    def test_relocation_output_is_deterministically_bound_to_native_identity(self):
        with self.sources.activated():
            result = scoring_output(self.sources, RELOCATION_TASK, self.output)
            digest = hashlib.sha256(str(self.output).encode('utf-8')).hexdigest()
            self.assertEqual(result, self.root / RELOCATION_ROOT / 'model_evidence' / digest / 'scoring')
            second = self.output.parent.parent / 'two/native'
            second.mkdir(parents=True)
            self.assertNotEqual(result, scoring_output(self.sources, RELOCATION_TASK, second))

    def test_relocation_foreign_owner_is_rejected_before_output_creation(self):
        with self.sources.activated():
            with patch.object(self.sources, 'module_file', return_value=('unrelated/native_backend.py', False)):
                with self.assertRaisesRegex(ValueError, 'Evidence owner differs'):
                    scoring_output(self.sources, RELOCATION_TASK, self.output)
        self.assertFalse((self.root / RELOCATION_ROOT / 'model_evidence').exists())

    def test_relocation_source_tampering_is_not_re_pinned(self):
        path = self.root / RELOCATION_ROOT / 'native_backend.py'
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'Pinned dependency changed'), self.sources.activated():
            scoring_output(self.sources, RELOCATION_TASK, self.output)
        self.assertFalse((self.root / RELOCATION_ROOT / 'model_evidence').exists())


if __name__ == '__main__':
    unittest.main()

"""CPU-only evidence routing against the unchanged original lighting guard."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.simulation.source_tree import NativeSourceTree
from robotagentbench.simulation.task_output import (
    LIGHTING_MODULE, LIGHTING_ROOT, LIGHTING_TASK, scoring_output,
)


class TaskOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        bundled = (Path(__file__).resolve().parents[1] / 'src/robotagentbench/backends/profiles/room')
        self.pins = {}
        for name in ('lighting.py', 'cycle.py'):
            relative = (LIGHTING_ROOT / name).as_posix()
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
            scoring_output(self.sources, LIGHTING_TASK, self.output)
            with self.assertRaisesRegex(ValueError, 'Fresh evidence'):
                scoring_output(self.sources, LIGHTING_TASK, self.output)

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
            for output in (self.root, self.root.parent, alias, self.root / 'missing'):
                with self.subTest(output=str(output)), self.assertRaises((ValueError, FileNotFoundError)):
                    scoring_output(self.sources, LIGHTING_TASK, output)

    def test_task_evidence_parent_cannot_escape_through_a_link(self):
        (self.root / LIGHTING_ROOT / 'model_evidence').symlink_to(self.output, target_is_directory=True)
        with self.sources.activated(), self.assertRaisesRegex(ValueError, 'Fresh evidence'):
            scoring_output(self.sources, LIGHTING_TASK, self.output)


if __name__ == '__main__':
    unittest.main()

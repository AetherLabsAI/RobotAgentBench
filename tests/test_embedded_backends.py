import ast
from pathlib import Path
import sys
import unittest

from robotagentbench.backends import describe, root, verify
from robotagentbench.backends import sources


class EmbeddedBackendTests(unittest.TestCase):
    def test_all_selected_source_profiles_ship_and_match(self):
        self.assertEqual(set(describe()), {'room','atomic_coffee','failure_recovery','cloth'})
        for name in describe():
            self.assertGreater(describe()[name]['files'], 0)
            self.assertTrue(verify(name)['verified'])
            self.assertFalse(verify(name)['task_code_imported'])

    def test_actual_scoring_and_initialization_code_not_only_catalogs(self):
        room = root('room')
        for name in ('short_tasks/registry.py', 'short_tasks/scene.py', 'openend/dispatch.py',
                     'evaluation/tasks/coffee_machine/process_rules.py', 'evaluation/tasks/air_fryer/process_rules.py'):
            self.assertTrue((room/name).is_file(), name)
            ast.parse((room/name).read_text())
        cloth = root('cloth')/'owner'
        for name in ('approved_rules.py','facts_reader.py','folding.py','placement.py','source_owner.py'):
            self.assertTrue((cloth/name).is_file(), name)

    def test_no_private_answers_native_binary_or_rollouts_in_source_manifest(self):
        for profile in sources._manifest()['profiles'].values():
            for name in profile['files']:
                self.assertNotIn('answers_private', name)
                self.assertNotIn('__pycache__', name)
                self.assertNotIn(Path(name).suffix, ('.so','.npz','.mp4','.usdc'))

    def test_backend_queries_do_not_import_isaac(self):
        before = set(sys.modules)
        describe(); root('room')
        added = set(sys.modules)-before
        self.assertFalse(any(name.split('.')[0] in {'pxr','omni','isaacsim','torch'} for name in added))

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            root('../room')
        with self.assertRaises(ValueError):
            verify('unknown')


if __name__ == '__main__':
    unittest.main()

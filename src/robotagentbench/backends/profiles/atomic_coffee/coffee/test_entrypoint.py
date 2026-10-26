"""Administrative routing tests, without launching Isaac or changing task rules."""
import argparse
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from . import __main__ as entry


class EntrypointTests(unittest.TestCase):
    def module(self, name, **members):
        module = types.ModuleType(__package__ + '.' + name)
        module.__dict__.update(members)
        return {module.__name__: module}

    def args(self, **changes):
        values = dict(initial=Path('initial.usda'), output=Path('output'), gpu=2,
                      authorized_shared_gpu2=False, reuse_cache=None, hd=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_description_does_not_claim_registration_or_score(self):
        task = entry.describe()
        self.assertEqual(task['source_atomic_api'], '1-2')
        self.assertFalse(task['registered'])
        self.assertIsNone(task['runtime_id'])
        self.assertFalse(task['evaluation']['scoring_integrated'])
        self.assertTrue(task['user_accepted_current_interference'])

    def test_prepare_delegates_exact_existing_sampler_inputs(self):
        build = Mock(return_value={'scene': 'scene'})
        with patch.dict(sys.modules, self.module('build_initial', build=build)):
            result = entry.prepare(Path('new'), 7, 'random')
        build.assert_called_once_with(Path('new'), seed=7, role='random')
        self.assertEqual(result, {'scene': 'scene'})

    def test_preview_checks_source_before_native_launch(self):
        calls = []
        preflight = Mock(side_effect=lambda path: calls.append('preflight'))
        native = Mock(side_effect=lambda args: calls.append('native'))
        modules = self.module('isolated_preflight', preflight=preflight)
        modules.update(self.module('native_diagnostic', main=native))
        with patch.dict(sys.modules, modules):
            entry.preview(self.args())
        self.assertEqual(calls, ['preflight', 'native'])
        preflight.assert_called_once_with(Path('initial.usda'))
        native.assert_called_once_with(['--initial', 'initial.usda', '--output',
                                      'output', '--gpu', '2', '--atomic-source-replay'])

    def test_preview_rejects_source_before_any_native_launch(self):
        native = Mock()
        modules = self.module('isolated_preflight', preflight=Mock(side_effect=ValueError('bad source')))
        modules.update(self.module('native_diagnostic', main=native))
        with patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            entry.preview(self.args())
        native.assert_not_called()

    def test_preview_forwards_only_existing_optional_flags(self):
        native = Mock(return_value=0)
        modules = self.module('isolated_preflight', preflight=Mock())
        modules.update(self.module('native_diagnostic', main=native))
        with patch.dict(sys.modules, modules):
            self.assertEqual(entry.preview(self.args(hd=True, authorized_shared_gpu2=True,
                                                    reuse_cache=Path('cache'))), 0)
        self.assertEqual(native.call_args.args[0][-4:],
                         ['--authorized-shared-gpu2', '--reuse-cache', 'cache', '--hd-presentation'])

    def test_verify_distinguishes_mechanism_from_old_stress_and_score(self):
        result = dict(real_trigger_checks_passed=True, mechanism_checks={'contact': True},
                      farther_push_policy_checks={'same_frame': True},
                      requested_farther_push_verified=False, full_task_scoring_verified=False)
        verify = Mock(return_value=result)
        with patch.dict(sys.modules, self.module('verify_atomic_replay', verify=verify)):
            self.assertEqual(entry.verify(entry.HERE, entry.HERE / 'definition.json'), 0)
            self.assertFalse(result['requested_farther_push_verified'])
            self.assertFalse(result['full_task_scoring_verified'])
            for field in ('mechanism_checks', 'farther_push_policy_checks'):
                result[field] = {'failed': False}
                self.assertEqual(entry.verify(entry.HERE, entry.HERE / 'definition.json'), 1)
                result[field] = {'passed': True}
            result['real_trigger_checks_passed'] = False
            self.assertEqual(entry.verify(entry.HERE, entry.HERE / 'definition.json'), 1)

    def test_verify_rejects_directory_outside_task_before_writing(self):
        verify = Mock()
        with patch.dict(sys.modules, self.module('verify_atomic_replay', verify=verify)):
            with self.assertRaises(ValueError):
                entry.verify(entry.HERE.parent, entry.HERE / 'definition.json')
        verify.assert_not_called()

    def test_help_and_description_need_no_native_import(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(['describe']), 0)
        self.assertIn('scored_registration_pending', output.getvalue())
        for arguments in (['--help'], ['preview', '--help']):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                entry.main(arguments)
            self.assertEqual(error.exception.code, 0)


if __name__ == '__main__':
    unittest.main()

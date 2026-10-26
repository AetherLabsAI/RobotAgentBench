import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch
from .isolated_preflight import environment, preflight


class IsolatedPreflightTests(unittest.TestCase):
    def test_dependency_paths_only_in_child_environment(self):
        parent = {'PYTHONPATH': 'sources', 'CUDA_VISIBLE_DEVICES': 'unchanged'}
        child = environment(parent, Path('/usd'), Path('/physx'))
        self.assertEqual(parent, {'PYTHONPATH': 'sources', 'CUDA_VISIBLE_DEVICES': 'unchanged'})
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], 'unchanged')
        self.assertIn('/physx/plugins/PhysxSchema/resources', child['PXR_PLUGINPATH_NAME'])
        self.assertTrue(child['PYTHONPATH'].endswith('sources'))

    def test_child_uses_same_existing_preflight_and_no_bytecode_writes(self):
        with patch(__package__ + '.isolated_preflight.installation', return_value=(Path('/usd'), Path('/physx'))), \
                patch('subprocess.run', return_value=Mock(stdout=json.dumps({'rows': 813}))) as run:
            self.assertEqual(preflight(Path('/initial.usda')), {'rows': 813})
        args = run.call_args.args[0]
        self.assertEqual(args[1:4], ['-B', '-m', __package__ + '.preflight_atomic'])
        self.assertTrue(run.call_args.kwargs['check'])

    def test_failed_admission_is_not_ignored(self):
        with patch(__package__ + '.isolated_preflight.installation', return_value=(Path('/usd'), Path('/physx'))), \
                patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'admission')):
            with self.assertRaises(subprocess.CalledProcessError):
                preflight(Path('/initial.usda'))

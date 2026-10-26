"""Native environment tests use stdlib metadata and a pinned local interpreter."""
import hashlib
from pathlib import Path
import platform
import sys
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from robotagentbench.simulation.dependencies import Resource, RuntimeDependencies
from robotagentbench.simulation.environment import Directory, NativeEnvironment
from robotagentbench.simulation.source_tree import SourceDependencyError


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # A venv may copy its executable rather than symlink it. The test's
        # temporary launcher must target the base installation with its stdlib,
        # not a copied binary detached from that venv's pyvenv.cfg.
        self.binary = Path(getattr(sys, '_base_executable', sys.executable)).resolve()
        self.launcher = self.root / 'python'
        self.launcher.symlink_to(self.binary)
        metadata = self.root / 'example_native-1.0.dist-info'
        metadata.mkdir()
        path = metadata / 'METADATA'
        path.write_text('Metadata-Version: 2.1\nName: example-native\nVersion: 1.0\n')
        resources = {
            'python': Resource('binary', self.binary.name,
                hashlib.sha256(self.binary.read_bytes()).hexdigest(), 'native'),
            'metadata': Resource('runtime', 'example_native-1.0.dist-info/METADATA',
                hashlib.sha256(path.read_bytes()).hexdigest(), 'native')}
        self.dependencies = RuntimeDependencies({'binary': self.binary.parent, 'runtime': self.root}, resources)

    def environment(self, **overrides):
        values = dict(launcher=self.launcher, binary='python', python_version=sys.version_info[:2],
                      platform=sys.platform, machine=platform.machine(), packages={'example-native': '1.0'},
                      metadata_paths=(Directory('runtime'),), required_resources=('metadata',))
        values.update(overrides)
        return NativeEnvironment(self.dependencies, **values)

    def test_isolated_metadata_probe_never_imports_native_or_project_modules(self):
        report = self.environment().probe()
        self.assertEqual(report['packages'], {'example-native': '1.0'})
        self.assertEqual(report['native_modules_loaded'], [])
        self.assertFalse(list(self.root.rglob('*.pyc')))

    def config(self):
        return dict(schema='robotagentbench.native_environment.v1',launcher=str(self.launcher),
                    binary='python',python_version=list(sys.version_info[:2]),platform=sys.platform,
                    machine=platform.machine(),packages={'example-native':'1.0'},
                    metadata_paths=[dict(root='runtime',path='')],python_paths=[],library_paths=[],
                    plugin_paths=[],asset_paths=[],material_paths=[],required_resources=['metadata'])

    def test_operator_configuration_preserves_exact_metadata_and_explicit_empty_paths(self):
        profile=NativeEnvironment.from_config(self.dependencies,self.config())
        self.assertEqual(profile.probe()['packages'],{'example-native':'1.0'})
        self.assertEqual(profile.python_paths,())
        self.assertEqual(profile.environment({'LD_PRELOAD':'ambient'})['PYTHONPATH'],'')

    def test_operator_configuration_rejects_extra_missing_or_malformed_fields(self):
        for change in ('extra','missing','directory','version','resources','schema'):
            config=self.config()
            if change=='extra':config['auto_download']=True
            elif change=='missing':del config['material_paths']
            elif change=='directory':config['metadata_paths']=[dict(root='runtime',path='',fallback='.')]
            elif change=='version':config['python_version']='3.11'
            elif change=='resources':config['required_resources']=['metadata','metadata']
            elif change=='schema':config['schema']='other'
            with self.subTest(change=change),self.assertRaises(ValueError):
                NativeEnvironment.from_config(self.dependencies,config)

    def test_wrong_abi_architecture_version_or_missing_package_is_rejected(self):
        for overrides in ({'python_version': (3, 99)}, {'machine': 'different-architecture'},
                          {'packages': {'example-native': '2.0'}}, {'packages': {'missing-native': '1.0'}}):
            with self.subTest(overrides=overrides), self.assertRaises(SourceDependencyError):
                self.environment(**overrides).probe()
        duplicate = self.root / 'duplicate_native-1.0.dist-info'
        duplicate.mkdir()
        (duplicate / 'METADATA').write_text('Metadata-Version: 2.1\nName: example-native\nVersion: 1.0\n')
        with self.assertRaises(SourceDependencyError):
            self.environment().probe()

    def test_ambient_python_and_native_injection_paths_are_not_appended(self):
        original = {'PYTHONPATH': 'unreviewed', 'PYTHONHOME': 'unreviewed', 'LD_PRELOAD': 'unreviewed',
                    'LD_LIBRARY_PATH': 'unreviewed', 'PXR_PLUGINPATH_NAME': 'unreviewed', 'PATH': 'keep'}
        env = self.environment(python_paths=(Directory('runtime'),)).environment(original)
        self.assertEqual(env['PYTHONPATH'], str(self.root))
        self.assertEqual(env['PATH'], 'keep')
        self.assertNotIn('PYTHONHOME', env)
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('LD_LIBRARY_PATH', env)
        self.assertNotIn('PXR_PLUGINPATH_NAME', env)
        self.assertEqual(original['PYTHONPATH'], 'unreviewed')

    def test_launcher_and_required_metadata_cannot_drift(self):
        environment = self.environment()
        (self.root / 'example_native-1.0.dist-info/METADATA').write_text('changed')
        with self.assertRaises(SourceDependencyError):
            environment.verify()
        self.launcher.unlink()
        self.launcher.write_text('not the pinned interpreter')
        with self.assertRaises(SourceDependencyError):
            environment.verify()

    def test_directory_links_duplicate_paths_and_unpinned_binary_are_not_accepted(self):
        (self.root / 'linked').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(SourceDependencyError):
            self.environment(python_paths=(Directory('runtime', 'linked'),)).verify()
        with self.assertRaises(ValueError):
            self.environment(python_paths=(Directory('runtime'), Directory('runtime'))).verify()
        with self.assertRaises(ValueError):
            self.environment(binary='unreviewed')

    def test_probe_does_not_execute_installed_sitecustomize(self):
        (self.root / 'sitecustomize.py').write_text('raise AssertionError("must not execute")\n')
        self.environment(python_paths=(Directory('runtime'),)).probe()

    def test_asset_and_material_resolver_paths_are_explicit_not_ambient(self):
        original = {'MDL_SYSTEM_PATH': 'ambient', 'PXR_AR_DEFAULT_SEARCH_PATH': 'ambient', 'LD_AUDIT': 'ambient'}
        environment = self.environment(asset_paths=(Directory('runtime'),), material_paths=(Directory('runtime'),))
        env = environment.environment(original)
        self.assertEqual(env['MDL_SYSTEM_PATH'], str(self.root))
        self.assertEqual(env['PXR_AR_DEFAULT_SEARCH_PATH'], str(self.root))
        self.assertNotIn('LD_AUDIT', env)

    def test_failed_or_timed_out_probe_does_not_retry_or_import_a_fallback(self):
        for error in (OSError('unavailable'), subprocess.TimeoutExpired('native-metadata', 1)):
            with self.subTest(error=type(error).__name__), patch(
                    'robotagentbench.simulation.environment.subprocess.run', side_effect=error) as run:
                with self.assertRaises(SourceDependencyError):
                    self.environment().probe(timeout=1)
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()

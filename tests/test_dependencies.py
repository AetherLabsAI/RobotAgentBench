"""Portable dependency checks never download assets or initialize a simulator."""
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robotagentbench.simulation.dependencies import Resource, RuntimeDependencies
from robotagentbench.simulation.source_tree import SourceDependencyError


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.py'
        self.source.write_text('VALUE = 1\n')
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.resources = {'scorer': Resource('code', 'source.py', self.digest, 'source')}
        self.aliases = {'/original/evaluator/source.py': 'scorer'}

    def catalog(self):
        return RuntimeDependencies({'code': self.root}, self.resources, legacy_sources=self.aliases)

    def test_original_manifest_uses_an_exact_file_binding_without_rewriting_it(self):
        original = {'/original/evaluator/source.py': self.digest}
        before = deepcopy(original)
        catalog = self.catalog()
        self.assertEqual(catalog.verify_original_sources(original), {'verified': True, 'file_count': 1})
        self.assertEqual(original, before)
        self.assertEqual(catalog.file('scorer'), self.source)

    def test_unchanged_source_native_and_asset_resources_are_tracked_separately(self):
        resources = dict(self.resources)
        for kind in ('native', 'asset'):
            path = self.root / kind
            path.write_bytes(kind.encode())
            resources[kind] = Resource('code', kind, hashlib.sha256(path.read_bytes()).hexdigest(), kind)
        catalog = RuntimeDependencies({'code': self.root}, resources)
        self.assertEqual(catalog.verify()['kinds'], {'asset': 1, 'native': 1, 'source': 1})

    def test_changed_missing_unbound_and_symlinked_files_fail_closed(self):
        catalog = self.catalog()
        original = self.source.read_bytes()
        self.source.write_bytes(b'changed')
        with self.assertRaises(SourceDependencyError):
            catalog.file('scorer')
        self.source.unlink()
        with self.assertRaises(SourceDependencyError):
            catalog.file('scorer')
        target = self.root / 'target.py'
        target.write_bytes(original)
        self.source.symlink_to(target)
        with self.assertRaises(SourceDependencyError):
            catalog.file('scorer')
        with self.assertRaises(SourceDependencyError):
            catalog.file('unbound')

    def test_aliases_cannot_fall_back_to_original_paths_or_refresh_expected_hashes(self):
        catalog = self.catalog()
        for manifest in ({str(self.source): self.digest}, {'/original/evaluator/source.py': '0' * 64}):
            with self.assertRaises(SourceDependencyError):
                catalog.verify_original_sources(manifest)
        with self.assertRaises(ValueError):
            catalog.verify_original_sources({})

    def test_configuration_is_explicit_and_snapshotted(self):
        config = dict(schema='robotagentbench.runtime_dependencies.v1', roots={'code': str(self.root)},
                      resources={'scorer': dict(root='code', path='source.py', sha256=self.digest, kind='source')},
                      legacy_sources=dict(self.aliases))
        catalog = RuntimeDependencies.from_config(config)
        config['resources']['scorer']['sha256'] = '0' * 64
        self.assertEqual(catalog.file('scorer'), self.source)
        with self.assertRaises(TypeError):
            catalog.resources['scorer'] = None
        config['score_override'] = 100
        with self.assertRaises(ValueError):
            RuntimeDependencies.from_config(config)

    def test_unsafe_paths_and_unreviewed_alias_kinds_are_rejected(self):
        for path in ('../escape', '/absolute', 'a//b', 'a/./b', 'a\\b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                Resource('code', path, self.digest, 'asset')
        for original in ('relative', '/original/../escape', '/original//source.py'):
            with self.assertRaises(ValueError):
                RuntimeDependencies({'code': self.root}, self.resources, legacy_sources={original: 'scorer'})
        with self.assertRaises(ValueError):
            RuntimeDependencies({'code': self.root}, {'asset': Resource('code', 'source.py', self.digest, 'asset')},
                                legacy_sources={'/original/asset': 'asset'})

    def test_directories_are_explicit_and_cannot_escape_or_use_links(self):
        catalog = self.catalog()
        self.assertEqual(catalog.directory('code'), self.root)
        with self.assertRaises(ValueError):
            catalog.directory('code', '../outside')
        (self.root / 'link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(SourceDependencyError):
            catalog.directory('code', 'link')

    def test_pointer_text_is_not_an_installed_dependency_even_if_its_hash_matches(self):
        pointer = (b'version https://git-lfs.github.com/spec/v1\n'
                   b'oid sha256:' + b'a' * 64 + b'\nsize 348537607\n')
        for raw in (pointer, pointer.replace(b'\n', b'\r\n'),
                    pointer + b'ext-0-example sha256:' + b'b' * 64 + b'\n'):
            self.source.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            for kind in ('source', 'native', 'asset'):
                with self.subTest(kind=kind, pointer=raw):
                    resource = Resource('code', 'source.py', digest, kind)
                    catalog = RuntimeDependencies({'code': self.root}, {'required': resource})
                    with self.assertRaisesRegex(SourceDependencyError, 'unresolved Git LFS pointer'):
                        catalog.file('required')
                    self.assertEqual(self.source.read_bytes(), raw)

    def test_real_asset_uses_its_explicit_content_pin_without_pointer_fallback(self):
        raw = b'#usda 1.0\ndef Xform "World" {}\n'
        asset = self.root / 'room.usda'
        asset.write_bytes(raw)
        catalog = RuntimeDependencies({'code': self.root}, {
            'room': Resource('code', 'room.usda', hashlib.sha256(raw).hexdigest(), 'asset')})
        self.assertEqual(catalog.file('room'), asset)
        asset.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaises(SourceDependencyError):
            catalog.file('room')


if __name__ == '__main__':
    unittest.main()

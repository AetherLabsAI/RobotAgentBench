"""No native extension or simulator is loaded by these source-boundary tests."""
import hashlib
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import py_compile
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
import unittest

from robotagentbench.simulation.source_tree import NativeSourceTree, SourceDependencyError
from source_fixtures import SourceFixture


class SourceTreeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SourceFixture(self.temp.name)

    def test_original_module_is_pinned_and_no_bytecode_cache_is_created(self):
        f = self.fixture
        f.write('short_tasks/example.py', 'VALUE = "original"\n')
        before = list(sys.path)
        tree = f.tree()
        with tree.activated():
            module = tree.import_module('short_tasks.example')
            self.assertEqual(module.VALUE, 'original')
            self.assertEqual(Path(module.__file__), f.root / 'short_tasks/example.py')
        self.assertNotIn('short_tasks', sys.modules)
        self.assertEqual(sys.path, before)
        self.assertFalse(list(f.root.rglob('*.pyc')))

    def test_unpinned_modules_cannot_fall_back_to_path_search(self):
        f = self.fixture
        f.write('short_tasks/unpinned.py', 'raise AssertionError("must not execute")\n', pinned=False)
        with f.tree().activated():
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module('short_tasks.unpinned')

    def test_changed_module_is_rejected_even_when_already_imported(self):
        f = self.fixture
        path = f.write('short_tasks/example.py', 'VALUE = 1\n')
        tree = f.tree()
        with tree.activated():
            tree.import_module('short_tasks.example')
            path.write_text('VALUE = 2\n')
            with self.assertRaises(SourceDependencyError):
                tree.import_module('short_tasks.example')

    def test_unverified_bytecode_is_ignored_and_not_rewritten(self):
        f = self.fixture
        original = f.write('short_tasks/example.py', 'VALUE = "original"\n')
        poison = f.write('poison.py', 'VALUE = "unverified bytecode"\n', pinned=False)
        cache = Path(importlib.util.cache_from_source(str(original)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(poison), cfile=str(cache), doraise=True)
        before = cache.read_bytes()
        tree = f.tree()
        with tree.activated():
            self.assertEqual(tree.import_module('short_tasks.example').VALUE, 'original')
        self.assertEqual(cache.read_bytes(), before)

    def test_preloaded_namespace_is_not_replaced_or_removed(self):
        foreign = ModuleType('short_tasks')
        with patch.dict(sys.modules, {'short_tasks': foreign}):
            with self.assertRaises(SourceDependencyError):
                with self.fixture.tree().activated():
                    self.fail('Preloaded namespace must not be accepted')
            self.assertIs(sys.modules['short_tasks'], foreign)

    def test_second_binding_is_rejected_even_before_first_import(self):
        with self.fixture.tree().activated():
            with self.assertRaises(SourceDependencyError):
                with self.fixture.tree().activated():
                    self.fail('Concurrent source bindings are not supported')

    def test_symlinks_and_escaped_paths_are_not_dependency_pins(self):
        f = self.fixture
        target = f.write('short_tasks/real.py', 'VALUE = 1\n')
        alias = f.root / 'short_tasks/link.py'
        alias.symlink_to(target)
        pins = dict(f.pins, **{'short_tasks/link.py': hashlib.sha256(target.read_bytes()).hexdigest()})
        with self.assertRaises(SourceDependencyError):
            NativeSourceTree(f.root, pins).verify()
        for relative in ('../outside.py', '/outside.py', 'a//b.py', 'a/./b.py', 'a\\b.py'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                NativeSourceTree(f.root, {relative: '0' * 64})

    def test_unpinned_package_initializer_is_not_skipped(self):
        f = self.fixture
        f.write('short_tasks/extra/__init__.py', 'raise AssertionError("must not run")', pinned=False)
        f.write('short_tasks/extra/child.py', 'VALUE = 1\n')
        with f.tree().activated():
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module('short_tasks.extra.child')

    def test_native_extensions_are_disabled_until_explicit_dependency_admission(self):
        f = self.fixture
        relative = 'short_tasks/native' + importlib.machinery.EXTENSION_SUFFIXES[0]
        f.write(relative, 'not an actual native extension')
        with f.tree().activated():
            with self.assertRaises(SourceDependencyError):
                importlib.import_module('short_tasks.native')

    def test_original_source_path_changes_are_restored_on_exit(self):
        f = self.fixture
        f.write('short_tasks/path_change.py', 'import sys\nsys.path.insert(0, "test-vendor-path")\n')
        before = list(sys.path)
        tree = f.tree()
        with tree.activated():
            tree.import_module('short_tasks.path_change')
            self.assertEqual(sys.path[0], 'test-vendor-path')
        self.assertEqual(sys.path, before)

    def test_injected_module_cache_entry_is_not_accepted(self):
        f = self.fixture
        f.write('short_tasks/example.py', 'VALUE = 1\n')
        tree = f.tree()
        with tree.activated():
            sys.modules['short_tasks.example'] = ModuleType('short_tasks.example')
            with self.assertRaises(SourceDependencyError):
                tree.import_module('short_tasks.example')

    def test_missing_dependency_groups_are_not_waived(self):
        with self.assertRaises(SourceDependencyError):
            self.fixture.tree().require_group('native')
        with self.assertRaises(ValueError):
            NativeSourceTree(self.fixture.root, self.fixture.pins, groups={'native': ['absent']})

    def test_reviewed_pins_and_namespace_paths_cannot_be_refreshed_in_place(self):
        tree = self.fixture.tree()
        with self.assertRaises(TypeError):
            tree.pins['short_tasks/registry.py'] = '0' * 64
        with self.assertRaises(TypeError):
            tree.namespace_paths['short_tasks'] = 'unreviewed'

    def test_restored_namespace_can_use_its_reviewed_physical_location(self):
        f = self.fixture
        f.write('restored/short_tasks/example.py', 'VALUE = "restored original"\n')
        tree = NativeSourceTree(f.root, f.pins, namespace_paths={'short_tasks': 'restored/short_tasks'})
        with tree.activated():
            module = tree.import_module('short_tasks.example')
            self.assertEqual(module.VALUE, 'restored original')
            self.assertEqual(Path(module.__file__), f.root / 'restored/short_tasks/example.py')

    def test_nested_dependency_relocation_keeps_original_module_identity(self):
        f = self.fixture
        f.write('dependency/producer/__init__.py', '')
        f.write('dependency/producer/value.py', 'VALUE = 17\n')
        tree = NativeSourceTree(f.root, f.pins,
            namespace_paths={'short_tasks.restored': 'dependency/producer'})
        with tree.activated():
            module = tree.import_module('short_tasks.restored.value')
            self.assertEqual(module.__name__, 'short_tasks.restored.value')
            self.assertEqual(module.VALUE, 17)
        with self.assertRaises(ValueError):
            NativeSourceTree(f.root, f.pins, namespace_paths={'os': 'dependency/producer'})


if __name__ == '__main__':
    unittest.main()

"""Metadata-only original resource aliases; no SDK or native process."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from robotagentbench.simulation.diagnostic_instance_bank import relocation_resources as resources


class RelocationResourceTests(TestCase):
    tail = 'layout_task_activation_v1/task_scene_registration_v1/layout04/room.usda'
    raw = b'original scene bytes'

    def fixture(self, directory, *, layout='source', recorded_root=None):
        root = Path(directory).resolve()/'Runtime'
        root.mkdir()
        original = recorded_root or root
        source_relative = Path('short_tasks/agent_probe_v1/deployment_release/source')
        (root/source_relative).mkdir(parents=True)
        target = resources.LOGICAL_ROOT+self.tail
        key = resources.RUNTIME_PREFIX+target.lstrip('/')
        relative = Path(key) if layout == 'library' else source_relative/self.tail
        actual = root/relative
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_bytes(self.raw)
        metadata = dict(schema='robotagentbench.local_installation.v1', profile='room',
                        runtime=str(original), source_root=str(original/source_relative),
                        resources={key: str(original/relative)})
        self.write(root, metadata)
        return root, metadata, key, actual, target

    def write(self, root, metadata):
        (root/'runtime.json').write_text(json.dumps(metadata))

    def aliases(self, root):
        with patch.object(resources, 'ALIASES', ((self.tail, hashlib.sha256(self.raw).hexdigest()),)):
            return resources.original_resource_aliases(root)

    def test_old_library_and_current_installed_source_are_exact_readonly_aliases(self):
        for layout in ('library', 'source'):
            with self.subTest(layout=layout), TemporaryDirectory() as directory:
                root, metadata, key, actual, target = self.fixture(directory, layout=layout)
                self.assertEqual(self.aliases(root), [dict(source=str(actual), target=target, read_only=True)])

    def test_relocated_runtime_rebases_only_known_recorded_paths(self):
        with TemporaryDirectory() as directory:
            root, metadata, key, actual, target = self.fixture(directory, recorded_root=Path('/previous/Runtime'))
            self.assertEqual(self.aliases(root)[0]['source'], str(actual))

    def test_metadata_cannot_redirect_to_an_arbitrary_matching_file(self):
        with TemporaryDirectory() as directory:
            root, metadata, key, actual, target = self.fixture(directory)
            other = root/'different.usda'; other.write_bytes(self.raw)
            metadata['resources'][key] = str(other); self.write(root, metadata)
            with self.assertRaisesRegex(ValueError, 'mapping changed'):
                self.aliases(root)

    def test_original_hash_gate_remains_enforced_for_new_layout(self):
        with TemporaryDirectory() as directory:
            root, metadata, key, actual, target = self.fixture(directory)
            actual.write_bytes(b'different scene')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                self.aliases(root)

    def test_symlinked_source_and_resource_are_rejected(self):
        for replace in ('source', 'resource'):
            with self.subTest(replace=replace), TemporaryDirectory() as directory:
                root, metadata, key, actual, target = self.fixture(directory)
                selected = Path(metadata['source_root']) if replace == 'source' else actual
                moved = selected.with_name(selected.name+'-moved')
                selected.rename(moved); selected.symlink_to(moved, target_is_directory=replace == 'source')
                with self.assertRaisesRegex(ValueError, 'Canonical installed'):
                    self.aliases(root)

    def test_parent_traversal_and_outside_runtime_mappings_are_rejected(self):
        for recorded in ('/outside/room.usda', 'inside/../different.usda'):
            with self.subTest(recorded=recorded), TemporaryDirectory() as directory:
                root, metadata, key, actual, target = self.fixture(directory)
                metadata['resources'][key] = str(root/recorded); self.write(root, metadata)
                with self.assertRaises(ValueError): self.aliases(root)

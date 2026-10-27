from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from robotagentbench.simulation.saved_instance_release import context_assets as assets


class TrashAssetIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runtime = self.root / 'Runtime'; self.runtime.mkdir()
        self.source = self.runtime / 'source'; self.source.mkdir()
        self.logical = self.root / 'original/all_assets'
        self.metadata = dict(profile='room', source_files={}, library_files={})
        self.catalog = dict(schema='context_reasoning.trash_assets.v1', assets=[])
        self.manifest = dict(scene='unchanged.usda', objects=[])
        for index in range(5):
            identifier = 'context_trash_fixture_' + str(index)
            relative = 'all_assets/' + identifier + '/asset.usda'
            raw = ('original asset ' + str(index)).encode()
            pin = hashlib.sha256(raw).hexdigest()
            installed = self.source / relative
            library = self.runtime / 'libraries/current_assets' / relative
            logical = self.logical / identifier / 'asset.usda'
            for path in (installed, library, logical):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            self.metadata['source_files'][relative] = pin
            self.metadata['library_files']['libraries/current_assets/' + relative] = pin
            self.catalog['assets'].append(dict(id=str(index), all_assets_id=identifier,
                                               entry=str(logical), sha256=pin))
            self.manifest['objects'].append(dict(task_id='select_trash', instance_id='item_' + str(index),
                id=identifier, path='/World/ContextSelection/Item_' + str(index),
                source='coffee_table.objects', entry=str(installed), position=[index, 0, 1]))
        catalog_path = self.source / assets._CATALOG
        catalog_path.parent.mkdir(parents=True); catalog_path.write_text(json.dumps(self.catalog))
        pin = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        self.metadata['source_files'][assets._CATALOG] = pin
        for name, value in (('_CATALOG_SHA', pin), ('_LOGICAL_ASSETS', self.logical)):
            item = patch.object(assets, name, value); item.start(); self.addCleanup(item.stop)

    def restore(self):
        return assets.restore_trash_entries(self.manifest, source=self.source,
                                           runtime=self.runtime, metadata=self.metadata)

    def test_only_generated_entry_names_change(self):
        before = deepcopy(self.manifest)
        result, evidence = self.restore()
        self.assertEqual(self.manifest, before)
        self.assertEqual(evidence['all_catalog_aliases_verified'], 5)
        self.assertFalse(evidence['physics_scoring_modified'])
        for row, original, asset in zip(result['objects'], before['objects'], self.catalog['assets']):
            self.assertEqual(row['entry'], asset['entry'])
            row['entry'] = original['entry']
        self.assertEqual(result, before)

    def test_original_adapter_rejects_relocated_names_and_accepts_exact_aliases(self):
        import robotagentbench
        directory = Path(robotagentbench.__file__).parent / 'backends/profiles/room/short_task_ContextReasoning/tasks/select_trash'
        package = ModuleType('_trash_identity_test'); package.__path__ = [str(directory)]
        self.addCleanup(lambda: [sys.modules.pop(name, None) for name in tuple(sys.modules)
                               if name == package.__name__ or name.startswith(package.__name__ + '.')])
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location(package.__name__ + '.adapter', directory / 'adapter.py')
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        candidates = [dict(asset, instance_id=row['instance_id'], path=row['path'])
                      for asset, row in zip(self.catalog['assets'], self.manifest['objects'])]
        plan = dict(selection=dict(candidates=candidates))
        with self.assertRaisesRegex(ValueError, 'Candidate entry differs'):
            adapter.bind(deepcopy(plan), self.manifest)
        result, _ = self.restore()
        self.assertEqual(adapter.bind(plan, result)[0]['type'], 'select_trash')

    def test_original_catalogue_tampering_rejected(self):
        (self.source / assets._CATALOG).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): self.restore()

    def test_source_pin_mismatch_rejected(self):
        self.metadata['source_files']['all_assets/context_trash_fixture_0/asset.usda'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'metadata differs'): self.restore()

    def test_library_pin_mismatch_rejected(self):
        self.metadata['library_files']['libraries/current_assets/all_assets/context_trash_fixture_0/asset.usda'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'metadata differs'): self.restore()

    def test_logical_alias_content_mismatch_rejected(self):
        Path(self.catalog['assets'][0]['entry']).write_bytes(b'not the approved asset')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): self.restore()

    def test_installed_asset_content_mismatch_rejected(self):
        Path(self.manifest['objects'][0]['entry']).write_bytes(b'not the approved asset')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): self.restore()

    def test_unrelated_candidate_path_rejected(self):
        self.manifest['objects'][0]['entry'] = self.manifest['objects'][1]['entry']
        with self.assertRaisesRegex(ValueError, 'exact installed or original identity'): self.restore()

    def test_unknown_asset_is_not_repaired(self):
        self.manifest['objects'][0]['id'] = 'context_trash_unknown'
        with self.assertRaisesRegex(ValueError, 'absent from the original'): self.restore()


if __name__ == '__main__': unittest.main()

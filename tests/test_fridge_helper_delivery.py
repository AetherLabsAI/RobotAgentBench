"""Deliver the exact helper/config bytes pinned by the original fridge task."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robotagentbench.backends import sources
from robotagentbench.installation import configure


class FridgeHelperDeliveryTests(unittest.TestCase):
    def test_original_component_pins_and_fresh_runtime_paths(self):
        folder = sources.root('room')
        original = json.loads((folder/'short_tasks/tasks/fridge_to_heated_box/component_sources.json').read_text())
        records = sources._manifest()['profiles']['room']['files']
        names = ['vendor/l1_scene/original_fridge/'+n for n in
                 ('fridge_only.py', 'environment.json', 'fr_tune.json')]
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assets = root/'Assets'; assets.mkdir()
            raw = json.dumps(dict(schema='robotagentbench.asset_index.v1',
                                  version='test', objects={}, locations={})).encode()
            (assets/'resource-index.json').write_bytes(raw)
            manifest = dict(version='test', archives={'index.tar.gz': dict(members={
                'resource-index.json': dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())})})
            installed = configure(manifest, assets, root/'Runtime', 'room')
            metadata = json.loads((root/'Runtime/runtime.json').read_text())
            self.assertEqual(metadata['parameter_overrides'], {})
            for name in names:
                with self.subTest(name=name):
                    data = (folder/name).read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    self.assertEqual(digest, original['files_sha256'][name])
                    self.assertEqual(records[name], dict(sha256=digest, bytes=len(data)))
                    self.assertEqual((Path(installed['source_root'])/name).read_bytes(), data)
                    self.assertEqual(metadata['source_files'][name], digest)
            self.assertFalse(installed['native_started'])


if __name__ == '__main__':
    unittest.main()

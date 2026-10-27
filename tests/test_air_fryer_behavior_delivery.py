"""The air-fryer rail is loaded by filename, not by a Python import graph."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robotagentbench.backends import sources
from robotagentbench.installation import configure, _source_identity


class AirFryerBehaviorDeliveryTests(unittest.TestCase):
    # Original source, also pinned by the pre-release source inventory.
    DIGEST = 'f8395b7677c52f01fd53d838619b5e9a66161d4ac68fbf9cd4a97ecb85715ff5'
    BEHAVIOR = 'all_assets/air_fryer/behaviors/rail_release145.py'
    ADAPTER = 'evaluation/tasks/air_fryer/continuous_openended_v1/native.py'

    def test_original_behavior_is_pinned_in_every_consumer_profile(self):
        manifest = sources._manifest()
        for profile in ('room', 'atomic_coffee', 'failure_recovery'):
            with self.subTest(profile=profile):
                prefix = '' if profile == 'room' else 'generic/'
                name = prefix + self.BEHAVIOR
                raw = (sources.root(profile) / name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), self.DIGEST)
                self.assertEqual(manifest['profiles'][profile]['files'][name],
                                 dict(sha256=self.DIGEST, bytes=len(raw)))
                tree = ast.parse(raw)
                self.assertIn('RailRelease145', [n.name for n in tree.body
                                               if isinstance(n, ast.ClassDef)])

    def test_fresh_install_preserves_the_adapters_dynamic_load_path(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            assets = base / 'Assets'
            assets.mkdir()
            raw = json.dumps(dict(schema='robotagentbench.asset_index.v1',
                                  version='test', objects={}, locations={})).encode()
            (assets / 'resource-index.json').write_bytes(raw)
            manifest = dict(version='test', archives={'index.tar.gz': dict(members={
                'resource-index.json': dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())})})
            for profile in ('room', 'atomic_coffee', 'failure_recovery'):
                with self.subTest(profile=profile):
                    result = configure(manifest, assets, base / profile, profile)
                    selected = sources._manifest()['profiles'][profile]
                    adapter = (self.ADAPTER if profile == 'room' else
                               _source_identity('generic/' + self.ADAPTER, selected))
                    native = Path(result['source_root']) / adapter
                    self.assertTrue(native.is_file())
                    # Native adapter uses Path(__file__).parents[4] / BEHAVIOR.
                    behavior = native.parents[4] / self.BEHAVIOR
                    self.assertEqual(hashlib.sha256(behavior.read_bytes()).hexdigest(), self.DIGEST)
                    self.assertFalse(behavior.is_symlink())
                    self.assertFalse(result['native_started'])
                    metadata = json.loads((base / profile / 'runtime.json').read_text())
                    self.assertEqual(metadata['parameter_overrides'], {})


if __name__ == '__main__':
    unittest.main()

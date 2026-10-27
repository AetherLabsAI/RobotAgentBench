"""CPU-only nested owner dependencies; no USD, simulator or model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from robotagentbench.simulation.saved_instance_release import room_recovery_closure as closure


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RoomRecoveryClosure(unittest.TestCase):
    def fixture(self, root):
        original = closure.LEGACY_GENERIC + '/saved/manifest.json'
        generated = closure.LEGACY_GENERIC + '/saved/room.usdc'
        asset = closure.ROOTS[1] + '/layouts/ID_20_LIVING_STRAIGHT.usdc'
        path = root / 'manifest.json'
        path.write_text(json.dumps({'source_hashes': {asset: 'a' * 64, generated: 'b' * 64}}))
        selected = {'state': {'task_id': 'failure_recovery_sofa',
                             'saved_initialization': {'initial': {'original_launch': {'manifest': original}}},
                             'generated_scene_patches': [{'original_path': generated}]},
                    'attachments': [{'role': 'owner_json', 'original_path': original,
                                     'source': str(path), 'sha256': digest(path)}]}
        return selected, asset, generated, path

    def test_exact_nested_pin_kept_generated_rebinding_remains_owner_owned(self):
        with tempfile.TemporaryDirectory() as temp:
            selected, asset, _, _ = self.fixture(Path(temp).resolve())
            before = deepcopy(selected)
            self.assertEqual(closure.source_requirements(selected), {asset: 'a' * 64})
            self.assertEqual(selected, before)

    def test_other_family_unchanged(self):
        self.assertEqual(closure.source_requirements({'state': {'task_id': 'coffee_bag_interference'}}), {})

    def test_tamper_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            selected, _, _, path = self.fixture(root)
            link = root / 'link.json'; link.symlink_to(path)
            selected['attachments'][0]['source'] = str(link)
            with self.assertRaisesRegex(ValueError, 'changed'):
                closure.source_requirements(selected)
            selected['attachments'][0]['source'] = str(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                closure.source_requirements(selected)

    def test_conflicting_hashes_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            selected, asset, _, _ = self.fixture(Path(temp).resolve())
            selected['state']['nested'] = {'source_hashes': {asset: 'c' * 64}}
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                closure.source_requirements(selected)

    def test_unknown_namespace_and_invalid_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            selected, asset, _, _ = self.fixture(Path(temp).resolve())
            for path, pin in [('/etc/passwd', 'a' * 64), (asset, 'not-a-sha'),
                              (closure.ROOTS[0] + '/../secret', 'a' * 64)]:
                changed = deepcopy(selected); changed['state']['source_hashes'] = {path: pin}
                with self.assertRaises(ValueError):
                    closure.source_requirements(changed)

    def test_duplicate_json_key_is_rejected_even_if_hash_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            selected, _, _, path = self.fixture(Path(temp).resolve())
            path.write_text('{"source_hashes":{},"source_hashes":{}}')
            selected['attachments'][0]['sha256'] = digest(path)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                closure.source_requirements(selected)

    def test_original_verifier_is_invoked_without_sdk_or_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); source = root / 'source'; package = source / 'short_tasks'
            package.mkdir(parents=True); (package / '__init__.py').write_text('')
            (package / 'scene.py').write_text(
                'import hashlib,sys\nfrom pathlib import Path\n'
                'def verify_manifest(manifest):\n'
                ' assert not any(x in sys.modules for x in ("isaacsim","omni","pxr"))\n'
                ' for path,pin in manifest["source_hashes"].items():\n'
                '  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,"original source changed"\n')
            asset = root / 'asset'; asset.write_bytes(b'original asset')
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps({'source_hashes': {str(asset): digest(asset)}}))
            request = root / 'request.json'; request.write_text('{}')
            scene = root / 'scene.usda'; scene.write_text('#usda 1.0')
            initial = {'original_launch': {'manifest': str(manifest), 'request': str(request),
                                          'scene': str(scene), 'scene_sha256': digest(scene)}}
            before = deepcopy(initial)
            result = closure.verify_original_launch(initial, source_root=source)
            self.assertTrue(result['source_hashes_verified']); self.assertFalse(result['native_started'])
            self.assertEqual(initial, before)
            self.assertFalse((package / '__pycache__').exists())
            # Current native room owners keep scene in initial/manifest, not
            # original_launch. Do not require a field absent from that schema.
            without_scene = deepcopy(initial)
            del without_scene['original_launch']['scene']
            del without_scene['original_launch']['scene_sha256']
            without_scene['original_launch']['preparation_receipt'] = str(request)
            self.assertTrue(closure.verify_original_launch(without_scene, source_root=source)['source_hashes_verified'])
            asset.write_bytes(b'tampered')
            with self.assertRaises(subprocess.CalledProcessError):
                closure.verify_original_launch(initial, source_root=source)
            asset.unlink()
            with self.assertRaises(subprocess.CalledProcessError):
                closure.verify_original_launch(initial, source_root=source)

    def test_missing_or_rehashed_launch_input_never_runs_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(closure.subprocess, 'run', side_effect=AssertionError('must not run')):
                with self.assertRaises(ValueError):
                    closure.verify_original_launch({}, source_root=root)
                path = root / 'input'; path.write_text('{}')
                initial = {'original_launch': dict(manifest=str(path), request=str(path), scene=str(path),
                                                    scene_sha256='0' * 64)}
                with self.assertRaisesRegex(ValueError, 'changed'):
                    closure.verify_original_launch(initial, source_root=root)


if __name__ == '__main__':
    unittest.main()

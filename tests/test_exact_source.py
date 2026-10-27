"""Bounded original-byte reconstruction, without USD, models or simulation."""
import base64
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robotagentbench.simulation.saved_instance_release import exact_source as exact


class ExactSource(unittest.TestCase):
    def fixture(self, root):
        assets = root / 'Assets'; assets.mkdir()
        output = root / 'Plan'; output.mkdir()
        index = {'objects': {}, 'source_reconstruction': {}}

        def put(data):
            digest = hashlib.sha256(data).hexdigest()
            relative = 'objects/' + digest[:2] + '/' + digest
            path = assets / relative; path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
            index['objects'][digest] = dict(path=relative, sha256=digest, bytes=len(data))
            return digest

        target = b'abc-X-fghabc'
        digest = hashlib.sha256(target).hexdigest()
        base = put(b'abcdefgh')
        payload = dict(schema=exact.PATCH_SCHEMA, base_sha256=base, target_sha256=digest,
            target_bytes=len(target), operations=[['copy', 0, 3],
                ['literal', base64.b64encode(b'-X-').decode()], ['copy', 5, 3], ['copy', 0, 3]])

        def seal(value):
            patch_hash = put(gzip.compress(json.dumps(value).encode(), mtime=0))
            index['source_reconstruction'][digest] = dict(schema=exact.RECIPE_SCHEMA,
                base_sha256=base, patch_sha256=patch_hash, target_bytes=len(target))

        seal(payload)
        return index, assets, output, digest, target, payload, put, seal

    def test_exact_original_bytes_readonly_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            index, assets, output, digest, target, *_ = self.fixture(Path(temp).resolve())
            before = deepcopy(index)
            with patch('subprocess.run', side_effect=AssertionError('No SDK or subprocess')):
                path = exact.reconstruct(index, assets, output, digest)
            self.assertEqual(path.read_bytes(), target)
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            self.assertEqual(index, before)
            self.assertFalse(path.with_suffix('.partial').exists())
            with self.assertRaisesRegex(ValueError, 'fresh'):
                exact.reconstruct(index, assets, output, digest)
            self.assertEqual(path.read_bytes(), target)

    def test_no_recipe_returns_none_but_malformed_index_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            index, assets, output, digest, *_ = self.fixture(Path(temp).resolve())
            self.assertIsNone(exact.reconstruct(index, assets, output, 'e' * 64))
            index['source_reconstruction'] = []
            with self.assertRaises(ValueError):
                exact.reconstruct(index, assets, output, digest)

    def test_base_and_patch_bytes_are_verified_and_never_fallback(self):
        for role in ('base_sha256', 'patch_sha256'):
            for missing in (False, True):
                with self.subTest(role=role, missing=missing), tempfile.TemporaryDirectory() as temp:
                    index, assets, output, digest, *_ = self.fixture(Path(temp).resolve())
                    source = assets / index['objects'][index['source_reconstruction'][digest][role]]['path']
                    source.unlink() if missing else source.write_bytes(b'corrupt')
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        exact.reconstruct(index, assets, output, digest)
                    self.assertFalse((output / 'exact_source_inputs').exists())

    def test_paths_symlinks_unknown_fields_and_installed_target_rejected(self):
        for mutation in ('path', 'symlink', 'recipe', 'installed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                index, assets, output, digest, _, payload, *_ = self.fixture(Path(temp).resolve())
                base_record = index['objects'][payload['base_sha256']]
                if mutation == 'path': base_record['path'] = '../outside'
                elif mutation == 'symlink':
                    original = assets / base_record['path']; other = output / 'base'
                    original.rename(other); original.symlink_to(other)
                elif mutation == 'recipe': index['source_reconstruction'][digest]['path'] = '/untrusted'
                else: index['objects'][digest] = deepcopy(base_record)
                with self.assertRaises(ValueError):
                    exact.reconstruct(index, assets, output, digest)

    def test_invalid_operations_and_identity_fail_before_output(self):
        changes = [dict(schema='prototype'), dict(target_sha256='a' * 64), dict(target_bytes=True),
                   dict(operations=[['copy', -1, 12]]), dict(operations=[['copy', True, 12]]),
                   dict(operations=[['copy', 0, 9]]), dict(operations=[['copy', 0, 0]]),
                   dict(operations=[['eval', 'unsafe']]), dict(operations=[['literal', '###']]),
                   dict(operations=[['copy', 0, 1]]), dict(operations=[])]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                index, assets, output, digest, _, payload, _, seal = self.fixture(Path(temp).resolve())
                payload.update(change); seal(payload)
                with self.assertRaises(ValueError):
                    exact.reconstruct(index, assets, output, digest)
                self.assertFalse((output / 'exact_source_inputs').exists())

    def test_wrong_output_sha_never_publishes_target(self):
        with tempfile.TemporaryDirectory() as temp:
            index, assets, output, digest, _, payload, _, seal = self.fixture(Path(temp).resolve())
            payload['operations'][1][1] = base64.b64encode(b'-Y-').decode(); seal(payload)
            with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                exact.reconstruct(index, assets, output, digest)
            self.assertFalse((output / 'exact_source_inputs' / digest).exists())

    def test_expanded_json_and_source_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            index, assets, output, digest, *_ = self.fixture(Path(temp).resolve())
            with patch.object(exact, 'MAX_PATCH_BYTES', 200):
                with self.assertRaises(ValueError):
                    exact.reconstruct(index, assets, output, digest)
            index['source_reconstruction'][digest]['target_bytes'] = exact.MAX_SOURCE_BYTES + 1
            with self.assertRaisesRegex(ValueError, 'bounded'):
                exact.reconstruct(index, assets, output, digest)

    def test_duplicate_patch_keys_and_output_symlink_fail(self):
        for duplicate in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                index, assets, output, digest, _, payload, put, _ = self.fixture(Path(temp).resolve())
                if duplicate:
                    raw = json.dumps(payload)[:-1] + ',"schema":"' + exact.PATCH_SCHEMA + '"}'
                    index['source_reconstruction'][digest]['patch_sha256'] = put(gzip.compress(raw.encode(), mtime=0))
                else:
                    (output / 'exact_source_inputs').symlink_to(assets, target_is_directory=True)
                with self.assertRaises(ValueError):
                    exact.reconstruct(index, assets, output, digest)


if __name__ == '__main__':
    unittest.main()

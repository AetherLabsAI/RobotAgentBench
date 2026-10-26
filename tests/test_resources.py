import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from robotagentbench import resources


class AssetReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.cache = self.base / 'cache'; self.cache.mkdir()

    def archive(self, files, *, entries=None):
        path = self.cache / 'assets.tar.gz'
        with tarfile.open(path, 'w:gz') as output:
            for name, data in (entries or files.items()):
                member = tarfile.TarInfo(name); member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        raw = path.read_bytes()
        return dict(schema=resources.SCHEMA, version='test-v1', hosting=None,
                    archives={'assets.tar.gz':dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        members={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})})

    def manifest(self, value):
        path = self.base / 'release.json'; path.write_text(json.dumps(value))
        return resources.read_manifest(path)

    def test_offline_install_verify_and_no_overwrite(self):
        value = self.manifest(self.archive({'objects/mesh.usda':b'USD test bytes', 'objects/texture.bin':b'abc'}))
        destination = self.base / 'assets'
        result = resources.install(value, self.cache, destination)
        self.assertTrue(result['verified']); self.assertFalse(result['native_started'])
        self.assertEqual(resources.verify(value, destination)['files'], 2)
        with self.assertRaises(ValueError):
            resources.install(value, self.cache, destination)
        (destination / 'objects/texture.bin').write_bytes(b'abd')
        with self.assertRaises(ValueError):
            resources.verify(value, destination)

    def test_unpublished_hosting_fails_before_network(self):
        value = self.manifest(self.archive({'file':b'x'}))
        with self.assertRaisesRegex(ValueError, 'not published'):
            resources.fetch(value, self.cache)

    def test_v2_index_must_select_an_exact_pinned_member(self):
        value=self.archive({'resource-index-v2.json':b'{}'})
        value.update(schema='robotagentbench.asset_release.v2',index='resource-index-v2.json')
        self.assertEqual(self.manifest(value)['index'],'resource-index-v2.json')
        for name in ('missing.json','../resource-index-v2.json'):
            value['index']=name
            with self.assertRaises(ValueError):self.manifest(value)

    def test_mutable_or_unapproved_hosting_rejected(self):
        for revision in ('main', 'v1', '../bad'):
            value = self.archive({'file':b'x'})
            value['hosting'] = dict(provider='huggingface',repository='team/assets',revision=revision)
            with self.assertRaises(ValueError):
                self.manifest(value)

    def test_duplicate_json_fields_rejected(self):
        path = self.base / 'duplicate.json'; path.write_text('{"schema":1,"schema":2}')
        with self.assertRaises(ValueError):
            resources.read_manifest(path)

    def test_traversal_and_alias_manifest_names_rejected(self):
        for name in ('../escape','/absolute','a/../b','a//b','a\\b'):
            value = self.archive({name:b'x'})
            with self.assertRaises(ValueError):
                self.manifest(value)

    def test_bad_archive_and_member_hash_do_not_promote_installation(self):
        value = self.manifest(self.archive({'file':b'x'}))
        value['archives']['assets.tar.gz']['members']['file']['sha256'] = '0'*64
        destination = self.base / 'assets'
        with self.assertRaises(ValueError):
            resources.install(value, self.cache, destination)
        self.assertFalse(destination.exists())
        self.assertTrue(list(self.base.glob('assets.partial-*')))

    def test_extra_duplicate_and_traversal_archive_members_rejected(self):
        for entries in ([('file',b'x'),('extra',b'y')], [('file',b'x'),('file',b'x')], [('../escape',b'x')]):
            value = self.manifest(self.archive({'file':b'x'}, entries=entries))
            with self.assertRaises(ValueError):
                resources.install(value, self.cache, self.base/'assets')
            self.assertFalse((self.base/'assets').exists())
            self.assertFalse((self.base.parent/'escape').exists())

    def test_archive_symlink_rejected(self):
        value = self.archive({'file':b'x'})
        archive = self.cache/'assets.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            member = tarfile.TarInfo('file'); member.type=tarfile.SYMTYPE; member.linkname='../escape'
            output.addfile(member)
        raw=archive.read_bytes(); value['archives']['assets.tar.gz'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            resources.install(self.manifest(value), self.cache, self.base/'assets')

    def test_zero_or_boolean_sizes_rejected(self):
        value = self.archive({'file':b'x'}); value['archives']['assets.tar.gz']['bytes']=True
        with self.assertRaises(ValueError):
            self.manifest(value)

    def test_duplicate_members_across_archives_rejected(self):
        value = self.archive({'file':b'x'})
        value['archives']['other.tar.gz'] = value['archives']['assets.tar.gz']
        with self.assertRaises(ValueError):
            self.manifest(value)


if __name__ == '__main__':
    unittest.main()

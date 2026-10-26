import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.installation import configure,_source_identity


class LocalInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.assets = self.base/'Assets'; self.assets.mkdir()
        self.code = self.base/'code'; self.code.mkdir()
        raw = b'VALUE = 1\n'; (self.code/'module.py').write_bytes(raw)
        self.pin = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        source = dict(profiles={'room':dict(files={'module.py':self.pin},compatibility_namespaces=[])})
        for name,value in (('_manifest',source),('verify',{'verified':True}),('root',self.code)):
            p=patch('robotagentbench.installation.sources.'+name,return_value=value)
            p.start();self.addCleanup(p.stop)

    def manifest(self, files):
        locations,objects,members = {},{},{}
        for location,raw in files.items():
            digest=hashlib.sha256(raw).hexdigest();path='objects/'+digest[:2]+'/'+digest
            target=self.assets/path;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(raw)
            row=dict(sha256=digest,bytes=len(raw))
            objects[digest]=dict(path=path,**row);members[path]=row;locations[location]=digest
        index=dict(schema='robotagentbench.asset_index.v1',version='test',objects=objects,locations=locations)
        raw=json.dumps(index).encode();(self.assets/'resource-index.json').write_bytes(raw)
        members['resource-index.json']=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        return dict(version='test',archives={'assets.tar.gz':dict(members=members)})

    def test_materializes_bundled_code_data_and_local_aliases(self):
        manifest=self.manifest({'profiles/room/resources/object.usda':b'usd test',
                                'libraries/frozen_assets/asset.bin':b'123'})
        result=configure(manifest,self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        self.assertEqual((root/'module.py').read_bytes(),b'VALUE = 1\n')
        self.assertEqual((root/'resources/object.usda').read_bytes(),b'usd test')
        self.assertEqual((root/'api-protocol/resources').resolve(),root/'resources')
        self.assertFalse(result['native_started'])
        configuration=json.loads((self.base/'Runtime/runtime.json').read_text())
        self.assertEqual(configuration['parameter_overrides'],{})
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Runtime','room')

    def test_canonicalizes_existing_source_asset_alias(self):
        manifest=self.manifest({'profiles/room/api-protocol/all_assets/texture.png':b'png bytes'})
        result=configure(manifest,self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        self.assertEqual((root/'all_assets/texture.png').read_bytes(),b'png bytes')
        self.assertEqual((root/'api-protocol/all_assets').resolve(),root/'all_assets')

    def test_explicit_index_selects_pinned_supplement_without_overwriting_old_index(self):
        manifest=self.manifest({'profiles/room/data.json':b'123'})
        original=self.assets/'resource-index.json'
        selected=self.assets/'resource-index-v2.json'
        selected.write_bytes(original.read_bytes())
        row=manifest['archives']['assets.tar.gz']['members'].pop('resource-index.json')
        manifest['archives']['supplement.tar.gz']={'members':{'resource-index-v2.json':row}}
        manifest['index']='resource-index-v2.json'
        original.write_text('old index retained, not selected')
        result=configure(manifest,self.assets,self.base/'Runtime','room')
        self.assertEqual((Path(result['source_root'])/'data.json').read_bytes(),b'123')
        self.assertEqual(original.read_text(),'old index retained, not selected')

    def test_explicit_profile_data_overrides_older_library_alias_only(self):
        manifest=self.manifest({'libraries/generic_current/resources/object.usda':b'old',
                                'profiles/room/resources/object.usda':b'current'})
        result=configure(manifest,self.assets,self.base/'Runtime','room')
        self.assertEqual((Path(result['source_root'])/'resources/object.usda').read_bytes(),b'current')

    def test_current_asset_library_is_available_at_canonical_source_and_retained(self):
        manifest=self.manifest({'libraries/generic_current/api-protocol/all_assets/bean/asset.usda':b'old',
                                'libraries/current_assets/all_assets/bean/asset.usda':b'current',
                                'profiles/room/all_assets/bean/asset.usda':b'selected'})
        result=configure(manifest,self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        self.assertEqual((root/'api-protocol/all_assets/bean/asset.usda').read_bytes(),b'selected')
        self.assertEqual((self.base/'Runtime/libraries/current_assets/all_assets/bean/asset.usda').read_bytes(),b'current')
        self.assertEqual((self.base/'Runtime/libraries/generic_current/api-protocol/all_assets/bean/asset.usda').read_bytes(),b'old')

    def test_tampered_content_and_index_rejected(self):
        manifest=self.manifest({'profiles/room/data.json':b'123'})
        target=next((self.assets/'objects').rglob('*'))
        target=next(p for p in (self.assets/'objects').rglob('*') if p.is_file())
        target.write_bytes(b'124')
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())
        (self.assets/'resource-index.json').write_text('{}')
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Other','room')

    def test_unknown_profile_rejected(self):
        manifest=self.manifest({'file':b'x'})
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Runtime','unknown')

    def test_conflicting_same_priority_aliases_rejected(self):
        manifest=self.manifest({'profiles/room/resources/file':b'a',
                                'profiles/room/api-protocol/resources/file':b'b'})
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_specialized_identity_stays_inside_local_root(self):
        profile=dict(compatibility_namespaces=[
            dict(source='generic',target='/home/lingjun/robot_agent_arena/source'),
            dict(source='coffee',target='/mnt/agent-bench-data/lingjun/coffee'),
            dict(source='coffee',target='/home/lingjun/robot_agent_arena/source/coffee')])
        self.assertEqual(_source_identity('generic/task/score.py',profile),
                         'home/lingjun/robot_agent_arena/source/task/score.py')
        self.assertEqual(_source_identity('coffee/api.py',profile),
                         'mnt/agent-bench-data/lingjun/coffee/api.py')
        self.assertFalse(Path(_source_identity('coffee/api.py',profile)).is_absolute())


if __name__=='__main__':unittest.main()

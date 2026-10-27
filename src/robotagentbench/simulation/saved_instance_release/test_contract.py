"""Small synthetic CPU packaging boundaries; no real task/admission claims."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from . import Builder,materialize,sha

ROOT=Path(__file__).resolve().parent/'private'

class ReleaseContracts(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.root=Path(self.tmp.name)
        self.assets=self.root/'Assets';self.assets.mkdir();self.package=self.root/'package';self.package.mkdir()
        self.asset=b'mock-already-released-asset';self.digest=hashlib.sha256(self.asset).hexdigest()
        path='objects/'+self.digest[:2]+'/'+self.digest;p=self.assets/path;p.parent.mkdir(parents=True);p.write_bytes(self.asset)
        self.manifest=self.root/'assets.json';self.manifest.write_text(json.dumps(dict(schema='robotagentbench.asset_release.v1',version='mock',hosting=None,
            archives={'mock.tar.gz':dict(bytes=1,sha256='0'*64,members={path:dict(bytes=len(self.asset),sha256=self.digest)})})))
        self.launch=self.root/'launch.json';self.launch.write_text('{"scene":"/original/scene.usda"}\n')
    def tearDown(self):self.tmp.cleanup()
    def builder(self,**kwargs):return Builder(self.root/'release',asset_manifest=self.manifest,**kwargs)
    def test_asset_not_repacked_and_launch_unchanged(self):
        sdk=self.root/'sdk';sdk.mkdir();(sdk/'piece.bin').write_bytes(b'mock-sdk-material')
        b=self.builder(sdk_sources={sha(sdk/'piece.bin'):'piece.bin'})
        launch=b.add('/original/launch.json',self.launch,sha(self.launch),'launch')
        asset=b.add('/original/scene.usda',self.assets/('objects/'+self.digest[:2]+'/'+self.digest),self.digest,'scene')
        asset_alias=b.add('/another/scene.usda',self.assets/('objects/'+self.digest[:2]+'/'+self.digest),self.digest,'scene')
        sdk_node=b.add('/opt/mock-sdk/piece.bin',sdk/'piece.bin',sha(sdk/'piece.bin'),'material')
        b.group('original-common-source-and-assets',[asset,asset_alias,sdk_node])
        b.instance('mock',1,7,launch,[],groups=['original-common-source-and-assets'],provenance={'synthetic':True})
        result=b.finish(closure_evidence={'synthetic_only':True})
        self.assertEqual(result['nodes'][asset]['storage']['kind'],'assets')
        self.assertNotIn(self.digest,result['objects'])
        out=materialize(self.root/'release/release.json',self.root/'offline','mock',1,assets=self.assets,package=self.package,sdk=sdk)
        self.assertEqual((self.root/'offline/namespace/original/launch.json').read_bytes(),self.launch.read_bytes())
        self.assertEqual(out['operator']['launch'],'/original/launch.json');self.assertFalse(out['native_started'])
        self.assertIn(dict(logical_root='/opt/mock-sdk',backing_root=str(sdk)),out['operator']['readonly_bindings'])
        copied=self.root/'offline/namespace/original/scene.usda'
        self.assertNotEqual(copied.stat().st_ino,(self.assets/('objects/'+self.digest[:2]+'/'+self.digest)).stat().st_ino)
        self.assertEqual((self.root/'offline/namespace/another/scene.usda').read_bytes(),self.asset)
    def test_one_namespace_cannot_mix_two_versions(self):
        b=self.builder();first=b.add('/original/launch.json',self.launch,sha(self.launch),'launch')
        other=self.root/'other';other.write_bytes(b'other')
        second=b.add('/original/launch.json',other,sha(other),'dependency')
        with self.assertRaises(ValueError):b.instance('mock',1,1,first,[second],provenance={})
    def test_logs_and_corruption_refused(self):
        b=self.builder()
        with self.assertRaises(ValueError):b.add('/original/operator.log',self.launch,sha(self.launch))
        with self.assertRaises(ValueError):b.add('/original/model_events.jsonl',self.launch,sha(self.launch))
        with self.assertRaises(ValueError):b.add('/home/example/.codex/auth.json',self.launch,sha(self.launch))
        with self.assertRaises(ValueError):b.add('/original/launch.json',self.launch,'0'*64,'launch')
    def test_materialized_public_input_has_exact_worker_binding(self):
        b=self.builder();launch=b.add('/original/launch.json',self.launch,sha(self.launch),'launch')
        public=self.root/'public.json';public.write_text('{"prompt":"Public request","context":"Current memory"}')
        node=b.add('/original/public.json',public,sha(public),'public_input')
        b.instance('mock',1,1,launch,[],public_input_node=node,provenance={})
        b.finish(closure_evidence={'synthetic_only':True})
        out=materialize(self.root/'release/release.json',self.root/'offline','mock',1,
            assets=self.assets,package=self.package)
        binding=json.loads(Path(out['operator']['public_input_binding']).read_text())
        self.assertEqual(binding,dict(schema='robotagentbench.public_input_binding.v1',
            source='/original/public.json',sha256=sha(public),launch_sha256=sha(self.launch)))
        self.assertEqual(out['model_input'],dict(instruction='Public request',context='Current memory'))
        self.assertEqual((self.root/'offline/namespace/original/launch.json').read_bytes(),self.launch.read_bytes())
    def test_runtime_is_derived_with_original_source_pins(self):
        from robotagentbench.example import runtime
        source=self.root/'entry.py';source.write_text('VALUE=1\n');pin=sha(source)
        b=self.builder();launch=b.add('/original/launch.json',self.launch,sha(self.launch),'launch')
        code=b.add('/original/source/entry.py',source,pin,'source');b.group('source-profile',[code])
        b.instance('mock',1,1,launch,[],groups=['source-profile'],provenance={})
        b.instances[0]['runtime_profile']='source-profile'
        b.finish(closure_evidence={'runtime_profiles':{'source-profile':dict(source_root='/original/source',
            profile='room',source_files={'entry.py':pin},resources={})}})
        out=materialize(self.root/'release/release.json',self.root/'offline','mock',1,
            assets=self.assets,package=self.package)
        base,root,value=runtime(out['operator']['runtime'])
        self.assertEqual(base,self.root/'offline')
        self.assertEqual(root,self.root/'offline/namespace/original/source')
        self.assertEqual(value['source_files'],{'entry.py':pin})

if __name__=='__main__':unittest.main()

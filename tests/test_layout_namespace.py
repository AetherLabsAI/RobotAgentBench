"""Focused CPU checks for exact original layout and dependency namespace wiring."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.installation import configure,ORIGINAL_LAYOUT_LIBRARY
from robotagentbench.namespace import command


class OriginalLayoutNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.assets=self.base/'Assets';self.assets.mkdir()
        self.code=self.base/'code';self.code.mkdir();raw=b'ORIGINAL_SOURCE = True\n'
        (self.code/'original.py').write_bytes(raw)
        self.source_pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        profile=dict(profiles={'room':dict(files={'original.py':self.source_pin},compatibility_namespaces=[])})
        for name,value in(('_manifest',profile),('verify',{'verified':True}),('root',self.code)):
            mocked=patch('robotagentbench.installation.sources.'+name,return_value=value)
            mocked.start();self.addCleanup(mocked.stop)

    def manifest(self,files):
        locations={};objects={};members={}
        for location,raw in files.items():
            digest=hashlib.sha256(raw).hexdigest();relative='objects/'+digest[:2]+'/'+digest
            path=self.assets/relative;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(raw)
            record=dict(bytes=len(raw),sha256=digest)
            locations[location]=digest;objects[digest]=dict(path=relative,**record);members[relative]=record
        index=dict(schema='robotagentbench.asset_index.v1',version='test-layouts',locations=locations,objects=objects)
        raw=json.dumps(index).encode();(self.assets/'resource-index.json').write_bytes(raw)
        members['resource-index.json']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return dict(version='test-layouts',archives={'assets.tar.gz':dict(members=members)})

    def fifty(self):
        return {ORIGINAL_LAYOUT_LIBRARY+f'ID_{number:02d}_LIVING_STRAIGHT.usdc':
                ('# exact original layout '+str(number)).encode()for number in range(1,51)}

    def mounts(self,runtime):
        argv=command(runtime,'example:image','plan',env={})
        self.assertNotIn('--gpus',argv);self.assertIn('runc',argv)
        return [argv[i+1]for i,value in enumerate(argv[:-1])if value=='--mount']

    def test_all_fifty_layouts_and_relative_dependencies_share_one_readonly_original_root(self):
        files=self.fifty();files[ORIGINAL_LAYOUT_LIBRARY+'adjacent/material.usda']=b'exact adjacent layer'
        files['libraries/original_absolute/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/assets/wall.png']=b'exact texture'
        manifest=self.manifest(files);index_before=(self.assets/'resource-index.json').read_bytes()
        result=configure(manifest,self.assets,self.base/'Runtime','room');runtime=Path(result['runtime'])
        metadata=json.loads((runtime/'runtime.json').read_text())
        for location,raw in files.items():
            self.assertEqual((runtime/location).read_bytes(),raw)
            self.assertEqual(metadata['library_files'][location],hashlib.sha256(raw).hexdigest())
        mounts=self.mounts(runtime);target='/mnt/agent-bench-data/lingjun'
        matching=[row for row in mounts if ',dst='+target+','in row]
        self.assertEqual(matching,['type=bind,src='+str(runtime/'libraries/original_absolute/mnt/agent-bench-data/lingjun')+',dst='+target+',readonly'])
        self.assertEqual(metadata['parameter_overrides'],{})
        self.assertEqual(metadata['source_files'],{'original.py':self.source_pin['sha256']})
        self.assertEqual((self.assets/'resource-index.json').read_bytes(),index_before)

    def test_recovery_layout_names_and_legacy_textures_are_merged_without_hiding_canonical_files(self):
        files={}
        for number in range(1,51):
            name=f'ID_{number:02d}_LIVING_STRAIGHT.usdc';raw=('exact layout '+str(number)).encode()
            files[('libraries/recovery_original_layouts/'if number<=33 else ORIGINAL_LAYOUT_LIBRARY)+name]=raw
        old='libraries/legacy_data/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/assets/texture.png'
        files[old]=b'original texture'
        result=configure(self.manifest(files),self.assets,self.base/'Runtime','room');runtime=Path(result['runtime'])
        metadata=json.loads((runtime/'runtime.json').read_text())
        for number in range(1,51):
            canonical=ORIGINAL_LAYOUT_LIBRARY+f'ID_{number:02d}_LIVING_STRAIGHT.usdc'
            raw=('exact layout '+str(number)).encode()
            self.assertEqual((runtime/canonical).read_bytes(),raw)
            self.assertEqual(metadata['library_files'][canonical],hashlib.sha256(raw).hexdigest())
        canonical=old.replace('libraries/legacy_data/','libraries/original_absolute/',1)
        self.assertEqual((runtime/old).read_bytes(),(runtime/canonical).read_bytes())
        self.assertEqual(metadata['resources'][old],str(runtime/old))
        mounts=self.mounts(runtime)
        self.assertTrue(any('/original_absolute/'in row and ',dst=/mnt/agent-bench-data/lingjun,readonly'in row for row in mounts))
        self.assertFalse(any('/legacy_data/'in row for row in mounts))

    def test_same_original_identity_with_different_library_bytes_fails_before_staging(self):
        canonical='libraries/original_absolute/mnt/agent-bench-data/lingjun/material.png'
        legacy=canonical.replace('original_absolute','legacy_data')
        with self.assertRaisesRegex(ValueError,'Conflicting immutable original namespace'):
            configure(self.manifest({canonical:b'original',legacy:b'different'}),self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists());self.assertFalse(list(self.base.glob('Runtime.partial-*')))

    def test_conflicting_layout_alias_is_not_silently_preferred(self):
        name='ID_25_LIVING_STRAIGHT.usdc'
        with self.assertRaisesRegex(ValueError,'Conflicting immutable original namespace'):
            configure(self.manifest({ORIGINAL_LAYOUT_LIBRARY+name:b'one',
                'libraries/recovery_original_layouts/'+name:b'two'}),self.assets,self.base/'Runtime','room')

    def test_identical_duplicate_locations_preserve_original_bytes(self):
        name='ID_25_LIVING_STRAIGHT.usdc';raw=b'exact original'
        result=configure(self.manifest({ORIGINAL_LAYOUT_LIBRARY+name:raw,
            'libraries/recovery_original_layouts/'+name:raw}),self.assets,self.base/'Runtime','room')
        self.assertEqual((Path(result['runtime'])/ORIGINAL_LAYOUT_LIBRARY/name).read_bytes(),raw)

    def test_old_partial_runtime_requests_reconfiguration_instead_of_hiding_a_layout(self):
        runtime=self.base/'Runtime';source=runtime/'source';source.mkdir(parents=True)
        partial=runtime/'libraries/recovery_original_layouts/ID_25_LIVING_STRAIGHT.usdc'
        partial.parent.mkdir(parents=True);partial.write_bytes(b'original')
        metadata=dict(schema='robotagentbench.local_installation.v1',profile='room',runtime=str(runtime),
            source_root=str(source),library_files={str(partial.relative_to(runtime)):hashlib.sha256(b'original').hexdigest()})
        (runtime/'runtime.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'Reconfigure this Runtime'):
            self.mounts(runtime)

    def test_changed_content_object_is_rejected_before_alias_installation(self):
        name='ID_25_LIVING_STRAIGHT.usdc';manifest=self.manifest({'libraries/recovery_original_layouts/'+name:b'original'})
        path=next(path for path in(self.assets/'objects').rglob('*')if path.is_file());path.write_bytes(b'tampered')
        with self.assertRaises(ValueError):configure(manifest,self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())


if __name__=='__main__':unittest.main()

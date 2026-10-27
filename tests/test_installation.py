import hashlib
import importlib.util
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
        self.assertEqual(configuration['library_files'], {
            'libraries/frozen_assets/asset.bin': hashlib.sha256(b'123').hexdigest()})
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

    def test_original_room_templates_are_available_in_mounted_source_families(self):
        prefix='libraries/original_absolute/home/lingjun/robot_agent_arena/'
        files={
            'openend/tasks/all_objects_integration/room_all_dynamic_v2/room.usda':b'prototype',
            'openend/tasks/all_objects_integration/room_with_existing_drawer_v1/room.usda':b'sublayer',
            'layout_task_activation_v1/task_scene_registration_v1/other_layouts_v4/layout04/room.usda':b'base',
        }
        data={prefix+path:raw for path,raw in files.items()}
        data[prefix+'unrelated/not_mounted.usda']=b'library only'
        result=configure(self.manifest(data),self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        for path,raw in files.items():
            self.assertEqual((root/path).read_bytes(),raw)
            self.assertEqual((self.base/'Runtime'/prefix/path).read_bytes(),raw)
        self.assertFalse((root/'unrelated/not_mounted.usda').exists())
        meta=json.loads((self.base/'Runtime/runtime.json').read_text())
        self.assertEqual(meta['parameter_overrides'],{})
        for path,raw in files.items():
            self.assertEqual(meta['source_files'][path],hashlib.sha256(raw).hexdigest())

    def test_unknown_profile_rejected(self):
        manifest=self.manifest({'file':b'x'})
        with self.assertRaises(ValueError):
            configure(manifest,self.assets,self.base/'Runtime','unknown')

    def long_profile(self):
        from robotagentbench.installation import _ROOM_LONG_ENTRY
        (self.code/_ROOM_LONG_ENTRY).parent.mkdir(parents=True)
        (self.code/_ROOM_LONG_ENTRY).write_bytes(b'VALUE = 1\n')
        return dict(profiles={'room':dict(files={_ROOM_LONG_ENTRY:self.pin},compatibility_namespaces=[])})

    def test_long_admission_receives_exact_bundled_calibration(self):
        from robotagentbench.installation import _ROOM_HOTWAKE
        from robotagentbench.simulation.composite_instance_bank.source_manifest import EMBEDDED_DATA
        original=EMBEDDED_DATA['embedded/hotwake_probe_inputs.json']
        with patch('robotagentbench.installation.sources._manifest',return_value=self.long_profile()):
            result=configure(self.manifest({}),self.assets,self.base/'Runtime','room')
        self.assertEqual((Path(result['source_root'])/_ROOM_HOTWAKE).read_bytes(),original)
        meta=json.loads((self.base/'Runtime/runtime.json').read_text())
        self.assertEqual(meta['source_files'][_ROOM_HOTWAKE],hashlib.sha256(original).hexdigest())
        self.assertEqual(meta['parameter_overrides'],{})
        self.assertFalse(result['native_started'])

    def test_long_calibration_rejects_conflicting_installed_resource(self):
        from robotagentbench.installation import _ROOM_HOTWAKE
        with patch('robotagentbench.installation.sources._manifest',return_value=self.long_profile()):
            with self.assertRaisesRegex(ValueError,'Conflicting original long-task calibration'):
                configure(self.manifest({'profiles/room/'+_ROOM_HOTWAKE:b'changed sensor'}),
                          self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_long_calibration_rejects_changed_embedded_bytes(self):
        from robotagentbench.simulation.composite_instance_bank.source_manifest import EMBEDDED_DATA
        with patch('robotagentbench.installation.sources._manifest',return_value=self.long_profile()), \
             patch.dict(EMBEDDED_DATA,{'embedded/hotwake_probe_inputs.json':b'changed calibration'}):
            with self.assertRaisesRegex(ValueError,'Original long-task calibration input changed'):
                configure(self.manifest({}),self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_discovery_original_inputs_load_from_installed_source_data(self):
        import robotagentbench
        package=Path(robotagentbench.__file__).parent
        original=package/'backends/profiles/room'
        manifest=json.loads((package/'backends/manifest.json').read_text())
        prefix='short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/task_discovery_v1/'
        names=[prefix+'inputs.py',prefix+'person_context_v4/context.txt',prefix+'person_context_v4/prompts.json']
        pins={name:manifest['profiles']['room']['files'][name] for name in names}
        for name,pin in pins.items():
            raw=(original/name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),pin['sha256'])
            self.assertEqual(len(raw),pin['bytes'])
            target=self.code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        source=dict(profiles={'room':dict(files=pins,compatibility_namespaces=[])})
        with patch('robotagentbench.installation.sources._manifest',return_value=source):
            result=configure(self.manifest({}),self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        spec=importlib.util.spec_from_file_location('_discovery_original_inputs',root/prefix/'inputs.py')
        inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(inputs)
        prompts=json.loads((root/prefix/'person_context_v4/prompts.json').read_text())['prompts']
        self.assertEqual(len(prompts),10)
        for prompt in prompts:
            public=inputs.model_input(prompt['id'])
            self.assertEqual(public['context'],(original/prefix/'person_context_v4/context.txt').read_text())
            self.assertEqual(public['prompt'],prompt['prompt']+'\n\n'+inputs.DISCOVERY_SUFFIX)
            self.assertTrue(inputs.wire_instruction(public).startswith(inputs.PROTECTED_DISCOVERY_DIRECTIVE))

    def hand_profile(self):
        from robotagentbench.installation import _ROOM_HAND_ROOT
        module = _ROOM_HAND_ROOT + 'adaptive_hand.py'
        (self.code/module).parent.mkdir(parents=True)
        (self.code/module).write_bytes(b'VALUE = 1\n')
        return dict(profiles={'room':dict(files={module:self.pin},compatibility_namespaces=[])})

    def hand_resources(self):
        from robotagentbench.installation import _ROOM_HAND_ORIGINAL
        files = {
            'profiles/room/'+_ROOM_HAND_ORIGINAL+'assets/derived/left_hand_side_entry_012.json':
                b'{"texture":"../model/new_diff.png"}',
            'profiles/room/'+_ROOM_HAND_ORIGINAL+'assets/model/new_diff.png':b'original png',
            'specialized/asset_00865a85a1b6275f0f75769f':b'{"frames":[],"rest_bones":{}}',
            'specialized/asset_3206da7aeb769f0401e9f74e':b'{"policy":"fixture","products":{}}',
        }
        tails = ('assets/derived/left_hand_side_entry_012.json','assets/model/new_diff.png',
                 'static_short_hand_011/cropped_rig.json','source_product_ranges.json')
        aliases = tuple((location,hashlib.sha256(raw).hexdigest(),(tail,))
                        for (location,raw),tail in zip(files.items(),tails))
        return files, aliases

    def test_shelf_interference_runtime_restores_exact_existing_asset_closure(self):
        from robotagentbench.installation import _ROOM_HAND_ROOT, _ROOM_HAND_ORIGINAL
        files, aliases = self.hand_resources()
        with patch('robotagentbench.installation.sources._manifest',return_value=self.hand_profile()), \
             patch('robotagentbench.installation._ROOM_HAND_RESOURCES',aliases):
            result=configure(self.manifest(files),self.assets,self.base/'Runtime','room')
        root=Path(result['source_root'])
        meta=json.loads((self.base/'Runtime/runtime.json').read_text())
        for location,digest,tails in aliases:
            for prefix in (_ROOM_HAND_ROOT,_ROOM_HAND_ORIGINAL):
                path=prefix+tails[0]
                self.assertEqual((root/path).read_bytes(),files[location])
                self.assertEqual(meta['source_files'][path],digest)
        hand=root/_ROOM_HAND_ROOT/'assets/derived/left_hand_side_entry_012.json'
        texture=json.loads(hand.read_text())['texture']
        self.assertTrue((hand.parent/texture).is_file())
        self.assertEqual(meta['parameter_overrides'],{})
        self.assertFalse(result['native_started'])

    def test_shelf_product_range_alias_uses_existing_original_resource_pin(self):
        from robotagentbench.installation import _ROOM_HAND_RESOURCES, _ROOM_HAND_ORIGINAL
        from robotagentbench.simulation import dedicated_delivery
        rows=[row for row in _ROOM_HAND_RESOURCES if row[2]==('source_product_ranges.json',)]
        self.assertEqual(len(rows),1)
        location,digest,_=rows[0]
        self.assertEqual(location,'specialized/asset_3206da7aeb769f0401e9f74e')
        original=('/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/'
                  'deployment_generic_20261010/source/'+_ROOM_HAND_ORIGINAL+'source_product_ranges.json')
        self.assertEqual(digest,dedicated_delivery.template()['aliases'][original])

    def test_shelf_product_range_alias_rejects_another_sampling_range(self):
        files,aliases=self.hand_resources()
        location=next(row[0] for row in aliases if row[2]==('source_product_ranges.json',))
        files[location]=b'{"policy":"different range","products":{}}'
        with patch('robotagentbench.installation.sources._manifest',return_value=self.hand_profile()), \
             patch('robotagentbench.installation._ROOM_HAND_RESOURCES',aliases):
            with self.assertRaisesRegex(ValueError,'Original shelf-interference resource binding changed'):
                configure(self.manifest(files),self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_shelf_interference_alias_requires_original_pins(self):
        files, aliases = self.hand_resources()
        files[aliases[0][0]]=b'changed mesh'
        with patch('robotagentbench.installation.sources._manifest',return_value=self.hand_profile()), \
             patch('robotagentbench.installation._ROOM_HAND_RESOURCES',aliases):
            with self.assertRaisesRegex(ValueError,'Original shelf-interference resource binding changed'):
                configure(self.manifest(files),self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_shelf_interference_alias_rejects_conflicting_task_local_asset(self):
        from robotagentbench.installation import _ROOM_HAND_ROOT
        files, aliases = self.hand_resources()
        files['profiles/room/'+_ROOM_HAND_ROOT+aliases[0][2][0]]=b'other version'
        with patch('robotagentbench.installation.sources._manifest',return_value=self.hand_profile()), \
             patch('robotagentbench.installation._ROOM_HAND_RESOURCES',aliases):
            with self.assertRaisesRegex(ValueError,'Conflicting original shelf-interference resource'):
                configure(self.manifest(files),self.assets,self.base/'Runtime','room')
        self.assertFalse((self.base/'Runtime').exists())

    def test_shelf_helpers_preserve_original_sibling_imports_and_assets(self):
        helper='vendor/shelf_generation/tools/family3/collision_scene.py'
        (self.code/helper).parent.mkdir(parents=True)
        (self.code/helper).write_bytes(b'VALUE = 1\n')
        source=dict(profiles={'room':dict(files={helper:self.pin},compatibility_namespaces=[])})
        native='vendor/shelf_generation/work/family3_repair/collision/python_deps/fcl/native.so'
        prefix='libraries/original_absolute/mnt/agent-bench-data/lingjun/ziche/'
        manifest=self.manifest({'profiles/room/'+native:b'native test',prefix+'runtime/scene.usda':b'original scene'})
        with patch('robotagentbench.installation.sources._manifest',return_value=source):
            configure(manifest,self.assets,self.base/'Runtime','room')
        data=self.base/'Runtime';metadata=json.loads((data/'runtime.json').read_text())
        for relative,raw in ((helper,b'VALUE = 1\n'),(native,b'native test')):
            alias=prefix+relative.removeprefix('vendor/shelf_generation/')
            self.assertEqual((data/alias).read_bytes(),raw)
            self.assertEqual(metadata['library_files'][alias],hashlib.sha256(raw).hexdigest())
        self.assertEqual((data/prefix/'runtime/scene.usda').read_bytes(),b'original scene')
        self.assertEqual(metadata['parameter_overrides'],{})

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

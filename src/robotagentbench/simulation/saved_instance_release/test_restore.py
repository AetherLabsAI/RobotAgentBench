"""Small metadata-only contracts; no composer, USD, simulation or model."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .restore import describe,select,_write_saved_json_sidecars,_sha,_restore_scene

try:
    from pxr import Usd,UsdGeom,UsdPhysics
except ImportError:
    Usd=None


class CompactRestoreContract(unittest.TestCase):
    def fixture(self,root):
        runtime=root/'Runtime';runtime.mkdir();source=runtime/'source';source.mkdir()
        (runtime/'runtime.json').write_text(json.dumps(dict(schema='robotagentbench.local_installation.v1',profile='room',
            runtime=str(runtime),source_root=str(source),source_files={'synthetic.py':'a'*64})))
        bank=root/'bank';bank.mkdir()
        state=dict(schema='robotagentbench.compact_initial_state.v1',task_id='prepare_coffee',episode=1,actual_seed=7,
            kind='short',source_profile='room',scene_recipe={'layout':{},'plan':{}},
            task_parameters={'task_name':'prepare_coffee'},launch_options={'schema':'openend.short_task.launch.v1','task_name':'prepare_coffee'})
        raw=json.dumps(state).encode();(bank/'state.json').write_bytes(raw)
        entry=dict(task_id='prepare_coffee',episode=1,actual_seed=7,kind='short',source_profile='room',
            state=dict(path='state.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),attachments=[])
        release=bank/'release.json';release.write_text(json.dumps(dict(schema='robotagentbench.compact_initial_states.v1',instances=[entry])))
        return runtime,release,bank,state,entry

    def test_description_does_not_create_scene_or_output(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();runtime,release,bank,state,entry=self.fixture(root)
            output=root/'selected';result=describe(release,runtime,'prepare_coffee',1,output)
            self.assertEqual(result['selected']['state']['actual_seed'],7)
            self.assertEqual(result['mode'],'ordinary')
            self.assertFalse(output.exists());self.assertFalse(result['owner'].exists())
            self.assertFalse(list(runtime.rglob('*.usd*')))

    def test_changed_input_or_unsupported_owner_is_not_accepted(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();runtime,release,bank,state,entry=self.fixture(root)
            (bank/'state.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):select(release,'prepare_coffee',1)
            state['kind']='cloth';raw=json.dumps(state).encode();(bank/'state.json').write_bytes(raw)
            entry['state'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            release.write_text(json.dumps(dict(schema='robotagentbench.compact_initial_states.v1',instances=[entry])))
            result=describe(release,runtime,'prepare_coffee',1,root/'selected')
            self.assertEqual(result['mode'],'cloth')
            self.assertFalse((root/'selected').exists())

    def test_owner_path_cycle_keeps_real_new_hashes_and_original_code_pins(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();first=root/'first.json';second=root/'second.json'
            first.write_text(json.dumps({'receipt':'/original/second.json'}))
            second.write_text(json.dumps({'launch':'/original/first.json','launch_sha256':'a'*64,
                                         'code_sources':{'/original/owner.py':'c'*64}}))
            sidecars={'/original/first.json':{'source':str(first)},'/original/second.json':{'source':str(second)}}
            paths={'/original/first.json':str(root/'new_first.json'),'/original/second.json':str(root/'new_second.json')}
            hashes={};_write_saved_json_sidecars(sidecars,paths,hashes)
            actual=json.loads((root/'new_second.json').read_text())
            self.assertEqual(actual['launch'],str(root/'new_first.json'))
            self.assertEqual(actual['launch_sha256'],_sha(root/'new_first.json'))
            self.assertEqual(actual['code_sources'],{'/original/owner.py':'c'*64})


@unittest.skipIf(Usd is None,'Requires the native SDK USD modules; no simulator is started')
class RestoreStageMetadata(unittest.TestCase):
    def test_delta_layer_keeps_original_units_axes_time_and_physics(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);source=root/'source.usda';out=root/'restored.usda'
            original=Usd.Stage.CreateNew(str(source))
            original.SetDefaultPrim(UsdGeom.Xform.Define(original,'/World').GetPrim())
            for key,value in dict(upAxis='Z',metersPerUnit=1.,timeCodesPerSecond=60.,framesPerSecond=30.).items():
                original.SetMetadata(key,value)
            cube=UsdGeom.Cube.Define(original,'/World/Cube').GetPrim()
            UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(2.5)
            original.GetRootLayer().Save();before=source.read_bytes()
            checks=_restore_scene(source,{'objects':[]},{'objects':[]},out)
            restored=Usd.Stage.Open(str(out))
            self.assertEqual(UsdGeom.GetStageUpAxis(restored),'Z')
            self.assertEqual(UsdGeom.GetStageMetersPerUnit(restored),1.)
            self.assertEqual(restored.GetDefaultPrim().GetPath(),original.GetDefaultPrim().GetPath())
            for key,value in checks['stage_metadata'].items():
                self.assertEqual(restored.GetMetadata(key),value)
            self.assertEqual(restored.GetPrimAtPath('/World/Cube').GetAttribute('physics:mass').Get(),2.5)
            self.assertEqual(source.read_bytes(),before)


if __name__=='__main__':unittest.main()

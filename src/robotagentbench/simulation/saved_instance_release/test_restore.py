"""Small metadata-only contracts; no composer, USD, simulation or model."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from .restore import describe,select,_write_saved_json_sidecars,_sha,_restore_scene,_portable_state,_refresh_launch_hashes,_native_instruction_copy

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

    def test_long_recipe_keeps_seeded_logical_identity_while_other_paths_rebind(self):
        with TemporaryDirectory()as directory:
            runtime=Path(directory).resolve();source=runtime/'source';source.mkdir()
            logical='/mnt/original/layouts/ID_04.usdc'
            relative='libraries/original_absolute'+logical
            installed=runtime/relative;installed.parent.mkdir(parents=True);installed.write_bytes(b'original layout')
            metadata={'profile':'room','resources':{relative:{'sha256':_sha(installed)}}}
            recipe={'layout':{'id':4,'scene':logical},'plan':{'seed':1,'poses':[[.1,.2,.3]]}}
            state={'scene_recipe':recipe,'manifest_template':{'source_layout':logical},'actual_seed':1}
            original=json.loads(json.dumps(state))
            rebound=_portable_state(state,source,runtime,metadata,'long')
            self.assertEqual(rebound['scene_recipe'],original['scene_recipe'])
            self.assertEqual(rebound['manifest_template']['source_layout'],str(installed))
            self.assertEqual(rebound['actual_seed'],1)
            self.assertEqual(state,original)
            rebound['scene_recipe']['plan']['poses'][0][0]=9
            self.assertEqual(state,original)
            for mode in('ordinary','task','room','context'):
                with self.subTest(mode=mode):
                    other=_portable_state(state,source,runtime,metadata,mode)
                    self.assertEqual(other['scene_recipe']['layout']['scene'],str(installed))
                    self.assertEqual(other['scene_recipe']['plan'],recipe['plan'])

    def test_scene_hash_without_duplicated_scene_path_keeps_original_launch_shape(self):
        with TemporaryDirectory()as directory:
            scene=Path(directory)/'restored.usda';scene.write_bytes(b'original restored scene')
            request=Path(directory)/'request.json';request.write_bytes(b'original request')
            launch={'task_name':'stock_wooden_tray_lighting','scene_sha256':'a'*64,
                    'request':str(request),'request_sha256':'b'*64}
            keys=set(launch);_refresh_launch_hashes(launch,scene)
            self.assertEqual(set(launch),keys)
            self.assertEqual(launch['scene_sha256'],_sha(scene))
            self.assertEqual(launch['request_sha256'],_sha(request))
            self.assertNotIn('scene',launch)
            explicit={'scene':str(scene),'scene_sha256':'c'*64}
            _refresh_launch_hashes(explicit,Path(directory)/'unused.usda')
            self.assertEqual(explicit['scene_sha256'],_sha(scene))

    def public_instruction_fixture(self):
        task='stock_wooden_tray_lighting';instruction='Pick up the designated shelf products: Sprite.'
        state=dict(launch_options=dict(schema='openend.short_task.launch.v1',task_name=task),
                   task_parameters=dict(task_name=task,seed=7),
                   public_input=dict(instruction=instruction,context=''))
        registry=SimpleNamespace(CONTEXT_TASKS={'select_drink':'selection.owner'},FAILURE_RECOVERY_TASKS=frozenset({'recover_cup'}),
                                 ATOMIC_COFFEE_TASKS=frozenset({'coffee_moved'}),
                                 resolve=Mock(return_value=dict(task_name=task,instruction=instruction)))
        return state,registry

    def test_empty_native_context_copy_keeps_original_request_and_public_text(self):
        for mode in('ordinary','task'):
            with self.subTest(mode=mode):
                state,registry=self.public_instruction_fixture();original=json.loads(json.dumps(state))
                self.assertTrue(_native_instruction_copy(state,mode,registry))
                registry.resolve.assert_called_once_with(original['task_parameters'])
                self.assertIsNot(registry.resolve.call_args.args[0],state['task_parameters'])
                self.assertEqual(state,original)

    def test_native_copy_rejects_extra_context_and_unmatched_instruction(self):
        for context in('A different user need.',' ',None,{}):
            state,registry=self.public_instruction_fixture();state['public_input']['context']=context
            self.assertFalse(_native_instruction_copy(state,'task',registry))
            registry.resolve.assert_not_called()
        for instruction in('A new request.','Pick up the designated shelf products: Sprite. ',None,''):
            state,registry=self.public_instruction_fixture();state['public_input']['instruction']=instruction
            self.assertFalse(_native_instruction_copy(state,'task',registry))
        state,registry=self.public_instruction_fixture();state['public_input']['answer']='private target'
        self.assertFalse(_native_instruction_copy(state,'task',registry))
        registry.resolve.assert_not_called()

    def test_native_copy_cannot_bypass_other_owner_context_binding(self):
        for mode in('context','room','long','specialized','cloth'):
            state,registry=self.public_instruction_fixture()
            self.assertFalse(_native_instruction_copy(state,mode,registry))
            registry.resolve.assert_not_called()
        for task in('select_drink','recover_cup','coffee_moved'):
            state,registry=self.public_instruction_fixture()
            state['task_parameters']['task_name']=task;state['launch_options']['task_name']=task
            self.assertFalse(_native_instruction_copy(state,'task',registry))
            registry.resolve.assert_not_called()
        state,registry=self.public_instruction_fixture();state['launch_options']['schema']='openend.coding_room.launch.v1'
        self.assertFalse(_native_instruction_copy(state,'task',registry));registry.resolve.assert_not_called()
        state,registry=self.public_instruction_fixture();state['task_parameters']['task_name']='other_task'
        self.assertFalse(_native_instruction_copy(state,'task',registry));registry.resolve.assert_not_called()


@unittest.skipIf(Usd is None,'Requires the native SDK USD modules; no simulator is started')
class RestoreStageMetadata(unittest.TestCase):
    def background_fixture(self,root):
        prototype=root/'prototype.usda'
        source=Usd.Stage.CreateNew(str(prototype))
        UsdGeom.Xform.Define(source,'/World')
        prim=UsdGeom.Cube.Define(source,'/World/Notebook').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.12)
        source.GetRootLayer().Save()
        scene=root/'scene.usda';base=Usd.Stage.CreateNew(str(scene))
        UsdGeom.Xform.Define(base,'/World')
        UsdGeom.Xform.Define(base,'/World/RoomEpisode');base.GetRootLayer().Save()
        row=dict(id='notebook_green',path='/World/RoomEpisode/SofaBackground_notebook_green',
                 source='sofa',role='background',task_role='background',task_id=None,
                 required_by=[],scoring_target=False,prototype_scene=str(prototype),
                 prototype_path='/World/Notebook',transform_frame='world',
                 world_matrix_row_major_4x4=[1,0,0,0,0,1,0,0,0,0,1,0,.2,.3,.4,1])
        state=dict(task_id='hang_cap',objects=[row],manifest_template={
            'sofa_background':dict(version='sofa_background.v1',assets=['notebook_green'])})
        return scene,prototype,state

    def test_saved_background_reuses_pinned_prototype_and_exact_pose(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);scene,prototype,state=self.background_fixture(root)
            before=prototype.read_bytes();out=root/'restored.usda'
            checks=_restore_scene(scene,state,{'objects':[]},out,source_hashes={'prototype.usda':_sha(prototype)})
            restored=Usd.Stage.Open(str(out));row=state['objects'][0]
            prim=restored.GetPrimAtPath(row['path'])
            self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertAlmostEqual(prim.GetAttribute('physics:mass').Get(),.12)
            self.assertEqual(checks['restored_background_objects'],1)
            self.assertEqual(checks['maximum_world_matrix_error'],0)
            self.assertEqual(checks['background_source_hashes'],{str(prototype):_sha(prototype)})
            self.assertEqual(prototype.read_bytes(),before)

    def test_unpinned_background_and_target_substitution_are_rejected(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);scene,prototype,state=self.background_fixture(root)
            with self.assertRaisesRegex(ValueError,'not pinned'):
                _restore_scene(scene,state,{'objects':[]},root/'bad.usda',source_hashes={'prototype.usda':'a'*64})
            state['objects'][0]['scoring_target']=True
            with self.assertRaisesRegex(ValueError,'Missing original composed subtree'):
                _restore_scene(scene,state,{'objects':[]},root/'target.usda',source_hashes={'prototype.usda':_sha(prototype)})

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

"""Synthetic filesystem-only saved-bank contracts; never task preparation."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

from . import saved_instances as saved


class SavedInstances(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.sources=[]

    def tearDown(self):self.temp.cleanup()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def source(self,kind,path,public_inputs=()):
        row=dict(kind=kind,index=str(path),sha256=saved.digest(path),public_inputs=list(public_inputs))
        self.sources.append(row);return row

    def plan(self):return dict(schema=saved.SCHEMA,sources=self.sources,readonly_bindings=[])

    def short(self,coordinates=(1.,2.)):
        rows=[]
        for seed,x in enumerate(coordinates,1):
            launch=self.write(f'short/{seed}/launch.json',dict(task_name='prepare_coffee'))
            state=dict(layout_id=seed,objects=[dict(id='target',matrix=[x,0.,0.])],task_bindings=[],initial_base_xyyaw=[0.,0.,0.])
            rows.append(dict(task_name='prepare_coffee',seed=seed,launch=str(launch),launch_sha256=saved.digest(launch),
                status='prepared_cpu_not_native_accepted',output_hashes={str(launch):saved.digest(launch)},
                physical_configuration=state,physical_fingerprint=saved.fingerprint(state),
                semantic_fingerprint=saved.fingerprint({k:v for k,v in state.items()if k!='layout_id'})))
        path=self.write('short/index.json',dict(schema='robotagentbench.short_instance_bank.v1',instances=rows))
        return self.source('short',path),rows

    def composite(self):
        folder='composite';launch=self.write(folder+'/launch.json',dict(task_name='openend',initial_base_xyyaw=[0.,0.,0.]))
        manifest=self.write(folder+'/manifest.json',dict(layout_id=43,objects=[dict(id='cup',visible_bounds_world_m=[[1.,2.,3.],[2.,3.,4.]])]))
        public=self.write(folder+'/public_input.json',dict(prompt='Public current v4 request',context='Public household memory'))
        files={name:dict(path=str(path),sha256=saved.digest(path))for name,path in [('launch',launch),('manifest',manifest),('public_input.json',public)]}
        geometry=dict(layout_id=43,initial_base_xyyaw=[0.,0.,0.],objects=json.loads(manifest.read_text())['objects'])
        signature=saved.fingerprint(geometry)
        instance=self.write(folder+'/instance.json',dict(schema='robotagentbench.composite_instance.v31',task_name='host_guests',
            seed=1,launch=str(launch),actual_runtime_task='openend',status='actual_scene_cpu_admitted',files=files,geometry_sha256=signature))
        index=self.write(folder+'/index.json',dict(schema='robotagentbench.composite_instance_index.v31',rows=[dict(task_name='host_guests',seed=1,
            status='actual_scene_cpu_admitted',instance=str(instance),instance_hash=saved.digest(instance),geometry_sha256=signature)]))
        self.source('composite',index)

    def diagnostic(self):
        root=self.root/'diagnostic';launch=self.write('diagnostic/generated/launch.json',dict(task_name='select_drink'))
        public=self.write('diagnostic/case/model_input.json',dict(instruction='Choose a drink',context='Public preference'))
        variation=dict(rigid_body_world_transforms={'target':[[1.,0.,0.,1.]]},public_question={'instruction':'Choose a drink'})
        geometry=self.write('diagnostic/case/variation.private.json',variation)
        private=self.write('diagnostic/case/resolved.private.json',dict(private_answer='NEVER PASS TO MODEL'))
        pins={str(p.relative_to(root)):saved.digest(p)for p in (launch,public,geometry,private)}
        signature=__import__('hashlib').sha256(json.dumps(variation,sort_keys=True).encode()).hexdigest()
        record=self.write('diagnostic/case/record.json',dict(schema='robotagentbench.diagnostic_instance.v1',task_name='select_drink',seed=31,
            scenario=None,status='prepared_cpu_not_native',files_sha256=pins,launch='generated/launch.json',model_input='case/model_input.json',variation_sha256=signature))
        index=self.write('diagnostic/index.json',dict(schema='robotagentbench.diagnostic_instance_bank.v1',entries=[dict(task_name='select_drink',seed=31,
            actual_seed=31,episode_ordinal=1,selected_for_bank=True,scenario=None,status='prepared_cpu_not_native',record='case/record.json',
            record_sha256=saved.digest(record),hash=saved.digest(launch),variation_sha256=signature)]))
        self.source('diagnostic',index)

    def cloth(self):
        root=self.root/'cloth';root.mkdir()
        initial=root/'initial.npz';header=repr(dict(descr='<f8',fortran_order=False,shape=(1,3))).encode()+b'\n'
        npy=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<ddd',1.,2.,3.)
        with zipfile.ZipFile(initial,'w')as archive:archive.writestr('initial_cloth_q.npy',npy)
        record=self.write('cloth/variation.json',dict(task_name='collect_tablecloth',seed=1,initial_sha256=saved.digest(initial),geometry=dict(accepted=True)))
        launch=self.write('cloth/launch.json',dict(task_name='collect_tablecloth',variation=dict(record=str(record),record_sha256=saved.digest(record),
            initial=str(initial),initial_sha256=saved.digest(initial))))
        index=self.write('cloth/index.json',dict(schema='robotagentbench.cloth_instance_bank.v1',task_name='collect_tablecloth',instances=[dict(
            seed=1,status='cpu_geometry_prepared',launch=str(launch),launch_sha256=saved.digest(launch),initial_sha256=saved.digest(initial))]))
        self.source('cloth',index)

    def test_all_four_schemas_and_public_v4_binding(self):
        self.short();self.composite();self.diagnostic();self.cloth()
        self.assertEqual(len(saved.list_tasks(self.plan())),4)
        result=saved.verify(self.plan());self.assertEqual(sum(r['verified']for r in result['tasks']),5)
        selected=saved.select(self.plan(),'host_guests',1)
        self.assertEqual(selected['model_input'],dict(instruction='Public current v4 request',context='Public household memory'))
        self.assertEqual(selected['model_input_binding']['launch_sha256'],selected['operator']['launch_sha256'])
        choice=saved.select(self.plan(),'select_drink',1)
        self.assertEqual(choice['actual_seed'],31);self.assertNotIn('private_answer',json.dumps(choice))
        self.assertEqual(set(choice['model_input']),{'instruction','context'})
        short=saved.select(self.plan(),'prepare_coffee',2)
        self.assertIsNone(short['model_input']);self.assertEqual(short['model_input_binding']['status'],'pending_worker_public_instruction')
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules);self.assertNotIn('isaacsim',sys.modules)

    def test_tamper_and_nonphysical_seed_only_duplicates_reject(self):
        source,rows=self.short((1.,1.))
        with self.assertRaisesRegex(ValueError,'Duplicate actual'):saved.verify(self.plan())
        Path(rows[0]['launch']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):saved.select(self.plan(),'prepare_coffee',1)

    def test_index_seal_and_episode_bounds(self):
        source,_=self.short();Path(source['index']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):saved.list_tasks(self.plan())
        for episode in (0,31,True):
            with self.assertRaises(ValueError):saved.select(self.plan(),'prepare_coffee',episode)

    def test_explicit_readonly_locator_is_not_asset_relocation(self):
        self.composite();plan=self.plan()
        logical='/downloaded/catalog.json';actual=self.sources[0]['index'];plan['sources'][0]['index']=logical
        plan['readonly_bindings']=[dict(logical_root=logical,backing_root=actual)]
        selection=saved.select(plan,'host_guests',1)
        self.assertFalse(selection['operator']['dependency_relocation_verified'])
        manifest=saved.release_plan(plan)
        self.assertEqual(manifest['schema'],saved.RELEASE);self.assertFalse(manifest['relocation_complete'])
        self.assertTrue(all(row['release_relative_path']is None for row in manifest['immutable_files']))


if __name__=='__main__':unittest.main()

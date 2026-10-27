"""CPU-only synthetic filesystem fixtures; no model/SDK/native execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from robotagentbench.interfaces.action_contract import ORDER
from robotagentbench.interfaces import robot_profiles as rp
from .configure import build,save
from .config import sha
from .operator import configuration


class ConfigurationSeam(unittest.TestCase):
    def fixture(self,folder):
        root=folder/'Runtime'/'source';root.mkdir(parents=True)
        def write(path,value):
            path.write_text(json.dumps(value));return path
        write(root.parent/'runtime.json',dict(schema='robotagentbench.local_installation.v1',
            profile='room',runtime='/installed/Runtime',source_root='/installed/Runtime/source',source_files={}))
        launch=write(root/'launch.json',dict(schema='openend.short_task.launch.v1',task_name='synthetic'))
        sdk=folder/'sdk';cache=sdk/'isaacsim/extscache'
        for name in ('omni.usd.libs-fixture','omni.usd.schema.physx-fixture'):(cache/name).mkdir(parents=True)
        model=folder/'model';model.mkdir();vlm=folder/'vlm';vlm.mkdir();locks=folder/'locks';locks.mkdir()
        write(model/'config.json',dict(type='internvla_a1_5',chunk_size=50,max_state_dim=32,max_action_dim=32))
        (model/'model.safetensors').write_bytes(b'SYNTHETIC FIXTURE, NOT A CHECKPOINT')
        stats=write(model/'stats.json',{'fixture':{key:dict(mean=[0.]*n,std=[1.]*n)
            for key,n in [('observation.state',23),('action',22)]}})
        units=[rp.NATIVE_UNITS[n]for n in ORDER[:20]]+['m/s','rad/s']
        schema=write(model/'control_schema.json',dict(version=rp.MOBILE22,action_dim=22,state_dim=23,control_hz=20,
            action_channels=[dict(name=n,unit=u,representation=r)for n,u,r in zip(ORDER,units,
                ['absolute_joint_position_target']*20+['body_forward_velocity','world_z_yaw_rate'])],
            state_channels=[dict(name=n,unit=u,representation=r)for n,u,r in zip(
                ORDER[:20]+['base_x_world_m','base_y_world_m','base_yaw_world_unwrapped_rad'],units[:20]+['m','m','rad'],
                ['measured_joint_position']*20+['measured_world_position','measured_world_position','measured_world_unwrapped_yaw'])]))
        return dict(runtime=root.parent,sdk_site=sdk,launch=launch,native_python=sys.executable,
            model_python=sys.executable,checkpoint=model,stats=stats,stats_key='fixture',control_schema=schema,
            vlm=vlm,lock_root=locks,worker_gpu=0,model_gpu=1,task='short01',output=root/'run001')

    def test_actual_configuration_helpers_and_explicit_argv(self):
        with tempfile.TemporaryDirectory()as tmp:
            args=self.fixture(Path(tmp).resolve());value,receipt=build(**args)
            row=value['tasks']['short01'];argv=row['worker_argv']
            self.assertEqual(argv[:5],[sys.executable,'-B','-u','-m','robotagentbench.model_examples.native_worker'])
            self.assertEqual(argv[argv.index('--output')+1],'{output}/native')
            self.assertEqual(argv[-3:],['--gpu','{worker_gpu}','--execute'])
            self.assertEqual(value['output_root'],str(args['output'].parent))
            self.assertEqual(row['native_report'],'{output}/native/native_report.json')
            self.assertIsNone(row['public_prompt'])
            self.assertEqual(receipt['model_artifacts_sha256'][str(args['checkpoint']/'model.safetensors')],
                sha(args['checkpoint']/'model.safetensors'))
            self.assertFalse(receipt['native_started']);self.assertFalse(receipt['model_loaded'])
            self.assertIsNotNone(importlib.util.find_spec('robotagentbench.model_examples.native_worker'))
            self.assertNotIn('torch',sys.modules);self.assertNotIn('isaacsim',sys.modules)

    def test_save_reload_and_outside_workspace_reject(self):
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp).resolve();args=self.fixture(folder)
            result=save(folder/'run.json',**args)
            value,row,pins=configuration(result['config'],'short01')
            self.assertEqual(pins,result['model_artifacts_sha256'])
            self.assertEqual(value['output_root'],str(args['output'].parent))
            with self.assertRaises(ValueError):save(folder/'run.json',**args)
            args['output']=folder/'outside'
            with self.assertRaises(ValueError):build(**args)

    def test_saved_public_input_is_pinned_and_forwarded(self):
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp).resolve();args=self.fixture(folder)
            inputs=folder/'saved';inputs.mkdir()
            launch=inputs/'launch.json'
            launch.write_text(json.dumps(dict(schema='openend.coding_room.launch.v1',task_name='openend')))
            public=inputs/'public.json';public.write_text(json.dumps(dict(prompt='Host guests',context='Public memory')))
            binding=inputs/'binding.json';binding.write_text(json.dumps(dict(
                schema='robotagentbench.public_input_binding.v1',source=str(public),
                sha256=sha(public),launch_sha256=sha(launch))))
            args.update(launch=launch,input_roots=[inputs],public_input_binding=binding)
            value,receipt=build(**args);argv=value['tasks']['short01']['worker_argv']
            self.assertEqual(argv[argv.index('--input-root')+1],str(inputs))
            self.assertEqual(argv[argv.index('--public-input-binding')+1],str(binding))
            self.assertEqual(receipt['public_input_binding'],str(binding))
            public.write_text(json.dumps(dict(prompt='Changed',context='Public memory')))
            with self.assertRaises(ValueError):build(**args)


if __name__=='__main__':unittest.main()

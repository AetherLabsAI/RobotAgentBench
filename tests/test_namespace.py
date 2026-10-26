from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.namespace import command


class ExampleNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve()
        self.source=self.base/'source';self.source.mkdir()
        (self.base/'libraries/frozen_assets').mkdir(parents=True)
        (self.base/'locks').mkdir()
        p=patch('robotagentbench.namespace.runtime',return_value=(self.base,self.source,{}))
        p.start();self.addCleanup(p.stop)

    def test_plan_is_cpu_only_and_preserves_interactive_stdio(self):
        cmd=command(self.base,'team/image:v1','plan',env={})
        self.assertIn('--interactive',cmd)
        self.assertIn('runc',cmd)
        self.assertNotIn('--gpus',cmd)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',cmd)
        self.assertIn('--plan',cmd)
        self.assertNotIn('--privileged',cmd)
        self.assertNotIn('--tty',cmd)

    def test_native_requires_explicit_license_gpu_and_lock(self):
        with self.assertRaises(ValueError):
            command(self.base,'image','run',name='one',gpu=0,locks=self.base/'locks',env={})
        with self.assertRaises(ValueError):
            command(self.base,'image','run',name='one',env={'OMNI_KIT_ACCEPT_EULA':'YES'})
        cmd=command(self.base,'image','run',name='one',gpu=0,locks=self.base/'locks',env={'OMNI_KIT_ACCEPT_EULA':'YES'})
        self.assertIn('--gpus',cmd);self.assertIn('--gpu',cmd)
        self.assertIn('OMNI_KIT_ACCEPT_EULA=YES',cmd)
        self.assertIn('--lock-root',cmd)

    def test_image_and_episode_input_rejection(self):
        for image in('--privileged','image; command','image name'):
            with self.assertRaises(ValueError):command(self.base,image,'plan')
        for name in('../bad','a/b','name with spaces'):
            with self.assertRaises(ValueError):command(self.base,'image','prepare',name=name)

    def test_host_runtime_is_the_only_writable_data_namespace(self):
        cmd=command(self.base,'image','prepare',name='new')
        mounts=[cmd[i+1]for i,v in enumerate(cmd)if v=='--mount']
        for row in mounts:
            actual=row.split('src=',1)[1].split(',dst=',1)[0]
            self.assertTrue(Path(actual).is_relative_to(self.base))
        self.assertTrue(any('frozen_assets' in row and row.endswith(',readonly')for row in mounts))

    def test_original_absolute_assets_remain_container_only_and_readonly(self):
        data=self.base/'libraries/original_absolute/mnt/agent-bench-data/lingjun'
        data.mkdir(parents=True)
        cmd=command(self.base,'image','prepare',name='new')
        mounts=[cmd[i+1]for i,v in enumerate(cmd)if v=='--mount']
        self.assertIn('type=bind,src='+str(data)+',dst=/mnt/agent-bench-data/lingjun,readonly',mounts)

    def test_room_background_layer_family_keeps_relative_sublayers_together(self):
        (self.source/'assets').mkdir()
        cmd=command(self.base,'image','prepare',name='new')
        mounts=[cmd[i+1]for i,v in enumerate(cmd)if v=='--mount']
        self.assertIn('type=bind,src='+str(self.source/'assets')+',dst=/home/lingjun/robot_agent_arena/api-protocol/assets,readonly',mounts)


if __name__=='__main__':unittest.main()

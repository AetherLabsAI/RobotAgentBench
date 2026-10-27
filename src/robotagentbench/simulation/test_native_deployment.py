"""Small synthetic filesystem/argv checks; no SDK, Docker or device execution."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench import namespace
from robotagentbench.model_examples import native_cache
from robotagentbench.simulation import native_deployment as deployment


class NativeDeploymentTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.site=self.root/'sdk'
        self.kit=self.site/'isaacsim/kit';self.kit.mkdir(parents=True)
        (self.site/'isaacsim/extscache/omni.mdl-fixture/search_paths').mkdir(parents=True)
        for relative in native_cache.CORE_FILES:
            (self.kit/relative).write_bytes(('synthetic original '+relative).encode())
        self.runtime=self.root/'Runtime';self.runtime.mkdir()
        self.source=self.runtime/'source';self.source.mkdir()
        self.locks=self.root/'locks';self.locks.mkdir()
        self.image='sha256:'+'b'*64
        for p in (patch.object(deployment,'SDK_SITE',str(self.site)),
                  patch.object(deployment,'SDK_ALIAS',str(self.site)),
                  patch.object(namespace,'runtime',return_value=(self.runtime,self.source,{}))):
            p.start();self.addCleanup(p.stop)
        self.baseline=self.root/'image_baseline.json'
        deployment.create_baseline(self.site,self.baseline)

    def prepare(self):
        def copy(argv,**kwargs):
            self.assertEqual(argv[:3],['cp','-a','--reflink=auto'])
            shutil.copytree(argv[3],argv[4],symlinks=True)
        with patch.object(native_cache.subprocess,'run',side_effect=copy):
            record=deployment.prepare(self.runtime,'one',sdk_site=str(self.site),image=self.image,baseline=self.baseline)
        # prepare executes with Runtime mounted at /job. Reproduce the same
        # logical metadata after the temporary test tree moves back to host.
        out=self.runtime/'native_deployments/one'
        logical=Path('/job/native_deployments/one')
        def relocate(value):
            if isinstance(value,str):return value.replace(str(out),str(logical))
            if isinstance(value,list):return [relocate(x)for x in value]
            if isinstance(value,dict):return {k:relocate(v)for k,v in value.items()}
            return value
        for relative in ('cache/provenance.json','cache/binding.json','graphics/binding.json'):
            path=out/relative;path.write_text(json.dumps(relocate(json.loads(path.read_text()))))
        binding=json.loads((out/'cache/binding.json').read_text())
        binding['provenance']['sha256']=deployment.sha(out/'cache/provenance.json')
        (out/'cache/binding.json').write_text(json.dumps(binding))
        record['files_sha256']={relative:deployment.sha(out/relative)for relative in deployment.FILES}
        (out/'deployment.json').write_text(json.dumps(record))
        return out,record

    def test_image_original_baseline_is_independent_of_writable_copy(self):
        original=deployment.verify_image_baseline(sdk_site=str(self.site),baseline=self.baseline,check_visible_sdk=True)
        self.assertEqual(set(original['core_sha256']),set(native_cache.CORE_FILES))
        (self.kit/'libcarb.so').write_bytes(b'changed code')
        with self.assertRaises(ValueError):
            deployment.verify_image_baseline(sdk_site=str(self.site),baseline=self.baseline,check_visible_sdk=True)
        with self.assertRaises(ValueError):deployment.create_baseline(self.site,self.baseline)

    def test_prepared_cache_validates_without_host_sdk_and_rejects_reuse(self):
        out,record=self.prepare()
        shutil.rmtree(self.site)
        checked=deployment.validate(self.runtime,'one',image=self.image,sdk_site=str(self.site))
        self.assertEqual(checked['record'],record);self.assertEqual(len(checked['mounts']),2)
        self.assertFalse(list((out/'cache/mdl_search').iterdir()))
        with self.assertRaises(ValueError):
            namespace.command(self.runtime,self.image,'prepare-native',name='one')
        with self.assertRaises(ValueError):
            deployment.validate(self.runtime,'one',image='sha256:'+'c'*64,sdk_site=str(self.site))
        (out/'cache/kit/libcarb.so').write_bytes(b'changed copied code')
        with self.assertRaises(ValueError):
            deployment.validate(self.runtime,'one',image=self.image,sdk_site=str(self.site))

    def test_cpu_preparation_and_native_mounts_have_separate_entries(self):
        cpu=namespace.command(self.runtime,self.image,'prepare-native',name='one')
        self.assertIn('runc',cpu);self.assertNotIn('--gpus',cpu)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',cpu)
        self.assertIn('robotagentbench.simulation.native_deployment',cpu)
        out,_=self.prepare()
        native=namespace.command(self.runtime,self.image,'run',name='one',gpu=0,
            locks=self.locks,sdk_site=str(self.site),native_deployment=True,env={'OMNI_KIT_ACCEPT_EULA':'YES'})
        mounts=[native[i+1]for i,x in enumerate(native[:-1])if x=='--mount']
        self.assertIn('nvidia',native);self.assertIn('--deployment-sha256',native)
        for source,target in deployment.validate(self.runtime,'one',image=self.image,sdk_site=str(self.site))['mounts']:
            self.assertIn('type=bind,src='+str(source)+',dst='+target,mounts)
        self.assertIn('type=bind,src='+str(out/'graphics/nvidia_original_icd.json')+',dst=/task_graphics/nvidia_original_icd.json,readonly',mounts)
        # Model namespace uses command() without opting into the generic
        # entry, and remains responsible for its own cache/ICD configuration.
        plain=namespace.command(self.runtime,self.image,'run',name='other',gpu=0,
            locks=self.locks,sdk_site=str(self.site),env={'OMNI_KIT_ACCEPT_EULA':'YES'})
        self.assertIn('robotagentbench.example',plain)
        self.assertNotIn('robotagentbench.simulation.native_deployment',plain)


if __name__=='__main__':unittest.main()

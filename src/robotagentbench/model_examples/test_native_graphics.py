"""Pure CPU file/argv seams, not Vulkan, GPU or Isaac acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import native_graphics as graphics
from .codex.namespace import command


class NativeGraphics(unittest.TestCase):
    def test_original_bytes_fresh_prepare_and_source(self):
        self.assertEqual(len(graphics.ICD_BYTES),142)
        self.assertEqual(hashlib.sha256(graphics.ICD_BYTES).hexdigest(),graphics.SHA256)
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp).resolve()/'owned';binding=graphics.prepare(out)
            self.assertEqual(graphics.validate(binding).read_bytes(),graphics.ICD_BYTES)
            self.assertEqual(json.loads((out/'source.json').read_text())['source'],graphics.SOURCE)
            with self.assertRaises(ValueError):graphics.prepare(out)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules);self.assertNotIn('isaacsim',sys.modules)

    def test_reject_replacement_or_canonical_runtime_managed_target(self):
        with tempfile.TemporaryDirectory()as tmp:
            binding=graphics.prepare(Path(tmp).resolve()/'owned')
            for key,value in [('target','/etc/vulkan/icd.d/nvidia_icd.json'),('runtime','runc'),('sha256','0'*64),('environment',{})]:
                bad=deepcopy(binding);bad[key]=value
                with self.assertRaises(ValueError):graphics.validate(bad)
            path=Path(binding['source']);path.chmod(0o600);path.write_bytes(b'new ICD')
            with self.assertRaises(ValueError):graphics.validate(binding)

    def test_namespace_adds_only_fixed_driver_binding(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();dirs={name:root/name for name in ('runtime','source','sdk','package','locks')}
            for p in dirs.values():p.mkdir()
            binding=graphics.prepare(root/'graphics');image='sha256:'+'a'*64
            value=dict(schema='robotagentbench.codex_namespace.v1',runtime=str(dirs['runtime']),image=image,name='test',gpu=0,
                lock_root=str(dirs['locks']),sdk_site=str(dirs['sdk']),package_root=str(dirs['package']),entry_argv=['python','trusted_worker'],
                readonly_roots=[],readonly_aliases={},docker_cli=None,docker_host=None,intent=str(root/'intent.json'),graphics_binding=binding)
            base=['docker','run','--rm','--interactive','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics',image,'ignored']
            with patch('robotagentbench.example.runtime',return_value=(dirs['runtime'],dirs['source'],{})),patch('robotagentbench.namespace.command',return_value=base):
                result=command(value)
            self.assertEqual(result[result.index('--runtime')+1],'nvidia')
            self.assertIn('VK_DRIVER_FILES='+graphics.TARGET,result)
            self.assertIn('type=bind,src='+binding['source']+',dst='+graphics.TARGET+',readonly',result)
            self.assertIn('NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics',result)
            self.assertEqual(result[-3:],[image,'python','trusted_worker'])
            self.assertFalse(Path(value['intent']).exists())


if __name__=='__main__':unittest.main()

"""Synthetic filesystem/copy/argv seams, not an SDK or simulator test."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import native_cache as cache
from .codex.namespace import command


class NativeCache(unittest.TestCase):
    def fixture(self,root):
        site=root/'sdk';kit=site/'isaacsim/kit';kit.mkdir(parents=True)
        (site/'isaacsim/extscache/omni.mdl-fixture/search_paths').mkdir(parents=True)
        for name in cache.CORE_FILES:(kit/name).write_bytes(('SYNTHETIC '+name).encode())
        def copy(argv,**kwargs):
            self.assertEqual(argv[:3],['cp','-a','--reflink=auto'])
            self.assertEqual(kwargs,dict(check=True,timeout=90))
            shutil.copytree(argv[3],argv[4],symlinks=True)
        with patch('robotagentbench.model_examples.native_cache.subprocess.run',side_effect=copy):
            binding=cache.prepare(site,root/'owned')
        return site,binding

    def test_fresh_copy_provenance_and_exact_two_targets(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();site,binding=self.fixture(root)
            pairs=cache.validate(binding,sdk_site=site)
            self.assertEqual(len(pairs),2);self.assertFalse(list((root/'owned/mdl_search').iterdir()))
            self.assertEqual((root/'owned/kit/libcarb.so').read_bytes(),(site/'isaacsim/kit/libcarb.so').read_bytes())
            with self.assertRaises(ValueError):cache.prepare(site,root/'owned')

    def test_arbitrary_write_alias_or_changed_core_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();site,binding=self.fixture(root)
            bad=deepcopy(binding);bad['kit']['target']='/unrelated'
            with self.assertRaises(ValueError):cache.validate(bad,sdk_site=site)
            (root/'owned/kit/libcarb.so').write_bytes(b'changed')
            with self.assertRaises(ValueError):cache.validate(binding,sdk_site=site)

    def test_namespace_retains_sdk_ro_with_only_fixed_nested_rw(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();site,binding=self.fixture(root)
            for name in ('runtime','source','package','locks'):(root/name).mkdir()
            image='sha256:'+'b'*64
            value=dict(schema='robotagentbench.codex_namespace.v1',runtime=str(root/'runtime'),image=image,name='test',gpu=0,
                lock_root=str(root/'locks'),sdk_site=str(site),package_root=str(root/'package'),entry_argv=['python','trusted'],
                readonly_roots=[],readonly_aliases={},docker_cli=None,docker_host=None,intent=str(root/'intent.json'),native_cache_binding=binding)
            with patch('robotagentbench.example.runtime',return_value=(root/'runtime',root/'source',{})),patch('robotagentbench.namespace.command',return_value=['docker','run',image,'ignored']):
                argv=command(value)
            mounts=[argv[i+1]for i,x in enumerate(argv[:-1])if x=='--mount']
            self.assertIn('type=bind,src='+str(site)+',dst='+str(site)+',readonly',mounts)
            for source,target in cache.validate(binding,sdk_site=site):
                self.assertIn('type=bind,src='+str(source)+',dst='+str(target),mounts)
            self.assertFalse((root/'intent.json').exists())


if __name__=='__main__':unittest.main()

"""Path-only restoration keeps generated stain bytes and scoring data intact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from robotagentbench.simulation.saved_instance_release.restore import _rebind_generated_stain


class StainRebindingTests(unittest.TestCase):
    def fixture(self, root):
        files={}
        for name in ('asset.usda','asset_layout.json','textures/stain.png'):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        record=dict(version='openend.wipe_stain.randomization.v1',room_seed=2,shape_seed=939204053,
                    appearance={'family':'coffee_brown'},source_hashes={'/source/generator.py':'a'*64},files=files)
        receipt=root/'randomization.json';receipt.write_text(json.dumps(record))
        group=dict(layout_file=str(root/'asset_layout.json'),layout_sha256=files[str(root/'asset_layout.json')],
                   maximum_score=6,table_transform=[[1,0,0,0]],randomization=record,
                   randomization_file=str(receipt),randomization_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        saved=deepcopy(group);prefix='/old/episode'
        saved['layout_file']=prefix+'/asset_layout.json';saved['randomization_file']=prefix+'/randomization.json'
        saved['randomization']['files']={prefix+'/'+Path(p).relative_to(root).as_posix():h for p,h in files.items()}
        saved['randomization_sha256']='b'*64
        return dict(manifest_template={'wipe_table':saved},objects=[{'entry':prefix+'/asset.usda'}]),{'wipe_table':group}

    def test_relocates_same_content_without_mutating_input(self):
        with tempfile.TemporaryDirectory()as folder:
            state,composed=self.fixture(Path(folder));original=deepcopy(state)
            actual=_rebind_generated_stain(state,composed)
            self.assertEqual(state,original)
            self.assertEqual(actual['manifest_template']['wipe_table'],composed['wipe_table'])
            self.assertEqual(actual['objects'][0]['entry'],str(Path(folder)/'asset.usda'))

    def test_different_shape_score_or_file_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            state,composed=self.fixture(Path(folder))
            changed=deepcopy(composed);changed['wipe_table']['maximum_score']=7
            with self.assertRaises(ValueError):_rebind_generated_stain(state,changed)
            changed=deepcopy(composed);changed['wipe_table']['randomization']['shape_seed']=1
            with self.assertRaises(ValueError):_rebind_generated_stain(state,changed)
            (Path(folder)/'asset_layout.json').write_text('different scoring cells')
            with self.assertRaises(ValueError):_rebind_generated_stain(state,composed)

    def test_other_states_unchanged(self):
        for state in ({},{'manifest_template':{}},{'manifest_template':{'wipe_table':{'maximum_score':6}}}):
            self.assertIs(_rebind_generated_stain(state,{}),state)

if __name__=='__main__':unittest.main()

"""Saved-public-input admission checks only; no SDK, GPU, or model execution."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from admission_fixtures import AdmissionFixture
from robotagentbench.simulation.admission import admit_prepared_launch
from robotagentbench.simulation.public_input import SCHEMA, from_selection, load


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedPublicInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.f=AdmissionFixture(self.temp.name)
        self.f.configure(task='openend',schema='openend.coding_room.launch.v1')
        self.f.with_room_backend()
        self.public=self.f.inputs/'public_input.json'
        self.public.write_text(json.dumps(dict(prompt='Current v4 scenario request',context='Current v4 household memory')))
        self.binding=self.f.inputs/'public_binding.json'
        self.seal()

    def seal(self):
        self.binding.write_text(json.dumps(dict(schema=SCHEMA,source=str(self.public),
            sha256=sha(self.public),launch_sha256=sha(self.f.launch))))

    def admit(self):
        f=self.f
        return admit_prepared_launch(f.launch,f.output,backend=f.backend,workspace=f.workspace,
            public_input_binding=self.binding)

    def test_current_text_reaches_room_public_boundary_without_legacy_fallback(self):
        f=self.f;f.backend=replace(f.backend,public_context=lambda *args:self.fail('No legacy public-input fallback'))
        before={path:path.read_bytes()for path in(f.launch,f.request,f.manifest,f.ready,f.scene)}
        admitted=self.admit()
        self.assertEqual(admitted.plan['instruction'],'Current v4 scenario request')
        self.assertEqual(admitted.context,'Current v4 household memory')
        self.assertEqual(admitted.plan['score_policy'],f.plan['score_policy'])
        self.assertEqual(admitted.request,{'compiled_components':[]})
        self.assertEqual(f.plan['instruction'],'Test request')
        self.assertEqual(f.events,['room_admit'])
        for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
        self.assertFalse(f.output.exists())

    def test_binding_public_and_launch_remain_pinned_until_startup(self):
        for path in(self.public,self.binding,self.f.launch):
            with self.subTest(path=path.name):
                admitted=self.admit();raw=path.read_bytes();path.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):admitted.verify_inputs()
                path.write_bytes(raw)

    def test_wrong_launch_or_public_sha_fails_before_original_owner(self):
        for key in('launch_sha256','sha256'):
            with self.subTest(key=key):
                self.seal();value=json.loads(self.binding.read_text());value[key]='0'*64
                self.binding.write_text(json.dumps(value));self.f.events.clear()
                with self.assertRaises(ValueError):self.admit()
                self.assertEqual(self.f.events,[])

    def test_gt_extra_keys_duplicate_keys_nontext_and_outside_paths_are_rejected(self):
        invalid=(dict(instruction='Choose',context='',answer='secret'),
                 dict(instruction='Choose',context={'targets':['secret']}),
                 dict(tasks=['secret'],context=''),dict(instruction='',context=''))
        for value in invalid:
            self.public.write_text(json.dumps(value));self.seal()
            with self.assertRaises(ValueError):self.admit()
        self.public.write_text('{"prompt":"one","prompt":"two","context":""}');self.seal()
        with self.assertRaises(ValueError):self.admit()
        outside=self.f.root/'outside_public.json';outside.write_text('{"prompt":"x","context":""}')
        value=json.loads(self.binding.read_text());value.update(source=str(outside),sha256=sha(outside))
        self.binding.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.admit()

    def test_old_default_is_unchanged_and_standalone_override_is_not_silently_ignored(self):
        f=self.f;f.backend=replace(f.backend,public_context=lambda *args:'Original memory')
        admitted=admit_prepared_launch(f.launch,f.output,backend=f.backend,workspace=f.workspace)
        self.assertEqual(admitted.plan['instruction'],'Test request');self.assertEqual(admitted.context,'Original memory')
        f.configure();self.seal()
        with self.assertRaisesRegex(ValueError,'room owner'):self.admit()

    def test_instruction_form_works_without_supplementing_private_data(self):
        self.public.write_text('{"instruction":"Explicit public request","context":""}');self.seal()
        value,pins=load(self.binding,self.f.launch,workspace=self.f.workspace)
        self.assertEqual(value,dict(instruction='Explicit public request',context=''))
        self.assertEqual(set(path for path,_ in pins),{self.binding,self.public,self.f.launch})

    def test_unified_selection_binding_does_not_infer_missing_public_text(self):
        selection=dict(schema='robotagentbench.saved_instances.selection.v1',
            operator=dict(launch_sha256=sha(self.f.launch)),
            model_input=dict(instruction='Current public request',context='Current public memory'),
            model_input_binding=dict(status='saved_public_input',source=str(self.public),
                sha256=sha(self.public),launch_sha256=sha(self.f.launch)))
        self.assertEqual(from_selection(selection),json.loads(self.binding.read_text()))
        selection['model_input']=None
        with self.assertRaisesRegex(ValueError,'separately saved'):from_selection(selection)


if __name__=='__main__':unittest.main()

"""Admission/routing fixtures only; do not claim native FEM validation."""
from dataclasses import replace
import hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from admission_fixtures import AdmissionFixture
from source_fixtures import SourceFixture
from robotagentbench.simulation.admission import AdmittedLaunch
from robotagentbench.simulation.cloth_binding import NativeClothBinding
from robotagentbench.simulation.original_backend import run_original_worker
from robotagentbench.simulation.worker import NativeRunOptions
from robotagentbench.interfaces.errors import TaskError


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.f=AdmissionFixture(self.temp.name);self.tree=SourceFixture(self.temp.name).tree()
    def admitted(self,name):
        f=self.f;paths=(f.launch,f.request,f.manifest,f.scene)
        return AdmittedLaunch(f.launch,f.output,json.loads(f.launch.read_text()),json.loads(f.request.read_text()),
            json.loads(f.manifest.read_text()),dict(task_name=name,execution_mode='native_cloth_scored.v1',available=True,scoring_available=True),
            'native_cloth',tuple((p,hashlib.sha256(p.read_bytes()).hexdigest())for p in paths))
    def call(self,binding=None):
        return run_original_worker(self.f.launch,self.f.output,options=NativeRunOptions(0),sources=self.tree,
            workspace=self.f.workspace,native_cloth_owner=binding)
    def test_all_three_scored_tasks_need_their_native_owner_before_generic_import(self):
        for name in('spread_tablecloth','collect_tablecloth','long_task_18'):
            schema='openend.coding_room.launch.v1'if name=='long_task_18'else'openend.short_task.launch.v1'
            self.f.configure(task=name,schema=schema)
            with patch('robotagentbench.simulation.original_backend.bind_original_backend',side_effect=AssertionError('No CPU fallback')):
                with self.assertRaises(TaskError)as error:self.call()
            self.assertEqual(error.exception.reason,'native_cloth_owner_required');self.assertFalse(self.f.output.exists())
    def test_bound_cloth_owner_gets_exact_admitted_identity_without_source_imports(self):
        self.f.configure(task='collect_tablecloth');admitted=self.admitted('collect_tablecloth');calls=[]
        def execute(value,**kwargs):
            self.assertIs(value,admitted);self.assertFalse(self.tree.active);calls.append('execute');raise SystemExit(0)
        binding=NativeClothBinding(lambda *args,**kwargs:admitted,execute)
        with patch('robotagentbench.simulation.original_backend.bind_original_backend',side_effect=AssertionError('No generic imports')):
            with self.assertRaises(SystemExit):self.call(binding)
        self.assertEqual(calls,['execute']);self.assertFalse(self.f.output.exists())
    def test_preview_wrong_owner_or_changed_inputs_cannot_be_promoted(self):
        self.f.configure(task='collect_tablecloth');admitted=self.admitted('collect_tablecloth')
        variants=(replace(admitted,owner_kind='standalone'),
                  replace(admitted,plan=dict(admitted.plan,execution_mode='gpu_cloth_preview')),
                  replace(admitted,plan=dict(admitted.plan,scoring_available=False)))
        for value in variants:
            with self.assertRaises(ValueError):self.call(NativeClothBinding(lambda *a,**k:value,lambda *a,**k:self.fail('Must not execute')))
        self.f.scene.write_text('changed')
        with self.assertRaises(ValueError):self.call(NativeClothBinding(lambda *a,**k:admitted,lambda *a,**k:self.fail('Changed source executed')))
    def test_wrong_scope_and_returning_executor_fail_closed(self):
        self.f.configure(task='long_task_18')
        with self.assertRaises(ValueError):self.call()
        self.f.configure(task='collect_tablecloth');admitted=self.admitted('collect_tablecloth')
        with self.assertRaisesRegex(RuntimeError,'handoff returned'):
            self.call(NativeClothBinding(lambda *a,**k:admitted,lambda *a,**k:None))
    def test_json_or_callback_in_place_of_trusted_binding_is_rejected(self):
        self.f.configure(task='collect_tablecloth')
        for value in ({'module':'arbitrary'},lambda *a:None):
            with self.assertRaises(ValueError):self.call(value)
    def test_new_formal_schema_cannot_route_a_noncloth_task(self):
        self.f.configure(task='stock_wooden_tray')
        data=json.loads(self.f.launch.read_text());data['schema']='robotagentbench.native_cloth.launch.v1'
        self.f.launch.write_text(json.dumps(data))
        with patch('robotagentbench.simulation.original_backend.bind_original_backend',side_effect=AssertionError('No generic fallback')):
            with self.assertRaisesRegex(ValueError,'another task family'):self.call()
    def test_new_formal_schema_is_passed_to_trusted_cloth_owner_only(self):
        self.f.configure(task='collect_tablecloth')
        data=json.loads(self.f.launch.read_text());data['schema']='robotagentbench.native_cloth.launch.v1';data.pop('request')
        self.f.launch.write_text(json.dumps(data));a=self.admitted('collect_tablecloth')
        def execute(*args,**kwargs):raise SystemExit(0)
        with self.assertRaises(SystemExit):self.call(NativeClothBinding(lambda *args,**kwargs:a,execute))


if __name__=='__main__':unittest.main()

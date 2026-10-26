"""Task path wiring tests use fixture factories, never native scoring."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.simulation.backend_paths import BackendPaths, ResourceGroups, TASKS
from robotagentbench.simulation.dependencies import Resource, RuntimeDependencies
from robotagentbench.simulation.original_backend import bind_original_backend, run_original_worker
from robotagentbench.simulation.source_tree import SourceDependencyError
from robotagentbench.simulation.worker import NativeRunOptions
from admission_fixtures import AdmissionFixture
from source_fixtures import SourceFixture


class BackendPathTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture=SourceFixture(self.temp.name)
        original=(self.fixture.root/'short_tasks/registry.py').read_text()
        self.fixture.write('short_tasks/registry.py',original+
            'def resolve(request):\n    return dict(task_name=request["task_name"], score_policy={"maximum": 17})\n'
            'def bind(plan, manifest):\n    return plan\n'
            'def verify_sources(plan):\n    return {"original_source_check": True}\n')
        self.resources={}
        for index,(path,digest) in enumerate(self.fixture.pins.items()):
            self.resources['source_'+str(index)]=Resource('code',path,digest,'source')
        for kind in ('asset','native'):
            path=self.fixture.root/kind
            path.write_bytes(kind.encode())
            self.resources[kind]=Resource('code',kind,hashlib.sha256(path.read_bytes()).hexdigest(),kind)
        self.dependencies=RuntimeDependencies({'code':self.fixture.root},self.resources)
        self.source_keys=tuple(key for key in self.resources if key.startswith('source_'))

    def paths(self,*,assets=True,native=True):
        common=ResourceGroups(source=self.source_keys,assets=('asset',) if assets else (),
                              native=('native',) if native else ())
        return BackendPaths(self.dependencies,common=common,tasks={name:ResourceGroups() for name in TASKS})

    def test_complete_explicit_routing_table_and_matching_import_tree(self):
        paths=self.paths()
        paths.validate_source_tree(self.fixture.tree())
        self.assertEqual(len(paths.tasks),44)
        self.assertTrue(paths.verify('prepare_coffee','native')['verified'])
        for name in ('openend','task_discovery',*(f'long_task_{i}' for i in range(15,23))):
            self.assertTrue(paths.verify(name,'assets')['verified'])
        with self.assertRaises(ValueError):
            BackendPaths(self.dependencies,common=paths.common,tasks={'prepare_coffee':ResourceGroups()})
        fixture=AdmissionFixture(self.temp.name)
        tree=self.fixture.tree()
        for task in ('coffee_bag_interference','failure_recovery_sofa'):
            fixture.configure(task=task)
            def handoff(family,path,output,**kwargs):
                self.assertIs(kwargs['paths'],paths)
                self.assertFalse(tree.active)
                raise SystemExit(0)
            with self.assertRaises(SystemExit):
                run_original_worker(fixture.launch,fixture.output,options=NativeRunOptions(0),
                    sources=tree,workspace=fixture.workspace,paths=paths,specialized_owner=handoff)

    def test_operator_configuration_is_complete_and_does_not_infer_missing_groups(self):
        config=dict(schema='robotagentbench.backend_paths.v1',
                    common=dict(source=list(self.source_keys),assets=['asset'],native=['native'],coding=[]),
                    tasks={name:dict(source=[],assets=[],native=[],coding=[]) for name in TASKS})
        paths=BackendPaths.from_config(self.dependencies,config)
        self.assertTrue(paths.verify('prepare_coffee','native')['verified'])
        config['tasks']['prepare_coffee']['assets']=['asset','asset']
        with self.assertRaises(ValueError):BackendPaths.from_config(self.dependencies,config)
        config['tasks']['prepare_coffee']['assets']=[]
        del config['tasks']['prepare_coffee']['native']
        with self.assertRaises(ValueError):BackendPaths.from_config(self.dependencies,config)

    def test_operator_configuration_rejects_unknown_schema_fields_and_unbound_resources(self):
        config=dict(schema='robotagentbench.backend_paths.v1',
                    common=dict(source=list(self.source_keys),assets=['unbound'],native=[],coding=[]),
                    tasks={name:dict(source=[],assets=[],native=[],coding=[]) for name in TASKS})
        with self.assertRaises(ValueError):BackendPaths.from_config(self.dependencies,config)
        config['common']['assets']=[];config['fallback']=True
        with self.assertRaises(ValueError):BackendPaths.from_config(self.dependencies,config)
        del config['fallback'];config['schema']='other'
        with self.assertRaises(ValueError):BackendPaths.from_config(self.dependencies,config)

    def test_original_registry_and_generic_context_factories_use_configured_paths(self):
        with bind_original_backend(self.fixture.tree(),paths=self.paths()) as backend:
            for name in ('prepare_coffee','bake_bread','select_drink','select_snack','select_trash','select_sofa'):
                plan=backend.registry.resolve({'task_name':name})
                self.assertEqual(plan['score_policy'],{'maximum':17})
                self.assertIs(backend.registry.bind(plan,{}),plan)
                self.assertEqual(backend.registry.verify_sources(plan),{'original_source_check':True})
                result=backend.build_short_session(name,{'task_name':name},'stage')
                self.assertEqual(result[0],'trash' if name=='select_trash' else
                                 'selection' if name.startswith('select_') else 'standalone')

    def test_missing_asset_or_native_profile_never_reaches_the_original_factory(self):
        with bind_original_backend(self.fixture.tree(),paths=self.paths(assets=False)) as backend:
            plan=backend.registry.resolve({'task_name':'prepare_coffee'})
            with self.assertRaisesRegex(SourceDependencyError,'asset profile'):
                backend.registry.bind(plan,{})
        with bind_original_backend(self.fixture.tree(),paths=self.paths(native=False)) as backend:
            plan=backend.registry.resolve({'task_name':'prepare_coffee'})
            backend.registry.bind(plan,{})
            with self.assertRaisesRegex(SourceDependencyError,'native environment'):
                backend.build_short_session('prepare_coffee',{'task_name':'prepare_coffee'})

    def test_asset_drift_and_task_profile_switch_are_rejected(self):
        with bind_original_backend(self.fixture.tree(),paths=self.paths()) as backend:
            plan=backend.registry.resolve({'task_name':'prepare_coffee'})
            backend.registry.bind(plan,{})
            with self.assertRaisesRegex(SourceDependencyError,'differs'):
                backend.build_short_session('bake_bread',{'task_name':'bake_bread'})
            (self.fixture.root/'asset').write_bytes(b'changed')
            with self.assertRaises(SourceDependencyError):
                backend.build_short_session('prepare_coffee',{'task_name':'prepare_coffee'})

    def test_room_admission_build_and_acceptance_keep_the_selected_profile(self):
        # This test isolates path/profile routing; real budget state and task
        # termination are exercised by test_discovery_budget, not tuple fixtures.
        with patch('robotagentbench.tasks.discovery.binding.describe_plan', side_effect=lambda plan:plan), \
             patch('robotagentbench.tasks.discovery.binding.install', side_effect=lambda room:room), \
             patch('robotagentbench.tasks.discovery.binding.DiscoveryControlSession', side_effect=lambda session:session), \
             bind_original_backend(self.fixture.tree(),paths=self.paths()) as backend:
            for name in ('openend','task_discovery','long_task_15','long_task_22'):
                backend.admit_room({'task_name':name},'output')
                self.assertEqual(backend.build_room('request',evaluation_mode=name)[0],'room_build')
                self.assertEqual(backend.accept_room('room')[0],'room_accept')
            with self.assertRaises(SourceDependencyError):
                backend.build_room('request',evaluation_mode='openend')

    def test_import_tree_cannot_use_unbound_files_or_wrong_resource_kinds(self):
        tree=self.fixture.tree()
        self.fixture.write('short_tasks/unreviewed.py','VALUE=1\n')
        with self.assertRaises(SourceDependencyError):
            self.paths().validate_source_tree(self.fixture.tree())
        with self.assertRaises(ValueError):
            BackendPaths(self.dependencies,common=ResourceGroups(source=('asset',)),
                         tasks={name:ResourceGroups() for name in TASKS})
        self.paths().validate_source_tree(tree)


if __name__=='__main__':unittest.main()

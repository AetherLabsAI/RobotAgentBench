"""CPU delivery planning only; no simulator, model, mounts or source changes."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from robotagentbench.simulation import dedicated_delivery as delivery
from robotagentbench.simulation.cloth import config as cloth_config
from robotagentbench.simulation.specialized import config
from robotagentbench.simulation.specialized import camera_release


class Delivery(unittest.TestCase):
    def test_reviewed_templates_and_original_roles_are_packaged(self):
        value = delivery.template()
        self.assertEqual(value['schema'], 'robotagentbench.dedicated_delivery_template.v1')
        for family, identity in config.identities().items():
            self.assertEqual(set(value['families'][family]['tasks']), set(identity['tasks']))
        path = delivery.HERE / 'dedicated_cloth_template.json'
        self.assertEqual(delivery.sha(path), delivery.CLOTH_TEMPLATE_SHA)
        cloth = delivery.read(path)
        self.assertEqual(set(cloth['tasks']), cloth_config.TASKS)
        for roles in cloth['tasks'].values():
            self.assertEqual(set(roles), cloth_config.FILES)
        self.assertEqual(len(cloth['geometry_pins']), 139)
        self.assertFalse(any(name.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr'} for name in sys.modules))

    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'ambiguous.json'
            p.write_text('{"gpu":0,"gpu":1}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                delivery.read(p)

    def test_native_merge_preserves_pins_and_explicit_paths(self):
        dependencies = {'roots': {}, 'resources': {}}
        original = self.native_fixture(Path('/isolated/native'))
        snapshot = deepcopy(original)
        profile = delivery.merge_environment(dependencies, original)
        self.assertEqual(original, snapshot)
        self.assertEqual(dependencies['roots']['native_sdk'], '/isolated/native')
        self.assertEqual(dependencies['resources']['native_python']['sha256'], 'a' * 64)
        self.assertEqual(profile['binary'], 'native_python')
        self.assertEqual(profile['metadata_paths'], [{'root': 'native_sdk', 'path': ''}])

    @staticmethod
    def native_fixture(sdk):
        paths = ('metadata_paths', 'python_paths', 'library_paths', 'plugin_paths', 'asset_paths', 'material_paths')
        return {'dependencies': {'schema': 'robotagentbench.runtime_dependencies.v1',
            'roots': {'sdk': str(sdk)}, 'resources': {'python': {'root': 'sdk',
            'path': 'python', 'kind': 'native', 'sha256': 'a' * 64}}, 'legacy_sources': {}},
            'profile': dict(binary='python', required_resources=['python'],
                **{key: [{'root': 'sdk', 'path': ''}] if key == 'metadata_paths' else [] for key in paths})}

    def fixture(self, root):
        assets = root / 'Assets'; assets.mkdir()
        lock = root / 'shared-locks'; lock.mkdir()
        sdk = root / 'sdk'; sdk.mkdir()
        native = root / 'native.json'; native.write_text(json.dumps(self.native_fixture(sdk)))
        approved = {'resources': {}, 'aliases': {}, 'families': {}}
        index = {'objects': {}}
        runtimes = {}

        def asset(key, value):
            data = json.dumps(value).encode()
            digest = hashlib.sha256(data).hexdigest()
            rel = 'objects/' + digest[:2] + '/' + digest
            p = assets / rel; p.parent.mkdir(exist_ok=True, parents=True); p.write_bytes(data)
            index['objects'][digest] = {'path': rel}
            approved['resources'][key] = {'sha256': digest, 'kind': 'asset'}
            return digest

        for family, identity in config.identities().items():
            base = root / ('Runtime-' + family); base.mkdir()
            source = base / 'sources' / family; source.mkdir(parents=True)
            original = '/__rab_delivery_test__/' + family + '/source.py'
            p = source / original.lstrip('/'); p.parent.mkdir(parents=True)
            p.write_text('# ' + family)
            digest = delivery.sha(p)
            (base / 'runtime.json').write_text(json.dumps({'schema': 'robotagentbench.local_installation.v1',
                'profile': family, 'runtime': str(base), 'source_root': str(source),
                'source_files': {original.lstrip('/'): digest}}))
            runtimes[family] = str(base)
            asset(family + '_manifest', {'profiles': {family: {'files': [{'original': original, 'sha256': digest}]}}})
            asset(family + '_bindings', {'source_resources': {original: family + '_source'},
                'namespace_mounts': [{'source_relative': original.lstrip('/'), 'target': original}],
                'startup_resources': {}})
            approved['families'][family] = {'manifest': family + '_manifest',
                'bindings': family + '_bindings', 'required': [], 'tasks': {task: {
                'launch': '/__rab_delivery_test__/tasks/' + task + '/prepared/launch.json',
                'output_parent': '/__rab_delivery_test__/tasks/' + task + '/outputs', 'extra': []}
                for task in identity['tasks']}}
        return approved, index, dict(assets=assets, manifest=root / 'manifest.json', runtimes=runtimes,
            bank=root / 'Bank', native_config=native, family='atomic_coffee', gpu=0,
            lock_root=lock, output=root / 'Plan', episode=17)

    def patches(self, stack, approved, index):
        # Tiny fake assemblies test delivery orchestration; production fixed
        # camera provenance and derivation are covered separately.
        stack.enter_context(patch.object(camera_release, 'derive_assembly',
                                        side_effect=lambda path, family: Path(path).read_bytes()))
        stack.enter_context(patch.object(delivery, 'template', return_value=approved))
        stack.enter_context(patch.object(delivery.resources, 'read_manifest', return_value={}))
        stack.enter_context(patch.object(delivery.installation, '_index', return_value=index))
        stack.enter_context(patch.object(delivery, 'native', side_effect=lambda value: value))
        stack.enter_context(patch('subprocess.run', side_effect=AssertionError('No native subprocess')))
        stack.enter_context(patch('os.execve', side_effect=AssertionError('No native handoff')))
        return stack.enter_context(patch.object(delivery, 'select', side_effect=lambda bank, task, episode: {
            'state': {'launch_options': {'task_name': task, 'seed': 123, 'sensor': 'saved-camera.json',
                'episode_selection': episode}}}))

    def test_specialized_uses_exact_saved_options_and_fresh_readonly_input_plan(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            approved, index, kwargs = self.fixture(root)
            selected = self.patches(stack, approved, index)
            result = delivery.specialized(**kwargs)
            self.assertEqual(selected.call_count, 9)
            self.assertFalse(result['execution_ready'])
            self.assertTrue(result['bootstrap_only'])
            self.assertFalse(result['native_started'])
            self.assertFalse(result['resampled'])
            self.assertEqual(result['mount_phase'], 'cpu_preparation_only')
            operator = config.read_operator(result['operator'])
            self.assertEqual(operator['outer_lifecycle'], config.OUTER_CONTRACT)
            self.assertNotIn('/__rab_delivery_test__/atomic_coffee/source.py',operator['workspace']['input_roots'])
            self.assertIn('/__rab_delivery_test__/atomic_coffee',operator['workspace']['input_roots'])
            for family, roles in approved['families'].items():
                for task in roles['tasks']:
                    launch = kwargs['output'] / 'generated' / task / 'launch.json'
                    self.assertEqual(delivery.read(launch), {'task_name': task, 'seed': 123,
                        'sensor': 'saved-camera.json', 'episode_selection': 17})
            readonly_targets = {r['target'] for r in result['mounts'] if r['read_only']}
            self.assertIn(str(kwargs['assets']), readonly_targets)
            self.assertIn('/__rab_delivery_test__/atomic_coffee/source.py', readonly_targets)
            self.assertNotIn('/__rab_delivery_test__/failure_recovery/source.py', readonly_targets)
            self.assertFalse(Path('/__rab_delivery_test__').exists())
            self.assertFalse(any(root.rglob('*.pyc')))

    def test_changed_installed_source_fails_before_saved_state_is_read(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved, index, kwargs = self.fixture(Path(temp).resolve())
            selected = self.patches(stack, approved, index)
            p = Path(kwargs['runtimes']['atomic_coffee']) / 'sources/atomic_coffee/__rab_delivery_test__/atomic_coffee/source.py'
            p.write_text('not the reviewed bytes')
            with self.assertRaisesRegex(ValueError, 'Configured source member changed'):
                delivery.specialized(**kwargs)
            selected.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved, index, kwargs = self.fixture(Path(temp).resolve())
            self.patches(stack, approved, index)
            kwargs['output'].mkdir()
            sentinel = kwargs['output'] / 'keep.txt'; sentinel.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'Fresh output'):
                delivery.specialized(**kwargs)
            self.assertEqual(sentinel.read_text(), 'existing')

    def test_wrong_saved_task_is_rejected_without_resampling(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved, index, kwargs = self.fixture(Path(temp).resolve())
            selected = self.patches(stack, approved, index)
            selected.side_effect = lambda *a: {'state': {'launch_options': {'task_name': 'other'}}}
            with self.assertRaisesRegex(ValueError, 'Saved launch/task identity differs'):
                delivery.specialized(**kwargs)
            self.assertEqual(selected.call_count, 1)

    def test_image_native_roots_are_not_misrepresented_as_host_binds(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp).resolve()
            approved,index,kwargs=self.fixture(root)
            self.patches(stack,approved,index)
            material=root/'sdk/material.mdl';material.write_text('exact SDK material')
            digest=delivery.sha(material)
            approved['resources']['sdk_material']={'sha256':digest,'kind':'asset','sdk_relative':'material.mdl'}
            approved['aliases']['/__rab_delivery_test__/material.mdl']=digest
            result=delivery.specialized(**kwargs,native_in_image=True)
            self.assertEqual(result['required_native_roots_in_image'],[str(root/'sdk')])
            self.assertFalse(any(r['source']==str(root/'sdk')for r in result['mounts']))
            mounted=next(r for r in result['mounts']if r['target']=='/__rab_delivery_test__/material.mdl')
            self.assertTrue(Path(mounted['source']).is_relative_to(kwargs['output']))
            self.assertEqual(delivery.sha(mounted['source']),digest)
            self.assertTrue(mounted['read_only'])

    def test_native_owner_lock_is_one_fresh_file_not_writable_source_tree(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved,index,kwargs=self.fixture(Path(temp).resolve())
            self.patches(stack,approved,index)
            delivery.specialized(**kwargs)
            plan=kwargs['output']/'namespace-plan.json'
            task='coffee_bag_interference'
            uuid='GPU-00000000-0000-0000-0000-000000000000'
            rows=delivery.owner_lock_bindings(plan,task,hostname='VM-0-10-ubuntu',gpu_uuid=uuid)
            self.assertEqual(len(rows),1)
            self.assertEqual(Path(rows[0]['source']).stat().st_size,0)
            self.assertTrue(Path(rows[0]['source']).is_relative_to(kwargs['output']))
            self.assertTrue(rows[0]['target'].endswith('/short_tasks/agent_probe_v1/gpu-'+uuid+'.lock'))
            self.assertFalse(rows[0]['read_only'])
            self.assertTrue(rows[0]['external_shared_gpu_lease_still_required'])
            with self.assertRaises(FileExistsError):
                delivery.owner_lock_bindings(plan,task,hostname='VM-0-10-ubuntu',gpu_uuid=uuid)
            with self.assertRaisesRegex(ValueError,'real approved host'):
                delivery.owner_lock_bindings(plan,task,hostname='another-host',gpu_uuid=uuid)

    def test_readonly_namespace_gets_only_fresh_nested_mountpoint_scaffold(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();source=root/'Source';source.mkdir()
            original=source/'entry.py';original.write_text('unchanged source')
            asset=root/'asset.usda';asset.write_text('immutable asset')
            plan=root/'Plan';plan.mkdir()
            rows=[dict(source=str(source),target=delivery.LEGACY_GENERIC,read_only=True),
                  dict(source=str(asset),target=delivery.LEGACY_GENERIC+'/resources/asset.usda',read_only=True)]
            mounted,scaffolds=delivery.mountpoint_scaffolds(rows,plan)
            self.assertEqual(len(scaffolds),1)
            view=Path(mounted[0]['source'])
            self.assertEqual(delivery.sha(view/'entry.py'),delivery.sha(original))
            self.assertEqual((view/'resources/asset.usda').stat().st_size,0)
            self.assertFalse((source/'resources').exists())
            self.assertEqual(mounted[1],rows[1])
            self.assertTrue(mounted[0]['read_only'])
            self.assertTrue(view.is_relative_to(plan/'namespace_scaffolds'))

    def test_global_lease_does_not_shadow_pinned_sources_with_live_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();source=root/'Source';(source/'locks').mkdir(parents=True)
            output=root/'Plan';output.mkdir()
            lock=Path('/original/source/locks')
            rows=[dict(source=str(source),target='/original/source',read_only=True),
                  dict(source='/shared/live/locks',target=str(lock),read_only=False)]
            result=delivery.external_lock_namespace(rows,lock,output)
            self.assertEqual(result,[rows[0]])
            self.assertFalse((output/'external_lock_namespace').exists())

    def test_cloth_roles_rebind_only_to_fresh_runtime_and_geometry(self):
        from robotagentbench.backends import sources
        from robotagentbench.simulation.dependencies import RuntimeDependencies
        from robotagentbench.simulation.cloth import launcher
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            assets = root / 'Assets'; assets.mkdir()
            base = root / 'Runtime'; base.mkdir()
            source = base / 'sources/cloth'; source.mkdir(parents=True)
            geometry = root / 'geometry'; geometry.mkdir()
            wheel = geometry / 'shapely.so'; wheel.write_bytes(b'reviewed test wheel')
            sdk = root / 'sdk'; (sdk / 'isaacsim').mkdir(parents=True)
            lock = root / 'lock'; lock.mkdir()
            env = root / 'native.json'; env.write_text(json.dumps(self.native_fixture(sdk)))
            template_path = delivery.HERE / 'dedicated_cloth_template.json'
            roles = delivery.read(template_path)
            roles['geometry_pins'] = {'shapely.so': delivery.sha(wheel)}
            expected = {}
            for files in roles['tasks'].values():
                for name in files.values():
                    member = source / name.lstrip('/')
                    member.parent.mkdir(parents=True, exist_ok=True)
                    if not member.exists():member.write_text(name)
                    expected[name.lstrip('/')] = delivery.sha(member)
            (base / 'runtime.json').write_text(json.dumps({'schema': 'robotagentbench.local_installation.v1',
                'profile': 'cloth', 'runtime': str(base), 'source_root': str(source), 'source_files': expected}))
            actual_read = delivery.read
            stack.enter_context(patch.object(delivery, 'read', side_effect=lambda p: roles if Path(p) == template_path else actual_read(p)))
            stack.enter_context(patch.object(delivery.resources, 'read_manifest', return_value={}))
            stack.enter_context(patch.object(delivery.installation, '_index', return_value={
                'locations': {'profiles/cloth/' + p: h for p, h in expected.items()}}))
            stack.enter_context(patch.object(sources, 'verify', return_value=None))
            stack.enter_context(patch.object(sources, '_manifest', return_value={'profiles': {'cloth': {'files': {}}}}))
            stack.enter_context(patch.object(delivery, 'native', side_effect=lambda v: v))
            stack.enter_context(patch.object(RuntimeDependencies, 'from_config', return_value=SimpleNamespace(resources={})))
            original_glob = Path.rglob
            # Other independent test processes may create checkout bytecode;
            # production still rejects it and the fixture represents clean src.
            stack.enter_context(patch.object(Path, 'rglob', lambda p, pattern:
                iter(()) if pattern == '*.pyc' else original_glob(p, pattern)))
            stack.enter_context(patch('subprocess.run', side_effect=AssertionError('No SDK subprocess')))
            result = delivery.cloth(assets=assets, manifest=root / 'manifest.json', cloth_runtime=base,
                geometry_root=geometry, output=root / 'Plan', native_config=env, gpu=0, lock_root=lock)
            operator = launcher.read_operator(result['operator'])
            self.assertFalse(result['execution_ready'])
            self.assertFalse(result['native_started'])
            self.assertEqual(result['mount_phase'], 'cpu_preparation_only')
            self.assertEqual(len(result['source_aliases']), 8)
            for task, template in operator['launch_templates'].items():
                for role, path in template['files'].items():
                    self.assertEqual(path, str(source / roles['tasks'][task][role].lstrip('/')))
                    self.assertEqual(template['hashes'][path], delivery.sha(path))
                self.assertEqual(template['directories']['geometry_python_root'], str(geometry))
            for alias in result['source_aliases']:
                self.assertTrue(Path(alias['path']).is_relative_to(source))
                self.assertFalse(Path(alias['path']).exists())
            with self.assertRaisesRegex(ValueError, 'Reviewed cloth geometry wheel differs'):
                wheel.write_bytes(b'different wheel')
                delivery.cloth(assets=assets, manifest=root / 'manifest.json', cloth_runtime=base,
                    geometry_root=geometry, output=root / 'Other', native_config=env, gpu=0, lock_root=lock)


if __name__ == '__main__':
    unittest.main()

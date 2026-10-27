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
    def test_recovery_vendor_modules_match_original_upstream_and_are_explicit_inputs(self):
        from robotagentbench.backends import sources
        value=delivery.template()
        base='/home/lingjun/robot_agent_arena/api-protocol-st18-20261013-frozen4/evaluation/tasks/coffee_machine/vendor_producer/'
        relative='evaluation/tasks/coffee_machine/vendor_producer'
        provenance=delivery.read(sources.root('room')/relative/'MANIFEST.json')['files']
        required=value['families']['failure_recovery']['tasks']['failure_recovery_coffee_bag']['extra']
        for name in ('machine_contract.py','coffee_logic.py'):
            original=sources.root('atomic_coffee')/'frozen'/relative/name
            digest=delivery.sha(original)
            self.assertEqual(digest,provenance[name]['sha256'])
            self.assertEqual(value['aliases'][base+name],digest)
            key='source_recovery_vendor_'+Path(name).stem
            self.assertEqual(value['resources'][key],dict(kind='source',sha256=digest))
            self.assertIn(key,required)

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
        stack.enter_context(patch.object(camera_release, 'profile', return_value={'files': {}}))
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

    def test_nested_room_sources_are_exact_readonly_and_required(self):
        from robotagentbench.simulation.saved_instance_release import room_recovery_closure
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            approved, index, kwargs = self.fixture(root)
            self.patches(stack, approved, index); kwargs['family'] = 'failure_recovery'
            content = kwargs['assets'] / 'original-room.usdc'; content.write_bytes(b'original scene')
            digest = delivery.sha(content); index['objects'][digest] = {'path': content.name}
            target = '/__rab_delivery_test__/saved-source/room.usdc'
            stack.enter_context(patch.object(room_recovery_closure, 'source_requirements', return_value={target: digest}))
            result = delivery.specialized(**kwargs)
            row, = [r for r in result['mounts'] if r['target'] == target]
            self.assertTrue(row['read_only']); self.assertEqual(delivery.sha(row['source']), digest)
            operator = delivery.read(result['operator']); key = 'saved_room_source_' + digest
            for task in operator['families']['failure_recovery']['tasks'].values():
                self.assertIn(key, task['required_resources'])
            self.assertEqual(operator['dependencies']['resources'][key]['sha256'], digest)

    def test_nested_room_source_missing_or_conflicting_alias_fails_before_native(self):
        from robotagentbench.simulation.saved_instance_release import room_recovery_closure
        for conflict in (False, True):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp).resolve()
                approved, index, kwargs = self.fixture(root)
                self.patches(stack, approved, index); kwargs['family'] = 'failure_recovery'
                target = '/__rab_delivery_test__/saved-source/room.usdc'; digest = 'c' * 64
                if conflict:
                    content = kwargs['assets'] / 'original-room.usdc'; content.write_bytes(b'original scene')
                    digest = delivery.sha(content); index['objects'][digest] = {'path': content.name}
                    approved['aliases'][target] = 'd' * 64
                stack.enter_context(patch.object(room_recovery_closure, 'source_requirements', return_value={target: digest}))
                with self.assertRaisesRegex(ValueError, 'absent|conflicts'):
                    delivery.specialized(**kwargs)

    def test_nested_source_exact_byte_patch_is_required_and_readonly(self):
        import base64
        import gzip
        from robotagentbench.simulation.saved_instance_release import room_recovery_closure, exact_source
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            approved, index, kwargs = self.fixture(root)
            self.patches(stack, approved, index); kwargs['family'] = 'failure_recovery'
            original = b'original scene bytes'; digest = hashlib.sha256(original).hexdigest()
            def put(data):
                pin = hashlib.sha256(data).hexdigest(); relative = 'objects/' + pin[:2] + '/' + pin
                path = kwargs['assets'] / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                index['objects'][pin] = dict(path=relative, bytes=len(data), sha256=pin)
                return pin
            base = put(b'original scene')
            patch_pin = put(gzip.compress(json.dumps(dict(schema=exact_source.PATCH_SCHEMA,
                base_sha256=base,target_sha256=digest,target_bytes=len(original),
                operations=[['copy',0,14],['literal',base64.b64encode(b' bytes').decode()]])).encode(),mtime=0))
            index['source_reconstruction'] = {digest: dict(schema=exact_source.RECIPE_SCHEMA,
                base_sha256=base,patch_sha256=patch_pin,target_bytes=len(original))}
            target='/__rab_delivery_test__/saved-source/composed.usda'
            stack.enter_context(patch.object(room_recovery_closure,'source_requirements',return_value={target:digest}))
            result = delivery.specialized(**kwargs)
            row, = [r for r in result['mounts'] if r['target']==target]
            self.assertTrue(row['read_only']); self.assertEqual(Path(row['source']).read_bytes(), original)
            self.assertTrue(Path(row['source']).is_relative_to(kwargs['output'] / 'exact_source_inputs'))
            self.assertNotIn(digest, index['objects'])
            operator=delivery.read(result['operator'])
            for task in operator['families']['failure_recovery']['tasks'].values():
                self.assertIn('saved_room_source_'+digest,task['required_resources'])

    def test_camera_source_upgrade_needs_original_assembly_not_old_code_bytes(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved, index, kwargs = self.fixture(Path(temp).resolve())
            self.patches(stack, approved, index)
            family = 'atomic_coffee'
            original = '/__rab_delivery_test__/atomic_coffee/source.py'
            source = Path(kwargs['runtimes'][family]) / 'sources' / family / original.lstrip('/')
            after = {'sha256': delivery.sha(source), 'bytes': source.stat().st_size}
            before = {'sha256': hashlib.sha256(b'old absent code').hexdigest(), 'bytes': 15}
            approved['resources'][family + '_source'] = dict(kind='source', sha256=before['sha256'])
            camera_release.profile.side_effect = lambda f: {'files': {original: dict(before=before, after=after)} if f == family else {}}
            result = delivery.specialized(**kwargs)
            operator = delivery.read(result['operator'])
            resources = operator['dependencies']['resources']
            self.assertEqual(resources[family + '_source']['sha256'], after['sha256'])
            self.assertEqual(resources[family + '_manifest_upstream']['sha256'],
                             approved['resources'][family + '_manifest']['sha256'])
            self.assertNotIn(before['sha256'], index['objects'])

    def test_camera_upgrade_rejects_unknown_predecessor_and_wrong_new_bytes(self):
        for wrong_predecessor in (True, False):
            with self.subTest(wrong_predecessor=wrong_predecessor), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                approved, index, kwargs = self.fixture(Path(temp).resolve())
                selected = self.patches(stack, approved, index)
                family = 'atomic_coffee'
                original = '/__rab_delivery_test__/atomic_coffee/source.py'
                approved['resources'][family + '_source'] = dict(kind='source', sha256='a' * 64)
                change = dict(before={'sha256': ('b' if wrong_predecessor else 'a') * 64, 'bytes': 15},
                              after={'sha256': 'c' * 64, 'bytes': 15})
                camera_release.profile.side_effect = lambda f: {'files': {original: change} if f == family else {}}
                with self.assertRaisesRegex(ValueError, 'reviewed predecessor|fixed release'):
                    delivery.specialized(**kwargs)
                selected.assert_not_called()

    def test_camera_upgrade_does_not_bypass_missing_unrelated_resource(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            approved, index, kwargs = self.fixture(Path(temp).resolve())
            selected = self.patches(stack, approved, index)
            approved['resources']['unrelated_source'] = dict(kind='source', sha256='d' * 64)
            with self.assertRaisesRegex(ValueError, 'Reviewed content is absent'):
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
        from robotagentbench.simulation.cloth import launcher, source_data
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
                for role, name in files.items():
                    member = source / name.lstrip('/')
                    member.parent.mkdir(parents=True, exist_ok=True)
                    if not member.exists():
                        member.write_text(json.dumps({'files': {}}) if role == 'source_manifest' else name)
                    expected[name.lstrip('/')] = delivery.sha(member)
            for name in source_data.FILES:
                relative = 'original/' + name
                member = source / relative; member.parent.mkdir(parents=True, exist_ok=True)
                member.write_bytes((sources.root('cloth') / relative).read_bytes())
                expected[relative] = delivery.sha(member)
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
                    upstream = str(source / roles['tasks'][task][role].lstrip('/'))
                    if role == 'source_manifest':
                        self.assertTrue(Path(path).is_relative_to(root / 'Plan/generated'))
                        self.assertEqual(template['hashes'][upstream], delivery.sha(upstream))
                        self.assertEqual(delivery.read(upstream), {'files': {}})
                        derived = delivery.read(path)
                        self.assertEqual(derived['files'], source_data.FILES)
                        self.assertEqual(derived['static_dependency_delivery']['upstream_manifest_sha256'],
                                         delivery.sha(upstream))
                    else:
                        self.assertEqual(path, upstream)
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

"""CPU-only camera release and provenance checks; no SDK, render or model."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.backends import sources
from robotagentbench.simulation.specialized import camera_release as release, config

ROOT = Path(sources.__file__).parent / 'profiles'
RECOVERY = ROOT / 'failure_recovery/generic/short_task_robustness/tasks/grasp_failure_recovery'
ATOMIC = ROOT / 'atomic_coffee/coffee/public_integration_v1'
SIZES = dict(head=(1280, 800), left=(848, 480), right=(848, 480))


def digest(data):
    return hashlib.sha256(data).hexdigest()


class NativeCameraRelease(unittest.TestCase):
    def test_packaged_policy_only_covers_reviewed_source_members(self):
        policy = release.policy()
        self.assertEqual(policy['resolutions'], {k: list(v) for k, v in SIZES.items()})
        self.assertEqual({f: len(r['files']) for f, r in policy['profiles'].items()},
                         dict(atomic_coffee=3, failure_recovery=8))
        manifest = sources._manifest()
        for family, selected in policy['profiles'].items():
            self.assertEqual(selected['base_assembly_sha256'], config.identities()[family]['manifest_sha256'])
            for original, row in selected['files'].items():
                path = ROOT / family / row['relative']
                self.assertTrue(original.startswith('/'))
                self.assertEqual(digest(path.read_bytes()), row['after']['sha256'])
                self.assertEqual(path.stat().st_size, row['after']['bytes'])
                self.assertEqual(manifest['profiles'][family]['files'][row['relative']], row['after'])
                self.assertNotEqual(row['before']['sha256'], row['after']['sha256'])

    def test_component_records_match_current_pins_and_preserve_source_history(self):
        atomic = json.loads((ATOMIC / 'component_sources.json').read_text())
        for original, row in release.profile('atomic_coffee')['files'].items():
            if original.endswith('.py'):
                self.assertEqual(atomic['absolute_sources'][original], row['after']['sha256'])
        self.assertIn('task_owned_revision_history', atomic)
        first = (RECOVERY / 'public_component_sources.json').read_bytes()
        self.assertEqual(first, (RECOVERY / 'closeout_v1/component_sources.json').read_bytes())
        record = json.loads(first)
        for original, row in release.profile('failure_recovery')['files'].items():
            if '/grasp_failure_recovery/' in original and original.endswith('.py'):
                self.assertEqual(record['task_sources'][original.split('/grasp_failure_recovery/')[1]],
                                 row['after']['sha256'])

    def test_recovery_furniture_delta_is_explicit_and_other_family_unchanged(self):
        selected = release.profile('failure_recovery')
        self.assertEqual(selected['source_binding_release']['id'], 'manifest_source_furniture_v1')
        self.assertEqual(release.provenance('failure_recovery')['source_binding_release'],
                         selected['source_binding_release'])
        self.assertNotIn('source_binding_release', release.provenance('atomic_coffee'))
        self.assertEqual(release.profile('atomic_coffee')['derived_assembly_sha256'],
                         'b4fdbd6ba890caf308c6009d8c8913e8868cd4b03aacc0649571f32e70125ad1')
        room = ROOT / 'room'
        for relative in ('openend/tasks/all_objects_integration/geometry.py',
                         'short_tasks/tasks/sofa_to_drawer/native_geometry.py'):
            self.assertEqual((ROOT / 'failure_recovery/generic' / relative).read_bytes(),
                             (room / relative).read_bytes())
        before = deepcopy(selected['source_binding_release'])
        provenance = release.provenance('failure_recovery')
        provenance['source_binding_release']['id'] = 'not-a-policy-update'
        self.assertEqual(selected['source_binding_release'], before)

    def test_selected_native_products_and_head_instance_mask_use_standard_dimensions(self):
        workers = [ATOMIC / 'native_worker.py', RECOVERY / 'shelf_continuous_v2/owner.py',
                   RECOVERY / 'closeout_v1/atomic_owner_v2.py']
        for path in workers:
            tree = ast.parse(path.read_text())
            value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'CAMERA_RESOLUTIONS' for t in n.targets))
            sizes = {k.arg: ast.literal_eval(k.value) for k in value.keywords}
            self.assertEqual(sizes, SIZES)
            products = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == 'render_product']
            self.assertEqual(len(products), 1)
            self.assertEqual(ast.unparse(products[0].args[1]), 'CAMERA_RESOLUTIONS[name]')
            if path != workers[0]:
                masks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == 'instance_summary']
                self.assertEqual(len(masks), 1)
                self.assertEqual({k.arg: ast.literal_eval(k.value) for k in masks[0].keywords},
                                 {'expected_shape': (800, 1280)})

    def test_public_contract_accepts_only_standard_native_frames(self):
        for path in (ATOMIC / 'contract.py', RECOVERY / 'public_contract.py'):
            # Load only the CPU contract, not the dependency-heavy native package.
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.level == 0]
            namespace = {'rp': SimpleNamespace(MOBILE22='robot22', NATIVE_JOINTS=('joint',),
                         NATIVE_UNITS={}, get=lambda _: SimpleNamespace(order=('joint',))),
                         'mobile_base': SimpleNamespace(NAMES=('base',))}
            exec(compile(tree, str(path), 'exec'), namespace)
            def row(sizes):
                images = {n: base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR'
                          + struct.pack('>II', *size) + b'\x08\x02\x00\x00\x00' + b'0000').decode()
                          for n, size in sizes.items()}
                return dict(episode_id='cpu', step=0, time_s=0., instruction='Pick up.',
                    joints={'joint': 0.}, image_step=0, images=images,
                    base_state=dict(position_world_m=[0., 0., 0.], orientation_world_xyzw=[0., 0., 0., 1.],
                        linear_velocity_world_m_s=[0., 0., 0.], angular_velocity_world_rad_s=[0., 0., 0.],
                        yaw_world_unwrapped_rad=0.))
            standard = row(SIZES)
            self.assertEqual(namespace['public_observation'](standard)['images'], standard['images'])
            for wrong in ({n: (640, 480) for n in SIZES}, {n: (1280, 800) for n in SIZES},
                          dict(SIZES, right=(640, 480))):
                with self.assertRaisesRegex(ValueError, '1280x800 / wrist 848x480'):
                    namespace['public_observation'](row(wrong))

    def fixture_policy(self):
        original = dict(schema='fixture', owner='unchanged', physics={'dt': 1/240},
            profiles={'atomic_coffee': {'files': [
                dict(original='/original/worker.py', sha256='a'*64, bytes=10, read_only=True),
                dict(original='/original/scorer.py', sha256='b'*64, bytes=20, read_only=True)]}})
        raw = json.dumps(original).encode()
        selected = dict(base_assembly_sha256=digest(raw), files={
            '/original/worker.py': dict(relative='worker.py',
                before=dict(sha256='a'*64, bytes=10), after=dict(sha256='c'*64, bytes=11))})
        policy = dict(id='fixture-camera', resolutions={k: list(v) for k, v in SIZES.items()},
                      profiles={'atomic_coffee': selected})
        derived = deepcopy(original)
        derived['profiles']['atomic_coffee']['files'][0].update(selected['files']['/original/worker.py']['after'])
        derived['camera_release'] = dict(id=policy['id'], base_assembly_sha256=selected['base_assembly_sha256'],
            resolutions=policy['resolutions'], changed_sources=['/original/worker.py'],
            native_validation_claimed=False, prior_acceptance_scope='upstream_only')
        expected = (json.dumps(derived, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        selected['derived_assembly_sha256'] = digest(expected)
        return policy, raw, expected

    def test_assembly_derivation_preserves_every_other_field_and_original_bytes(self):
        policy, raw, expected = self.fixture_policy()
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'policy', return_value=policy):
            original = Path(temp) / 'upstream.json'; original.write_bytes(raw)
            self.assertEqual(release.derive_assembly(original, 'atomic_coffee'), expected)
            self.assertEqual(original.read_bytes(), raw)
            original.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError, 'exact retained original'):
                release.derive_assembly(original, 'atomic_coffee')

    def test_fixed_derived_hash_cannot_be_refreshed_after_an_unapproved_change(self):
        policy, raw, _ = self.fixture_policy()
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'policy', return_value=policy):
            original = Path(temp) / 'upstream.json'; original.write_bytes(raw)
            policy['profiles']['atomic_coffee']['files']['/original/worker.py']['after']['bytes'] = 12
            with self.assertRaisesRegex(ValueError, 'fixed release identity'):
                release.derive_assembly(original, 'atomic_coffee')

    def test_saved_pins_change_only_known_camera_source_fields(self):
        for family in ('atomic_coffee', 'failure_recovery'):
            changes = release.profile(family)['files']
            code = {p: row['before']['sha256'] for p, row in changes.items()}
            record = next(row for p, row in changes.items() if p.endswith('component_sources.json'))
            old = dict(seed=901, positions={'joint': 1.25}, score={'maximum': 5},
                code_sources=dict(code, **{'/original/scorer.py': 'e'*64}),
                source_layers={'/original/scene.usda': 'd'*64},
                source_record_sha256=record['before']['sha256'],
                unrelated_hash=record['before']['sha256'])
            snapshot = deepcopy(old)
            value, evidence = release.rebind_saved_source_pins(old, family)
            self.assertEqual(old, snapshot)
            self.assertEqual(value['source_layers'], old['source_layers'])
            self.assertEqual(value['unrelated_hash'], old['unrelated_hash'])
            self.assertEqual(value['code_sources']['/original/scorer.py'], 'e'*64)
            self.assertEqual(value['positions'], old['positions'])
            self.assertEqual(value['score'], old['score'])
            self.assertEqual(value['seed'], 901)
            self.assertEqual(len(evidence), len(changes)+1)
            for path, row in changes.items():
                self.assertEqual(value['code_sources'][path], row['after']['sha256'])
            self.assertEqual(value['source_record_sha256'], record['after']['sha256'])
            again, changes_again = release.rebind_saved_source_pins(value, family)
            self.assertEqual(again, value)
            self.assertEqual(changes_again, [])
            old['code_sources'][next(iter(code))] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'approved predecessor'):
                release.rebind_saved_source_pins(old, family)

    def test_unknown_component_record_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'approved camera release predecessor'):
            release.rebind_saved_source_pins({'source_record_sha256': '0'*64}, 'failure_recovery')

    def test_unrecognized_code_file_is_not_rebound_even_if_hash_matches(self):
        row = next(iter(release.profile('atomic_coffee')['files'].values()))
        value = {'code_sources': {'/some/other/worker.py': row['before']['sha256']}}
        result, evidence = release.rebind_saved_source_pins(value, 'atomic_coffee')
        self.assertEqual(result, value)
        self.assertEqual(evidence, [])

    def test_old_source_pin_mismatch_is_not_accepted_by_derived_assembly(self):
        policy, raw, _ = self.fixture_policy()
        policy['profiles']['atomic_coffee']['files']['/original/worker.py']['before']['sha256'] = 'f'*64
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'policy', return_value=policy):
            path = Path(temp) / 'upstream.json'; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'reviewed original'):
                release.derive_assembly(path, 'atomic_coffee')

    def test_restore_rebinds_sidecar_initial_and_launch_then_runs_normal_sealer(self):
        from robotagentbench.simulation.saved_instance_release import restore
        family = 'failure_recovery'
        code_path, code_change = next((p, r) for p, r in release.profile(family)['files'].items() if p.endswith('.py'))
        record = next(r for p, r in release.profile(family)['files'].items() if p.endswith('component_sources.json'))
        class Layer:
            def __init__(self, path): self.path = path
            def TransferContent(self, original): self.data = original.read_bytes()
            def GetCompositionAssetDependencies(self): return []
            def Save(self): Path(self.path).write_bytes(self.data)
        sdf = SimpleNamespace(Layer=SimpleNamespace(CreateNew=Layer, FindOrOpen=lambda path: Path(path)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'pxr': SimpleNamespace(Sdf=sdf)}):
            root = Path(temp); destination = root / 'result'; destination.mkdir()
            scene = root / 'saved.usda'; scene.write_bytes(b'# fixture scene; no SDK\n')
            old_code = {code_path: code_change['before']['sha256'], '/unchanged/scorer.py': 'e'*64}
            old_record = record['before']['sha256']
            sidecar = root / 'sidecar.json'
            sidecar.write_text(json.dumps(dict(code_sources=old_code, source_record_sha256=old_record)))
            state = dict(saved_initialization={'initial': dict(code_sources=old_code,
                source_record_sha256=old_record, scene='/old/scene.usda',
                scene_sha256='a'*64, positions={'joint': .25})},
                manifest_template=dict(initial='/old/initial.json', source_record_sha256=old_record),
                launch_options=dict(task_name='fixture', seed=57, source_record_sha256=old_record),
                task_parameters=dict(task_name='fixture', seed=57))
            selected = dict(state=state, attachments=[
                dict(role='scene_root', original_path='/old/scene.usda', source=str(scene)),
                dict(role='owner_json', original_path='/old/sidecar.json', source=str(sidecar))])
            snapshot = deepcopy(state)
            observed = {}
            def seal(operator, selected_family, task, launch, files, dest):
                observed.update(launch=json.loads(launch.read_text()), files=list(files))
                return {'launch': str(launch)}
            with patch.object(config, 'verify_sources') as verify, patch.object(restore, '_seal_specialized', side_effect=seal) as sealer:
                result = restore._specialized_saved({'selected': selected, 'runtime': root, 'metadata': {}},
                    {}, family, 'fixture', root/'bootstrap.json', destination)
            verify.assert_called_once_with({}, family)
            sealer.assert_called_once()
            launch = observed['launch']
            self.assertEqual(launch['source_record_sha256'], record['after']['sha256'])
            self.assertEqual(launch['seed'], 57)
            initial = json.loads((Path(launch['scene']).parent/'initial.json').read_text())
            self.assertEqual(initial['positions'], {'joint': .25})
            self.assertEqual(initial['code_sources'][code_path], code_change['after']['sha256'])
            self.assertEqual(initial['code_sources']['/unchanged/scorer.py'], 'e'*64)
            owner = json.loads((Path(launch['scene']).parent/'owner_0.json').read_text())
            self.assertEqual(owner['source_record_sha256'], record['after']['sha256'])
            self.assertEqual(owner['code_sources'], initial['code_sources'])
            self.assertEqual(state, snapshot)
            self.assertTrue(result['camera_release']['saved_source_pin_changes'])


if __name__ == '__main__':
    unittest.main()

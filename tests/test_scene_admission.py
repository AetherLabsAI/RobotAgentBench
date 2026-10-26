"""CPU admission boundary tests; no scene generation or native acceptance."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robotagentbench.interfaces.errors import TaskError
from robotagentbench.simulation.admission import admit_prepared_launch, verify_manifest
from robotagentbench.simulation.workspace import NativeWorkspace
from admission_fixtures import AdmissionFixture


class SceneAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = AdmissionFixture(self.temp.name)

    def admit(self):
        f = self.fixture
        return admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)

    def test_original_registry_and_source_checks_run_without_creating_output(self):
        admitted = self.admit()
        self.assertEqual(self.fixture.events, ['resolve', 'bind', 'verify_task_sources', 'verify_scene_sources'])
        self.assertEqual(admitted.plan['score_policy'], {'maximum': 17})
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(admitted.owner_kind, 'standalone')

    def test_unavailable_task_is_not_reinterpreted_as_valid_zero(self):
        self.fixture.plan.update(available=False, unavailable={'message': 'not available',
                                 'kind': 'load', 'reason': 'missing_native_source'})
        with self.assertRaises(TaskError) as error:
            self.admit()
        self.assertEqual(error.exception.reason, 'missing_native_source')
        self.assertEqual(self.fixture.events, ['resolve'])
        self.assertFalse(self.fixture.output.exists())

    def test_gpu_preview_is_not_admitted_to_cpu_owner(self):
        self.fixture.plan['preview'] = True
        with self.assertRaises(TaskError) as error:
            self.admit()
        self.assertEqual(error.exception.reason, 'gpu_cloth_scene_owner_required')

    def test_changed_or_missing_source_is_rejected_without_repinning(self):
        f = self.fixture
        before = f.manifest.read_bytes()
        f.scene.write_text('# changed source\n')
        with self.assertRaises(TaskError) as error:
            self.admit()
        self.assertEqual(error.exception.reason, 'scene_source_changed')
        self.assertEqual(f.manifest.read_bytes(), before)
        with self.assertRaises(TaskError):
            verify_manifest({'source_hashes': {str(f.inputs / 'absent.usda'): 'missing'}})
        with self.assertRaises(TaskError):
            verify_manifest({})

    def test_all_frozen_inputs_are_pinned_until_startup(self):
        f = self.fixture
        for path in (f.launch, f.request, f.manifest, f.ready, f.scene):
            with self.subTest(path=path.name):
                admitted = self.admit()
                before = path.read_bytes()
                path.write_bytes(before + b' ')
                with self.assertRaises(ValueError):
                    admitted.verify_inputs()
                path.write_bytes(before)

    def test_existing_and_outside_output_are_rejected(self):
        f = self.fixture
        f.output.mkdir()
        with self.assertRaises(ValueError):
            self.admit()
        with self.assertRaises(ValueError):
            f.workspace.output(f.root / 'outside')
        with self.assertRaises(ValueError):
            f.workspace.output(f.outputs)

    def test_symlinked_inputs_outputs_and_roots_are_rejected(self):
        f = self.fixture
        linked = f.inputs / 'linked.json'
        linked.symlink_to(f.launch)
        with self.assertRaises(ValueError):
            f.workspace.input(linked)
        output_link = f.outputs / 'linked'
        output_link.symlink_to(f.inputs, target_is_directory=True)
        with self.assertRaises(ValueError):
            f.workspace.output(output_link / 'new')
        root_link = f.root / 'linked_root'
        root_link.symlink_to(f.inputs, target_is_directory=True)
        with self.assertRaises(ValueError):
            NativeWorkspace((root_link,), f.outputs, f.locks)

    def test_wrong_task_identity_fails_before_registry_binding(self):
        f = self.fixture
        launch = json.loads(f.launch.read_text())
        launch['task_name'] = 'other'
        f.launch.write_text(json.dumps(launch))
        with self.assertRaises(ValueError):
            self.admit()
        self.assertEqual(f.events, [])

    def test_room_owner_is_required_and_runtime_request_is_preserved(self):
        f = self.fixture
        f.configure(task='openend', schema='openend.coding_room.launch.v1')
        with self.assertRaises(TaskError):
            self.admit()
        original = f.request.read_bytes()
        f.with_room_backend()
        admitted = self.admit()
        self.assertEqual(admitted.owner_kind, 'room')
        self.assertEqual(admitted.request, {'compiled_components': []})
        self.assertEqual(f.request.read_bytes(), original)
        self.assertEqual(f.events, ['room_admit'])

    def test_special_owners_never_fall_back_to_generic_runtime(self):
        f = self.fixture
        for name, kind in (('recovery_case', 'failure_recovery'), ('atomic_case', 'atomic_coffee')):
            with self.subTest(kind=kind):
                f.configure(task=name)
                with self.assertRaises(TaskError) as error:
                    self.admit()
                self.assertEqual(error.exception.reason, 'special_native_owner_required')
                f.backend = replace(f.backend, exec_special_owner=lambda *args, **kwargs: None)
                admitted = self.admit()
                self.assertEqual(admitted.owner_kind, kind)
                f.backend = replace(f.backend, exec_special_owner=None)

    def test_task_owner_cannot_mutate_frozen_input_files_during_admission(self):
        f = self.fixture
        f.configure(task='openend', schema='openend.coding_room.launch.v1')
        f.with_room_backend()
        original = f.backend.admit_room
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            f.ready.write_text('{"changed": true}')
            return result
        f.backend = replace(f.backend, admit_room=mutate)
        with self.assertRaises(ValueError):
            self.admit()

    def test_special_owner_uses_its_own_reset_without_generic_ready_pose(self):
        f = self.fixture
        f.backend = replace(f.backend, exec_special_owner=lambda *args, **kwargs: None)
        for name in ('atomic_case', 'recovery_case'):
            with self.subTest(task=name):
                f.configure(task=name)
                launch = json.loads(f.launch.read_text())
                del launch['ready_pose']
                f.launch.write_text(json.dumps(launch))
                f.events.clear()
                admitted = self.admit()
                self.assertEqual(f.events, ['resolve', 'special_admit'])
                self.assertNotIn(f.ready, dict(admitted.input_hashes))
                self.assertEqual(set(dict(admitted.input_hashes)),
                                 {f.launch, f.request, f.manifest, f.scene})
                original = f.manifest.read_bytes()
                f.manifest.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    admitted.verify_inputs()
                f.manifest.write_bytes(original)
                self.assertFalse(f.output.exists())

    def test_generic_and_room_owners_still_require_their_ready_pose(self):
        f = self.fixture
        for schema in ('openend.short_task.launch.v1', 'openend.coding_room.launch.v1'):
            with self.subTest(schema=schema):
                f.configure(schema=schema)
                launch = json.loads(f.launch.read_text())
                del launch['ready_pose']
                f.launch.write_text(json.dumps(launch))
                with self.assertRaisesRegex(ValueError, 'ready pose'):
                    self.admit()
                self.assertFalse(f.output.exists())

    def test_unbound_special_owner_does_not_run_legacy_path_admission(self):
        f = self.fixture
        f.configure(task='atomic_case')
        def wrong_environment(*args, **kwargs):
            self.fail('An unbound legacy interpreter must not admit in the generic environment')
        f.registry.admit_launch = wrong_environment
        with self.assertRaises(TaskError) as error:
            self.admit()
        self.assertEqual(error.exception.reason, 'special_native_owner_required')
        self.assertEqual(f.events, ['resolve'])

    def test_output_is_reserved_once_without_overwriting(self):
        f = self.fixture
        f.workspace.create_output(f.output)
        sentinel = f.output / 'sentinel.txt'
        sentinel.write_text('keep')
        with self.assertRaises(ValueError):
            f.workspace.create_output(f.output)
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

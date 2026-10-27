import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotagentbench.simulation.saved_instance_release import original_scene_patch as scene


class OriginalScenePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root/'Runtime'; self.runtime.mkdir()
        self.work = self.root/'work'; self.work.mkdir()
        self.base = self.runtime/'base.usdc'; self.base.write_bytes(b'original base')
        self.delta = self.root/'delta.usdc'; self.delta.write_bytes(b'original delta')
        self.resource = 'libraries/recovery_saved_rooms/base.usdc'
        self.logical = '/original/runs/recovery5_123/base/native/room.usdc'
        self.recipe = dict(schema=scene.SCHEMA, original_path=self.logical,
            original_scene_sha256='c'*64, base_resource=self.resource,
            base_sha256=scene.sha(self.base), delta_sha256=scene.sha(self.delta),
            removed_specs=[],changed_specs=[],borrowed_specs={},root_order=[],root_fields=[],
            child_orders={},property_orders={})
        self.state = dict(source_profile='failure_recovery',task_id='failure_recovery_sofa',
            saved_initialization=dict(initial=dict(source_layers={self.logical:'c'*64})),
            generated_scene_patches=[self.recipe])
        self.metadata = dict(resources={self.resource:str(self.base)},
            library_files={self.resource:scene.sha(self.base)})
        self.attachments=[dict(role='generated_scene_patch',source=str(self.delta),sha256=scene.sha(self.delta))]

    def restore(self):
        return scene.restore_generated_layers(self.state,attachments=self.attachments,
            runtime=self.runtime,metadata=self.metadata,work=self.work)

    def test_old_states_without_patches_remain_supported(self):
        self.state.pop('generated_scene_patches')
        self.assertEqual(self.restore(), ({},{},[]))

    def test_original_source_identity_cannot_change(self):
        self.recipe['original_scene_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'saved initial source'):self.restore()

    def test_uninstalled_shared_base_rejected(self):
        self.metadata['library_files'][self.resource]='d'*64
        with self.assertRaisesRegex(ValueError,'not installed'):self.restore()

    def test_shared_base_must_stay_in_runtime(self):
        outside=self.root/'outside.usdc';outside.write_bytes(self.base.read_bytes())
        self.metadata['resources'][self.resource]=str(outside)
        with self.assertRaisesRegex(ValueError,'leaves the installed'):self.restore()

    def test_missing_or_duplicate_indexed_patch_rejected(self):
        self.attachments=[]
        with self.assertRaisesRegex(ValueError,'exact indexed'):self.restore()

    def test_other_families_are_not_silently_converted(self):
        self.state['task_id']='prepare_coffee'
        with self.assertRaisesRegex(ValueError,'only for'):self.restore()

    def test_only_pinned_members_pass_to_sdf_helper(self):
        def reconstruct(recipe,*,base,delta,output):
            self.assertEqual(base,self.base)
            self.assertEqual(delta,str(self.delta))
            output.write_bytes(b'reconstructed original values')
            return output
        with patch.object(scene,'restore_layer',side_effect=reconstruct) as called:
            paths,hashes,files=self.restore()
        self.assertEqual(called.call_count,1)
        self.assertEqual(paths[self.logical],str(files[0]))
        self.assertEqual(hashes[self.logical],scene.sha(files[0]))

    def test_recipe_rejects_unlisted_borrow(self):
        self.recipe['borrowed_specs']['/unlisted.points']='/original.points'
        with self.assertRaisesRegex(ValueError,'explicitly changed'):scene.validate_recipe(self.recipe)

    def test_hash_change_is_rejected_before_sdk_import(self):
        self.base.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Pinned original'):self.restore()


if __name__=='__main__':unittest.main()

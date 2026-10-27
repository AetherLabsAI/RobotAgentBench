"""CPU-only recovery snapshot binding checks; no sampling, physics or scoring."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from robotagentbench.simulation.saved_instance_release import restore


class SavedRecoverySidecars(unittest.TestCase):
    def test_exact_initial_sibling_keeps_runtime_required_filename_and_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            source=work/'indexed.json'
            snapshot={'source': {'seed': 4}, 'items': [{'prim_path': '/World/target', 'support': 'shelf'}]}
            source.write_text(json.dumps(snapshot))
            original='/old/episode04/initial/source_snapshot.json'
            sidecars={original: {'source': str(source)}}
            before=source.read_bytes()
            paths=restore._specialized_sidecar_paths(sidecars,'/old/episode04/initial/initial.json',work,
                family='failure_recovery',task='failure_recovery_shelf_drink')
            self.assertEqual(paths[original],str(work/'source_snapshot.json'))
            hashes={}
            files=restore._write_saved_json_sidecars(sidecars,paths,hashes)
            self.assertEqual(files,[work/'source_snapshot.json'])
            self.assertEqual(json.loads(files[0].read_text()),snapshot)
            self.assertEqual(hashes[original],hashlib.sha256(files[0].read_bytes()).hexdigest())
            self.assertEqual(source.read_bytes(),before)

    def test_missing_or_other_episode_snapshot_is_rejected_before_native_start(self):
        for sidecars in ({}, {'/old/episode05/initial/source_snapshot.json': {}}):
            with self.assertRaisesRegex(ValueError,'indexed source_snapshot.json sibling'):
                restore._specialized_sidecar_paths(sidecars,'/old/episode04/initial/initial.json',Path('/new'),
                    family='failure_recovery',task='failure_recovery_shelf_drink')

    def test_other_task_bindings_remain_unchanged(self):
        sidecars={'/old/request.json': {}, '/old/manifest.json': {}}
        for family,task in [('atomic_coffee','coffee_button_lighting_interference'),
                            ('failure_recovery','failure_recovery_coffee_bag')]:
            result=restore._specialized_sidecar_paths(sidecars,'/old/initial.json',Path('/new'),
                family=family,task=task)
            self.assertEqual(result,{'/old/request.json': '/new/owner_0.json',
                                     '/old/manifest.json': '/new/owner_1.json'})


if __name__=='__main__':
    unittest.main()

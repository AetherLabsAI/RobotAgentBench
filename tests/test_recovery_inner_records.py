"""Fixed base-task metadata aliases never relax original source verification."""
import hashlib
import json
from pathlib import Path
import unittest

from robotagentbench import backends
from robotagentbench.simulation.dedicated_delivery import source_alias_pin
from robotagentbench.simulation.specialized import camera_release


class RecoveryInnerRecords(unittest.TestCase):
    def test_only_two_exact_metadata_records_change_known_geometry_fields(self):
        changes = camera_release.profile('failure_recovery')['resource_aliases']
        self.assertEqual(len(changes), 2)
        self.assertNotIn('resource_aliases', camera_release.profile('atomic_coffee'))
        profiles = Path(backends.__file__).parent / 'profiles'
        expected_fields = {
            'sofa_to_coffee_table': {'openend/tasks/all_objects_integration/geometry.py',
                                   'short_tasks/tasks/sofa_to_drawer/native_geometry.py'},
            'clear_table_trash': {'openend/tasks/all_objects_integration/geometry.py',
                                 'openend/tasks/trash_transfer/geometry.py'},
        }
        for target, change in changes.items():
            with self.subTest(target=target):
                task = Path(target).parent.name
                self.assertEqual(set(change['source_fields']), expected_fields[task])
                raw = (profiles / change['packaged']).read_bytes()
                self.assertEqual(len(raw), change['after']['bytes'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), change['after']['sha256'])
                data = json.loads(raw)
                original = raw
                for relative, pins in change['source_fields'].items():
                    self.assertEqual(data[change['pin_field']][relative], pins['after'])
                    self.assertEqual(original.count(pins['after'].encode()), 1)
                    original = original.replace(pins['after'].encode(), pins['before'].encode())
                self.assertEqual(len(original), change['before']['bytes'])
                self.assertEqual(hashlib.sha256(original).hexdigest(), change['before']['sha256'])
                if task == 'sofa_to_coffee_table':
                    self.assertEqual(data[change['pin_field']]['openend/tasks/scene_requirements/task_choices.py'],
                                     '8e199a9ad2e87e6c9e1e17d390e0939e2d24d4d443a13f66a612a27d81119856')

    def test_aliases_reject_unknown_predecessors_and_do_not_rebind_other_paths(self):
        changes = camera_release.profile('failure_recovery')['resource_aliases']
        for target, change in changes.items():
            before, after = change['before']['sha256'], change['after']['sha256']
            self.assertEqual(source_alias_pin(target, before, 'failure_recovery'), after)
            for wrong in ('0' * 64, after):
                with self.assertRaisesRegex(ValueError, 'exact reviewed release predecessor'):
                    source_alias_pin(target, wrong, 'failure_recovery')
            self.assertEqual(source_alias_pin('/foreign/component_sources.json', before,
                                             'failure_recovery'), before)
            self.assertEqual(source_alias_pin(target, before, 'atomic_coffee'), before)

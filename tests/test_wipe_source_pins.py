"""The wipe scorer is unchanged; its scene-list dependency follows the approved release list."""
from copy import deepcopy
import hashlib
import json
import unittest

from robotagentbench.backends import sources


class WipeSourcePinsTests(unittest.TestCase):
    def test_sofa_and_storage_tasks_only_rebase_approved_source_pins(self):
        root=sources.root('room');name='openend/tasks/scene_requirements/task_choices.py'
        for task in ('hang_cap','pack_object','retrieve_from_drawer','sofa_to_coffee_table','sofa_to_drawer','table_to_storage_basket'):
            folder=root/'short_tasks/tasks'/task
            old=json.loads((folder/'component_sources.original.json').read_text())
            current=json.loads((folder/'component_sources.json').read_text())
            key=next(k for k in ('sources','files_sha256','source_files')if k in old)
            expected=deepcopy(old)
            expected[key][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            geometry='openend/tasks/all_objects_integration/geometry.py'
            if geometry in expected[key]:
                expected[key][geometry]=hashlib.sha256((root/geometry).read_bytes()).hexdigest()
            self.assertEqual(current,expected)

    def test_only_scene_inventory_dependency_changes(self):
        root = sources.root('room')
        folder = root/'short_tasks/tasks/wipe_table'
        old = json.loads((folder/'component_sources.original.json').read_text())
        current = json.loads((folder/'component_sources.json').read_text())
        expected = deepcopy(old)
        inventory = root/'short_tasks/scene_sources.json'
        expected['sources']['short_tasks/scene_sources.json'] = hashlib.sha256(inventory.read_bytes()).hexdigest()
        self.assertEqual(current, expected)
        for name,pin in current['sources'].items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), pin)

    def test_scene_inventory_only_records_existing_cloth_clear_admission(self):
        root = sources.root('room')
        old = json.loads((root/'short_tasks/scene_sources.original.json').read_text())
        current = json.loads((root/'short_tasks/scene_sources.json').read_text())
        name = 'openend/tasks/scene_requirements/task_choices.py'
        # Preserve the exact original catalogue except for the previously
        # approved cloth_clear entry; no task evaluator or physics is changed.
        raw = (root/name).read_bytes()
        lines = raw.splitlines(keepends=True)
        addition = [line for line in lines if b'dict(type=' in line and b'cloth_clear' in line]
        self.assertEqual(len(addition), 1)
        pins_key = next(key for key,value in old.items() if isinstance(value,dict) and name in value)
        self.assertEqual(hashlib.sha256(b''.join(line for line in lines if line not in addition)).hexdigest(),
                         old[pins_key][name])
        expected = deepcopy(old)
        expected[pins_key][name] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(current, expected)


if __name__ == '__main__':
    unittest.main()

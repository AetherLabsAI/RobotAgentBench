import copy
import unittest

from compose_plan import build


class PlanTests(unittest.TestCase):
    def request(self):
        return {'layout_id': 1, 'seed': 7, 'tasks': [
            {'key': 'coffee.place_cup'},
            {'key': 'microwave.take_plate', 'payload_variant': 'beef_box'}]}

    def test_two_stations(self):
        plan = build(self.request())
        self.assertEqual(len(plan['station_scenarios']), 2)
        self.assertIn('beef_box', plan['assets'])

    def test_dependencies_deduplicate(self):
        request = self.request()
        request['tasks'].append({'key': 'coffee.press_button'})
        plan = build(request)
        self.assertEqual(plan['assets']['coffee_machine']['required_by'],
                         ['coffee.place_cup', 'coffee.press_button'])
        self.assertEqual(sum(s['station'] == 'coffee' for s in plan['station_scenarios']), 1)

    def test_initial_conflict_not_silently_resolved(self):
        request = self.request()
        request['tasks'].append({'key': 'coffee.retrieve_cup'})
        plan = build(request)
        self.assertIn('initial_task_required', [b['code'] for b in plan['blockers']])
        self.assertIsNone(plan['station_scenarios'][0]['initial_state'])

    def test_explicit_initial_scenario(self):
        request = self.request()
        request['tasks'].append({'key': 'coffee.retrieve_cup'})
        request['initial_task_by_station'] = {'coffee': 'coffee.place_cup'}
        plan = build(request)
        self.assertEqual(plan['station_scenarios'][0]['initial_state'], 'empty_cup_on_counter')
        self.assertNotIn('initial_task_required', [b['code'] for b in plan['blockers']])

    def test_put_requires_held_object_prerequisite(self):
        request = self.request()
        request['tasks'][1]['key'] = 'microwave.put_box'
        self.assertIn('held_payload_prerequisite', [b['code'] for b in build(request)['blockers']])

    def test_seed_trace_deterministic(self):
        request = self.request()
        self.assertEqual(build(request), build(copy.deepcopy(request)))
        original = build(request)['seed_trace']
        request['seed'] = 8
        self.assertNotEqual(original, build(request)['seed_trace'])

    def test_not_execution_or_sampling(self):
        plan = build(self.request())
        for flag in ('runtime_ready', 'usd_written', 'random_poses_sampled'):
            self.assertFalse(plan[flag])
        self.assertIsNone(plan['randomization']['new_ranges'])

    def test_control_20hz_is_separate_from_physics(self):
        contract = build(self.request())['control_contract']
        self.assertEqual(contract['frequency_hz'], 20)
        self.assertEqual(contract['period_seconds'], 0.05)
        self.assertEqual(contract['frequency_hz'] * contract['period_seconds'], 1)
        self.assertTrue(contract['independent_of_physics_hz'])
        self.assertFalse(contract['runtime_applied'])

    def test_invalid_requests_rejected(self):
        cases = []
        for field, value in [('layout_id', 2), ('layout_id', True), ('seed', -1),
                             ('seed', True), ('tasks', []), ('unexpected', 1),
                             ('initial_task_by_station', {'other': 'coffee.place_cup'}),
                             ('initial_task_by_station', {'coffee': 'microwave.take_plate'})]:
            request = self.request(); request[field] = value; cases.append(request)
        for task in [{'key': 'unknown'}, {'key': 'microwave.take_plate'},
                     {'key': 'microwave.take_plate', 'payload_variant': 'unknown'},
                     {'key': 'coffee.place_cup', 'payload_variant': 'beef_box'},
                     {'key': 'coffee.place_cup', 'extra': True}]:
            request = self.request(); request['tasks'] = [task]; cases.append(request)
        request = self.request(); request['tasks'].append(request['tasks'][0]); cases.append(request)
        for request in cases:
            with self.subTest(request=request), self.assertRaises(ValueError):
                build(request)

    def test_all_catalog_dependencies_resolve(self):
        import json
        from compose_plan import HERE
        catalog = json.loads((HERE / 'registry.json').read_text())
        for task in catalog['tasks']:
            for variant in task.get('payload_options', [None]):
                selection = {'key': task['key']}
                if variant:
                    selection['payload_variant'] = variant
                request = {'layout_id': 1, 'seed': 7, 'tasks': [selection]}
                plan = build(request)
                self.assertTrue(plan['assets'])


if __name__ == '__main__':
    unittest.main()

import unittest
from copy import deepcopy
from .catalog import CASES, base_request, definition, validate_tipped_history


class FiveCases(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(set(CASES), {'shelf_drink', 'coffee_bag', 'sofa',
            'table_waste_tipped_cup', 'table_waste_empty_grasp'})
        self.assertEqual(sorted(d['number'] for d in CASES.values()), [1, 2, 3, 4, 5])

    def test_independent_copy(self):
        d = definition('coffee_bag'); d['maximum_score'] = 0
        self.assertEqual(definition('coffee_bag')['maximum_score'], 6)

    def test_deferred_cases_not_current(self):
        for name in ('coffee_cup', 'drawer', 'basket', 'coffee_button', 'cutlery'):
            with self.assertRaises(ValueError): definition(name)

    def test_tipped_is_one_cup_original_disposal_request(self):
        q = base_request('table_waste_tipped_cup', 401, waste_categories=['cup'])
        self.assertEqual(q['task_name'], 'clear_table_trash')
        self.assertEqual(q['targets'], [dict(category='cup', quantity=1)])
        self.assertEqual(definition('table_waste_tipped_cup')['destination_role'], 'ordinary_trash_bin')
        for categories in (None, [], ['cup', 'cup'], ['can']):
            with self.assertRaises(ValueError):
                base_request('table_waste_tipped_cup', 401, waste_categories=categories)

    def test_sofa_needs_explicit_existing_asset(self):
        with self.assertRaises(ValueError): base_request('sofa', 401)
        for asset in ('remote_tv', 'book_red', 'wallet', 'glasses_case'):
            q = base_request('sofa', 401, sofa_asset=asset)
            self.assertEqual(q['options'], dict(asset=asset))

    def test_invalid_seed_or_category_refused(self):
        for seed in (0, True, -1, 2**32, '401'):
            with self.assertRaises(ValueError): base_request('table_waste_tipped_cup', seed, waste_categories=['cup'])
        with self.assertRaises(ValueError):
            base_request('table_waste_empty_grasp', 401, waste_categories=['bottle'])

    def test_authored_pose_is_not_a_physical_failure_witness(self):
        with self.assertRaises(ValueError): validate_tipped_history({}, target='/cup', bin_path='/bin')
        h = dict(target='/cup', destination='/bin', synthetic_pose_only=False,
            events=[dict(kind=k, time_s=t) for t,k in enumerate((
                'physical_grasp', 'lift_without_support', 'release_before_bin', 'rest_on_side_on_table'))],
            **{k:True for k in ('measured_contacts','measured_supported_rest','measured_released',
                'visible','inside_workspace','same_dynamic_asset','physics_unchanged')})
        self.assertEqual(validate_tipped_history(h, target='/cup', bin_path='/bin')['initial_points'], 0)
        for field in ('measured_contacts', 'measured_supported_rest', 'measured_released', 'physics_unchanged'):
            bad = deepcopy(h); bad[field] = False
            with self.assertRaises(ValueError): validate_tipped_history(bad, target='/cup', bin_path='/bin')
        bad = deepcopy(h); bad['events'].reverse()
        with self.assertRaises(ValueError): validate_tipped_history(bad, target='/cup', bin_path='/bin')
        with self.assertRaises(ValueError): validate_tipped_history(h, target='/other', bin_path='/bin')


if __name__ == '__main__': unittest.main()

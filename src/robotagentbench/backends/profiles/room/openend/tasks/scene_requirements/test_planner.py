import copy
import unittest
from planner import (compile_plan, verify_placed_inventory, RequirementError,
                     INFRASTRUCTURE, CLOTH_POOL)


def request(*tasks, extras=(), seed=42):
    return dict(schema='openend.scene_requirements.request.v1', seed=seed,
                tasks=list(tasks), extras=list(extras))


def task(typ, name='t', **kw):
    return dict(id=name, type=typ, **kw)


def placed(plan):
    return [dict(instance_id=o['instance_id'], asset_id=o['asset_id'],
                 source=o['source'] or 'sofa', prim_path='/World/Objects/O'+str(i))
            for i, o in enumerate(plan['objects'])]


def roles(plan):
    return list(INFRASTRUCTURE) + [r['role'] for r in plan['scene_dependencies']]


class PlannerTests(unittest.TestCase):
    def test_no_tasks_only_infrastructure(self):
        p = compile_plan(request())
        self.assertEqual(p['objects'], [])
        self.assertEqual({r['role'] for r in p['infrastructure']}, set(INFRASTRUCTURE))

    def test_shared_bag_and_no_unrequested_objects(self):
        p = compile_plan(request(task('wallet_transfer', 'w'), task('glasses_transfer', 'g')))
        self.assertEqual([o['asset_id'] for o in p['objects']], ['wallet', 'glasses_case'])
        bag = next(r for r in p['infrastructure'] if r['role'] == 'fixed_cpu_bag')
        self.assertEqual(bag['required_by'], ['g', 'w'])

    def test_singleton_duplicate_and_conflicting_source(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('wallet_transfer', 'a', source='sofa'),
                                 task('wallet_transfer', 'b', source='small_table')))

    def test_singleton_count(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('wallet_transfer', count=2)))

    def test_extra_singleton_does_not_clone_target(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('wallet_transfer'), extras=[dict(id='e', asset='wallet')]))

    def test_multiple_books_explicit(self):
        p = compile_plan(request(task('book_transfer', count=3, source='sofa')))
        self.assertEqual(len(p['objects']), 3)
        self.assertEqual(len({o['instance_id'] for o in p['objects']}), 3)

    def test_book_default_not_invented(self):
        with self.assertRaisesRegex(RequirementError, 'Book count'):
            compile_plan(request(task('book_transfer')))

    def test_explicit_source_recorded_not_sampled(self):
        p = compile_plan(request(task('wallet_transfer', source='sofa')))
        self.assertEqual(p['objects'][0]['source'], 'sofa')
        self.assertIsNone(p['objects'][0]['pose'])

    def test_unspecified_source_owned_by_scene(self):
        p = compile_plan(request(task('wallet_transfer')))
        self.assertIsNone(p['objects'][0]['source'])

    def test_random_cleanup_counts_200_seeds(self):
        for typ in ('trash_transfer', 'cloth_clear'):
            seen = set()
            for seed in range(200):
                p = compile_plan(request(task(typ), seed=seed))
                n = len(p['objects']); seen.add(n)
                self.assertIn(n, (2, 3, 4))
            self.assertEqual(seen, {2, 3, 4})

    def test_exact_trash_no_top_up(self):
        p = compile_plan(request(task('trash_transfer', items=[dict(kind='paper', count=1)])))
        self.assertEqual(len(p['objects']), 1)

    def test_exact_trash_repeat_same_category(self):
        p = compile_plan(request(task('trash_transfer', items=[dict(kind='paper', count=3), dict(kind='can')])) )
        self.assertEqual([o['trash_category'] for o in p['objects']], ['paper']*3+['can'])

    def test_categories_all_present_only_selected(self):
        for seed in range(50):
            p = compile_plan(request(task('trash_transfer', categories=['paper', 'can']), seed=seed))
            self.assertEqual({o['trash_category'] for o in p['objects']}, {'paper', 'can'})

    def test_four_categories_fit_default(self):
        p = compile_plan(request(task('trash_transfer', categories=['paper', 'can', 'snack', 'cup'])))
        self.assertEqual(len(p['objects']), 4)

    def test_infeasible_category_count(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('trash_transfer', categories=['paper', 'can'], count=1)))

    def test_exact_cloth_and_membership(self):
        p = compile_plan(request(task('cloth_clear', items=[dict(kind='cloth_clear_train', count=2)])))
        self.assertEqual(len(p['objects']), 2)
        self.assertEqual(p['tasks'][0]['target_count'], 2)
        self.assertTrue(all(o['source'] == 'coffee_table.cloth' for o in p['objects']))

    def test_cloth_cannot_start_elsewhere(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('cloth_clear', source='sofa')))

    def test_seed_repeatable_input_unchanged(self):
        r = request(task('trash_transfer')); before = copy.deepcopy(r)
        self.assertEqual(compile_plan(r), compile_plan(r))
        self.assertEqual(r, before)

    def test_independent_random_stream(self):
        a = compile_plan(request(task('trash_transfer', 'trash')))
        b = compile_plan(request(task('wallet_transfer', 'wallet'), task('trash_transfer', 'trash')))
        self.assertEqual(a['objects'], [o for o in b['objects'] if o['required_by'] == ['trash']])

    def test_extra_without_task_not_scored(self):
        p = compile_plan(request(extras=[dict(id='e', asset='paper_cup', count=2)]))
        self.assertEqual(len(p['objects']), 2)
        self.assertTrue(all(o['required_by'] == [] for o in p['objects']))
        self.assertEqual(p['tasks'], [])

    def test_background_cup_same_table_ambiguous(self):
        with self.assertRaisesRegex(RequirementError, 'Ambiguous'):
            compile_plan(request(task('trash_transfer', source='coffee_table'),
                                 extras=[dict(id='bg', asset='paper_cup', source='coffee_table.objects')]))

    def test_hidden_packaging_label_cannot_exempt(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('trash_transfer', source='snack_shelf'),
                                 extras=[dict(id='bg', asset='zbar_t', source='snack_shelf')]))

    def test_separate_visible_sources_allowed(self):
        p = compile_plan(request(task('trash_transfer', source='coffee_table.objects'),
                                 extras=[dict(id='bg', asset='paper_cup', source='small_table')]))
        self.assertEqual(p['objects'][-1]['required_by'], [])

    def test_generic_cloth_includes_other_cloth_objects(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('cloth_clear'), extras=[dict(id='bg', asset='book_red', source='coffee_table.cloth')]))

    def test_random_source_ambiguity_deferred_not_passed(self):
        p = compile_plan(request(task('cloth_clear'), task('book_transfer', 'b', count=2)))
        self.assertTrue(p['unresolved_semantic_pairs'])
        rows = placed(p)
        for r in rows:
            r['source'] = 'coffee_table.cloth'
        with self.assertRaises(RequirementError):
            verify_placed_inventory(p, rows, roles(p))

    def test_duplicate_ids(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('wallet_transfer'), task('glasses_transfer')))

    def test_unsupported_task_explicit_failure(self):
        with self.assertRaisesRegex(RequirementError, 'No inventory adapter'):
            compile_plan(request(task('coffee')))

    def test_strict_fields_and_counts(self):
        for kw in (dict(count=True), dict(count=0), dict(source='typo'), dict(hidden_target=True)):
            with self.subTest(kw=kw), self.assertRaises(RequirementError):
                compile_plan(request(task('wallet_transfer', **kw)))

    def test_exact_and_random_conflict(self):
        with self.assertRaises(RequirementError):
            compile_plan(request(task('trash_transfer', items=[dict(kind='cup')], count=3)))

    def test_scene_gate_exact_manifest_only(self):
        p = compile_plan(request(task('wallet_transfer', source='sofa')))
        receipt = verify_placed_inventory(p, placed(p), roles(p))
        self.assertTrue(receipt['semantic_binding_checked'])
        self.assertFalse(receipt['physics_verified'])

    def test_scene_missing_target(self):
        p = compile_plan(request(task('wallet_transfer')))
        with self.assertRaises(RequirementError):
            verify_placed_inventory(p, [], roles(p))

    def test_scene_unrequested_objects(self):
        p = compile_plan(request(task('wallet_transfer')))
        rows = placed(p) + [dict(instance_id='old_room_cup', asset_id='paper_cup', source='sofa', prim_path='/World/Extra')]
        with self.assertRaises(RequirementError):
            verify_placed_inventory(p, rows, roles(p))

    def test_scene_wrong_source_or_asset(self):
        p = compile_plan(request(task('wallet_transfer', source='sofa')))
        for key, value in [('source', 'small_table'), ('asset_id', 'paper_cup'), ('source', None)]:
            rows = placed(p); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(RequirementError):
                verify_placed_inventory(p, rows, roles(p))

    def test_missing_bag_and_bins_even_no_tasks(self):
        p = compile_plan(request())
        with self.assertRaises(RequirementError):
            verify_placed_inventory(p, [], [])

    def test_duplicate_scene_paths(self):
        p = compile_plan(request(task('book_transfer', count=2)))
        rows = placed(p); rows[1]['prim_path'] = rows[0]['prim_path']
        with self.assertRaises(RequirementError):
            verify_placed_inventory(p, rows, roles(p))

    def test_not_runtime_or_physical_result(self):
        p = compile_plan(request(task('cloth_clear')))
        self.assertFalse(p['executable'])
        self.assertFalse(p['checks']['runtime_binding'])
        self.assertEqual(len(CLOTH_POOL), 20)


if __name__ == '__main__':
    unittest.main()

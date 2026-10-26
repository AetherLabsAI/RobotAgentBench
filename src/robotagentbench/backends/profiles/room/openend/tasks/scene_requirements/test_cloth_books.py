import copy
import unittest
from types import SimpleNamespace
from planner import compile_plan,RequirementError
from room_request import resolve

def request(n=2):
    return dict(schema='openend.scene_requirements.request.v2',seed=42,layout_id=48,
                tasks=[dict(id='cloth_clear',type='cloth_clear',items=[dict(kind='book_red',count=n)])],extras=[])

class ClothBooks(unittest.TestCase):
    def test_books_bound_in_one_cloth_group(self):
        from openend.tasks.cloth_clear.test_registration import fixture
        from openend.tasks.cloth_clear.registration import resolve_request
        r,m=fixture();m['cloth_clear']['asset_pool_ids'].append('book_red')
        for o in m['objects'][:2]:o['id']='book_red'
        group,=resolve_request(r,m)
        self.assertEqual(group['target_count'],2)
        self.assertEqual(group['maximum_score'],6)
        self.assertEqual(group['stage_points'],[1,1,1])
    def test_unapproved_asset_and_duplicate_pool_rejected(self):
        from openend.tasks.cloth_clear.test_registration import fixture
        from openend.tasks.cloth_clear.registration import resolve_request
        for extras in (['wallet'],['book_red','book_red']):
            r,m=fixture();m['cloth_clear']['asset_pool_ids']+=extras
            with self.assertRaises(ValueError):resolve_request(r,m)
    def test_both_task_registrations_claim_different_actual_books(self):
        from openend.tasks.cloth_clear.test_registration import fixture
        from openend.room import prepare_registration
        r,m=fixture();m['cloth_clear']['asset_pool_ids'].append('book_red')
        for o in m['objects'][:2]:o['id']='book_red'
        m['objects'].append(dict(id='book_red',path='/World/SofaBook'))
        r['tasks'].append(dict(instance_id='book',type='book_transfer',parameters={}))
        result=prepare_registration(r,m)
        self.assertEqual(result['resolved_objects']['book'],'/World/SofaBook')
        self.assertNotIn('/World/SofaBook',result['cloth_clear_groups'][0]['target_paths'])
    def test_exact_books_one_group_distinct_instances(self):
        for n in range(1,5):
            p=resolve(request(n))['requirements']['portable_inventory']
            self.assertEqual([o['asset_id'] for o in p['objects']],['book_red']*n)
            self.assertEqual({o['source'] for o in p['objects']},{'coffee_table.cloth'})
            self.assertEqual(len({o['instance_id'] for o in p['objects']}),n)
            self.assertEqual([t['type'] for t in p['tasks']],['cloth_clear'])
    def test_limit_and_no_count_items_ambiguity(self):
        for n in (0,5,True,1.5):
            with self.assertRaises(ValueError):resolve(request(n))
        r=request();r['tasks'][0]['count']=2
        with self.assertRaises(ValueError):resolve(r)
    def test_mixed_exact_inventory(self):
        r=request();r['tasks'][0]['items'].append(dict(kind='cloth_clear_eraser'))
        p=resolve(r)['requirements']['portable_inventory']
        self.assertEqual([o['asset_id'] for o in p['objects']],['book_red','book_red','cloth_clear_eraser'])
    def test_default_random_library_unchanged(self):
        for seed in range(20):
            r=request();r['seed']=seed+1;r['tasks'][0].pop('items')
            p=resolve(r)['requirements']['portable_inventory']
            self.assertTrue(2<=len(p['objects'])<=4)
            self.assertTrue(all(o['asset_id'].startswith('cloth_clear_') for o in p['objects']))
    def test_separate_book_task_source_can_coexist(self):
        r=request();r['tasks'].append(dict(id='book_transfer',type='book_transfer',source='sofa',count=1))
        p=resolve(r)['requirements']['portable_inventory']
        self.assertEqual(len(p['objects']),3)
        self.assertEqual([o['source'] for o in p['objects']],['coffee_table.cloth']*2+['sofa'])
    def test_port_filters_only_cloth_books_without_mutating_scene_inventory(self):
        from openend.tasks.all_objects_integration.ports import BookPort
        from openend.tasks.object_transfers.test_logic import Driver
        from openend.tasks.object_transfers.binding import bind
        d=Driver('book_to_small_table')
        manifest=copy.deepcopy(d.manifest)
        manifest['objects'] += [dict(id='book_red',path='/World/ClothBook'+str(i),task_role='cloth_clear_target') for i in range(2)]
        before=copy.deepcopy(manifest)
        self.assertEqual(BookPort.resolve_object({},manifest)['path'],d.b.target)
        view=BookPort.binding_inventory(SimpleNamespace(manifest=manifest))
        self.assertEqual(bind(view,**d.kw).target,d.b.target)
        self.assertEqual(manifest,before)
        manifest['objects'].append(dict(id='book_red',path='/World/ExtraUnassignedBook'))
        with self.assertRaises(ValueError):BookPort.resolve_object({},manifest)

if __name__=='__main__':unittest.main()

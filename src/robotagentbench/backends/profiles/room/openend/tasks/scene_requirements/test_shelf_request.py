import unittest
from room_request import resolve
from task_choices import SHELF_PRODUCTS,CHOICES

class ShelfRequests(unittest.TestCase):
    def request(self,product):
        return dict(schema='openend.scene_requirements.request.v2',seed=42,layout_id=26,tasks=[dict(id='shelf',type='snack_drink_to_tray',product=product)])
    def test_all_fourteen(self):
        self.assertEqual(len(SHELF_PRODUCTS),14)
        self.assertTrue(any(c['type']=='snack_drink_to_tray' for c in CHOICES))
        for key,_ in SHELF_PRODUCTS:
            result=resolve(self.request(key))['requirements']
            self.assertEqual(result['shelf_task_requirements'][0]['product'],key)
            self.assertEqual(result['portable_inventory']['objects'],[])
    def test_bad_products(self):
        for product in ('unknown','',None,12):
            with self.assertRaises(ValueError):resolve(self.request(product))
    def test_exact_parameters(self):
        request=self.request('coke_c3');request['tasks'][0]['count']=2
        self.assertEqual(len(resolve(request)['requirements']['shelf_task_requirements']),2)
        request['tasks'][0]['unexpected']=True
        with self.assertRaises(ValueError):resolve(request)
    def test_existing_tasks_coexist(self):
        request=self.request('coke_c3')
        request['tasks'] += [dict(id='fridge',type='fridge_to_microwave',payload_variant='beef_box'),dict(id='wallet',type='wallet_transfer',source='coffee_table.drawer')]
        result=resolve(request)['requirements']
        self.assertEqual(len(result['fixed_task_requirements']),1)
        self.assertEqual(len(result['shelf_task_requirements']),1)
        self.assertEqual(len(result['portable_inventory']['objects']),1)
    def test_four_categories(self):
        request=self.request('coke_c3')
        request['tasks']=[dict(id='shelf_'+p,type='snack_drink_to_tray',product=p) for p in ('coke_c3','gatorade_m','oreo_gf','zbar_t')]
        self.assertEqual(len(resolve(request)['requirements']['shelf_task_requirements']),4)
        request['tasks'].append(dict(id='fifth',type='snack_drink_to_tray',product='sprite_c1'))
        with self.assertRaisesRegex(ValueError,'最多'):resolve(request)
    def test_repeated_category_independent_ids(self):
        request=self.request('coke_c3');request['tasks'].append(dict(id='another',type='snack_drink_to_tray',product='coke_c3'))
        rows=resolve(request)['requirements']['shelf_task_requirements']
        self.assertEqual([r['id'] for r in rows],['shelf','another'])

if __name__=='__main__':unittest.main()

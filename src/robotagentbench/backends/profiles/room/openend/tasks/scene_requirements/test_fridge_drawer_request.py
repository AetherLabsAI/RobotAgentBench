import tempfile
import unittest
from pathlib import Path
from room_request import resolve
from generate_room import choose_plan
from task_choices import CHOICES

def req(*tasks):
    return dict(schema='openend.scene_requirements.request.v2',seed=42,layout_id=26,tasks=list(tasks))

class FridgeDrawerTests(unittest.TestCase):
    def test_current_catalog_choices_have_runtime(self):
        from openend.room import FACTORIES
        kinds=[row['type'] for row in CHOICES]
        self.assertEqual(len(kinds),len(set(kinds)))
        self.assertTrue({'fridge_to_microwave','glasses_transfer','wallet_transfer'}<=set(kinds))
        self.assertTrue(set(kinds)<=set(FACTORIES))
    def test_fridge_explicit_variants(self):
        for variant in ('beef_box','leftover_box'):
            r=resolve(req(dict(id='cold',type='fridge_to_microwave',payload_variant=variant)))
            self.assertEqual(r['requirements']['fixed_task_requirements'][0]['payload_variant'],variant)
            self.assertEqual(r['requirements']['portable_inventory']['objects'],[])
            self.assertEqual(r['requirements']['station_request']['stations'],[])
    def test_fridge_reject_unknown_missing_or_placement(self):
        for fields in ({},{'payload_variant':'salad_box'},{'payload_variant':'beef_box','placement':{}}):
            with self.assertRaises(ValueError):resolve(req(dict(id='cold',type='fridge_to_microwave',**fields)))
    def test_each_drawer_variant_exactly_one_not_on_table(self):
        for kind in ('glasses_transfer','wallet_transfer'):
            with tempfile.TemporaryDirectory() as d:
                p=choose_plan(resolve(req(dict(id='pick',type=kind,source='coffee_table.drawer'))),Path(d))
            self.assertEqual(len(p['living_plan']['objects']),1)
            self.assertEqual(p['living_plan']['objects'][0]['source'],'coffee_table.drawer')
            self.assertEqual(p['living_plan']['small_table'],[])
            self.assertEqual(p['living_plan']['sofa'],[])
    def test_drawer_two_objects_rejected(self):
        with self.assertRaisesRegex(ValueError,'最多一个'):
            resolve(req(*[dict(id=k,type=k,source='coffee_table.drawer') for k in ('glasses_transfer','wallet_transfer')]))
    def test_other_drawer_source_rejected(self):
        with self.assertRaises(ValueError):resolve(req(dict(id='r',type='remote_transfer',source='coffee_table.drawer')))
    def test_mixed_room(self):
        p=resolve(req(dict(id='cold',type='fridge_to_microwave',payload_variant='beef_box'),
            dict(id='wallet',type='wallet_transfer',source='coffee_table.drawer'),dict(id='glasses',type='glasses_transfer',source='sofa')))
        self.assertEqual(len(p['requirements']['portable_inventory']['objects']),2)
        self.assertEqual(len(p['requirements']['fixed_task_requirements']),1)

if __name__=='__main__':unittest.main()

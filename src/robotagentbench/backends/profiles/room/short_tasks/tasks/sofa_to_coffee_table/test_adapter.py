import json
import unittest
from . import adapter


class CombinationTests(unittest.TestCase):
    def test_four_explicit_variants(self):
        for asset,points in [('book_red',8),('remote_tv',6),('wallet',6),('glasses_case',6)]:
            p=adapter.resolve(dict(task_name='sofa_to_coffee_table',seed=42,options={'asset':asset}))
            self.assertTrue(p['available'])
            self.assertEqual(p['score_policy']['maximum'],points)
            self.assertEqual(p['destination'],'coffee_table.objects')
            manifest={'objects':[dict(id=asset,path='/World/Target',source='sofa',task_id=p['task_name'])]}
            self.assertEqual(len(adapter.bind(p,manifest)),1)
            self.assertEqual(p['target_bindings'][0]['destination_path'],'/World/Assets/CoffeeTable')
            manifest['objects'][0]['task_id']='background'
            with self.assertRaises(ValueError):adapter.bind(p,manifest)

    def test_unknown_asset_and_wrong_destination_rejected(self):
        for options in [dict(asset='cat'),dict(destination='tv_cabinet')]:
            with self.assertRaises(ValueError):
                adapter.resolve(dict(task_name='sofa_to_coffee_table',seed=42,options=options))


if __name__ == "__main__": unittest.main()

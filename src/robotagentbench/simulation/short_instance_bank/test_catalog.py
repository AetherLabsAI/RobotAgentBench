"""Small saved-artifact checks, not simulated preparation tests."""
import json
from pathlib import Path
import tempfile
import unittest

from . import bank,catalog


class SavedCatalogTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).parent/'private';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)

    def tearDown(self):self.temp.cleanup()

    def write(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return p

    def index(self,seed=31):
        launch=self.write('launch.json',dict(task_name='prepare_coffee'))
        row=dict(task_name='prepare_coffee',seed=seed,status='prepared_cpu_not_native_accepted',
                 launch=str(launch),launch_sha256=bank.sha(launch),output_hashes={str(launch):bank.sha(launch)},
                 physical_configuration=dict(objects=[dict(matrix=[1,0,0,1])],task_bindings=[],initial_base_xyyaw=[0,0,0]))
        return self.write('index.json',dict(schema=bank.SCHEMA,instances=[row]))

    def test_actual_seed_separate_from_ordinal(self):
        result=catalog.collect([self.index()])
        row=result['instances'][0]
        self.assertEqual((row['actual_seed'],row['episode_ordinal']),(31,1))
        self.assertFalse(result['all_30_distinct_valid']);self.assertFalse(result['resampled'])
        path=self.write('catalog.json',result)
        self.assertEqual(catalog.load(path,'prepare_coffee',1)['seed'],31)

    def test_tampering_is_rejected(self):
        index=self.index();(self.root/'launch.json').write_text('changed')
        with self.assertRaises(ValueError):catalog.collect([index])

    def test_duplicate_seed_paths_do_not_increase_count(self):
        index=self.index();result=catalog.collect([index,index])
        self.assertEqual(len(result['instances']),1)

    def test_layout_label_not_a_physical_dimension(self):
        self.assertEqual(bank.semantic_fingerprint(dict(layout_id=1,objects=[1])),
                         bank.semantic_fingerprint(dict(layout_id=50,objects=[1])))


if __name__=='__main__':unittest.main()

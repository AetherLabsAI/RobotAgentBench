import unittest
from planner import RequirementError
from planner_v2 import compile_request


def request(*tasks):
    return dict(schema='openend.scene_requirements.request.v2', layout_id=1, seed=42, tasks=list(tasks))


class StationEnvelopeTests(unittest.TestCase):
    def test_station_and_portable_separate_no_fabricated_instances(self):
        p = compile_request(request(dict(id='c', type='coffee', placement=dict(surface_group='island')),
                                    dict(id='w', type='wallet_transfer')))
        self.assertEqual(len(p['station_request']['stations']), 1)
        self.assertEqual([o['asset_id'] for o in p['portable_inventory']['objects']], ['wallet'])
        self.assertFalse(p['assembly_requirements'][0]['asset_instance_expansion_complete'])

    def test_duplicate_cross_family_id(self):
        with self.assertRaises(RequirementError):
            compile_request(request(dict(id='same', type='coffee'), dict(id='same', type='wallet_transfer')))

    def test_duplicate_device(self):
        with self.assertRaises(RequirementError):
            compile_request(request(dict(id='a', type='coffee'), dict(id='b', type='coffee')))

    def test_extra_id_conflict(self):
        r=request(dict(id='a', type='coffee')); r['extras']=[dict(id='a', asset='wallet')]
        with self.assertRaises(RequirementError):compile_request(r)

    def test_device_cannot_move_component_separately(self):
        with self.assertRaises(RequirementError):
            compile_request(request(dict(id='c', type='coffee', source='sofa')))

    def test_portable_not_device_pose(self):
        with self.assertRaises(RequirementError):
            compile_request(request(dict(id='w', type='wallet_transfer', placement={})))

    def test_existing_seed_contract(self):
        r=request(dict(id='c', type='coffee'));r['seed']=0
        with self.assertRaises(RequirementError):compile_request(r)

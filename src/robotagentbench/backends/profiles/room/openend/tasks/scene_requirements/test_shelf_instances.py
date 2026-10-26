"""CPU registration/scoring contracts; not real manipulation evidence."""
import unittest
from copy import deepcopy
from dataclasses import replace
from .shelf_instances import KIND, peer_bodies, scoped_pick
from .shelf_composite import Binding, Frame, SnackTrayComposite
from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
from evaluation.tasks.coffee_table_place.process_rules import HeldFacts
from openend.room import validate_request, prepare_registration, resolve_objects

PRODUCTS = ('coke_c3', 'gatorade_m', 'oreo_gf', 'zbar_t')


def fixture(n=4):
    tasks = [dict(instance_id='shelf_'+p, type=KIND, parameters={'product': p}) for p in PRODUCTS[:n]]
    request = dict(tasks=tasks, no_progress_timeout_s=120, selection_order='front_then_distance')
    objects = [dict(id=p, path='/World/'+p, task_role='requested_full_product', task_id='shelf_'+p)
               for p in PRODUCTS[:n]]
    objects.append(dict(id='coke_c3', path='/World/Background', task_role='background_stock'))
    manifest = dict(objects=objects, registration={'request': request})
    mapping = {o['path']: o['path']+'/Asset' for o in objects}
    return request, manifest, mapping


def frame(c, held=False, placed=False, others=frozenset()):
    n = 0 if c.last is None else c.last.pick.sample_index+1
    t = 0. if c.last is None else c.last.pick.time_s+1/60
    pick = PhysicalFrame('episode', n, t, held, held or placed, .1 if held else 0.,
                         frozenset(), others, False, False, active_target=c.binding.candidates[0])
    place = HeldFacts(n, t, 1., held, placed, placed, (1.,0.,0.,0.),
                      (0.,0.,0.), (1.,0.,0.,0.), False, False)
    return Frame(c.binding, pick, place, 1/60)


class ShelfInstances(unittest.TestCase):
    def test_one_two_four_register(self):
        for n in (1, 2, 4):
            request, manifest, _ = fixture(n)
            result = prepare_registration(request, manifest)
            self.assertEqual(len(result['resolved_objects']), n)
            self.assertEqual(set(result['resolved_objects'].values()), {'/World/'+p for p in PRODUCTS[:n]})

    def test_allow_same_product_but_reject_duplicate_id(self):
        request, _, _ = fixture(2)
        repeated = deepcopy(request); repeated['tasks'][1]['parameters'] = repeated['tasks'][0]['parameters']
        validate_request(repeated)
        repeated['tasks'][1]['instance_id'] = repeated['tasks'][0]['instance_id']
        with self.assertRaises(ValueError): validate_request(repeated)

    def test_fifth_rejected(self):
        request, _, _ = fixture()
        request['tasks'].append(dict(instance_id='fifth', type=KIND, parameters={'product':'sprite_c1'}))
        with self.assertRaises(ValueError): validate_request(request)

    def test_other_task_duplicates_still_rejected(self):
        request, _, _ = fixture(1)
        request['tasks'] = [dict(instance_id=k, type='coffee', parameters={}) for k in ('a','b')]
        with self.assertRaises(ValueError): validate_request(request)

    def test_same_body_still_rejected(self):
        request, manifest, _ = fixture(2)
        manifest['objects'][1]['path'] = manifest['objects'][0]['path']
        with self.assertRaises(ValueError): resolve_objects(validate_request(request), manifest)

    def test_peers_require_registration_and_exact_body(self):
        _, manifest, mapping = fixture()
        peers = peer_bodies(manifest, 'shelf_coke_c3', 'coke_c3', mapping)
        self.assertEqual(peers, frozenset('/World/'+p+'/Asset' for p in PRODUCTS[1:]))
        self.assertNotIn('/World/Background/Asset', peers)
        bad = deepcopy(manifest); bad['objects'][1]['task_id']='unregistered'
        with self.assertRaises(ValueError): peer_bodies(bad,'shelf_coke_c3','coke_c3',mapping)
        bad = deepcopy(manifest); bad['registration']['request']['tasks']=bad['registration']['request']['tasks'][:1]
        self.assertEqual(peer_bodies(bad,'shelf_coke_c3','coke_c3',mapping), frozenset())

    def test_missing_body_and_duplicate_body_fail_closed(self):
        _, manifest, mapping = fixture(2)
        del mapping['/World/gatorade_m']
        with self.assertRaises(ValueError): peer_bodies(manifest,'shelf_coke_c3','coke_c3',mapping)
        mapping['/World/gatorade_m']=mapping['/World/coke_c3']
        with self.assertRaises(ValueError): peer_bodies(manifest,'shelf_coke_c3','coke_c3',mapping)

    def test_four_scores_independent_no_group_completion(self):
        _, manifest, mapping = fixture()
        scorers = [SnackTrayComposite(Binding('episode','shelf_'+p,p,('/World/'+p+'/Asset',),
                    '/World/Board','/World/Tray'),room_managed=True) for p in PRODUCTS]
        for c in scorers: c.observe(frame(c))
        for i,c in enumerate(scorers):
            peers=peer_bodies(manifest,c.binding.task_instance_id,c.binding.product,mapping)
            # A different legitimately requested object can be held concurrently.
            for _ in range(182):
                raw=frame(c,held=True,others=peers)
                c.observe(replace(raw,pick=scoped_pick(raw.pick,peers)))
                if c.phase!='pick': break
            self.assertEqual(c.summary()['score'],4)
            for _ in range(80):
                if c.ended: break
                c.observe(frame(c,placed=True))
            self.assertTrue(c.success);self.assertEqual(c.summary()['score'],7)
            self.assertTrue(all(x.summary()['score']==0 and not x.success for x in scorers[i+1:]))
        self.assertEqual(sum(c.summary()['score'] for c in scorers),28)

    def test_peer_grasp_no_own_reward_background_still_wrong(self):
        c=SnackTrayComposite(Binding('episode','shelf_coke_c3','coke_c3',('/World/coke_c3/Asset',),
                            '/World/Board','/World/Tray'),room_managed=True)
        c.observe(frame(c))
        peer='/World/gatorade_m/Asset'
        raw=frame(c,others=frozenset([peer]))
        c.observe(replace(raw,pick=scoped_pick(raw.pick,frozenset([peer]))))
        self.assertFalse(c.ended);self.assertEqual(c.summary()['score'],0)
        self.assertEqual(raw.pick.wrong_drinks_grasped,frozenset([peer]))
        raw=frame(c,others=frozenset(['/World/Background/Asset']))
        c.observe(replace(raw,pick=scoped_pick(raw.pick,frozenset([peer]))))
        self.assertTrue(c.ended);self.assertFalse(c.success)

    def test_collision_facts_unchanged(self):
        c=SnackTrayComposite(Binding('episode','shelf_coke_c3','coke_c3',('/World/coke_c3/Asset',),
                            '/World/Board','/World/Tray'),room_managed=True)
        raw=replace(frame(c).pick,neighbor_contacts=frozenset(['/World/Peer']))
        self.assertEqual(scoped_pick(raw,frozenset(['/World/Peer'])).neighbor_contacts,raw.neighbor_contacts)


if __name__=='__main__': unittest.main()

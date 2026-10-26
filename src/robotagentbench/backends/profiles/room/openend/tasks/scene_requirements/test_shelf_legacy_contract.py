"""Synthetic CPU contract tests, NOT a physical trajectory acceptance."""
from dataclasses import replace
import math
import unittest
import importlib.util
if importlib.util.find_spec('openend') is None or importlib.util.find_spec('numpy') is None:
    raise unittest.SkipTest('Deployed 202 dependencies required')
from arena_api.physics_defaults import PHYSICS_DT as DT
from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
from evaluation.tasks.coffee_table_place.process_rules import HeldFacts
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS, normalize_parameters, parameter_schema, scene_requirements
from .shelf_composite import Binding, Frame, SnackTrayComposite


def binding(product='coke_c3'):
    return Binding('episode', 'task', product, ('/World/Target', '/World/OtherEligible'),
                   '/World/Shelf/Board', '/World/WoodenTray')


def fact(b, n, held=False, separated=False, placed=False, floor=False, outside=False,
         target='/World/Target', height=1., q=(1., 0., 0., 0.), wrong=frozenset(), shelf=False):
    return Frame(b, PhysicalFrame(b.episode_id, n, n*DT, held, separated,
        .1 if separated else 0., frozenset(), wrong, floor, outside,
        shelf_contact=shelf, active_target=target),
        HeldFacts(n, n*DT, height, held, placed, placed, q, (0., 0., 0.), q, floor, outside))


def picked(product='coke_c3'):
    c=SnackTrayComposite(binding(product))
    c.observe(fact(c.binding, 0))
    for n in range(1, 800):
        c.observe(fact(c.binding, n, held=True, separated=True))
        if c.phase=='carry_and_place': return c
    raise AssertionError('Original 3s hold did not hand off')


def finish(c, q=(1.,0.,0.,0.)):
    for _ in range(300):
        n=c.last.pick.sample_index+1
        c.observe(fact(c.binding, n, separated=True, placed=True, q=q))
        if c.ended: return c.summary()
    raise AssertionError('Original settle did not complete')


class Contracts(unittest.TestCase):
    def test_catalog(self):
        self.assertEqual(len(PRODUCTS),14)
        for i,(key,p) in enumerate(PRODUCTS.items(),1):
            self.assertEqual((p.pick_api,p.place_api),(f'4-{i}',f'5-{i}'))
            self.assertEqual(normalize_parameters({'product':key}),{'product':key})
            r=scene_requirements({'product':key})
            self.assertEqual(r['required_categories'],[key])
            self.assertFalse(r['create_objects'])
            self.assertEqual(r['destination_role'],'coffee_table_wooden_tray')
        schema=parameter_schema(); schema['properties']['product']['enum'].clear()
        self.assertEqual(len(parameter_schema()['properties']['product']['enum']),14)

    def test_bad_parameters(self):
        for p in ({},None,{'product':'unknown'},{'product':[]},{'product':'coke_c3','count':2}):
            with self.subTest(p=p),self.assertRaises(ValueError):normalize_parameters(p)

    def test_all_products_same_complete_pipeline(self):
        for product in PRODUCTS:
            with self.subTest(product=product):
                c=picked(product)
                self.assertFalse(c.success)
                self.assertEqual(c.summary()['score'],4)
                s=finish(c)
                self.assertTrue(s['success']);self.assertEqual(s['score'],7)
                self.assertEqual(len(s['transitions']),1)
                self.assertFalse(s['transitions'][0]['scene_reset'])
                self.assertFalse(s['runtime_registered']);self.assertFalse(s['physics_verified'])

    def test_invalid_samples_transactional(self):
        c=SnackTrayComposite(binding());c.observe(fact(c.binding,0))
        good=fact(c.binding,1,held=True,separated=True)
        bad=[replace(good,binding=replace(c.binding,product='pring_bbq')),
             fact(c.binding,2),fact(c.binding,0),
             replace(good,pick=replace(good.pick,time_s=float('nan'))),
             replace(good,pick=replace(good.pick,complete=False)),
             replace(good,place=replace(good.place,grasped=False)),
             replace(good,place=replace(good.place,quaternion_wxyz=(0.,0.,0.,0.)))]
        before=c.summary()
        for f in bad:
            with self.assertRaises(ValueError):c.observe(f)
            self.assertEqual(c.summary(),before)
        c.observe(good)
        with self.assertRaises(ValueError):c.observe(fact(c.binding,2,held=True,separated=True,target='/World/OtherEligible'))

    def test_identity_and_facts_after_handoff(self):
        c=picked();n=c.last.pick.sample_index+1
        f=fact(c.binding,n,held=True,separated=True)
        for bad in (replace(f,pick=replace(f.pick,complete=1)),
                    replace(f,pick=replace(f.pick,lift_m=float('nan'))),
                    fact(c.binding,n,held=True,separated=True,target='/World/OtherEligible')):
            with self.assertRaises(ValueError):c.observe(bad)
        c.observe(f)

    def test_initial_preheld_allowed(self):
        c=SnackTrayComposite(binding())
        s=c.observe(fact(c.binding,0,held=True))
        self.assertFalse(s['ended']);self.assertEqual(s['score'],2)
        self.assertFalse(s['initial_state_gate_enabled'])

    def test_wrong_grasp_fatal(self):
        c=SnackTrayComposite(binding());c.observe(fact(c.binding,0))
        s=c.observe(fact(c.binding,1,wrong=frozenset(['/World/Wrong'])))
        self.assertTrue(s['ended']);self.assertFalse(s['success'])

    def test_floor_failure_but_no_outside_failure_both_phases(self):
        for place in (False,True):
            for field in ('floor','outside'):
                c=picked() if place else SnackTrayComposite(binding())
                if not place:c.observe(fact(c.binding,0))
                s=c.observe(fact(c.binding,c.last.pick.sample_index+1,**{field:True}))
                self.assertEqual(s['ended'],field=='floor');self.assertFalse(s['success'])
                if field=='outside':
                    self.assertTrue(s['outside_workspace_raw'])
                    self.assertFalse(s['outside_workspace_failure_enabled'])

    def test_initial_already_on_tray_does_not_skip_pick(self):
        c=SnackTrayComposite(binding())
        s=c.observe(fact(c.binding,0,placed=True,separated=True))
        self.assertFalse(s['ended']);self.assertEqual(s['phase'],'pick');self.assertEqual(s['score'],0)

    def test_outside_diagnostic_cannot_cancel_valid_completion(self):
        c=picked()
        for _ in range(300):
            s=c.observe(fact(c.binding,c.last.pick.sample_index+1,placed=True,separated=True,outside=True))
            if s['ended']:break
        self.assertTrue(s['success']);self.assertEqual(s['score'],7)

    def test_settle_interrupt_and_regrasp(self):
        c=picked()
        for _ in range(100):c.observe(fact(c.binding,c.last.pick.sample_index+1,placed=True,separated=True))
        self.assertFalse(c.success)
        c.observe(fact(c.binding,c.last.pick.sample_index+1,held=True,separated=True))
        self.assertFalse(c.success)
        self.assertTrue(finish(c)['success'])

    def test_hold_interruption(self):
        c=SnackTrayComposite(binding());c.observe(fact(c.binding,0))
        for n in range(1,400):c.observe(fact(c.binding,n,held=True,separated=True))
        c.observe(fact(c.binding,400,separated=True))
        for n in range(401,800):c.observe(fact(c.binding,n,held=True,separated=True))
        self.assertEqual(c.phase,'pick');self.assertFalse(c.success)

    def test_upright_product_rule_retained(self):
        q=(math.cos(math.pi/6), math.sin(math.pi/6), 0., 0.)
        for product,score in [('coke_c3',6),('skit_littles',6),('pring_bbq',7)]:
            self.assertEqual(finish(picked(product),q)['score'],score)

    def test_finalize(self):
        c=SnackTrayComposite(binding());c.observe(fact(c.binding,0))
        with self.assertRaises(ValueError):c.finalize('success')
        s=c.finalize();self.assertFalse(s['success']);self.assertTrue(s['ended'])
        self.assertEqual(s,c.finalize())
        with self.assertRaises(ValueError):c.observe(fact(c.binding,1))

    def test_handoff_drop_preserves_boundary(self):
        c=picked()
        for i in range(1,20):
            c.observe(fact(c.binding,c.last.pick.sample_index+1,separated=True,height=1.-i*.01))
        self.assertEqual(c.summary()['place']['held_drop_penalty'],.5)

    def test_drop_back_charged_once(self):
        c=SnackTrayComposite(binding());c.observe(fact(c.binding,0))
        c.observe(fact(c.binding,1,held=True,separated=True))
        c.observe(fact(c.binding,2))
        before=c.summary()['score']
        for n in range(3,20):c.observe(fact(c.binding,n))
        self.assertEqual(c.summary()['score'],before)
        events=c.pick.events
        self.assertEqual(sum(e['kind']=='dropped_back_to_support' for e in events),1)

    def test_original_phase_timeouts(self):
        for place in (False,True):
            c=picked() if place else SnackTrayComposite(binding())
            if not place:c.observe(fact(c.binding,0))
            for _ in range(round(60/DT)+1):
                c.observe(fact(c.binding,c.last.pick.sample_index+1,held=place,separated=place))
                if c.ended:break
            self.assertTrue(c.ended);self.assertFalse(c.success)
            self.assertEqual(c.reason,'time_limit')
            t=c.last.pick.time_s-(c.place_start*DT if place else 0)
            self.assertAlmostEqual(t,60)


if __name__=='__main__':unittest.main()

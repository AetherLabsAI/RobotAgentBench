"""Synthetic 60/240 Hz score contracts, not native manipulation evidence."""
import unittest
import importlib.util
if importlib.util.find_spec('openend') is None or importlib.util.find_spec('numpy') is None:
    raise unittest.SkipTest('Run native component contracts in the deployed 202 Python environment')
from dataclasses import replace
from openend.tasks.scene_requirements.shelf_composite import Binding,Frame,SnackTrayComposite
from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
from evaluation.tasks.coffee_table_place.process_rules import HeldFacts
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
from openend.room import validate_request,resolve_objects

def new(product='coke_c3'):
    return SnackTrayComposite(Binding('episode','shelf',product,('/World/Target',),'/World/Board','/World/Tray'),room_managed=True)

def step(c,dt=1/60,held=False,placed=False):
    n=0 if c.last is None else c.last.pick.sample_index+1
    t=0. if c.last is None else c.last.pick.time_s+dt
    return c.observe(Frame(c.binding,
        PhysicalFrame('episode',n,t,held,held or placed,.1 if held else 0.,frozenset(),frozenset(),False,False,active_target='/World/Target'),
        HeldFacts(n,t,1.,held,placed,placed,(1.,0.,0.,0.),(0.,0.,0.),(1.,0.,0.,0.),False,False),dt))

class ShelfRuntime(unittest.TestCase):
    def test_all_products_at_sixty(self):
        for key in PRODUCTS:
            with self.subTest(product=key):
                c=new(key);step(c)
                for _ in range(180):step(c,held=True)
                self.assertEqual(c.phase,'pick')
                step(c,held=True);self.assertEqual(c.phase,'carry_and_place')
                self.assertEqual(c.summary()['score'],4)
                for _ in range(60):step(c,placed=True)
                self.assertFalse(c.success)
                # Existing support debounce (.1s) precedes the unchanged 1s settle window.
                for _ in range(10):
                    if c.ended:break
                    step(c,placed=True)
                self.assertTrue(c.success);self.assertEqual(c.summary()['score'],7)
    def test_mixed_rate_preserves_real_duration(self):
        c=new();step(c)
        for _ in range(360):step(c,1/240,held=True)
        for _ in range(80):step(c,1/60,held=True)
        self.assertEqual(c.phase,'pick')
        while c.phase=='pick':step(c,1/60,held=True)
        self.assertGreaterEqual(c.last.pick.time_s,3.)
        while not c.ended:step(c,1/60,placed=True)
        self.assertEqual(c.summary()['score'],7)
    def test_pause_cannot_bridge_hold(self):
        c=new();step(c)
        for _ in range(120):step(c,held=True)
        c.pause()
        for _ in range(120):step(c,held=True)
        self.assertEqual(c.phase,'pick')
    def test_bad_clock_transactional(self):
        c=new();step(c);before=c.summary()
        with self.assertRaises(ValueError):step(c,1/20,held=True)
        self.assertEqual(c.summary(),before)
    def test_exact_binding_not_background_packaging(self):
        request=dict(tasks=[dict(instance_id='s',type='snack_drink_to_tray',parameters={'product':'zbar_t'})],no_progress_timeout_s=120,selection_order='front_then_distance')
        tasks=validate_request(request)
        manifest={'objects':[dict(id='zbar_t',path='/World/Garbage',task_role='garbage'),dict(id='zbar_t',path='/World/FullProduct',task_role='requested_full_product',task_id='s')]}
        self.assertEqual(resolve_objects(tasks,manifest)['s']['path'],'/World/FullProduct')
        manifest['objects'][1]['task_id']='another_task'
        with self.assertRaises(ValueError):resolve_objects(tasks,manifest)
    def test_same_category_background_is_not_scored(self):
        from openend.tasks.scene_requirements.shelf_port import SnackTrayPort
        manifest={'objects':[
            dict(id='coke_c3',path='/World/Target',task_role='requested_full_product'),
            dict(id='coke_c3',path='/World/Extra',task_role='background_stock'),
            dict(id='coke_c3',path='/World/Garbage',task_role='garbage')]}
        self.assertEqual(SnackTrayPort.resolve_object({'product':'coke_c3'},manifest)['path'],'/World/Target')
        manifest['objects'][1]['task_role']='requested_full_product'
        with self.assertRaises(ValueError):SnackTrayPort.resolve_object({'product':'coke_c3'},manifest)

if __name__=='__main__':unittest.main()

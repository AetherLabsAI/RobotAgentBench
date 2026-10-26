"""Selected-room contracts, synthetic evidence, not manipulation validation."""
import io,unittest
from types import SimpleNamespace
from multi_task_room_v2 import validate_request,MultiTaskRoom
from multi_task_dispatch_v1 import MultiTaskDispatcher,TaskRegistration
from evaluation.tasks.shelf_pick.openended_composite_v1.composite import SnackTrayComposite
from evaluation.tasks.shelf_pick.openended_composite_v1.room_port import SnackTrayPort
from evaluation.tasks.shelf_pick.openended_composite_v1.test_composite import binding,fact,picked,finish

class Tests(unittest.TestCase):
    def test_registration_all_products(self):
        from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
        for p in PRODUCTS:
            r=dict(tasks=[dict(instance_id='snack',type='snack_drink_to_tray',parameters=dict(product=p))],no_progress_timeout_s=120,selection_order='front_then_distance')
            self.assertEqual(validate_request(r),r['tasks'])
        r['tasks'][0]['parameters']['product']='invalid'
        with self.assertRaises(ValueError):validate_request(r)

    def test_pause_does_not_complete_partial_hold(self):
        c=SnackTrayComposite(binding(),room_managed=True)
        for n in range(400):c.observe(fact(c.binding,n,held=True,separated=True))
        before=c.summary()['score'];c.pause()
        for n in range(400,800):c.observe(fact(c.binding,n,held=True,separated=True))
        self.assertEqual(c.phase,'pick');self.assertEqual(c.summary()['score'],before)
        for n in range(800,1200):
            c.observe(fact(c.binding,n,held=True,separated=True))
            if c.phase!='pick':break
        self.assertEqual(c.phase,'carry_and_place')
        self.assertTrue(finish(c)['success'])

    def test_pause_resets_placement_window_only(self):
        c=picked();c.room_managed=True
        for _ in range(150):c.observe(fact(c.binding,c.last.pick.sample_index+1,placed=True,separated=True))
        c.pause();self.assertEqual(c.completed_pick['score'],4)
        for _ in range(150):c.observe(fact(c.binding,c.last.pick.sample_index+1,placed=True,separated=True))
        self.assertFalse(c.success);self.assertTrue(finish(c)['success'])

    def port(self):
        p=SnackTrayPort.__new__(SnackTrayPort)
        p.composite=SnackTrayComposite(binding(),room_managed=True);p.samples=0;p.index=0
        p.score=p.composite.summary();p.buffer=[];p.enabled=False;p.clock=SimpleNamespace(hz=240,current_time=1.)
        p.trace=io.StringIO();p.source=SimpleNamespace(after_physics=lambda _:None,frame=lambda n:fact(p.composite.binding,n,outside=True))
        return p

    def test_native_samples_selected_only(self):
        p=self.port();p.begin_selected_control(False)
        for _ in range(3):p.after_physics({})
        self.assertEqual(p.buffer,[]);self.assertEqual(p.samples,0)
        p.begin_selected_control(True)
        for _ in range(12):p.after_physics({})
        self.assertEqual(p.samples,0)
        s=p.observe();self.assertEqual(p.samples,12);self.assertEqual(p.index,1)
        self.assertFalse(s['outside_workspace_failure_enabled']);self.assertFalse(s['initial_state_gate_enabled'])
        self.assertTrue(s['outside_workspace_raw']);p.pause()
        p.begin_selected_control(True)
        for _ in range(12):p.after_physics({})
        p.observe();self.assertEqual(p.samples,24)

    def test_no_initial_score_gate(self):
        p=self.port();p.scene=SimpleNamespace(identity=p.composite.binding)
        p.source.latest={'poses':{t:{'position_m':[0,0,1]} for t in p.composite.binding.candidates}}
        self.assertFalse(p.accept_initial()['initial_state_gate_enabled']);self.assertEqual(p.samples,0)

    def test_timeout_discards_unconsumed_samples(self):
        p=self.port();p.begin_selected_control(True);p.after_physics({});p.discard_control()
        self.assertEqual(p.samples,0);self.assertEqual(p.buffer,[])

    def test_room_owns_terminal_timeout(self):
        c=SnackTrayComposite(binding(),room_managed=True)
        for n in range(14402):c.observe(fact(c.binding,n))
        self.assertFalse(c.ended)
        self.assertEqual(c.summary()['global_timeout_policy'],'shared_room_no_progress')

    def test_activation_handoff_geometry(self):
        p=self.port();p.scene=SimpleNamespace(identity=binding(),tray='/World/WoodenTray')
        p.manifest={'objects':[{'path':'/World/Target'.removesuffix('/Asset'),'visible_bounds_world_m':[[1,0,0],[1.1,.1,1]]},
                               {'path':'/World/WoodenTray','visible_bounds_world_m':[[-1,0,0],[-.9,.1,1]]}]}
        # Real unified bodies are wrapper/Asset; use that canonical shape.
        from dataclasses import replace
        p.scene.identity=replace(binding(),candidates=('/World/Target/Asset',))
        room=MultiTaskRoom.__new__(MultiTaskRoom);room.ports={'snack':p}
        room.dispatch=MultiTaskDispatcher([TaskRegistration('snack','snack_drink_to_tray',(1,0),((1,0),(1.1,0),(1.1,.1)),240)],sector_radius=1.32,sector_half_angle=45)
        room._refresh_targets();self.assertGreater(room.dispatch.registrations[0].target_xy[0],0)
        p.composite.phase='carry_and_place';room._refresh_targets()
        self.assertLess(room.dispatch.registrations[0].target_xy[0],0)

if __name__=='__main__':unittest.main()

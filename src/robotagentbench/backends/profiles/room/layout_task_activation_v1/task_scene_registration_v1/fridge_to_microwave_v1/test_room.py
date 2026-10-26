import io,unittest
from types import SimpleNamespace
from multi_task_room_v2 import MultiTaskRoom,validate_request,FACTORIES
from multi_task_dispatch_v1 import MultiTaskDispatcher,TaskRegistration
from fridge_to_microwave_v1.room_port import FridgeMicrowavePort
from fridge_to_microwave_v1.composite import Composite


class Tests(unittest.TestCase):
    def test_registration_both_variants(self):
        for variant in ('beef_box','leftover_box'):
            r=dict(tasks=[dict(instance_id='food',type='fridge_to_microwave',parameters=dict(payload_variant=variant))],no_progress_timeout_s=120,selection_order='front_then_distance')
            self.assertEqual(validate_request(r),r['tasks'])
        r['tasks'][0]['parameters']['held_by_right']=True
        with self.assertRaises(ValueError):validate_request(r)
        self.assertIs(FACTORIES['fridge_to_microwave'],FridgeMicrowavePort)

    def port(self):
        p=FridgeMicrowavePort.__new__(FridgeMicrowavePort)
        p.composite=Composite('episode','/World/Box','beef_box');p.buffer=[];p.enabled=False
        p.scene=SimpleNamespace(activation={'refrigerator':[[1,0,0],[2,1,2]],'microwave':[[-2,0,0],[-1,1,1]]})
        return p

    def test_handoff_target_and_hz_together(self):
        p=self.port();room=MultiTaskRoom.__new__(MultiTaskRoom);room.ports={'food':p}
        room.dispatch=MultiTaskDispatcher([TaskRegistration('food','fridge_to_microwave',(1,0),((1,0),(2,0),(2,1)),60)],sector_radius=1.32,sector_half_angle=45)
        room._refresh_targets();a=room.dispatch.registrations[0]
        self.assertEqual(a.physics_hz,60);self.assertGreater(a.target_xy[0],0)
        # Contract-only fixture, not physical progress evidence.
        p.composite.phase='microwave';room._refresh_targets();b=room.dispatch.registrations[0]
        self.assertEqual(b.physics_hz,240);self.assertLess(b.target_xy[0],0)

    def test_pending_or_unselected_pauses_continuous_holds(self):
        p=self.port();p.begin_selected_control(False)
        self.assertTrue(p.composite.paused);self.assertFalse(p.enabled)

    def test_missing_substeps_never_scores(self):
        p=self.port();p.enabled=True;p.buffer=[{}]*2
        with self.assertRaises(RuntimeError):p.observe()
        self.assertIsNone(p.composite.last)

    def test_idle_readers_do_not_buffer_scores(self):
        p=self.port();p.source=SimpleNamespace(after_physics=lambda *a:None,latest={})
        p.clock=SimpleNamespace(hz=60)
        p.after_physics({});self.assertEqual(p.buffer,[])
        p.begin_selected_control(True);p.after_physics({});self.assertEqual(len(p.buffer),1)
        p.discard_control();self.assertEqual(p.buffer,[])

    def test_selected_wrong_phase_clock_rejected(self):
        p=self.port();p.source=SimpleNamespace(after_physics=lambda *a:None,latest={});p.clock=SimpleNamespace(hz=240)
        p.begin_selected_control(True)
        with self.assertRaises(RuntimeError):p.after_physics({})


if __name__=='__main__':unittest.main()

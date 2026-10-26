"""Synthetic main-loop routing only; no physical action success claims."""
import io
import unittest
from types import SimpleNamespace
from openend.room import MultiTaskRoom
from openend.dispatch import MultiTaskDispatcher, TaskRegistration
from .shelf_port import SnackTrayPort

POLY=((.8,-.1),(1.,-.1),(1.,.1),(.8,.1))


class SyntheticShelf(SnackTrayPort):
    def __init__(self):
        self.composite=SimpleNamespace(ended=False,phase='carry_and_place')
        self.score=dict(score=0,success=False);self.index=0;self.pauses=0
    def selection_geometry(self):return ((.9,0),POLY)
    def begin_selected_control(self,enabled):self.enabled=enabled
    def pause(self):self.pauses+=1
    def observe(self):
        self.composite.ended=True;self.index+=1
        self.score=dict(score=7,success=True,ended=True)
        return self.score


def room_fixture(count=4,order='front_then_distance'):
    room=MultiTaskRoom.__new__(MultiTaskRoom)
    room.started=True;room.closed=False;room.faulted=False;room.pending=None
    room.ports={f'shelf_{i}':SyntheticShelf() for i in range(count)}
    regs=[TaskRegistration(k,'table_wooden_tray',(.9,0),POLY,60) for k in room.ports]
    room.dispatch=MultiTaskDispatcher(regs,sector_radius=1.32,sector_half_angle=45,
                                     no_progress_timeout_s=120,selection_order=order)
    room.dispatch.start(accepted_scene=True,simulation_time_s=0.)
    room.clock=SimpleNamespace(hz=60,current_time=0.)
    room.workspace_selector=SimpleNamespace(select=lambda robot:dict(
        physics_rate_owner='single_frontal_workspace',requested_physics_hz=60,
        current_workspace='table_wooden_tray'))
    room.previous_active=set();room.frame=0;room.trace=io.StringIO()
    room.reader_counts={k:dict(full=0,idle=0) for k in room.ports}
    return room


def begin(room):
    return room.begin_control(dict(xy=(0,0),yaw=0),command_vw=(0,0),base_velocity=(0,)*6)


def step(room):
    plan=begin(room)
    room.clock.current_time+=.05
    return plan,room.finish_control(completed_substeps=plan['substeps'])


class ShelfRetirement(unittest.TestCase):
    def test_four_complete_at_same_tray_without_blocking(self):
        for order in ('front_then_distance','distance_then_front'):
            room=room_fixture(order=order)
            identities={k:id(p) for k,p in room.ports.items()}
            for i in range(4):
                plan,result=step(room)
                self.assertEqual(plan['current_task'],f'shelf_{i}')
                self.assertEqual(plan['scoring_tasks'],[f'shelf_{i}'])
                self.assertEqual((plan['physics_hz'],plan['substeps']),(60,3))
                self.assertEqual(result['outcome']['total_score'],7*(i+1))
            progress=room.dispatch.last_progress_time
            plan,result=step(room)
            self.assertIsNone(plan['current_task']);self.assertEqual(plan['scoring_tasks'],[])
            self.assertEqual(result['outcome']['total_score'],28)
            self.assertEqual(len(result['outcome']['scoreboard']),4)
            self.assertTrue(all(s['success'] for s in result['outcome']['scoreboard'].values()))
            self.assertEqual(room.dispatch.last_progress_time,progress)
            self.assertEqual({k:id(p) for k,p in room.ports.items()},identities)
            self.assertEqual([p.index for p in room.ports.values()],[1]*4)

    def test_failed_terminal_shelf_also_yields(self):
        room=room_fixture(2);room.ports['shelf_0'].composite.ended=True
        plan,_=step(room)
        self.assertEqual(plan['current_task'],'shelf_1')
        self.assertEqual(room.dispatch.scores['shelf_0']['score'],0)
        self.assertFalse(room.dispatch.scores['shelf_0']['success'])

    def test_retirement_does_not_reset_shared_timeout(self):
        room=room_fixture(1);step(room)
        progress=room.dispatch.last_progress_time
        room.clock.current_time=progress+119
        plan=begin(room)
        self.assertIsNone(plan['current_task'])
        self.assertAlmostEqual(plan['remaining_s'],1.)
        self.assertEqual(room.dispatch.last_progress_time,progress)

    def test_other_task_retirement_policy_unchanged(self):
        room=room_fixture(1)
        # Not a shelf port: the new shelf-only retirement rule must not affect it.
        room.ports['shelf_0']=SimpleNamespace(ended=True)
        plan=begin(room)
        self.assertEqual(plan['current_task'],'shelf_0')
        self.assertEqual(len(room.dispatch.registrations),1)


if __name__=='__main__':unittest.main()

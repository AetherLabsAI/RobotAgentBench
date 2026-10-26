"""Synthetic wiring/score regressions, not a physical rollout."""
import unittest
import importlib.util
if importlib.util.find_spec('numpy') is None or importlib.util.find_spec('openend') is None:
    raise unittest.SkipTest('Runtime wiring tests require the deployed 202 API environment')
from types import SimpleNamespace
from unittest.mock import patch
from openend.room import validate_request,resolve_objects,MultiTaskRoom
from openend.dispatch import TaskRegistration,MultiTaskDispatcher
from openend.tasks.scene_requirements.runtime_ports import FridgePort
from openend.tasks.glasses_transfer import test_composite as glasses
from openend.tasks.wallet_transfer import test_composite as wallet

class WiringTests(unittest.TestCase):
    def test_two_payloads_resolve_actual_target_not_appliance(self):
        for variant in ('beef_box','leftover_box'):
            tasks=validate_request(dict(tasks=[dict(instance_id='f',type='fridge_to_microwave',parameters={'payload_variant':variant})],no_progress_timeout_s=120,selection_order='front_then_distance'))
            result=resolve_objects(tasks,{'objects':[dict(id=variant,path='/payload'),dict(id='refrigerator',path='/fridge')]})
            self.assertEqual(result['f']['path'],'/payload')
    def test_payload_identity_ambiguity_rejected(self):
        with self.assertRaises(ValueError):FridgePort.resolve_object({'payload_variant':'beef_box'},{'objects':[]})
    def test_drawer_cannot_award_lift_below_rim_or_skip_hold(self):
        for module in (glasses,wallet):
            run=module.Rules();run.setUp()
            for _ in range(725):run.step(held=True,robot_contact=True,source_clear=False)
            self.assertEqual(run.task.summary()['score'],1)
            self.assertFalse(run.task.pick.ever_complete)
            for _ in range(400):run.step(held=True,robot_contact=True,source_clear=True)
            run.step(held=True,robot_contact=True,source_clear=False)
            for _ in range(400):run.step(held=True,robot_contact=True,source_clear=True)
            self.assertFalse(run.task.pick.ever_complete)
            run.carry()
            self.assertEqual(run.placed()['score'],7)
    def test_host_phase_route_keeps_score_identity_and_timer(self):
        port=SimpleNamespace(workspace_id='fridge',requested_physics_hz=60,ended=False,
                             selection_geometry=lambda:((1.,0.),((.8,-.2),(1.2,-.2),(1.2,.2),(.8,.2))),
                             pause=lambda:None,begin_selected_control=lambda v:enabled.append(v))
        owner=object.__new__(MultiTaskRoom)
        owner.started=True;owner.closed=False;owner.faulted=False;owner.pending=None
        owner.ports={'f':port};owner.previous_active=set();owner.frame=0
        owner.dispatch=MultiTaskDispatcher([TaskRegistration('f','fridge',(1.,0.),((.8,-.2),(1.2,-.2),(1.2,.2),(.8,.2)),60)],sector_radius=1.32,sector_half_angle=45)
        owner.dispatch.start(accepted_scene=True,simulation_time_s=0.)
        owner.clock=SimpleNamespace(hz=60,current_time=0.)
        # Use the real workspace selector contract from the existing unit fixture.
        from openend.tasks.all_objects_integration.workspace_physics import Workspace,WorkspacePhysics
        owner.workspace_selector=WorkspacePhysics([Workspace('fridge','/fridge',((.8,-.2),(1.2,-.2),(1.2,.2),(.8,.2)),60)],dict(range=1.32,halfAngle=45))
        enabled=[]
        plan=owner.begin_control({'xy':(0.,0.),'yaw':0.},command_vw=(0.,0.),base_velocity=(0.,)*6)
        self.assertEqual(plan['scoring_tasks'],['f']);self.assertEqual(plan['substeps'],3)
        owner.dispatch.complete_physics_frame(simulation_time_s=.05)
        owner.dispatch.record_scores({'f':{'score':3,'success':False}});owner.dispatch.finish_frame()
        owner.pending=None;owner.clock.current_time=.05;owner.clock.hz=240
        port.workspace_id='microwave';port.requested_physics_hz=240
        owner.workspace_selector=WorkspacePhysics([Workspace('microwave','/microwave',((.8,-.2),(1.2,-.2),(1.2,.2),(.8,.2)),240)],dict(range=1.32,halfAngle=45))
        before=owner.dispatch.last_progress_time
        plan=owner.begin_control({'xy':(0.,0.),'yaw':0.},command_vw=(0.,0.),base_velocity=(0.,)*6)
        self.assertEqual(plan['substeps'],12)
        self.assertEqual(owner.dispatch.registrations[0].kind,'microwave')
        self.assertEqual(owner.dispatch.scores['f']['score'],3)
        self.assertEqual(owner.dispatch.last_progress_time,before)
        self.assertEqual(enabled,[True,True])

if __name__=='__main__':unittest.main()

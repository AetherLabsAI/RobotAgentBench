"""Versioned ten-plus native rooms using the original discovery ledger/clock."""
import json
from openend.room import MultiTaskRoom, prepare_registration
from ..runtime import DiscoveryRoom as PilotRoom, DiscoveryDispatcher
from ..policy import CoverageLedger, RULE_VERSION
from .object_bindings import catalogue, within
from .contact_catalogue import background_catalogue


class DiscoveryRoom(PilotRoom):
    def __init__(self,stage,manifest,output,episode_id,*,request,sensor,workspace_selector):
        # Only goal identity is extended. The same ports initialize and validate
        # the real room; original operation scores are never consumed.
        MultiTaskRoom.__init__(self,stage,manifest,output,episode_id,request=request,
                               sensor=sensor,workspace_selector=workspace_selector)
        try:
            binding=catalogue(request,manifest,prepare_registration(request,manifest))
            self.task_to_object=binding['task_to_object']
            if set(self.task_to_object)!=set(self.ports):
                raise ValueError('Actual native ports differ from frozen physical owners')
            from ...bag_support_v2.policy import enable_for_ports
            kinds={t['instance_id']:t['type'] for t in json.loads((self.output/'registration.json').read_text())['request']['tasks']}
            support=enable_for_ports(kinds,self.ports)
            targets=binding['targets']
            from pxr import UsdPhysics
            background=background_catalogue(
                [str(prim.GetPath()) for prim in stage.Traverse()
                 if prim.HasAPI(UsdPhysics.CollisionAPI)],
                [r['path'] for r in manifest['objects']],targets)
            self.discovery=CoverageLedger(episode_id=episode_id,targets=targets,
                non_targets=background,fingers=self.contact_routes.native.fingers)
            old=self.dispatch
            self.dispatch=DiscoveryDispatcher(old.registrations,sector_radius=old.radius,
                sector_half_angle=old.half,no_progress_timeout_s=old.timeout,selection_order=old.order)
            self.event_trace=(self.output/'discovery_events.jsonl').open('x')
            (self.output/'discovery_rule.json').write_text(json.dumps(dict(binding,
                rule_version=RULE_VERSION,non_targets=background,denominator=len(targets),
                operation_points_enabled=False,wrong_touch_penalty=0,bag_support_policy=support,
                native_controls_and_workspace_rates_unchanged=True),indent=2)+'\n')
        except BaseException:
            MultiTaskRoom.close(self)
            raise

    def discovery_scores(self):
        return {key:dict(score=int(key==canonical and canonical in self.discovery.target_events),
                         maximum_score=int(key==canonical),
                         started=canonical in self.discovery.target_events,
                         success=canonical in self.discovery.target_events)
                for key,canonical in self.task_to_object.items()}

"""Continuous full-arm held transport and complete gravity release preflight."""
import numpy as np,json
from removal_open_mouth_bundle_v10 import AdaptiveRightBasket as Previous
from removal_continuous_held_transport_v1 import plan_transport
class AdaptiveRightBasket(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.held_path_end=max(i for i,x in enumerate(self.phases)if x=='R_higher_clearance_after_outward_lift');self.held_hold_end=max(i for i,x in enumerate(self.phases)if x=='R_hold_lifted_cloth');self.release_start=self.phases.index('R_release_in_original_basket');self.release_end=max(i for i,x in enumerate(self.phases)if x=='R_release_in_original_basket');self.withdraw_start=self.phases.index('R_withdraw_open_hand');self.withdraw_end=max(i for i,x in enumerate(self.phases)if x=='R_withdraw_open_hand');self.settle_start=self.phases.index('basket_settle_after_release');self.held_transport=None;self.last_native_values=None;self.transport_events=[]
 def on_frame(self,source_frame,physical_frame,cloth):
  nxt=super().on_frame(source_frame,physical_frame,cloth)
  if source_frame==self.vertical_start-1 and self.held_transport is None:
   try:
    self.held_transport=plan_transport(self.last_native_values,held_frames=self.held_path_end-self.vertical_start+1,open_frames=self.release_end-self.release_start+1,withdraw_frames=self.withdraw_end-self.withdraw_start+1);event=dict(event='full_held_transport_and_release_preflight',physical_frame=physical_frame,**self.held_transport['audit']);self.transport_events.append(event);print(json.dumps(event),flush=True)
   except (RuntimeError,AssertionError)as exc:
    self.record_geometry_failure(physical_frame,'continuous_held_transport_route_unavailable',dict(error=str(exc)));return len(self.phases)
  if self.held_transport is not None:
   if source_frame==self.held_hold_end:return self.release_start
   if source_frame==self.withdraw_end:return self.settle_start
  return nxt
 def command(self,source_frame,physical_frame,template):
  if self.held_transport is None or source_frame<self.vertical_start:
   result=super().command(source_frame,physical_frame,template)
  else:
   t=self.held_transport
   if source_frame<=self.held_path_end:q=t['held'][source_frame-self.vertical_start]
   elif source_frame<=self.held_hold_end:q=t['held'][-1]
   elif self.release_start<=source_frame<=self.release_end:q=t['opened'][source_frame-self.release_start]
   elif self.withdraw_start<=source_frame<=self.withdraw_end:q=t['withdraw'][source_frame-self.withdraw_start]
   elif source_frame>=self.settle_start:q=t['withdraw'][-1]
   else:raise RuntimeError(('Unexpected unskipped legacy transfer phase',source_frame,self.phases[source_frame]))
   result=np.array([q[n]for n in self.names]);self.right_prior={n:q[n]for n in self.right_arm}
  self.last_native_values=self.robot.calibrate_loops(dict(zip(self.names,map(float,result))));return result
 def report(self):
  return dict(super().report(),continuous_held_transport_events=self.transport_events,right_grasp_command_rad=0.,same_native_robot_and_material=True,legacy_folded_vertical_lift_replaced=True,transfer_and_lower_merged_into_full_arm_motion=True,skipped_redundant_postrelease_raise=True,source_phase_labels_retained_for_native_closure_and_retention_measurements=True,object_state_writes=0)

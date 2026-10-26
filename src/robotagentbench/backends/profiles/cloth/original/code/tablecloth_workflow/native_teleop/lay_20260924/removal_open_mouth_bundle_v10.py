"""Actual aperture and broad contact selection replaces pre-compressed layer heuristic."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from removal_bare_lower_bundle_v8 import AdaptiveRightBasket as Previous
from removal_adaptive_diagnostic_continuation_v4 import AdaptiveRightBasket as Root
from removal_open_mouth_selector_v2 import select_pair
from pregrasp_feasibility_v1 import checked_candidate_route
from cooked_closure_guard import CookedClosureGuard
class AdaptiveRightBasket(Previous):
 def __init__(self,action,*args,cooked_guard_path,**kwargs):
  super().__init__(action,*args,**kwargs);self.mouth_guard_path=Path(cooked_guard_path);self.mouth_guard=None;self.mouth_events=[]
  ix=max(i for i,x in enumerate(self.phases)if x=='R_horizontal_sweep_into_layers');values=self.robot.calibrate_loops(dict(zip(self.names,map(float,self.action_targets[ix]))));poses=self.robot.fk(values);self.mouth_names=list(poses);self.mouth_bodies=np.array([np.r_[h[:3,3],Rotation.from_matrix(h[:3,:3]).as_quat()]for h in poses.values()]);self.old_reference=self.right_ref.copy();pad_local=np.array([.03285,0,0,1]);self.right_ref=np.mean([(poses[f'gripper_r_{kind}_link4']@pad_local)[:3]for kind in ['inner','outer']],0)
 def on_frame(self,source_frame,physical_frame,cloth):
  if source_frame not in self.bundle_clear_ends:return super().on_frame(source_frame,physical_frame,cloth)
  if self.mouth_guard is None:self.mouth_guard=CookedClosureGuard(self.mouth_guard_path)
  selection=select_pair(cloth,self.uv,self.cloth_ids,self.mouth_guard,self.mouth_bodies,self.mouth_names,self.right_ref,float(self.table_bounds[0,0]));chosen=None;preflights=[]
  for candidate in selection['ranked_candidates']:
   route,audit=checked_candidate_route(self,candidate);preflights.append(dict(pair_indices=candidate['pair_indices'],**audit))
   if route is not None:
    chosen=candidate;self.checked_pre_route=route['route'];self.pre_correction=route['correction'];self.accepted_route_audit=route['audit'];break
  last=source_frame==self.bundle_end;event=dict(event='actual_open_mouth_two_layer_selection',physical_frame=physical_frame,pass_index=self.bundle_clear_ends.index(source_frame)+1,projected_pairs_both_over40mm=selection['projected_pairs_both_over40mm'],minimum_nodes_per_layer=10,decision='grasp'if chosen else'live_physical_repair_needed'if last else'repeat',force_confirmed=False,pregrasp_preflight=preflights)
  if chosen:event['selected']={k:v for k,v in chosen.items()if k!='inside_node_ids'}
  self.mouth_events.append(event);print(json.dumps(event),flush=True)
  if chosen:
   self.right_layer_indices=chosen['pair_indices'];self.right_shift=np.array(chosen['nominal_translation']);self.right_trace.append(dict(event='right_actual_two_layer_open_mouth_capture_target',physical_frame=physical_frame,**event['selected']));Root.on_frame(self,source_frame,physical_frame,cloth);return self.right_start
  if last:self.record_geometry_failure(physical_frame,'no_broad_native_open_mouth_capture',event);return len(self.phases)
  self.bundle_shift_from=self.bundle_shift.copy()
  try:self.bundle_shift,point,material=self.bare_lower_anchor(cloth)
  except RuntimeError as exc:self.record_geometry_failure(physical_frame,'bare_lower_anchor_unavailable',dict(error=str(exc)));return len(self.phases)
  self.bundle_ramp_start=source_frame+1;tag=self.phases[self.bundle_ramp_start];self.bundle_ramp_end=max(k for k,x in enumerate(self.phases)if x==tag);self.bundle_events.append(dict(event='next_measured_bare_lower_anchor',physical_frame=physical_frame,actual_point=point.tolist(),material_uv=material.tolist(),shift_m=self.bundle_shift.tolist(),raised_return_interpolation=True));return Root.on_frame(self,source_frame,physical_frame,cloth)
 def report(self):
  return dict(super().report(),actual_open_mouth_selection_events=self.mouth_events,grasp_reference_is_actual_pad_mid=True,old_corner_reference=self.old_reference.tolist(),nominal_pad_mid_reference=self.right_ref.tolist(),pregrasp_layers_need_not_touch=True,whole_arm_route_preflight_before_stopping_brush=True,no_uncaught_pregrasp_route_assertion=True,accepted_route_audit=getattr(self,'accepted_route_audit',None),table_edge_overhang_required_m=.04,min_measured_nodes_per_layer_inside_open_mouth=10,native_collision_and_physical_retention_checks_unchanged=True)

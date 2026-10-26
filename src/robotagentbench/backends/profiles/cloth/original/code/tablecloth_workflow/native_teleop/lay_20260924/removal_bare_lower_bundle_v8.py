"""Measure folded layer overlap, physically brush it out, then retarget right grasp."""
from removal_adaptive_diagnostic_continuation_v4 import AdaptiveRightBasket as Base
from scipy.spatial import cKDTree
import numpy as np,json
from pathlib import Path
class AdaptiveRightBasket(Base):
 def __init__(self,action,*args,**kwargs):
  super().__init__(action,*args,**kwargs);self.bundle_start=self.phases.index('R_bundle_brush_safe_approach');self.bundle_approach_end=max(i for i,x in enumerate(self.phases)if x=='R_bundle_brush_safe_approach');self.bundle_clear_ends=[i for i in range(len(self.phases)-1)if self.phases[i].startswith('R_bundle_brush')and self.phases[i].endswith('_clear')and self.phases[i+1]!=self.phases[i]];self.bundle_end=self.bundle_clear_ends[-1];self.bundle_shift=None;self.bundle_previous=None;self.bundle_events=[];self.brush_nominal=np.array(json.loads((Path(__file__).parent/'bottom_brush_pose175_v3.json').read_text())['chosen']['target']);self.bundle_shift_from=np.zeros(3);self.bundle_ramp_start=self.bundle_start;self.bundle_ramp_end=self.bundle_approach_end;self.uv=np.asarray(args[0]);old=np.flatnonzero(self.cloth_ids==0);self.bundle_layers=[old[self.uv[old,1]<.235],old[self.uv[old,1]>.275]]
 def bare_lower_anchor(self,cloth):
  a,b=self.bundle_layers;distance,_=cKDTree(cloth[b,:2]).query(cloth[a,:2]);valid=(distance>.015)&(abs(cloth[a,2]-.4505)<.004)&(cloth[a,0]>self.table_bounds[0,0]+.010)
  ids=a[valid]
  if len(ids)<10:raise RuntimeError('No measured uncovered lower-panel brush patch; native live correction needed')
  chosen=int(ids[np.argmin(np.linalg.norm(cloth[ids,:2]-self.brush_nominal[:2],axis=1))]);point=cloth[chosen].astype(float);shift=point-self.brush_nominal;shift[2]=0.
  if np.linalg.norm(shift[:2])>.10:raise RuntimeError('Bare lower panel outside validated local brush workspace')
  return shift,point,self.uv[chosen]
 def overlap(self,cloth):
  a,b=self.bundle_layers;d,j=cKDTree(cloth[b]).query(cloth[a]);pairs=np.c_[a,b[j]];center=cloth[pairs].mean(1);valid=(d<.004)&(center[:,2]>.32)&(center[:,2]<.475)&(abs(center[:,1]+.49)<.20);pairs=pairs[valid];center=center[valid]
  if len(pairs)<10:raise RuntimeError('Measured folded cloth has no accessible two-layer contact patch')
  return pairs,center
 def on_physics(self,source_frame,physical_frame,cloth,bodies,body_names):
  if source_frame<len(self.phases):return super().on_physics(source_frame,physical_frame,cloth,bodies,body_names)
 def on_frame(self,source_frame,physical_frame,cloth):
  if source_frame>=len(self.phases):return source_frame+1
  if source_frame==self.bundle_start-1:
   try:self.bundle_shift,point,material=self.bare_lower_anchor(cloth)
   except RuntimeError as exc:
    self.record_geometry_failure(physical_frame,'bare_lower_anchor_unavailable',dict(error=str(exc)));return len(self.phases)
   self.bundle_events.append(dict(event='measured_bare_lower_panel_brush_anchor',physical_frame=physical_frame,actual_point=point.tolist(),material_uv=material.tolist(),action_translation=self.bundle_shift.tolist(),top_panel_excluded=True));print(json.dumps(self.bundle_events[-1]),flush=True)
  if source_frame in self.bundle_clear_ends:
   try:pairs,center=self.overlap(cloth)
   except RuntimeError as exc:
    self.record_geometry_failure(physical_frame,'folded_layer_overlap_unavailable',dict(error=str(exc)));return len(self.phases)
   front=float(np.quantile(center[:,0],.02));band=(center[:,0]<=front+.008);candidates=np.flatnonzero(band);qualified=candidates[center[candidates,0]<=self.table_bounds[0,0]-.04];pool=qualified if len(qualified) else candidates;chosen=int(pool[np.argmin(abs(center[pool,1]-self.right_ref[1]))]);overhang=float(self.table_bounds[0,0]-front);selected_overhang=float(self.table_bounds[0,0]-center[chosen,0]);ready=overhang>=.04 and selected_overhang>=.04 and len(qualified)>=10;last=source_frame==self.bundle_end;row=dict(event='measured_two_layer_bundle_exposure',physical_frame=physical_frame,source_frame=source_frame,pass_index=self.bundle_clear_ends.index(source_frame)+1,front_x_m=front,overhang_m=overhang,threshold_m=.04,contact_pairs=len(pairs),qualified_pair_count=len(qualified),selected_overhang_m=selected_overhang,drooping_pairs_included=True,chosen_pair_uv=self.uv[pairs[chosen]].tolist(),chosen_pair_center=center[chosen].tolist(),decision='grasp'if ready else'live_physical_repair_needed'if last else'repeat');self.bundle_events.append(row);print(json.dumps(row),flush=True)
   if last and not ready:
    self.record_geometry_failure(physical_frame,'two_layer_exposure_under40mm',row)
    return len(self.phases) # Enter existing native action-only live-repair interface; no blind pinch into table.
   if ready:
    self.right_layer_indices=pairs[chosen].tolist();self.right_shift=center[chosen].astype(float)-self.right_ref;self.right_trace.append(dict(event='right_actual_overlapping_layers_after_brush',physical_frame=physical_frame,layer_xyz_m=cloth[pairs[chosen]].tolist(),shift_m=self.right_shift.tolist()))
    if not ready:self.record_geometry_failure(physical_frame,'two_layer_exposure_under40mm',row)
    super().on_frame(source_frame,physical_frame,cloth);return self.right_start
   self.bundle_shift_from=self.bundle_shift.copy()
   try:self.bundle_shift,point,material=self.bare_lower_anchor(cloth)
   except RuntimeError as exc:
    self.record_geometry_failure(physical_frame,'bare_lower_anchor_unavailable',dict(error=str(exc)));return len(self.phases)
   self.bundle_ramp_start=source_frame+1;tag=self.phases[self.bundle_ramp_start];self.bundle_ramp_end=max(k for k,x in enumerate(self.phases)if x==tag)
   self.bundle_events.append(dict(event='next_measured_bare_lower_anchor',physical_frame=physical_frame,actual_point=point.tolist(),material_uv=material.tolist(),shift_m=self.bundle_shift.tolist(),raised_return_interpolation=True))
  return super().on_frame(source_frame,physical_frame,cloth)
 def command(self,source_frame,physical_frame,template):
  if source_frame>=len(self.phases):return template
  result=super().command(source_frame,physical_frame,template)
  if not self.bundle_start<=source_frame<=self.bundle_end:return result
  values=dict(zip(self.names,map(float,result)));original=self.robot.fk(values)['gripper_r_base_link'];blend=1.
  if source_frame<=self.bundle_ramp_end:
   u=np.clip((source_frame-self.bundle_ramp_start+1)/(self.bundle_ramp_end-self.bundle_ramp_start+1),0,1);blend=u*u*(3-2*u)
  shift=np.zeros(3)if self.bundle_shift is None else self.bundle_shift_from*(1-blend)+self.bundle_shift*blend;target=original[:3,3]+shift
  from local_bottom_brush_retarget_v2 import solve_local_brush_target
  solved,error=solve_local_brush_target(self.robot,values,target,original[:3,:3])
  if self.bundle_previous is not None:
   for n in self.right_arm:solved[n]=self.bundle_previous[n]+float(np.clip(solved[n]-self.bundle_previous[n],-.7/20,.7/20))
  poses=self.robot.fk(solved);error=float(np.linalg.norm(poses['gripper_r_base_link'][:3,3]-target))
  if error>.002:self.record_geometry_failure(physical_frame,'bundle_brush_ik_error',dict(error_m=error))
  hits=self.self_audit(poses)+self.cad_guard.check(poses)
  if hits:self.record_geometry_failure(physical_frame,'bundle_brush_self_collision',dict(hits=str(hits[:3])))
  for link,points in self.right_collision_points.items():
   h=poses[link];world=points@h[:3,:3].T+h[:3,3]
   for box,lo,hi in self.basket_boxes:
    if np.any(world.max(0)<=lo)or np.any(world.min(0)>=hi):continue
    eq=self.basket_planes[box]
    if np.any(np.max(world@eq[:,:3].T+eq[:,3],axis=1)<-.001):self.record_geometry_failure(physical_frame,'bundle_brush_basket_collision',dict(link=link,box=box))
   signed=world@self.table_planes[:,:3].T+self.table_planes[:,3]
   if np.any(np.max(signed,axis=1)<-.001):self.record_geometry_failure(physical_frame,'bundle_brush_table_interior',dict(link=link))
  for n in self.right_arm:result[self.names.index(n)]=solved[n]
  self.bundle_previous=dict(zip(self.names,map(float,result)));return result
 def report(self):return dict(super().report(),bundle_brush_events=self.bundle_events,bundle_brush_is_native_contact_only=True,overlap_grasp_replaces_separated_fixed_uv_mean=True,brush_retarget_reference_relative_max_joint_delta_rad=.35,brush_retarget_does_not_integrate_previous_nullspace_error=True,droop_aware_overlap=True,selected_grasp_patch_overhang_required_m=.04,minimum_qualified_pair_count=10,no_blind_pinch_after_last_failed_brush=True)

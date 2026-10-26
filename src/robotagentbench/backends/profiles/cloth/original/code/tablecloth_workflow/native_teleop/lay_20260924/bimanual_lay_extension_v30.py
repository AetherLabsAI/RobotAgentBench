"""State-feedback action continuation. Objects move only through physical contact.
Numerical planning pauses physics, changes only future joint targets, and never restarts state.
"""
import json,numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from bimanual_lay_plan_v30 import LayPlanner,DATA
class LayExtension:
 def __init__(self,output,cloth_ids):
  self.planner=LayPlanner();self.output=Path(output);self.ids=np.flatnonzero(cloth_ids==1);self.stage='right_eye';self.events=[];self.live=[];self.left=None;self.wait=0;self.retries=0;self.complete=False;self.released=False;self.failure=None;self.presentation_attempt=0;self.insertion_steps=0;self.align_retries=0;self.left_cleared_torso=False;self.corner_still_wait=0
  with np.load(DATA/'robot_native_collision.npz') as z:
   self.pad_points={s:[z['points_gripper_'+s+'_'+k+'_link4'].copy() for k in ['inner','outer']] for s in ['l','r']};self.trees={s:[cKDTree(a) for a in self.pad_points[s]] for s in ['l','r']}
 def grip(self,actual,cloth,uv,side):
  poses=self.planner.robot.fk(actual);ds=[]
  for k,tree in zip(['inner','outer'],self.trees[side]):
   t=poses['gripper_'+side+'_'+k+'_link4'];ds.append(tree.query((cloth-t[:3,3])@t[:3,:3])[0])
  near=(ds[0]<.005)&(ds[1]<.005);target=np.array([.32775,.028333]) if side=='r' or self.left is None else np.array(self.left['uv']);mask=np.linalg.norm(uv-target,axis=1)<.07
  return dict(patch_count=int((near&mask).sum()),other_patch_count=int((near&~mask).sum()),material_uv_bounds=[uv[near&mask].min(0).tolist(),uv[near&mask].max(0).tolist()] if (near&mask).any() else None, material_uv_median=np.median(uv[near&mask],axis=0).tolist() if (near&mask).any() else None)
 def save(self):
  (self.output/'bimanual_events.json').write_text(json.dumps(dict(stage=self.stage,events=self.events,failure=self.failure,complete=self.complete,object_state_writes=0),indent=2)+'\n')
  (self.output/'bimanual_live_grip.json').write_text(json.dumps(self.live,indent=2)+'\n')
 def observe(self,frame,phase,actual,cloth,uv):
  if self.stage!='right_eye' and (frame%5==0 or phase.startswith('L_verify')):
   row=dict(frame=frame,phase=phase,left=self.grip(actual,cloth,uv,'l'),right=self.grip(actual,cloth,uv,'r'));self.live.append(row)
   if frame%100==0:self.save()
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage=='left_torso_cleared':self.left_cleared_torso=True;self.stage='right_presented'
  rg=self.grip(actual,cloth,uv,'r');lg=self.grip(actual,cloth,uv,'l');rows=[];old=self.stage
  np.savez_compressed(self.output/('observed_'+self.stage+'.npz'),cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
  if self.stage=='right_eye':
   if not rg['patch_count']:self.failure='Right grasp not retained at eye pose';self.stage='failed';self.save();return []
   rows=self.planner.forward_right_presentation(command);rows+=[(rows[-1][0],'R_wait_free_corner')]*120;self.stage='right_presented';self.events.append(dict(stage='right_retained_presentation',plan=self.planner.last_presentation,object_state_writes=0))
  elif self.stage=='right_presented':
   if not rg['patch_count']:self.failure='Right grasp lost while presenting';self.stage='failed';self.save();return []
   if not self.left_cleared_torso:
    rows=self.planner.clear_left_torso(command,cloth);self.stage='left_torso_cleared';self.events.append(dict(stage=self.stage,plan=self.planner.last_torso_clearance_plan,planned_frames=len(rows),physics_state_writes=0));self.save();return rows
   near=self.ids[np.linalg.norm(uv[self.ids]-[.019,.49111111],axis=1)<.03]
   window=np.asarray([r[0][near]for r in recent[-41:]]);motion=float(np.linalg.norm(window-window[-1],axis=-1).max())if len(window)==41 else float('inf')
   self.events.append(dict(stage='free_corner_motion_before_left',wait_frames=self.corner_still_wait,last2s_max_motion_m=motion,selected_corner='opposite diagonal',object_state_writes=0))
   if motion>.001 and self.corner_still_wait<400:
    self.corner_still_wait+=20;self.save();return [(command,'R_hold_for_free_corner_settle')]*20
   self.corner_still_wait=0
   try:
    rows,self.left=self.planner.choose_left(command,cloth,uv);self.stage='left_prepared'
   except RuntimeError as exc:
    self.presentation_attempt+=1;self.events.append(dict(stage='left_plan_retry',attempt=self.presentation_attempt,reason=str(exc)))
    if self.presentation_attempt>2:raise
    idx=int(self.ids[np.linalg.norm(uv[self.ids]-[.019,.49111111],axis=1).argmin()]);delta=np.clip(np.array([-2.03,-.23,.70])-cloth[idx],-.06,.06)
    mid=self.planner.geom(command,'R')[0];target=np.clip(mid+delta,[-2.23,-.55,1.03],[-1.97,-.28,1.18])
    self.events.append(dict(stage='measured_presentation_correction',measured_corner=cloth[idx].tolist(),requested_delta_m=delta.tolist(),actual_target_delta_m=(target-mid).tolist(),workspace_bound_is_action_planning_only=True))
    rows=self.planner.cartesian(command,'R',target,120,'R_adjust_free_corner')
    rows+=[(rows[-1][0],'R_wait_free_corner')]*60
    self.stage='right_presented' 
  elif self.stage=='left_prepared':
   point=cloth[self.left['material_vertex']];self.left['insertion_observed_xyz']=point.tolist();self.insertion_steps=0
   try:
    rows=self.planner.recenter_left(command,cloth,point,self.left);self.stage='left_tracking'
   except RuntimeError as exc:
    self.align_retries+=1;self.events.append(dict(stage='recenter_path_rejected',attempt=self.align_retries,reason=str(exc)))
    if self.align_retries>3:raise
    rows,self.left=self.planner.choose_left(command,cloth,uv);self.stage='left_prepared'
  elif self.stage=='left_tracking':
   point=cloth[self.left['material_vertex']];mid,h,poses,_=self.planner.geom(actual,'L');local=(cloth-h[:3,3])@h[:3,:3];bounds=[]
   for k,pts in zip(['inner','outer'],self.pad_points['l']):
    t=np.linalg.inv(h)@poses['gripper_l_'+k+'_link4'];points=pts@t[:3,:3].T+t[:3,3];bounds.append((points.min(0),points.max(0)))
   lo=np.maximum(bounds[0][0],bounds[1][0]);hi=np.minimum(bounds[0][1],bounds[1][1]);lo[1]=bounds[0][1][1]+.003;hi[1]=bounds[1][0][1]-.003
   lo[0]+=.002;hi[0]-=.002;lo[2]+=.012;hi[2]-=.012
   inside=np.all((local>lo)&(local<hi),axis=1);patch=np.linalg.norm(uv-np.asarray(self.left['uv']),axis=1)<.07;count=int((inside&patch).sum());other=int((inside&~patch).sum())
   self.events.append(dict(stage='left_open_pad_corridor',single_patch_nodes=count,other_nodes=other,intended_interior_point_contained=bool(inside[self.left['material_vertex']]),step=self.insertion_steps,observed_corner=point.tolist(),pad_center=mid.tolist(),physics_state_writes=0))
   if count>=12 and other==0 and bool(inside[self.left['material_vertex']]):
    rows=self.planner.close_left(command);self.stage='left_closed';self.wait=0
   else:
    self.insertion_steps+=1
    if self.insertion_steps>35:raise RuntimeError('Left live insertion never contained a single cloth patch; no empty close')
    direction=np.asarray(self.left['direction']);remaining=float((point-mid)@direction);normal_error=float(abs((point-mid)@h[:3,1]))
    if normal_error>.018 and remaining<.04:
     self.align_retries+=1
     if self.align_retries>3:raise RuntimeError('Left corner repeatedly displaced outside aperture during approach')
     rows=self.planner.cartesian(command,'L',mid-.04*direction,30,'L_clear_before_realign',np.asarray(self.left['locked_rotation']));self.stage='left_prepared'
    else:
     # Reobserve after every <=5mm forward increment. Center cross-axis only while outside the cloth.
     delta=min(.005,max(.001,remaining+.008));goal=mid+delta*direction
     if remaining>.04:goal+=(point-mid-remaining*direction)*.5
     rows=self.planner.micro_left(command,goal,np.asarray(self.left['locked_rotation']));self.stage='left_tracking'
  elif self.stage=='left_inserted':
   rows=self.planner.close_left(command);self.stage='left_closed';self.wait=0
  elif self.stage in ('left_closed','both_reinforced'):
   angles=np.abs([actual[n] for n in ['idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1']]);names=list(actual);ii=[names.index(n) for n in ['idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1']];window=np.asarray([r[1][ii] for r in recent[-11:]]);stable=len(window)==11 and np.ptp(window,axis=0).max()<.001
   right_angles=np.abs([actual[n]for n in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']]);ri=[names.index(n)for n in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']];rw=np.asarray([r[1][ri]for r in recent[-11:]]);right_stable=len(rw)==11 and np.ptp(rw,axis=0).max()<.001
   if self.stage=='both_reinforced' and (not right_stable or max(right_angles)>.0175):
    self.wait+=20;self.events.append(dict(stage='reinforced_right_stabilizing',angles=right_angles.tolist(),stable=bool(right_stable),wait_frames=self.wait))
    if self.wait<=160:self.save();return [(command,'LR_verify_reinforced_closure')]*20
    self.failure='Right jaw did not stabilize after native reclosure';self.stage='failed';self.save();return []
   self.events.append(dict(stage='left_closure_readback',angles=angles.tolist(),stable=bool(stable),grip=lg))
   if max(angles)>.0175 or not stable or not lg['patch_count'] or lg['other_patch_count']:
    self.wait+=20
    if self.wait<=160:rows=[(command,'L_verify_grip')]*20
    elif self.retries<2:
     self.retries+=1;goal=self.planner.robot.calibrate_loops(command|{'idx31_gripper_l_inner_joint1':-.35,'idx41_gripper_l_outer_joint1':.35});rows=self.planner.interpolate(command,goal,'L',60,'L_reopen_for_retry');self.stage='right_presented'
    else:self.failure='Left grip not confirmed after physical retries';self.stage='failed';self.save();return []
   else:
    # Opposite diagonal corners: use actual captured material distance, no adjacent-edge span cap.
    if not rg['patch_count']:self.failure='Right grip lost before two-hand unfolding';self.stage='failed';self.save();return []
    if self.stage=='left_closed':
     # Real native closing commands only, after both hands already physically captured cloth.
     # Hold arms fixed while native pads settle; never modify material vertices or masses.
     goal=self.planner.robot.calibrate_loops(command|{'idx31_gripper_l_inner_joint1':-.005,'idx41_gripper_l_outer_joint1':.005,'idx71_gripper_r_inner_joint1':-.005,'idx81_gripper_r_outer_joint1':.005})
     keys=[n for n in command if '_gripper_' in n]
     for i in range(1,61):
      a=i/60;a=a*a*(3-2*a);q=self.planner.robot.calibrate_loops(command|{n:command[n]+a*(goal[n]-command[n])for n in keys});rows.append((q,'LR_reinforce_native_closure'))
     rows += [(rows[-1][0],'LR_verify_reinforced_closure')]*100
     self.stage='both_reinforced';self.wait=0;self.events.append(dict(stage='both_reinforced',target_opening_rad=.005,arm_targets_unchanged=True,native_drive_properties_unchanged=True,object_state_writes=0));self.save();return rows
    self.events.append(dict(stage='reinforced_closure_readback',left_angles=angles.tolist(),right_angles=[abs(actual[n])for n in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']],left=lg,right=rg,stable=bool(stable)))
    self.events.append(self.planner.configure_material_frame(actual,cloth,uv,self.output));self.captured_right_uv=np.asarray(rg['material_uv_median']);self.captured_left_uv=np.asarray(lg['material_uv_median'])
    rest_span=float(np.linalg.norm(self.captured_left_uv-self.captured_right_uv));span=.97*rest_span;self.span=span
    self.events.append(dict(stage='measured_material_span',right_uv=self.captured_right_uv.tolist(),left_uv=self.captured_left_uv.tolist(),rest_span_m=rest_span,commanded_span_m=span,source='median material coordinates of actual bilateral pad-proximity patches'))
    center=np.array([-2.04,-.20,.85]);left=center+self.planner.material_span_axis*span/2;right=center-self.planner.material_span_axis*span/2;rows=self.planner.both_cartesian(command,left,right,480,'LR_unfold_equal_height');rows+=[(rows[-1][0],'LR_unfold_hold')]*100;self.stage='unfolded'
  elif self.stage=='unfolded':
   if not lg['patch_count'] or not rg['patch_count']:self.failure='Grip lost during unfolding';self.stage='failed';self.save();return []
   # Position the free hanging edge over the tabletop before lowering.
   center=np.array([-2.04,-.20,.78]);rows=self.planner.both_cartesian(command,center+self.planner.material_span_axis*self.span/2,center-self.planner.material_span_axis*self.span/2,180,'LR_over_table');self.stage='over_table'
  elif self.stage=='over_table':
   center=np.array([-2.04,-.20,.70]);rows=self.planner.both_cartesian(command,center+self.planner.material_span_axis*self.span/2,center-self.planner.material_span_axis*self.span/2,180,'LR_lower_diagonal');rows+=[(rows[-1][0],'LR_diagonal_table_contact_hold')]*60;self.stage='diagonal_lowered'
  elif self.stage=='diagonal_lowered':
   center=np.array([-2.04,-.20,.68]);rows=self.planner.both_cartesian(command,center+self.planner.material_span_axis*self.span/2,center-self.planner.material_span_axis*self.span/2,240,'LR_lower_diagonal');rows+=[(rows[-1][0],'LR_table_contact_hold')]*60;self.stage='diagonal_aligned_near_table'
  elif self.stage=='diagonal_aligned_near_table':
   rows=[(command,'LR_table_hold')]*100;self.stage='laid_held';self.events.append(dict(stage='untwisted_release_height',target_hand_center_z=.68,reason='Maintain material orientation and native reach; gravity lowers sheet after axial pad exit.',object_state_writes=0))
  elif self.stage=='laid_held':
   start=command.copy()
   for i in range(1,61):
    t=i/60;t=t*t*(3-2*t);v=start.copy()
    for side,inner,outer in [('L','idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1'),('R','idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1')]:
     opening=abs(start[inner])+t*(.35-abs(start[inner]));v[inner]=-opening;v[outer]=opening
    q=self.planner.robot.calibrate_loops(v);poses=self.planner.robot.fk(q)
    if max(self.planner.collision(poses,side).max()for side in ['L','R'])>.004:raise RuntimeError('Native open-pad release sweep intersects robot/table')
    rows.append((q,'LR_release_on_table'))
   self.released=True;self.stage='released'
  elif self.stage=='released':
   rows=self.planner.tip_open_pads(command);rows+=[(rows[-1][0],'LR_wait_after_pad_tip')]*60;self.stage='open_pads_tipped';self.events.append(dict(stage='open_pads_tipped',local_roll_deg={'L':90,'R':-90},native_joint_targets_only=True,object_state_writes=0))
  elif self.stage=='open_pads_tipped':
   rows=self.planner.withdraw_axial(command);rows+=[(rows[-1][0],'LR_open_pads_clear_hold')]*80;self.stage='open_pads_withdrawn'
  elif self.stage=='open_pads_withdrawn':
   rows=[(command,'tablecloth_final_settle')]*240;self.stage='settled'
  elif self.stage=='settled':
   q=cloth[self.ids];on=(abs(q[:,2]-.451)<.012)&(q[:,0]>=-2.35)&(q[:,0]<=-1.574649)&(q[:,1]>=-.7210436)&(q[:,1]<=.4789564);self.events.append(dict(stage='final_table_readback',fraction_on_table=float(on.mean()),bounds=[q.min(0).tolist(),q.max(0).tolist()]));self.complete=True;self.stage='complete';self.save();return []
  else:return []
  self.events.append(dict(previous=old,stage=self.stage,planned_frames=len(rows),right=rg,left=lg,left_observation=self.left,physics_state_writes=0));self.save();return rows

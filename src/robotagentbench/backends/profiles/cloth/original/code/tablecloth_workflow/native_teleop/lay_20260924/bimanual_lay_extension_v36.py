"""Future diagnostic: fit native right closure to cloth before prolonged holding.
Action-only aperture .009 replaces the additional .005 eye reclosure. Native
robot drives, contact gates, materials and accumulated object state are unchanged.
"""
from bimanual_lay_extension_v35 import LayExtension as Base
import numpy as np,json
class LayExtension(Base):
 def grip(self,actual,cloth,uv,side):
  if side!='r':return super().grip(actual,cloth,uv,side)
  selected=self.output/'selected_grasp_material.json'
  target=np.array([.32775,.028333])
  if selected.exists():
   data=json.loads(selected.read_text());target=np.asarray(data.get('selected_material_uv_m',target))
  poses=self.planner.robot.fk(actual);ds=[]
  for k,tree in zip(['inner','outer'],self.trees[side]):
   t=poses['gripper_'+side+'_'+k+'_link4'];ds.append(tree.query((cloth-t[:3,3])@t[:3,:3])[0])
  near=(ds[0]<.005)&(ds[1]<.005);mask=np.linalg.norm(uv-target,axis=1)<.07
  return dict(patch_count=int((near&mask).sum()),other_patch_count=int((near&~mask).sum()),material_uv_bounds=[uv[near&mask].min(0).tolist(),uv[near&mask].max(0).tolist()]if(near&mask).any()else None,material_uv_median=np.median(uv[near&mask],axis=0).tolist()if(near&mask).any()else None,selected_material_uv=target.tolist(),left_target_registered=None)
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage=='right_eye' and not self.early_right_reinforced:
   np.savez_compressed(self.output/'observed_right_eye_before_reclosure.npz',cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
   goal=self.planner.robot.calibrate_loops(command|{'idx71_gripper_r_inner_joint1':-.009,'idx81_gripper_r_outer_joint1':.009})
   rows=self.planner.interpolate(command,goal,'R',60,'R_fit_closure_at_eye');rows += [(rows[-1][0],'R_verify_fitted_eye_grip')]*100
   self.stage='right_eye_reinforced';self.events.append(dict(stage=self.stage,target_opening_rad=.009,previous_candidate_target_rad=.005,arm_targets_unchanged=True,object_state_writes=0));self.save();return rows
  return super().append_at_end(command,actual,cloth,uv,recent)

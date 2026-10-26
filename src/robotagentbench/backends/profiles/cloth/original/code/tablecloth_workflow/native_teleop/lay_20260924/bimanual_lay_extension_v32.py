"""Candidate: reinforce native right closure before the long presentation/left approach.
Valid physical states remain available for corrective native joint commands if planning fails.
"""
from bimanual_lay_extension_v31 import LayExtension as BaseExtension
import numpy as np
class LayExtension(BaseExtension):
 def __init__(self,output,cloth_ids):
  super().__init__(output,cloth_ids);self.early_right_reinforced=False;self.early_right_wait=0
 def append_at_end(self,command,actual,cloth,uv,recent):
  original=self.stage
  try:
   if self.stage=='right_eye' and not self.early_right_reinforced:
    if not self.grip(actual,cloth,uv,'r')['patch_count']:raise RuntimeError('Right cloth patch lost before early reclosure')
    np.savez_compressed(self.output/'observed_right_eye_before_reclosure.npz',cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
    goal=self.planner.robot.calibrate_loops(command|{'idx71_gripper_r_inner_joint1':-.005,'idx81_gripper_r_outer_joint1':.005});keys=[n for n in command if '_gripper_r_'in n];rows=[]
    for i in range(1,61):
     t=i/60;t=t*t*(3-2*t);calibrated=self.planner.robot.calibrate_loops(command|{n:command[n]+t*(goal[n]-command[n])for n in keys});q=command|{n:calibrated[n]for n in keys}
     if i%5==0 and self.planner.collision(self.planner.robot.fk(q),'R').max()>.004:raise RuntimeError('Early native right reclosure intersects existing geometry')
     rows.append((q,'R_reinforce_closure_at_eye'))
    rows += [(rows[-1][0],'R_verify_reinforced_eye_grip')]*100;self.stage='right_eye_reinforced';self.events.append(dict(stage=self.stage,target_opening_rad=.005,arm_targets_unchanged=True,object_state_writes=0));self.save();return rows
   if self.stage=='right_eye_reinforced':
    names=list(actual);ii=[names.index(n)for n in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']];window=np.asarray([r[1][ii]for r in recent[-11:]]);angles=np.abs([actual[names[i]]for i in ii]);stable=len(window)==11 and np.ptp(window,axis=0).max()<.001;grip=self.grip(actual,cloth,uv,'r');self.events.append(dict(stage='early_right_reclosure_readback',angles=angles.tolist(),stable=bool(stable),grip=grip))
    if not stable or max(angles)>.0175 or not grip['patch_count']:
     self.early_right_wait+=20
     if self.early_right_wait<=160:self.save();return [(command,'R_verify_reinforced_eye_grip')]*20
     raise RuntimeError('Early right reclosure not stable or cloth not retained')
    self.early_right_reinforced=True;self.stage='right_eye'
   rows=super().append_at_end(command,actual,cloth,uv,recent)
   if not rows and self.failure and original not in ['settled','awaiting_physical_repair']:
    raise RuntimeError(self.failure)
   return rows
  except RuntimeError as exc:
   if not np.isfinite(cloth).all() or not np.isfinite(list(actual.values())).all():raise
   self.events.append(dict(stage='valid_state_needs_action_replan',previous_stage=original,reason=str(exc),physical_state_restores=0));self.failure=None;self.complete=False;self.stage='awaiting_physical_repair';self.save()
   return super().append_at_end(command,actual,cloth,uv,recent)

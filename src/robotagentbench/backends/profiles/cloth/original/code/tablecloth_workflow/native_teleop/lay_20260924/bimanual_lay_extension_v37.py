"""Fresh diagnostic: pause after verified R eye grip for measured long-edge actions.
Adds lower-torso planning geometry and a real self-intersection rejection to flatness.
No objects attached, teleported, pinned or reset during an episode.
"""
from bimanual_lay_extension_v36 import LayExtension as Base
from bimanual_lay_extension_v31 import LayExtension as LiveRepair
from native_body1_collision_guard_v1 import LayPlanner
from cloth_self_intersection_audit_v1 import audit_self_intersections
import numpy as np,json
class LayExtension(Base):
 def __init__(self,output,cloth_ids):
  super().__init__(output,cloth_ids);self.planner=LayPlanner()
 def flat_readback(self,actual,cloth,uv):
  row=super().flat_readback(actual,cloth,uv);audit=audit_self_intersections(cloth,self.repair_faces);row['cloth_self_intersection_pairs_over_50micron']=audit['pairs_over_50micron'];row['cloth_max_edge_piercing_depth_m']=audit['max_edge_piercing_depth'];row['flat_and_clear_diagnostic']=bool(row['flat_and_clear_diagnostic'] and audit['pairs_over_50micron']==0);(self.output/f'self_intersection_repair_{self.repair_index:03d}.json').write_text(json.dumps(audit,indent=2));return row
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage=='right_eye_reinforced':
   names=list(actual);ii=[names.index(n)for n in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']];window=np.asarray([r[1][ii]for r in recent[-11:]]);angles=np.abs([actual[names[i]]for i in ii]);stable=len(window)==11 and np.ptp(window,axis=0).max()<.001;grip=self.grip(actual,cloth,uv,'r');valid=bool(stable and max(angles)<=.0175 and grip['patch_count']>0)
   self.events.append(dict(stage='right_eye_retained_before_longedge_plan',angles=angles.tolist(),stable=bool(stable),valid=valid,grip=grip,object_state_writes=0))
   if not valid and self.early_right_wait<160:self.early_right_wait+=20;self.save();return [(command,'R_verify_fitted_eye_grip')]*20
   self.early_right_reinforced=valid;self.stage='awaiting_physical_repair';self.save();return LiveRepair.append_at_end(self,command,actual,cloth,uv,recent)
  return super().append_at_end(command,actual,cloth,uv,recent)

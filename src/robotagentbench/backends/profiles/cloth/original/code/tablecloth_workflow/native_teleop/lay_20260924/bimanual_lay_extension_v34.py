"""Parallel-face candidate plus read-only native body/force snapshots at planning stops."""
from bimanual_lay_extension_v32 import LayExtension as Base
from bimanual_lay_extension_v31 import LayExtension as LiveRepair
from parallel_face_presentation_plan_v1 import LayPlanner
import numpy as np
class LayExtension(Base):
 def __init__(self,output,cloth_ids):
  super().__init__(output,cloth_ids);self.planner=LayPlanner()
 def flat_readback(self,actual,cloth,uv):
  if hasattr(self,'force_readback')and self.force_readback.rows:
   recent=self.force_readback.rows[-3:];keys=set.intersection(*(set(r)for r in recent));data={k:np.asarray([r[k]for r in recent])for k in keys};data.update(body_names=self.force_readback.body_names,joint_names=self.force_readback.names,cloth_q=cloth,uv=uv)
   np.savez_compressed(self.output/('native_snapshot_repair_'+str(self.repair_index).zfill(3)+'.npz'),**data)
   from native_pad_penetration_snapshot_v1 import audit
   audit(self.output,cloth,uv,recent[-1]['get_link_transforms'],self.force_readback.body_names,self.repair_index)
  return super().flat_readback(actual,cloth,uv)
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage=='left_torso_cleared' or (self.stage=='right_presented' and self.left_cleared_torso):
   self.left_cleared_torso=True;self.events.append(dict(stage='parallel_face_left_access_observed',object_state_writes=0));self.stage='awaiting_physical_repair';self.save();return LiveRepair.append_at_end(self,command,actual,cloth,uv,recent)
  return super().append_at_end(command,actual,cloth,uv,recent)

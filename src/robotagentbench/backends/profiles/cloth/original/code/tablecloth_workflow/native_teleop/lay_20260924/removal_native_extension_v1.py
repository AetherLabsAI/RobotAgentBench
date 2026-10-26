"""Physical basket release checks and optional action-only live correction."""
from bimanual_lay_extension_v31 import LayExtension as Base
from native_body1_collision_guard_v1 import LayPlanner
from scipy.spatial import cKDTree
import numpy as np,json
class LayExtension(Base):
 def __init__(self,output,cloth_ids):
  super().__init__(output,cloth_ids);self.ids=np.flatnonzero(cloth_ids==0);self.planner=LayPlanner();self.stage='awaiting_physical_repair';self.released=False
 def observe(self,*args):pass
 def flat_readback(self,actual,cloth,uv):
  q=cloth[self.ids];inside=np.all((q>=[-2.419,-1.013,.285])&(q<=[-2.061,-.737,.465]),axis=1);xy=np.all((q[:,:2]>=[-2.419,-1.013])&(q[:,:2]<=[-2.061,-.737]),axis=1);poses=self.planner.robot.fk(actual);gaps={}
  for side in ['l','r']:
   pts=[]
   for kind,local in zip(['inner','outer'],self.pad_points[side]):
    t=poses['gripper_'+side+'_'+kind+'_link4'];pts.append(local@t[:3,:3].T+t[:3,3])
   gaps[side]=float(cKDTree(np.concatenate(pts)).query(q)[0].min())
  ok=bool(xy.mean()>=.9 and inside.mean()>=.9 and min(gaps.values())>.005)
  return dict(basket_xy_fraction=float(xy.mean()),basket_xyz_fraction=float(inside.mean()),minimum_pad_distances=gaps,flat_and_clear_diagnostic=ok,meaning='Removal basket containment and pad release, not laying flatness',owner_approved=False,production_count=0)
 def append_at_end(self,command,actual,cloth,uv,recent):
  row=self.flat_readback(actual,cloth,uv);self.events.append(dict(stage='removal_physical_readback',metrics=row))
  if row['flat_and_clear_diagnostic']:
   self.complete=True;self.stage='complete';self.save();return []
  self.stage='awaiting_physical_repair';return super().append_at_end(command,actual,cloth,uv,recent)

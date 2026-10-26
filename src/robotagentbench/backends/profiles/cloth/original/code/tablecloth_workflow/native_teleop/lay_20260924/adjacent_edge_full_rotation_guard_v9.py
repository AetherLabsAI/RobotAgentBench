"""Full hand poses with continuous redundant-elbow search; native collision gates retained."""
from adjacent_edge_full_rotation_guard_v7 import LayPlanner as Base
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(Base):
 def solve_captured_full(self,v,side,point,rotation,mid_seed):
  cap=self.captures[side];names=self.robot.arm_names(side)
  def evaluate(seed):
   q,err=self.robot.solve(seed,side,point,cap['point_local'],rotation,max_steps=320);_,h,poses,out=self.geom(q,side);pos=float(np.linalg.norm(h[:3,:3]@cap['point_local']+h[:3,3]-point));ang=float(np.rad2deg(Rotation.from_matrix(rotation.T@h[:3,:3]).magnitude()));coll=float(self.collision(poses,side).max());return q,(pos,ang,coll,out)
  q,metrics=evaluate(v)
  def valid(m):return m[0]<=.002 and m[1]<=2 and m[2]<=.004 and m[3]>=.005
  if valid(metrics):return q
  poses,axes=self.robot.fk(q,True);h=poses['gripper_'+side.lower()+'_base_link'];x=h[:3,:3]@cap['point_local']+h[:3,3];J=np.vstack([np.stack([np.cross(axes[n][1],x-axes[n][0])for n in names],axis=1),.12*np.stack([axes[n][1]for n in names],axis=1)]);null=np.linalg.svd(J,full_matrices=True)[2][-1];failures=[metrics]
  for step in [.06,-.06,.12,-.12,.20,-.20,.30,-.30,.5,-.5,.75,-.75,1.,-1.]:
   seed=q.copy()
   for n,d in zip(names,null):lo,hi=self.robot.by_name[n]['limits'];seed[n]=float(np.clip(q[n]+step*d,lo+1e-5,hi-1e-5))
   g,m=evaluate(seed);failures.append(m)
   if valid(m):return g
  raise RuntimeError(('Full hand pose redundant search failed',side,failures))

"""Local analytic normal Jacobian plus gradual outward-elbow clearance."""
from adjacent_edge_direct_normal_path_v29 import LayPlanner as Base
from adjacent_edge_local_normal_ik_v30 import LayPlanner as Refine
import numpy as np
class LayPlanner(Base):
 def solve_captured_normal(self,v,side,point,normal,mid_seed):
  names=self.robot.arm_names(side);cap=self.captures[side];q=v.copy();prior=np.array([v[n]for n in names]);bounds=np.array([self.robot.by_name[n]['limits']for n in names]);lo=np.maximum(bounds[:,0]+1e-6,prior-.12);hi=np.minimum(bounds[:,1]-1e-6,prior+.12);normal=np.asarray(normal)/np.linalg.norm(normal);out0=self.geom(v,side)[3];out_goal=min(.025,out0+.004);sign=1 if side=='L'else -1
  for k in range(180):
   poses,axes=self.robot.fk(q,True);h=poses['gripper_'+side.lower()+'_base_link'];actual=h[:3,:3]@cap['point_local']+h[:3,3];n=h[:3,1];elbow=axes[names[2]][0];shoulder=axes[names[0]][0];out=sign*(elbow[1]-shoulder[1]);active=out<out_goal;jout=np.array([sign*np.cross(axes[j][1],elbow-axes[j][0])[1]if index<2 and active else 0. for index,j in enumerate(names)]);error=np.r_[point-actual,.12*(normal-n),.25*max(0,out_goal-out)];jac=np.vstack([np.stack([np.cross(axes[j][1],actual-axes[j][0])for j in names],axis=1),.12*np.stack([np.cross(axes[j][1],n)for j in names],axis=1),.25*jout])
   if np.linalg.norm(error)<.00015:break
   delta=jac.T@np.linalg.solve(jac@jac.T+.006**2*np.eye(7),error);delta*=min(1,.04/max(abs(delta).max(),1e-12));x=np.clip(np.array([q[j]for j in names])+delta,lo,hi);q.update(zip(names,map(float,x)))
  _,h,poses,out=self.geom(q,side);err=float(np.linalg.norm(h[:3,:3]@cap['point_local']+h[:3,3]-point));angle=float(np.rad2deg(np.arccos(np.clip(h[:3,1]@normal,-1,1))));coll=float(self.collision(poses,side).max())
  if err<=.002 and angle<=2 and coll<=.004 and out>=.005:return q
  print('LOCAL REFINE',side,err,angle,coll,out,flush=True)
  return Refine.solve_captured_normal(self,v,side,point,normal,mid_seed)

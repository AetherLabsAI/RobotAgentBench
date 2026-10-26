"""Full hand pose planning with collision-aware continuous refinement."""
from adjacent_edge_coherent_rotation_guard_v12 import LayPlanner as Base
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(Base):
 def solve_captured_full(self,v,side,point,rotation,mid_seed):
  try:return super().solve_captured_full(v,side,point,rotation,mid_seed)
  except RuntimeError as e:first=str(e)
  names=self.robot.arm_names(side);cap=self.captures[side];prior=np.array([v[n]for n in names]);bounds=np.array([self.robot.by_name[n]['limits']for n in names]);lo=np.maximum(bounds[:,0]+1e-6,prior-.5);hi=np.minimum(bounds[:,1]-1e-6,prior+.5)
  def residual(x):
   q=v|dict(zip(names,x));_,h,poses,out=self.geom(q,side);actual=h[:3,:3]@cap['point_local']+h[:3,3]
   return np.r_[2*(actual-point),.16*Rotation.from_matrix(rotation.T@h[:3,:3]).as_rotvec(),3*np.maximum(0,self.collision(poses,side)-.0032),2*max(0,.007-out),(x-prior)*1e-5]
  fit=least_squares(residual,np.clip(prior,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=120,ftol=1e-9,gtol=1e-9,xtol=1e-9);q=v|dict(zip(names,fit.x));_,h,poses,out=self.geom(q,side);pos=float(np.linalg.norm(h[:3,:3]@cap['point_local']+h[:3,3]-point));ang=float(np.rad2deg(Rotation.from_matrix(rotation.T@h[:3,:3]).magnitude()));coll=float(self.collision(poses,side).max())
  if pos>.002 or ang>2 or coll>.004 or out<.005:raise RuntimeError(('Full hand collision-aware solve rejected',side,pos,ang,coll,out))
  return q

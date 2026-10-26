"""Bounded least-squares pose IK with unchanged physical limits and clearance gates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from held_hand_cartesian_guard_v1 import LayPlanner as Base
class LayPlanner(Base):
 def solve_full(self,v,side,target,rotation):
  mid,h,_,_=self.geom(v,side);local=(mid-h[:3,3])@h[:3,:3];arm=self.robot.arm_names(side);seed=np.array([v[n]for n in arm]);bounds=np.array([self.robot.by_name[n]['limits']for n in arm]);lo=np.maximum(bounds[:,0]+1e-7,seed-.25);hi=np.minimum(bounds[:,1]-1e-7,seed+.25);link='gripper_'+side.lower()+'_base_link'
  def residual(x):
   hh=self.robot.fk(v|dict(zip(arm,x)))[link];point=hh[:3,3]+hh[:3,:3]@local;return np.r_[point-target,.1*Rotation.from_matrix(rotation@hh[:3,:3].T).as_rotvec(),1e-6*(x-seed)]
  fit=least_squares(residual,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=200,ftol=1e-10,gtol=1e-10,xtol=1e-10);q=v|dict(zip(arm,fit.x.tolist()));gm,gh,poses,out=self.geom(q,side);error=float(np.linalg.norm(gm-target));angle=float(np.rad2deg(Rotation.from_matrix(rotation.T@gh[:3,:3]).magnitude()));collision=float(self.collision(poses,side).max());assert error<.002 and angle<2 and collision<=.004 and out>=.005,('bounded pose',side,error,angle,collision,out);return q

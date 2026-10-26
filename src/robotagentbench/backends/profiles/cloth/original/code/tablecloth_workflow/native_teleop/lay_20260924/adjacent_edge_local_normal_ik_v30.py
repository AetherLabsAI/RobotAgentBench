"""Continuous local captured-point normal solve without alternate branch seeding."""
from adjacent_edge_direct_normal_path_v29 import LayPlanner as Base
from scipy.optimize import least_squares
import numpy as np
class LayPlanner(Base):
 def solve_captured_normal(self,v,side,point,normal,mid_seed):
  names=self.robot.arm_names(side);prior=np.array([v[n]for n in names]);cap=self.captures[side];bounds=np.array([self.robot.by_name[n]['limits']for n in names]);lo=np.maximum(bounds[:,0]+1e-6,prior-.12);hi=np.minimum(bounds[:,1]-1e-6,prior+.12);normal=np.asarray(normal)/np.linalg.norm(normal)
  def measure(x):
   q=v|dict(zip(names,x));_,h,poses,out=self.geom(q,side);return q,h[:3,:3]@cap['point_local']+h[:3,3]-point,h[:3,1],self.collision(poses,side),out
  def residual(x):
   q,pos,n,coll,out=measure(x)
   return np.r_[12*pos,.6*(n-normal),12*np.maximum(0,coll-.0025),4*max(0,.022-out),.004*(x-prior)]
  fit=least_squares(residual,np.clip(prior,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=100,gtol=1e-9,xtol=1e-9,ftol=1e-9);q,pos,n,coll,out=measure(fit.x);err=float(np.linalg.norm(pos));ang=float(np.rad2deg(np.arccos(np.clip(n@normal,-1,1))))
  if err>.002 or ang>2 or coll.max()>.004 or out<.005:raise RuntimeError(('local normal IK',side,err,ang,float(coll.max()),float(out)))
  return q

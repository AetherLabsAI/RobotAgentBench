"""Continuous IK branch and per-frame held-hand Cartesian path validation.
Native joint actions only. Body/contact/geometry gates remain unchanged.
"""
from native_body1_collision_guard_v1 import LayPlanner as Base
from scipy.spatial.transform import Rotation,Slerp
import numpy as np
class LayPlanner(Base):
 def solve_full(self,v,side,target,rotation):
  mid,h,_,_=self.geom(v,side);local=(mid-h[:3,3])@h[:3,:3]
  q,err=self.robot.solve(v,side,target,local,rotation,max_steps=320)
  gm,gh,poses,out=self.geom(q,side);oe=np.rad2deg(Rotation.from_matrix(rotation.T@gh[:3,:3]).magnitude());coll=self.collision(poses,side).max()
  if np.linalg.norm(gm-target)>=.002 or oe>=2 or coll>.004 or out<.005:raise RuntimeError(('Current IK branch cannot track requested pose',side,list(target),float(err),float(oe),float(coll),float(out)))
  return q
 def interpolate(self,v,goal,side,frames,phase):
  rows=super().interpolate(v,goal,side,frames,phase);start,h,_,_=self.geom(v,side);end,hh,_,_=self.geom(goal,side);turn=Slerp([0,1],Rotation.from_matrix(np.array([h[:3,:3],hh[:3,:3]])));worst=0.;angle=0.
  for j,(q,_)in enumerate(rows,1):
   t=j/len(rows);t=t*t*(3-2*t);mid,actual,_,_=self.geom(q,side);err=float(np.linalg.norm(mid-(start+t*(end-start))));rot=float(np.rad2deg(Rotation.from_matrix(turn(t).as_matrix().T@actual[:3,:3]).magnitude()));worst=max(worst,err);angle=max(angle,rot)
  if worst>.004 or angle>2:raise RuntimeError(('Joint interpolation leaves held-hand Cartesian segment',phase,worst,angle,len(rows)))
  self.last_cartesian_segment=dict(max_position_deviation_m=worst,max_rotation_deviation_deg=angle,frames=len(rows));return rows

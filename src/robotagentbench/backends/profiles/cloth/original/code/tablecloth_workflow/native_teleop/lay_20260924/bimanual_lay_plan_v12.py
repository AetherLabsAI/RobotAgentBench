"""Run-local action planning accelerator. Same native collision/pose gates.
A nearby Jacobian solution is accepted only after all original hard checks;
otherwise retain the constrained optimizer. Never changes simulator state.
"""
from bimanual_lay_plan_v11 import LayPlanner as BasePlanner,DATA
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(BasePlanner):
 def solve(self,v,side,target,orientation=None,guide=None):
  mid,h,_,_=self.geom(v,side);local=(mid-h[:3,3])@h[:3,:3];goal,err=self.robot.solve(v,side,target,local,orientation,max_steps=60);mid,h,poses,out=self.geom(goal,side);names=self.robot.arm_names(side);delta=max(abs(goal[n]-v[n])for n in names);oe=0 if orientation is None else float(np.rad2deg(Rotation.from_matrix(orientation.T@h[:3,:3]).magnitude()));collision=float(self.collision(poses,side).max());bend=np.rad2deg(np.abs([goal[names[3]],goal[names[5]]]))
  if np.linalg.norm(mid-target)<=.002 and oe<=2 and out>=.005 and collision<=.004 and delta<=.20:
   self.last_metrics=dict(side=side,error_m=float(np.linalg.norm(mid-target)),outward_m=out,collision_residual_m=collision,orientation_deg=oe,bend_deg=bend.tolist(),method='nearby_jacobian_with_original_gates');return goal
  return super().solve(v,side,target,orientation,guide)

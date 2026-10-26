"""Run-local posture optimization candidate; no native joint-limit change."""
from bimanual_lay_plan_v15 import LayPlanner as BasePlanner,DATA
from bimanual_lay_plan_v11 import LayPlanner as ConstrainedPlanner
import numpy as np
class LayPlanner(BasePlanner):
 def solve(self,v,side,target,orientation=None,guide=None):
  goal=super().solve(v,side,target,orientation,guide)
  names=self.robot.arm_names(side);bend=[names[3],names[5]]
  old=np.maximum(0,np.abs([v[n]for n in bend])-np.pi/3).sum()
  new=np.maximum(0,np.abs([goal[n]for n in bend])-np.pi/3).sum()
  if orientation is None and new>old+np.deg2rad(.1):
   return ConstrainedPlanner.solve(self,v,side,target,orientation,guide)
  return goal

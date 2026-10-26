"""Keep143 right presentation solver; use checked nearby IK for left feedback."""
from bimanual_lay_plan_v13 import LayPlanner as BasePlanner,DATA
from bimanual_lay_plan_v11 import LayPlanner as ConstrainedPlanner
class LayPlanner(BasePlanner):
 def solve(self,v,side,target,orientation=None,guide=None):
  if side=='R' and getattr(self,'_presentation_solver',False):return ConstrainedPlanner.solve(self,v,side,target,orientation,guide)
  return super().solve(v,side,target,orientation,guide)
 def cartesian(self,v,side,target,frames,phase,orientation=None):
  old=getattr(self,'_presentation_solver',False);self._presentation_solver=phase in ['R_present_for_left','R_adjust_free_corner']
  try:return super().cartesian(v,side,target,frames,phase,orientation)
  finally:self._presentation_solver=old

"""Full-stroke L search plus continuous insertion branch, no held-point writes."""
from left_adjacent_body1_fullstroke_v1 import LayPlanner as Search
from held_hand_cartesian_guard_v1 import LayPlanner as Held
from native_body1_collision_guard_v1 import LayPlanner as Native
class LayPlanner(Search,Held):
 def interpolate(self,v,goal,side,frames,phase):
  changed=max(abs(goal[n]-v[n])for n in self.robot.arm_names(side))
  if changed<1e-9 or ('prepare'in phase and 'insertion'not in phase) or 'unfold_elbow'in phase:return Native.interpolate(self,v,goal,side,frames,phase)
  return Held.interpolate(self,v,goal,side,frames,phase)

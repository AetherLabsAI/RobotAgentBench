"""Continuous held-edge path with contact-plane orientation, free in-plane wrist roll.
The endpoint objective is physical pad-normal alignment. All collision, captured
span, native rate and positional gates remain. Full axial rotation is measured,
not prescribed, since the user accepts the native camera-side pose.
"""
from adjacent_edge_coherent_rotation_guard_v21 import LayPlanner as Base
from adjacent_edge_cartesian_guard_v4 import LayPlanner as Normal
class LayPlanner(Base):
 def solve_captured_full(self,v,side,point,rotation,mid_seed):
  return Normal.solve_captured_normal(self,v,side,point,rotation[:,1],mid_seed)

"""Additional action-planning guard for native lower torso, discovered in164.
Uses unchanged native convex collision vertices. No physics/asset overrides.
"""
import json,numpy as np
from scipy.spatial import ConvexHull
from bimanual_lay_plan_v30 import LayPlanner as BasePlanner,DATA
from left_adjacent_face_angles_v5 import LayPlanner as SearchPlanner
from adjacent_edge_lay_plan_v3 import LayPlanner as EdgePlanner
class Body1Guard:
 def __init__(self):
  super().__init__();meta=json.loads((DATA/'robot_native_collision.json').read_text());self.body1_planes=[ConvexHull(np.asarray(s['vertices'])).equations for s in meta['nonhand_collision_shapes']if s['link']=='body_link1'];assert self.body1_planes
 def body1_collision(self,poses,side):
  out=[];t=poses['body_link1'];s=side.lower();links=['arm_'+s+'_link'+str(k)for k in range(2,8)]+['arm_'+s+'_end_link']
  for n,points in self.shapes.items():
   if not(n.startswith('gripper_'+s+'_')or n in links):continue
   h=poses[n];local=(points@h[:3,:3].T+h[:3,3]-t[:3,3])@t[:3,:3]
   for eq in self.body1_planes:out.append(max(0.,.012-float((local@eq[:,:3].T+eq[:,3]).max(axis=1).min())))
  return np.asarray(out)
 def collision(self,poses,side):return np.r_[super().collision(poses,side),self.body1_collision(poses,side)]
class LayPlanner(Body1Guard,BasePlanner):pass
class LeftSearchPlanner(Body1Guard,SearchPlanner):pass
class AdjacentEdgePlanner(Body1Guard,EdgePlanner):pass

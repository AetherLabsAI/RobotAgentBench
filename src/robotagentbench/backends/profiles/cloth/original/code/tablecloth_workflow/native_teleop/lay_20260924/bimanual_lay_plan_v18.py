"""Action-only repair for measured native left-pad/torso collision in147.
Checks native torso convex surfaces and clears the still-open left hand first.
No native asset, collision filter, drive, solver or object state is changed.
"""
from bimanual_lay_plan_v16 import LayPlanner as BasePlanner,DATA
from scipy.spatial import ConvexHull,cKDTree
import numpy as np,json
class LayPlanner(BasePlanner):
 def __init__(self):
  super().__init__();meta=json.loads((DATA/'robot_native_collision.json').read_text())
  self.torso_planes=[ConvexHull(np.asarray(s['vertices'])).equations for s in meta['nonhand_collision_shapes']if s['link']=='body_link2']
 def collision(self,poses,side):
  out=super().collision(poses,side).tolist();t=poses['body_link2'];prefix='gripper_'+side.lower()+'_';links=['arm_'+side.lower()+'_link'+str(k)for k in [3,4,5,6]]+['arm_'+side.lower()+'_end_link']
  for n,points in self.shapes.items():
   if not(n.startswith(prefix)or n in links):continue
   h=poses[n];local=(points@h[:3,:3].T+h[:3,3]-t[:3,3])@t[:3,:3]
   for eq in self.torso_planes:out.append(max(0.,.012-float((local@eq[:,:3].T+eq[:,3]).max(axis=1).min())))
  return np.asarray(out)
 def clear_left_torso(self,v,cloth):
  mid,h,_,_=self.geom(v,'L');failures=[];self.cloth_tree=cKDTree(cloth)
  try:
   for offset in [[0,.08,0],[.08,.08,0],[.08,0,0],[.08,.12,.04],[.12,.08,.04]]:
    try:
     rows=self.cartesian(v,'L',mid+offset,120,'L_clear_torso_before_aperture')
     # Check the narrower aperture at the cleared endpoint before committing.
     end=rows[-1][0];narrow=self.robot.calibrate_loops(end|{'idx31_gripper_l_inner_joint1':-.35,'idx41_gripper_l_outer_joint1':.35})
     self.interpolate(end,narrow,'L',40,'offline_aperture_clearance_check')
     self.last_torso_clearance_plan=dict(offset_m=offset,frames=len(rows),keeps_original_aperture=True,native_collision_surfaces=True,object_state_writes=0)
     return rows
    except RuntimeError as e:failures.append(str(e))
   raise RuntimeError(('No collision-cleared left staging pose',failures))
  finally:self.cloth_tree=None

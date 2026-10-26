"""Action-only presentation turns plus conservative fixed head clearance."""
from bimanual_lay_plan_v9 import LayPlanner as BasePlanner,DATA
import numpy as np,json
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
class LayPlanner(BasePlanner):
 def __init__(self):
  super().__init__();meta=json.loads((DATA/'robot_native_collision.json').read_text());self.head_planes={}
  for row in meta['nonhand_collision_shapes']:
   if row['link'].startswith('head_'):
    self.head_planes.setdefault(row['link'],[]).append(ConvexHull(np.asarray(row['vertices'])).equations)
 def collision(self,poses,side):
  out=super().collision(poses,side).tolist()
  moving=[n for n in self.shapes if n.startswith('gripper_'+side.lower()+'_') or n in ['arm_'+side.lower()+'_link'+k for k in ['3','5','6','7']]]
  for head,sets in self.head_planes.items():
   h=poses[head]
   for n in moving:
    t=poses[n];world=self.shapes[n]@t[:3,:3].T+t[:3,3];local=(world-h[:3,3])@h[:3,:3]
    for eq in sets:out.append(max(0,.012-float((local@eq[:,:3].T+eq[:,3]).max(axis=1).min())))
  return np.asarray(out)
 def turn_right(self,v,degrees,frames=240,target=None):
  mid,h,_,_=self.geom(v,'R');rotation=h[:3,:3];current=v;rows=[];target=mid if target is None else np.asarray(target)
  for k in range(1,21):
   t=k/20;t=t*t*(3-2*t);orientation=Rotation.from_euler('z',degrees*t,degrees=True).as_matrix()@rotation
   goal=self.solve(current,'R',mid+t*(target-mid),orientation);part=self.interpolate(current,goal,'R',max(1,frames//20),'R_turn_cloth_for_left');rows+=part;current=rows[-1][0]
  return rows

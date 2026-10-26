"""Necessary geometry check before lift; never claims a force-confirmed grasp."""
import json,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
class CookedClosureGuard:
 def __init__(self,path):
  d=json.loads(path.read_text());self.planes={side:[ConvexHull(np.asarray(c['vertices_body'])).equations for r in d[side] for c in r['convexes']]for side in ('inner','outer')}
 def check(self,cloth_point,bodies,body_names):
  h=bodies[body_names.index('gripper_r_base_link')];axis=Rotation.from_quat(h[3:]).as_matrix()[:,1];ends={};depth={}
  for side,planes in self.planes.items():
   b=bodies[body_names.index('gripper_r_'+side+'_link4')];r=Rotation.from_quat(b[3:]).as_matrix();v=(cloth_point-b[:3])@r;ray=axis@r;intervals=[];signed=[]
   for eq in planes:
    x=eq[:,:3]@v+eq[:,3];a=eq[:,:3]@ray;signed.append(float(x.max()));pos=a>1e-10;neg=a<-1e-10
    if np.any(x[~(pos|neg)]>1e-9):continue
    lo=float(np.max(-x[neg]/a[neg])) if neg.any()else -np.inf;hi=float(np.min(-x[pos]/a[pos])) if pos.any()else np.inf
    if lo<=hi:intervals.append((lo,hi))
   ends[side]=intervals;depth[side]=min(signed)
  gap=None;between=False
  if ends['inner'] and ends['outer']:
   left=max(t[1]for t in ends['inner']);right=min(t[0]for t in ends['outer']);gap=right-left;between=left<=0<=right
  ready=bool(gap is not None and gap>0 and between and min(depth.values())>=0)
  return dict(geometry_allows_lift=ready,cooked_gap_m=gap,selected_center_between_surfaces=between,signed_plane_distances_m=depth,force_confirmed=False,scope='Necessary geometric check only; force/retention checks still required')

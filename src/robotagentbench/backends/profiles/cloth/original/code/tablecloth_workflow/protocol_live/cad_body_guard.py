"""CAD vertex versus native body primitives; exact conservative AABB broad phase."""
import itertools,numpy as np
from scipy.spatial.transform import Rotation
import audit_arm_posture as audit
class CadBodyGuard:
 def __init__(self,cal):
  self.vertices={l:np.unique(cal['points_'+l],axis=0) for l in cal['links'].tolist() if l.startswith('gripper_') and 'points_'+l in cal};self.bounds=[]
  for shape in audit.shapes:
   p=shape['primitive'];h=np.array(p['half_extents'] if p['type']=='box' else [p['radius'],p['radius'],p.get('half_height',p['radius'])]);cube=np.array(list(itertools.product([-1,1],repeat=3)))*h;rot=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();self.bounds.append(cube@rot.T+p['position'])
 def check(self,poses):
  hit=[];bounds=[]
  for shape,cube in zip(audit.shapes,self.bounds):
   T=poses[shape['link']];world=cube@T[:3,:3].T+T[:3,3];bounds.append((world.min(0),world.max(0)))
  for link,p in self.vertices.items():
   T=poses[link];world=p@T[:3,:3].T+T[:3,3];lo,hi=world.min(0),world.max(0)
   for shape,(blo,bhi) in zip(audit.shapes,bounds):
    other=shape['link']
    if frozenset((link,other)) in audit.adj or np.any(hi<blo) or np.any(lo>bhi):continue
    B=poses[other];depth=float(audit.inside_primitive(shape['primitive'],(world-B[:3,3])@B[:3,:3]).max())
    if depth>.002:hit.append(dict(source=link,target=other,penetration_mm=depth*1000))
  return hit

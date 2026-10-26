"""Read-only cloth vertex versus native cooked pad interiors, actual rigid body poses."""
import json,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
def audit(output,cloth,uv,bodies,body_names,index):
 results={}
 for side in ['l','r']:
  d=json.loads((output/('native_pad_cooked_convexes.json'if side=='r'else'native_left_pad_cooked_convexes.json')).read_text());parts={}
  for kind in ['inner','outer']:
   b=bodies[body_names.index('gripper_'+side+'_'+kind+'_link4')];local=(cloth-b[:3])@Rotation.from_quat(b[3:]).as_matrix();depth=np.full(len(cloth),-np.inf)
   for group in d[kind]:
    for c in group['convexes']:
     verts=np.asarray(c['vertices_body']);ids=np.flatnonzero(np.all(local>=verts.min(0),axis=1)&np.all(local<=verts.max(0),axis=1))
     if not len(ids):continue
     eq=ConvexHull(verts).equations;signed=(local[ids]@eq[:,:3].T+eq[:,3]).max(1);depth[ids]=np.maximum(depth[ids],-signed)
   ids=np.flatnonzero(depth>.0005);parts[kind]=dict(max_interior_depth_m=max(0.,float(depth.max())),vertices_beyond_0p5mm=int(len(ids)),vertices_beyond_1mm=int((depth>.001).sum()),vertices_beyond_2mm=int((depth>.002).sum()),material_uv=uv[ids].tolist(),indices=ids.tolist())
  results[side]=parts
 (output/('native_pad_penetration_repair_'+str(index).zfill(3)+'.json')).write_text(json.dumps(dict(actual_native_body_poses=True,read_only=True,force_confirmed=False,hands=results),indent=2))

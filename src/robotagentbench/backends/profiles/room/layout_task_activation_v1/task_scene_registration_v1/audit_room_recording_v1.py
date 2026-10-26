"""Audit actual visible object bounds after the native rigid-body recording."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Gf
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
r=json.loads((a.run/'navigation_result.json').read_text());data=np.load(a.run/'native_poses.npz')
scene=Usd.Stage.Open(str(a.run/'native_initial.usdc'));xf=UsdGeom.XformCache()
records=json.loads((a.run/'composition.json').read_text())['objects']
paths=r['body_paths'];path_ids={p:i for i,p in enumerate(paths)}
matrices={p:xf.GetLocalToWorldTransform(scene.GetPrimAtPath(p)) for p in paths}
def world(p,v):
 scale=Gf.Transform(matrices[p]).GetScale()
 m=Gf.Matrix4d().SetScale(scale)*Gf.Matrix4d().SetRotate(Gf.Quatd(float(v[6]),Gf.Vec3d(*map(float,v[3:6]))))
 m.SetTranslateOnly(Gf.Vec3d(*map(float,v[:3])));return np.array(m)
rows=[]
for record in records:
 meshes=[]
 for p in Usd.PrimRange(scene.GetPrimAtPath(record['path'])):
  im=UsdGeom.Imageable(p)
  if not im:continue
  if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render'):continue
  if p.IsA(UsdGeom.Mesh):vs=UsdGeom.Mesh(p).GetPointsAttr().Get()
  elif p.IsA(UsdGeom.Cube):
   h=UsdGeom.Cube(p).GetSizeAttr().Get()/2;vs=[(x,y,z) for x in (-h,h) for y in (-h,h) for z in (-h,h)]
  else:continue
  if vs is None or not len(vs):continue
  pts=np.c_[np.array(vs),np.ones(len(vs))]@np.array(xf.GetLocalToWorldTransform(p))
  ancestors=[x for x in paths if str(p.GetPath()).startswith(x+'/') or str(p.GetPath())==x]
  body=max(ancestors,key=len) if ancestors else None
  meshes.append((body,pts if body is None else pts@np.linalg.inv(np.array(matrices[body]))))
 bounds=[]
 for tick in [0,39,len(data['bodies'])-1]:
  pts=np.concatenate([(v if body is None else v@world(body,data['bodies'][tick,path_ids[body]]))[:,:3] for body,v in meshes])
  bounds.append([pts.min(axis=0).tolist(),pts.max(axis=0).tolist()])
 row={'id':record['id'],'path':record['path'],'bounds_at_start_2s_end':bounds}
 if '/Sofa_' in record['path']:
  row['seat_support_retained']=bounds[-1][0][2]>.42
 if '/Shelf_' in record['path']:
  expected=.75 if 'tier2' in record['path'] else 1.1825
  row['shelf_support_retained']=abs(bounds[-1][0][2]-expected)<.025
 rows.append(row)
out={'native_status':r['status'],'control_hz':r['control_hz'],'seconds':len(r['rows'])/20,'objects':rows,
     'support_checks_passed':all(x.get('seat_support_retained',True) and x.get('shelf_support_retained',True) for x in rows),
     'scope':'Measured final mesh bounds; not complete manipulation or full collision qualification'}
(a.run/'settled_bounds_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='objects'}))
for x in rows:
 if '/Sofa_' in x['path'] or not x.get('shelf_support_retained',True):print(x['id'],x['bounds_at_start_2s_end'][-1])

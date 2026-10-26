"""Read-only diagnosis of the failed fridge preview camera."""
import json,sys
from pathlib import Path
from pxr import Usd,UsdGeom,Gf
import numpy as np
from workspace_views import fit
folder=Path(sys.argv[1])
record=json.loads((folder/'composition.json').read_text())
row=next(o for o in record['objects'] if o['id']=='refrigerator')
stage=Usd.Stage.OpenMasked(str(folder/'room.usda'),Usd.StagePopulationMask([row['path']]))
prim=stage.GetPrimAtPath(row['path'])
m=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
front=list(m.TransformDir(Gf.Vec3d(0,-1,0)))
lo,hi=np.asarray(row['visible_bounds_world_m']);eye,target=fit([lo-.04,hi+.04],front[:2],.25)
print(json.dumps(dict(path=row['path'],active=prim.IsActive(),loaded=prim.IsLoaded(),
    meshes=sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim)),
    visibility=UsdGeom.Imageable(prim).ComputeVisibility(),
    transformed_camera_front=front,horizontal_front_length=float(np.linalg.norm(front[:2])),
    eye=eye,target=target,camera_distance=float(np.linalg.norm(np.array(eye)-target)),
    eye_inside_fridge_bounds=bool(np.all(np.array(eye)>=lo)&np.all(np.array(eye)<=hi))),indent=2))

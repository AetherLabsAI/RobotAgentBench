"""Read-only current layout anchor measurement for the tray workspace page."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
scene=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
s=Usd.Stage.Open(str(scene));xf=UsdGeom.XformCache();bc=UsdGeom.BBoxCache(0,['default','render','proxy']);result={}
for name,path in [('tray','/World/FoodTray'),('tray_visual','/World/FoodTray/Render'),('holder','/World/Cutlery/Holder')]:
    prim=s.GetPrimAtPath(path);assert prim,path
    m=np.array(xf.GetLocalToWorldTransform(prim)).T;assert np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-6)
    b=bc.ComputeWorldBound(prim).ComputeAlignedRange();assert not b.IsEmpty()
    result[name]=dict(path=path,matrix=m.tolist(),min=list(b.GetMin()),max=list(b.GetMax()))
result.update(scene=str(scene),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),status='measurement_only_no_layout_changes')
args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from build_room_demo_v1 import exact

root=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts')
result={}
for n in (1,2,3,4):
    s=Usd.Stage.Open(str(root/f'ID_{n:02d}_LIVING_STRAIGHT.usdc'))
    rows=[]
    for p in s.GetPrimAtPath('/World/Assets').GetChildren():
        if any(x in p.GetName().lower() for x in ('sofa','table','shel','microwave')):
            try:b=exact(s,str(p.GetPath())).tolist()
            except AssertionError:b=None
            rows.append(dict(path=str(p.GetPath()),bounds=b,matrix=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(p)).tolist()))
    result[n]=rows
Path(__file__).with_name('other_layouts_inventory_v1.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

import sys,json
from pxr import Usd,UsdGeom
import numpy as np
sys.path.insert(0,'/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
from build_room_demo_v1 import exact
for f in sys.argv[1:]:
    s=Usd.Stage.Open(f);p='/World/Assets/SmallCoffeeTable'
    print(f,exact(s,p).tolist(),np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(p))).tolist())

from pathlib import Path
import sys,numpy as np,json,subprocess
from scipy.spatial.transform import Rotation,Slerp
p=Path(__file__).parent;sys.path.insert(0,str(p))
from continuous_repair_actions_v3 import Repair
from held_hand_cartesian_guard_v1 import LayPlanner
r=Repair('mirrored_measured_lay_0174');r.planner=LayPlanner();pl=r.planner;ob=r.wait(0,14400)
subprocess.run([sys.executable,str(p/'audit_recovery_grip.py'),r.ep.name,'repair_000'],check=True)
a=json.loads((r.ep/'recovery_grip_repair_000.json').read_text());assert a['hands']['r']['cooked_between']>=3,a
z=np.load(p/'runs/nonintersecting_tilt35_0165/action.npz');goal=dict(zip(z['joint_names'],z['joint_target'][4504]));v=ob['command'];
from outward_eye_cartesian_v2 import plan
rows,info=plan(pl,v,goal);current=rows[-1][0];(r.ep/'outward_eye_guarded_plan.json').write_text(json.dumps(info,indent=2))
rows += [(current,'R_measure_retained_corner_for_left')]*20
r.submit(0,ob,rows);r.wait(1,14400);print('DIRECT PRESENTATION READY',flush=True)

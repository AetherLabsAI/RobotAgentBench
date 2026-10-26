"""Measured fresh-episode long-edge grasp; native robot joint actions only."""
from pathlib import Path
import sys,json,subprocess,numpy as np
p=Path(__file__).parent;sys.path.insert(0,str(p))
from continuous_repair_actions_v3 import Repair
from left_longedge_natural_search_v3 import LayPlanner
r=Repair('mirrored_measured_lay_0174');r.planner=LayPlanner();index=1
ob=r.wait(index,14400)
subprocess.run([sys.executable,str(p/'audit_recovery_grip.py'),r.ep.name,f'repair_{index:03d}'],check=True,stdout=subprocess.DEVNULL)
audit=json.loads((r.ep/f'recovery_grip_repair_{index:03d}.json').read_text())
assert audit['hands']['r']['cooked_between']>=3, ('Right physical grasp requires repair',audit)
pl=r.planner;pl.recovery_desired_uv=[[.03325,.0377778],[.038,.0425]]
rows,info=pl.choose_left(ob['command'],ob['cloth'],ob['uv'])
meta=dict(start_observation=index,after_observation=index+1,frames=len(rows),info=info,held_edge='adjacent short material edge v=min',object_state_writes=0,body1_and_full_insertion_preflight=True)
(r.ep/'left_longedge_body1_face_plan.json').write_text(json.dumps(meta,indent=2))
(r.ep/'left_parallel_face_plan.json').write_text(json.dumps(meta,indent=2))
r.submit(index,ob,rows);print('LEFT LONGEDGE PREGRASP SUBMITTED',meta,flush=True)
r.wait(index+1,7200);print('LEFT LONGEDGE PREGRASP READY',index+1,flush=True)

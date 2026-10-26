from pathlib import Path
import sys,subprocess,json,numpy as np
from continuous_repair_actions_v3 import Repair
from adjacent_edge_analytic_normal_ik_v31 import LayPlanner
p=Path(__file__).parent;n='mirrored_measured_lay_0174';r=Repair(n);ep=r.ep

def run(name,*args):
 print('CONTROLLER',name,args,flush=True)
 subprocess.run([sys.executable,str(p/name),*map(str,args)],check=True)
def finished():return (ep/'trial.json').exists()
for script in ['continuous174_direct_eye_v2.py','adjacent174_shortedge_pregrasp_v8.py','longedge174_left_longedge_insert_guard_v3.py','longedge174_left_longedge_probe_guard_v3.py']:run(script)
probe=json.loads((ep/'longedge_left_small_lift_observation.json').read_text());assert probe['material_point_delta_m'][2]>.02,probe
i=probe['after_observation'];run('preflight_mirrored_planes_v13.py',n,i)
z=np.load(ep/'mirrored_candidate_v13.npz');names=z['joint_names'].tolist();actions=[(dict(zip(names,row.tolist())),str(ph))for row,ph in zip(z['joint_target'],z['phase'])];r.planner=LayPlanner();ob=r.wait(i,14400);r.submit(i,ob,actions);i+=1;r.wait(i,14400)
for attempt in range(3):
 run('adaptive_freeedge_lay_contact_live_v3.py',n,i,'contact');i+=1;ob=r.wait(i,14400);c=ob['cloth'];contact=(np.abs(c[:,2]-.4505)<.0015)&(c[:,0]>-2.35)&(c[:,0]<-1.55)&(c[:,1]>-.8)&(c[:,1]<0)
 (ep/f'actual_freeedge_support_{i:03d}.json').write_text(json.dumps(dict(nodes=int(contact.sum()),surface_z=.4505,tolerance=.0015,object_state_writes=0),indent=2))
 if contact.sum()>=20:break
else:raise RuntimeError('No measured free edge table support after physical approaches')
run('measured_anchor_lower_live_v3.py',n,i,'lower');i+=1;r.wait(i,14400)
for script,extra in [('open_and_observe_live_v3.py',[]),('measured_openpad_exit_live_v4.py',['lower']),('measured_openpad_exit_live_v4.py',['side']),('measured_outward_openpad_roll_v1.py',[]),('measured_openpad_exit_live_v4.py',['side']),('measured_openpad_exit_live_v4.py',['side'])]:
 if finished():break
 run(script,n,i,*extra);i+=1
 if finished():break
 r.wait(i,14400)
print('Scheduled actions finished; measured terminal audit required',flush=True)

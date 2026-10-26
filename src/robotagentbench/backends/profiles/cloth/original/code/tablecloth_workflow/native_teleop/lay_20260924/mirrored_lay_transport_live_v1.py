"""Transport measured held edge using native contact, mirrored pad planes and one easing."""
from pathlib import Path
import sys,json,subprocess,numpy as np
from continuous_repair_actions_v3 import Repair
from adjacent_edge_analytic_normal_ik_v31 import LayPlanner
p=Path(__file__).parent;r=Repair(sys.argv[1]);i=int(sys.argv[2]);mode=sys.argv[3];ob=r.wait(i,14400);stage=f'repair_{i:03d}';subprocess.run([sys.executable,str(p/'audit_recovery_grip.py'),r.ep.name,stage],check=True,stdout=subprocess.DEVNULL);pl=LayPlanner();r.planner=pl;pl.configure(ob,json.loads((r.ep/f'recovery_grip_{stage}.json').read_text()));v=ob['command'];pp=pl.captured_points(v);edge=np.array([0.,-1.,0.]);failed=[]
if mode=='contact':centers=[[-1.85,-.4,.90],[-1.88,-.4,.90],[-1.90,-.4,.89]];frames=180
elif mode=='lower':centers=[[-2.26,-.4,.55],[-2.24,-.4,.55],[-2.28,-.4,.57]];frames=240
else:raise ValueError(mode)
for xyz in centers:
 center=np.array(xyz);targets={}
 for side,sgn in [('L',-1),('R',1)]:
  h=pl.geom(v,side)[1];rot=h[:3,:3];cap=pl.captures[side];point=center+sgn*.5*pl.rest_span*.98*edge;targets[side]=(point-rot@(cap['point_local']-cap['mid_local']),rot)
 try:rows=pl.move_normal(v,targets,frames,'LR_mirrored_plane_'+mode+'_continuous');break
 except(RuntimeError,AssertionError)as e:failed.append(dict(center=xyz,error=str(e)));print('MIRRORED TRANSPORT REJECT',mode,failed[-1],flush=True)
else:raise RuntimeError(failed)
rows += [(rows[-1][0],'LR_read_actual_'+mode+'_cloth')]*20;meta=dict(mode=mode,target_center=xyz,failed=failed,span_audit=pl.last_span_audit,actual_grasp_readback=True,object_state_writes=0);(r.ep/f'mirrored_transport_{mode}_{i:03d}.json').write_text(json.dumps(meta,indent=2));r.submit(i,ob,rows);print('MIRRORED TRANSPORT SUBMITTED',meta,flush=True);r.wait(i+1,14400);print('MIRRORED TRANSPORT READY',mode,i+1,flush=True)

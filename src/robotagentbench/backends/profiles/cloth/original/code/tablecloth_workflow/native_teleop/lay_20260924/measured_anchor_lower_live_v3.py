"""Transport measured held edge using native contact, mirrored pad planes and one easing."""
from pathlib import Path
import sys,json,subprocess,numpy as np
from continuous_repair_actions_v3 import Repair
from adjacent_edge_analytic_normal_ik_v31 import LayPlanner
p=Path(__file__).parent;r=Repair(sys.argv[1]);i=int(sys.argv[2]);mode=sys.argv[3];ob=r.wait(i,14400);stage=f'repair_{i:03d}';subprocess.run([sys.executable,str(p/'audit_recovery_grip.py'),r.ep.name,stage],check=True,stdout=subprocess.DEVNULL);pl=LayPlanner();r.planner=pl;pl.configure(ob,json.loads((r.ep/f'recovery_grip_{stage}.json').read_text()));v=ob['command'];pp=pl.captured_points(v);edge=np.array([0.,-1.,0.]);failed=[]
if mode=='contact':centers=[[-1.85,-.4,.90],[-1.88,-.4,.90],[-1.90,-.4,.89]];frames=180
elif mode=='lower':
 q=ob['cloth'];uv=ob['uv'];on=(abs(q[:,2]-.4505)<.0015)&(q[:,0]>=-2.35)&(q[:,0]<=-1.574649)&(q[:,1]>=-.7210436)&(q[:,1]<=.4789564);assert on.sum()>=20,('Actual free-edge table contact not established',int(on.sum()),float(q[:,2].min()))
 held_uv=(pl.captures['L']['uv']+pl.captures['R']['uv'])/2;depth=(uv[on]-held_uv)@pl.material_inward;anchor=np.median(q[on],axis=0);material_depth=float(np.median(depth));height=.55;horizontal=np.sqrt(max(0,material_depth**2-(height-.4505)**2))*.98;project=(uv-held_uv)@pl.material_inward;min_x=-2.35+.015-float(project.min());max_x=-1.574649-.015-float(project.max());chosen=float(np.clip(anchor[0]-horizontal,min_x,max_x));centers=[[chosen,-.4,height],[min(max_x,chosen+.02),-.4,height]];frames=240
 (r.ep/f'measured_table_anchor_{i:03d}.json').write_text(json.dumps(dict(actual_near_surface_nodes=int(on.sum()),actual_anchor=anchor.tolist(),material_depth_m=material_depth,target_x=chosen,full_material_x_bounds=[min_x,max_x],object_state_writes=0),indent=2))
else:raise ValueError(mode)
for xyz in centers:
 center=np.array(xyz);targets={}
 for side,sgn in [('L',-1),('R',1)]:
  h=pl.geom(v,side)[1];rot=h[:3,:3];cap=pl.captures[side];point=center+sgn*.5*pl.rest_span*.98*edge;targets[side]=(point-rot@(cap['point_local']-cap['mid_local']),rot)
 try:rows=pl.move_normal(v,targets,frames,'LR_mirrored_plane_'+mode+'_continuous');break
 except(RuntimeError,AssertionError)as e:failed.append(dict(center=xyz,error=str(e)));print('MIRRORED TRANSPORT REJECT',mode,failed[-1],flush=True)
else:raise RuntimeError(failed)
rows += [(rows[-1][0],'LR_read_actual_'+mode+'_cloth')]*20;meta=dict(mode=mode,target_center=xyz,failed=failed,span_audit=pl.last_span_audit,actual_grasp_readback=True,object_state_writes=0);(r.ep/f'mirrored_transport_{mode}_{i:03d}.json').write_text(json.dumps(meta,indent=2));r.submit(i,ob,rows);print('MIRRORED TRANSPORT SUBMITTED',meta,flush=True);r.wait(i+1,14400);print('MIRRORED TRANSPORT READY',mode,i+1,flush=True)

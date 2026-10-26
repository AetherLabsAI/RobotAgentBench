"""Smallest hand rotations aligning physical pad planes; native joints only."""
from pathlib import Path
import sys,json,numpy as np,subprocess
from scipy.spatial.transform import Rotation
from continuous_repair_actions_v3 import Repair
from adjacent_edge_analytic_normal_ik_v31 import LayPlanner
p=Path(__file__).parent;r=Repair(sys.argv[1]);i=int(sys.argv[2]);ob=r.wait(i,14400);stage=f'repair_{i:03d}'
subprocess.run([sys.executable,str(p/'audit_recovery_grip.py'),r.ep.name,stage],check=True,stdout=subprocess.DEVNULL)
audit=json.loads((r.ep/f'recovery_grip_{stage}.json').read_text());pl=LayPlanner();r.planner=pl;pl.configure(ob,audit);v=ob['command'];points=pl.captured_points(v);center0=(points['L']+points['R'])/2;edge=points['R']-points['L'];edge/=np.linalg.norm(edge);normals={s:pl.geom(v,s)[1][:3,1]for s in ['L','R']};common=sum(normals.values());common-=edge*(common@edge);common/=np.linalg.norm(common);failed=[];success=None
for normal in [np.array([0.,0.,-1.])]:
 for target_center in [[-2.,-.4,1.1],[-1.95,-.4,1.0],[-2.1,-.4,1.0],[-1.9,-.4,.95]]:
  center=np.asarray(target_center);edge=np.array([0.,-1.,0.]);targets={}
  for s,sign in [('L',-1),('R',1)]:
   h=pl.geom(v,s)[1];n=h[:3,1];desired_normal=normal if s=='L' else -normal;axis=np.cross(n,desired_normal);angle=np.arctan2(np.linalg.norm(axis),n@desired_normal);rot=Rotation.from_rotvec(axis/max(np.linalg.norm(axis),1e-12)*angle).as_matrix()@h[:3,:3];cap=pl.captures[s];point=center+sign*.5*pl.rest_span*.98*edge;targets[s]=(point-rot@(cap['point_local']-cap['mid_local']),rot)
  try:
   actions=pl.move_normal(v,targets,220,'LR_minimum_rotation_align_pad_planes_and_lift');success=dict(center=center.tolist(),normal_L=normal.tolist(),normal_R=(-normal).tolist(),edge=edge.tolist());break
  except (RuntimeError,AssertionError)as e:failed.append(dict(center=center.tolist(),normal=normal.tolist(),error=str(e)));print('MINIMAL PLANE REJECT',failed[-1],flush=True)
 if success:break
meta=dict(success=success,failed=failed,object_state_writes=0,orientation_objective='pad plane normal, free axial roll audited',span_audit=getattr(pl,'last_span_audit',None));(r.ep/f'mirrored_candidate_v13_{i:03d}.json').write_text(json.dumps(meta,indent=2))
if not success:raise RuntimeError('No clear minimal-rotation alignment path')
actions += [(actions[-1][0],'LR_measure_aligned_actual_cloth')]*20;np.savez_compressed(r.ep/'mirrored_candidate_v13.npz',joint_names=np.array(list(actions[0][0])),joint_target=np.array([[q[n]for n in actions[0][0]]for q,_ in actions]),phase=np.array([ph for _,ph in actions]),executed=False);print('OFFLINE MIRRORED PLAN READY',meta,flush=True);sys.exit(0);r.submit(i,ob,actions);print('MINIMAL PLANE SUBMITTED',meta,flush=True);r.wait(i+1,14400);print('MINIMAL PLANE READY',i+1,flush=True)

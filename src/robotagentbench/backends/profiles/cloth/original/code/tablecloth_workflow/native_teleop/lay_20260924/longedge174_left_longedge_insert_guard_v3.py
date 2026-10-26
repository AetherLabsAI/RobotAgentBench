from pathlib import Path
import sys,json,numpy as np,time,subprocess
p=Path('/media/aether/OS/agent_bench_runs/tablecloth_gripper_actuator_20260922/native_teleop/lay_20260924');sys.path.insert(0,str(p));from continuous_repair_actions_v3 import Repair
from left_longedge_cartesian_guard_v2 import LayPlanner
r=Repair('mirrored_measured_lay_0174');r.planner=LayPlanner();ready=r.ep/'left_longedge_body1_face_plan.json'
while not ready.exists():time.sleep(2)
plan=json.loads(ready.read_text());index=plan['after_observation'];ob=r.wait(index,timeout=7200);actual_poses=r.planner.robot.fk(ob['actual']);assert r.planner.body1_collision(actual_poses,'L').max()<=.004, 'Actual pregrasp lower torso clearance failed';info=plan['info'];target=np.array(info['uv']);d=np.array(info['direction']);rot=np.array(info['locked_rotation']);events=[]
for step in range(36):
 ob=r.wait(index,timeout=7200);v=ob['command'];poses=r.planner.robot.fk(ob['actual']);h=poses['gripper_l_base_link'];local=(ob['cloth']-h[:3,3])@h[:3,:3];bounds=[]
 for kind in ['inner','outer']:
  n='gripper_l_'+kind+'_link4';t=np.linalg.inv(h)@poses[n];q=r.pad['points_'+n]@t[:3,:3].T+t[:3,3];bounds.append((q.min(0),q.max(0)))
 lo=np.maximum(bounds[0][0],bounds[1][0]);hi=np.minimum(bounds[0][1],bounds[1][1]);lo[1]=bounds[0][1][1]+.003;hi[1]=bounds[1][0][1]-.003;lo[0]+=.002;hi[0]-=.002;lo[2]+=.012;hi[2]-=.012;inside=np.all((local>lo)&(local<hi),axis=1);patch=np.linalg.norm(ob['uv']-target,axis=1)<.07;count=int((inside&patch).sum());other=int((inside&~patch).sum());idx=int(np.linalg.norm(ob['uv']-target,axis=1).argmin());point=ob['cloth'][idx];mid=r.planner.geom(ob['actual'],'L')[0];remaining=float((point-mid)@d);cross=point-mid-remaining*d;ev=dict(index=index,step=step,strict_central_nodes=count,other_nodes=other,remaining=remaining,cross=float(np.linalg.norm(cross)),target_inside=bool(inside[idx]),point=point.tolist(),mid=mid.tolist());events.append(ev);(r.ep/'longedge_left_insertion_readback.json').write_text(json.dumps(events,indent=2));print(ev,flush=True)
 if count>=12 and other==0 and inside[idx] and np.all(ob['uv'][inside&patch].mean(0)>=[.015,.015]) and np.all(ob['uv'][inside&patch].mean(0)<=[.327,.495]):
  ids=np.flatnonzero(inside&patch);chosen=idx;goal=r.planner.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-.005,'idx41_gripper_l_outer_joint1':.005});rows=r.planner.interpolate(v,goal,'L',60,'L_adjacent_shortedge_close_broad_actual_patch');rows += [(rows[-1][0],'L_adjacent_shortedge_verify_full_closure')]*100;r.submit(index,ob,rows);after=r.wait(index+1,timeout=7200);(r.ep/'longedge_left_closed_ready.json').write_text(json.dumps(dict(next_observation=index+1,target_uv=ob['uv'][chosen].tolist(),original_corner_uv=target.tolist(),strict_central_nodes=count,other_nodes=other,object_state_writes=0)));print('CLOSED',index+1,flush=True);break
 if other:raise RuntimeError(('Other patch in strict corridor',ev))
 if remaining<-.015:raise RuntimeError(('Point passed without central containment',ev))
 if np.linalg.norm(cross)>.003 and remaining>.025:delta=cross*min(1,.005/np.linalg.norm(cross));phase='L_adjacent_shortedge_recenter_measured_corner'
 else:delta=d*min(.005,max(.001,remaining+.008));phase='L_adjacent_shortedge_forward_measured_insertion'
 goal=r.planner.solve_full(v,'L',r.planner.geom(v,'L')[0]+delta,rot);rows=r.planner.interpolate(v,goal,'L',10,phase);r.submit(index,ob,rows);index+=1
else:raise RuntimeError('Insertion did not reach broad single-patch containment')

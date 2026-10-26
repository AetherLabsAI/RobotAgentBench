from pathlib import Path
import sys,json,numpy as np,time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
p=Path('/media/aether/OS/agent_bench_runs/tablecloth_gripper_actuator_20260922/native_teleop/lay_20260924');sys.path.insert(0,str(p));from continuous_repair_actions_v3 import Repair
from cooked_closure_guard import CookedClosureGuard
from left_longedge_cartesian_guard_v2 import LayPlanner
r=Repair('continuous_natural_lay_0168');r.planner=LayPlanner();ready=r.ep/'longedge_left_closed_ready.json'
while not ready.exists():
 if (r.ep/'trial.json').exists():raise RuntimeError('Physical episode ended before recovery closure')
 time.sleep(1)
d=json.loads(ready.read_text());index=d['next_observation'];target=np.asarray(d['target_uv']);ob=r.wait(index);jaw_names=['idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1'];before_angles=np.array([ob['actual'][n]for n in jaw_names]);r.submit(index,ob,[(ob['command'],'L_adjacent_shortedge_measure_closure_stability')]*20);index+=1;ob=r.wait(index);after_angles=np.array([ob['actual'][n]for n in jaw_names]);stable=bool(np.max(np.abs(after_angles-before_angles))<.001 and max(abs(after_angles))<=.0175);(r.ep/'longedge_left_closure_stability.json').write_text(json.dumps(dict(stable=stable,before_angles=before_angles.tolist(),after_angles=after_angles.tolist(),sample_separation_s=1.,force_confirmed=False),indent=2));v=ob['command'];actual=ob['actual'];poses=r.planner.robot.fk(actual);names=list(poses);bodies=np.array([np.r_[h[:3,3],Rotation.from_matrix(h[:3,:3]).as_quat()]for h in poses.values()]);dist=[]
for kind in ['inner','outer']:
 n='gripper_l_'+kind+'_link4';t=poses[n];dist.append(cKDTree(r.pad['points_'+n]).query((ob['cloth']-t[:3,3])@t[:3,:3])[0])
near=np.flatnonzero((dist[0]<.005)&(dist[1]<.005)&(np.linalg.norm(ob['uv']-target,axis=1)<.07));g=CookedClosureGuard(r.ep/'native_left_pad_cooked_convexes.json');alias=[n.replace('gripper_l_','gripper_r_')if n.startswith('gripper_l_')else'unused_'+n for n in names];valid=[int(i)for i in near if g.check(ob['cloth'][i],bodies,alias)['geometry_allows_lift']];angles={n:actual[n]for n in ['idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1']};summary=dict(before_observation=index,target_uv=target.tolist(),bilateral_proximity=len(near),actual_cooked_between=len(valid),actual_jaw_angles=angles,force_confirmed=False,physical_lift_pending=True);(r.ep/'longedge_left_regrasp_geometry.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
assert stable and len(valid)>=3, ('Actual left closure not yet stable or held',stable,len(valid))
start,h,_,_=r.planner.geom(v,'L');rot=h[:3,:3];current=v;rows=[]
for k in range(1,11):
 t=k/10;t=t*t*(3-2*t);goal=r.planner.solve_full(current,'L',start+np.array([0,0,.03*t]),rot);part=r.planner.interpolate(current,goal,'L',6,'L_adjacent_shortedge_physical_lift_probe');rows+=part;current=part[-1][0]
rows += [(current,'L_adjacent_shortedge_verify_small_lift')]*40;r.submit(index,ob,rows);after=r.wait(index+1);idx=int(np.linalg.norm(ob['uv']-target,axis=1).argmin());delta=after['cloth'][idx]-ob['cloth'][idx];(r.ep/'longedge_left_small_lift_observation.json').write_text(json.dumps(dict(before_observation=index,after_observation=index+1,target_uv=target.tolist(),material_point_delta_m=delta.tolist(),requested_hand_lift_m=.03,retention_not_yet_verified=True,object_state_restores=0),indent=2));print('LEFT SMALL LIFT OBSERVATION READY',index+1,delta,flush=True)

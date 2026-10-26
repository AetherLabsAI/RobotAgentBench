"""Physical release: rotate each open lower pad away from the measured cloth center."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from continuous_repair_actions_v3 import Repair
from held_hand_cartesian_guard_v1 import LayPlanner
from adjacent_edge_cartesian_guard_v4 import solve_normal
r=Repair(sys.argv[1]);r.planner=LayPlanner();pl=r.planner;i=int(sys.argv[2]);ob=r.wait(i,14400);v=ob['command'];current=v;rows=[];choices=[];failed=[];clothcenter=np.median(ob['cloth'],axis=0)
for side in ['L','R']:
 start,h,poses,_=pl.geom(current,side);ds=[]
 for kind in ['inner','outer']:
  name='gripper_'+side.lower()+'_'+kind+'_link4';t=poses[name];ds.append(cKDTree(r.pad['points_'+name]).query((ob['cloth']-t[:3,3])@t[:3,:3])[0])
 gap=float(np.minimum(*ds).min())
 if gap>.008:continue
 assert abs(current['idx31_gripper_l_inner_joint1'if side=='L'else'idx71_gripper_r_inner_joint1'])>=.19
 n0=h[:3,1];outside=start-clothcenter;outside[2]=0;outside/=np.linalg.norm(outside);desired=outside*(-1 if n0[2]>0 else 1);selected=None
 for normal in [desired,np.array([0,1 if side=='L'else -1,0]),np.array([-1.,0,0]),-desired]:
  axis=np.cross(n0,normal);angle=np.arctan2(np.linalg.norm(axis),n0@normal);axis/=max(np.linalg.norm(axis),1e-12);part=[];state=current
  try:
   for k in range(1,31):
    a=k/30;a=a*a*(3-2*a);nn=Rotation.from_rotvec(axis*angle*a).apply(n0);goal,info=solve_normal(pl,state,side,start,nn,allow_flip=False);seg=pl.interpolate(state,goal,side,4,side+'_roll_open_lower_pad_away_from_measured_cloth');part+=seg;state=seg[-1][0]
   selected=dict(side=side,normal=normal.tolist(),angle_deg=float(np.rad2deg(angle)),start_mid=start.tolist(),measured_initial_gap=gap);break
  except(RuntimeError,AssertionError)as e:failed.append(dict(side=side,normal=normal.tolist(),error=str(e)));print('OPEN OUTWARD ROLL REJECT',failed[-1],flush=True)
 if selected is None:raise RuntimeError(('No validated physical outward pad roll',side,failed))
 rows+=part;current=state;choices.append(selected)
rows += [(current,'LR_read_gravity_release_after_outward_pad_roll')]*20
meta=dict(choices=choices,failed_candidates=failed,cloth_center=clothcenter.tolist(),object_state_writes=0,no_attachment=True,release_pending_measured_readback=True);(r.ep/f'measured_outward_openpad_roll_{i:03d}.json').write_text(json.dumps(meta,indent=2));r.submit(i,ob,rows);print('SUBMITTED OUTWARD PAD ROLL',meta,flush=True)
try:r.wait(i+1,14400);print('READY',i+1,flush=True)
except RuntimeError:
 if not(r.ep/'trial.json').exists():raise
 print('Native ended; inspect terminal record',flush=True)

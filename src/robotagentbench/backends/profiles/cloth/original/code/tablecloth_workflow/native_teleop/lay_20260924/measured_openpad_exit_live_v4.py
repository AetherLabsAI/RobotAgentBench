"""Withdraw open fingers from actual tabletop cloth, never move cloth state."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial import cKDTree
p=Path(__file__).parent;sys.path.insert(0,str(p))
from continuous_repair_actions_v3 import Repair
from held_hand_cartesian_guard_v1 import LayPlanner
from adjacent_edge_cartesian_guard_v4 import solve_normal
r=Repair(sys.argv[1]);r.planner=LayPlanner();pl=r.planner
index=int(sys.argv[2]);ob=r.wait(index,7200);v=ob['command'];mode=sys.argv[3]if len(sys.argv)>3 else 'side';poses=pl.robot.fk(ob['actual']);gaps={}
for side in ['L','R']:
 ds=[]
 for kind in ['inner','outer']:
  n='gripper_'+side.lower()+'_'+kind+'_link4';t=poses[n];ds.append(cKDTree(r.pad['points_'+n]).query((ob['cloth']-t[:3,3])@t[:3,:3])[0])
 gaps[side]=float(np.minimum(*ds).min())
rows=[];current=v;choices=[]
for side,sign in [('L',1),('R',-1)]:
 if gaps[side]>.008:continue
 # Preserve the native open aperture. No upward lift while a lower pad supports cloth.
 assert abs(current['idx31_gripper_l_inner_joint1'if side=='L'else'idx71_gripper_r_inner_joint1'])>=.19
 start,h,_,_=pl.geom(current,side);target=start.copy()
 if mode=='lower':target[2]=min(start[2],.51)
 elif mode=='side':target[1]+=.04*sign
 else:raise ValueError(mode)
 state=current;part=[]
 for k in range(1,17):
  a=k/16;a=a*a*(3-2*a);goal,info=solve_normal(pl,state,side,start+a*(target-start),h[:3,1],allow_flip=False);seg=pl.interpolate(state,goal,side,5,side+'_open_pad_'+mode+'_exit_with_table_support');part+=seg;state=seg[-1][0]
 rows+=part;current=state;choices.append(dict(side=side,start=start.tolist(),target=target.tolist(),initial_pad_gap_m=gaps[side]))
phase='LR_settle_after_verified_open_pad_clearance'if not choices else 'LR_read_actual_open_pad_clearance'
rows += [(current,phase)]*20
meta=dict(start_observation=index,after_observation=index+1,mode=mode,pad_normal_preserved=True,inplane_wrist_rotation_allowed=True,actual_pad_gaps_m=gaps,choices=choices,frames=len(rows),object_state_writes=0,cloth_pose_or_velocity_writes=0)
(r.ep/f'openpad_exit_{index:03d}_{mode}_normal_plan.json').write_text(json.dumps(meta,indent=2));r.submit(index,ob,rows);print('MEASURED OPENPAD EXIT SUBMITTED',meta,flush=True)
try:r.wait(index+1,7200);print('ACTUAL OPENPAD EXIT READY',index+1,flush=True)
except RuntimeError:
 if(r.ep/'trial.json').exists():print('Native ended; inspect final',flush=True)
 else:raise

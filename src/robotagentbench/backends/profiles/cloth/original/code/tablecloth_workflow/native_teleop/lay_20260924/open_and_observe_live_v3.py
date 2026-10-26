"""Open both native jaws over the measured tabletop; inspect before withdrawing."""
from pathlib import Path
import sys,json,numpy as np,subprocess
p=Path(__file__).parent;sys.path.insert(0,str(p))
from continuous_repair_actions_v3 import Repair
from held_hand_cartesian_guard_v1 import LayPlanner
r=Repair(sys.argv[1]);r.planner=LayPlanner();pl=r.planner
index=int(sys.argv[2]);ob=r.wait(index,7200);v=ob['command']
inside=np.all((ob['cloth'][:,:2]>=[-2.35,-.7210436])&(ob['cloth'][:,:2]<=[-1.574649,.4789564]),axis=1)
on=inside&(abs(ob['cloth'][:,2]-.451)<.012)
assert on.sum()>=5,'Actual tabletop contact required before open release'
arm_before={n:v[n]for s in ['L','R']for n in pl.robot.arm_names(s)}
g=pl.robot.calibrate_loops(v|dict(idx31_gripper_l_inner_joint1=-.20,idx41_gripper_l_outer_joint1=.20,idx71_gripper_r_inner_joint1=-.20,idx81_gripper_r_outer_joint1=.20))
names=[n for n in v if '_gripper_'in n];rows=[]
for k in range(1,41):
 a=k/40;a=a*a*(3-2*a);q=pl.robot.calibrate_loops(v|{n:v[n]+a*(g[n]-v[n])for n in names})
 assert all(abs(q[n]-val)<1e-10 for n,val in arm_before.items())
 assert max(pl.collision(pl.robot.fk(q),s).max()for s in ['L','R'])<=.004
 rows.append((q,'LR_open_native_jaws_with_fixed_arms_over_table'))
rows += [(rows[-1][0],'LR_measure_gravity_release_before_hand_withdrawal')]*20
meta=dict(start_observation=index,after_observation=index+1,actual_table_contact_nodes=int(on.sum()),frames=len(rows),object_state_writes=0,no_attachment=True,release_unverified_until_actual_readback=True)
(r.ep/'open_fixed_arm_release_plan.json').write_text(json.dumps(meta,indent=2));r.submit(index,ob,rows);print('OPEN RELEASE SUBMITTED',meta,flush=True)
try:r.wait(index+1,7200)
except RuntimeError:
 if(r.ep/'trial.json').exists():print('Native ended; inspect final measurements',flush=True);sys.exit(0)
 raise
for script in ['audit_recovery_grip.py','actual_cloth_landmarks_v1.py']:
 subprocess.run([sys.executable,str(p/script),r.ep.name,f'repair_{index+1:03d}'],check=True)
print('ACTUAL OPEN RELEASE READY',index+1,flush=True)

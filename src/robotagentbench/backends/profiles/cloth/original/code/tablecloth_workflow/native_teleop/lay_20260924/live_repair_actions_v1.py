"""Native-joint-only continuation helpers; same live physics, never state restoration."""
from pathlib import Path
import numpy as np,json,time,types,os
from bimanual_lay_plan_v30 import LayPlanner,DATA
P=Path(__file__).parent
class Repair:
 def __init__(self,trial):
  self.ep=P/'runs'/trial;self.planner=LayPlanner();self.pad=np.load(DATA/'robot_native_collision.npz');s=(P/'bimanual_lay_extension_v31.py').read_text();a=s.index('  rows=[];previous=command;allowed=');b=s.index('  assert rows;self.repair_index',a);self.validation='\n'.join(x[2:]for x in s[a:b].splitlines())
 def wait(self,index,timeout=1800):
  start=time.monotonic();path=self.ep/('observed_repair_'+str(index).zfill(3)+'.npz')
  while time.monotonic()-start<timeout:
   if (self.ep/'trial.json').exists():raise RuntimeError('Native episode ended')
   try:
    status=json.loads((self.ep/'repair_wait.json').read_text())
    if status['waiting']and Path(status['request_path']).name=='repair_request_'+str(index).zfill(3)+'.json':
     with np.load(path)as z:return dict(cloth=z['cloth_q'].copy(),uv=z['uv'].copy(),actual=dict(zip(z['joint_names'],map(float,z['joint_position']))),command=dict(zip(z['command_joint_names'],map(float,z['joint_target']))))
   except (FileNotFoundError,ValueError,KeyError):pass
   time.sleep(.5)
  raise RuntimeError('No next actual observation within timeout')
 def submit(self,index,observation,actions):
  command=observation['command'];allowed=self.planner.robot.arm_names('R')+self.planner.robot.arm_names('L')+['idx31_gripper_l_inner_joint1','idx41_gripper_l_outer_joint1','idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1'];data=dict(kind='joint_targets',actions=[dict(targets={n:float(q[n])for n in allowed if n in q},phase=phase)for q,phase in actions]);env=dict(self=types.SimpleNamespace(planner=self.planner),np=np,command=command,data=data);exec(self.validation,env);assert env['rows'];dst=self.ep/('repair_request_'+str(index).zfill(3)+'.json');assert not dst.exists();tmp=dst.with_suffix('.proposed.json');tmp.write_text(json.dumps(data));audit=dst.with_suffix('.validation.json');audit.write_text(json.dumps(dict(passed=True,exact_native_validator=True,frames=len(actions),object_state_writes=0,body_targets_unchanged=True),indent=2));os.replace(tmp,dst)
 def corridor(self,obs,side,target_uv):
  poses=self.planner.robot.fk(obs['actual']);h=poses['gripper_'+side.lower()+'_base_link'];local=(obs['cloth']-h[:3,3])@h[:3,:3];bounds=[]
  for kind in ['inner','outer']:
   n='gripper_'+side.lower()+'_'+kind+'_link4';t=np.linalg.inv(h)@poses[n];q=self.pad['points_'+n]@t[:3,:3].T+t[:3,3];bounds.append((q.min(0),q.max(0)))
  lo=np.maximum(bounds[0][0],bounds[1][0]);hi=np.minimum(bounds[0][1],bounds[1][1]);lo[1]=bounds[0][1][1]+.003;hi[1]=bounds[1][0][1]-.003;lo[0]+=.002;hi[0]-=.002;lo[2]+=.006;hi[2]-=.006;inside=np.all((local>lo)&(local<hi),axis=1);patch=np.linalg.norm(obs['uv']-target_uv,axis=1)<.07;idx=int(np.linalg.norm(obs['uv']-target_uv,axis=1).argmin());return dict(target_inside=bool(inside[idx]),patch=int((inside&patch).sum()),other=int((inside&~patch).sum()),idx=idx,inside_uv=obs['uv'][inside].tolist())

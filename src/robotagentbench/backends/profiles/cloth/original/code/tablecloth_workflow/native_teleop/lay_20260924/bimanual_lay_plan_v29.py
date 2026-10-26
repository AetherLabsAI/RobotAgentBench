"""Run-local material-frame unfold; native joint motion only, unchanged physical gates."""
from bimanual_lay_plan_v27 import LayPlanner as BasePlanner,DATA
from pad_normal_ik_continuous_candidate import solve_normal
from scipy.spatial.transform import Rotation,Slerp
import numpy as np

class LayPlanner(BasePlanner):
 flat_material_normal_sign=1
 def solve_full(self,v,side,target,rotation):
  mid,h,_,_=self.geom(v,side);local=(mid-h[:3,3])@h[:3,:3];nom=self.robot.nominal_values();names=self.robot.arm_names(side);options=[]
  for seed in [v,v|{n:nom[n]for n in names}]:
   q,err=self.robot.solve(seed,side,target,local,rotation,max_steps=240);gm,gh,poses,out=self.geom(q,side);oe=np.rad2deg(Rotation.from_matrix(rotation.T@gh[:3,:3]).magnitude());coll=self.collision(poses,side).max()
   if np.linalg.norm(gm-target)<.002 and oe<2 and coll<=.004 and out>=.005:options.append((np.linalg.norm([q[n]-v[n]for n in names]),q))
  if not options:raise RuntimeError(('No native-clear full material orientation',side,list(target)))
  return min(options,key=lambda x:x[0])[1]
 def both_cartesian(self,v,left,right,frames,phase):
  if phase!='LR_unfold_equal_height':return self.material_normal_move(v,left,right,frames,phase)
  a,lh,_,_=self.geom(v,'L');b,rh,_,_=self.geom(v,'R');c0=(a+b)/2;c1=(np.asarray(left)+np.asarray(right))/2;d0=b-a;d1=np.asarray(right)-np.asarray(left);l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);angle=np.arctan2(np.linalg.norm(axis),u0@u1);axis/=np.linalg.norm(axis)
  rotations={s:Slerp([0,1],Rotation.from_matrix(np.array([h[:3,:3],self.material_hand_rotations[s]])))for s,h in [('L',lh),('R',rh)]};current=v;rows=[]
  for k in range(1,41):
   t=k/40;t=t*t*(3-2*t);center=c0+t*(c1-c0);direction=Rotation.from_rotvec(axis*angle*t).apply(u0);distance=l0+t*(l1-l0);targets={'L':center-direction*distance/2,'R':center+direction*distance/2};goal=current
   for side in ['R','L']:goal=self.solve_full(goal,side,targets[side],rotations[side](t).as_matrix())
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n])for n in names);N=max(frames//40,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,N+1):
    t2=j/N;t2=t2*t2*(3-2*t2);q=current|{n:current[n]+t2*(goal[n]-current[n])for n in names}
    if j==N or j%5==0:
     if max(self.collision(self.robot.fk(q),s).max()for s in ['L','R'])>.004:raise RuntimeError(('Material unfold path collision',k,j))
    rows.append((q,phase))
   current=goal
  return rows

 def material_normal_move(self,v,left,right,frames,phase):
  if phase not in ['LR_unfold_equal_height','LR_over_table','LR_lower_diagonal']:return super().both_cartesian(v,left,right,frames,phase)
  a,lh,_,_=self.geom(v,'L');b,rh,_,_=self.geom(v,'R');c0=(a+b)/2;c1=(np.asarray(left)+np.asarray(right))/2;d0=b-a;d1=np.asarray(right)-np.asarray(left);l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sn=np.linalg.norm(axis);angle=np.arctan2(sn,np.clip(u0@u1,-1,1))
  if sn<1e-8:
   axis=np.cross(u0,[0,0,1])
   if np.linalg.norm(axis)<1e-8:axis=np.cross(u0,[1,0,0])
  axis/=np.linalg.norm(axis);current=v;rows=[];normal0={'L':lh[:3,1],'R':rh[:3,1]};normal1=getattr(self,'material_normal_targets',{s:np.array([0.,0.,1.])for s in ['L','R']});turn={}
  for s in ['L','R']:
   n0=normal0[s];n1=normal1[s];ax=np.cross(n0,n1);an=np.arctan2(np.linalg.norm(ax),np.clip(n0@n1,-1,1))
   if np.linalg.norm(ax)<1e-8:ax=np.cross(n0,[1,0,0])
   if np.linalg.norm(ax)<1e-8:ax=np.cross(n0,[0,1,0])
   turn[s]=ax/np.linalg.norm(ax)*an
  for k in range(1,31):
   t=k/30;t=t*t*(3-2*t);center=c0+t*(c1-c0)
   if phase=='LR_unfold_equal_height':direction=Rotation.from_rotvec(axis*angle*t).apply(u0);distance=l0+t*(l1-l0);targets={'L':center-direction*distance/2,'R':center+direction*distance/2}
   else:targets={'L':a+t*(np.asarray(left)-a),'R':b+t*(np.asarray(right)-b)}
   goal=current
   for side in ['R','L']:
    normal=Rotation.from_rotvec(turn[side]*t).apply(normal0[side]);goal,info=solve_normal(self,goal,side,targets[side],normal,allow_flip=False)
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n])for n in names);N=max(frames//30,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,N+1):
    u=j/N;u=u*u*(3-2*u);q=current|{n:current[n]+u*(goal[n]-current[n])for n in names}
    if j==N or j%5==0:
     poses=self.robot.fk(q)
     if max(self.collision(poses,s).max()for s in ['L','R'])>.004:raise RuntimeError(('coordinated normal path collision',phase,k,j))
    rows.append((q,phase))
   current=goal
  self.last_normal_motion=dict(phase=phase,frames=len(rows),target_normals={k:x.tolist()for k,x in normal1.items()},object_state_writes=0)
  return rows

 def configure_material_frame(self,actual,cloth,uv,output):
  from scipy.spatial import cKDTree
  from cooked_closure_guard import CookedClosureGuard
  from pathlib import Path
  poses=self.robot.fk(actual);names=list(poses);bodies=np.array([np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat()]for t in poses.values()]);cad=np.load(DATA/'robot_native_collision.npz');hands={}
  for side,target in [('l',[.019,.491111]),('r',[.32775,.028333])]:
   ds=[]
   for kind in ['inner','outer']:
    name='gripper_'+side+'_'+kind+'_link4';t=poses[name];ds.append(cKDTree(cad['points_'+name]).query((cloth-t[:3,3])@t[:3,:3])[0])
   near=np.flatnonzero((ds[0]<.005)&(ds[1]<.005)&(np.linalg.norm(uv-target,axis=1)<.07));guard=CookedClosureGuard(Path(output)/('native_pad_cooked_convexes.json'if side=='r'else'native_left_pad_cooked_convexes.json'));alias=names if side=='r'else[n.replace('gripper_l_','gripper_r_')if n.startswith('gripper_l_')else'unused_'+n for n in names];ids=[]
   for i in near:
    check=guard.check(cloth[i],bodies,alias)
    if check['geometry_allows_lift']and check['cooked_gap_m']<.0015:ids.append(i)
   if len(ids)<3:raise RuntimeError(('Insufficient physical patch to estimate material orientation',side,len(ids)))
   u=uv[ids]-uv[ids].mean(0);x=cloth[ids]-cloth[ids].mean(0)
   if np.linalg.matrix_rank(u)<2:raise RuntimeError(('Material patch has insufficient two-dimensional extent',side))
   jac=np.linalg.lstsq(u,x,rcond=None)[0].T;uu,sv,vv=np.linalg.svd(jac,full_matrices=False);basis=uu@vv;frame=np.c_[basis,np.cross(basis[:,0],basis[:,1])];local=poses['gripper_'+side+'_base_link'][:3,:3].T@frame;hands[side.upper()]=dict(nodes=len(ids),uv=uv[ids].mean(0),local=local,strains=sv)
  delta=hands['L']['uv']-hands['R']['uv'];angle=np.pi/2-np.arctan2(delta[1],delta[0])-np.pi/6;flat=Rotation.from_euler('z',angle).as_matrix();self.material_hand_rotations={s:flat@h['local'].T for s,h in hands.items()};self.material_normal_targets={s:r[:,1]for s,r in self.material_hand_rotations.items()};self.material_span_axis=Rotation.from_euler('z',-30,degrees=True).apply([0.,1.,0.]);cad.close()
  return dict(stage='actual_material_frame_for_unfold',hands={s:{k:(x.tolist()if isinstance(x,np.ndarray)else x)for k,x in h.items()}for s,h in hands.items()},flat_material_frame=flat.tolist(),hand_rotations={s:r.tolist()for s,r in self.material_hand_rotations.items()},object_state_writes=0)

 def withdraw_axial(self,v,distance=.12,frames=160):
  starts={s:self.geom(v,s)[0]for s in ['L','R']};rot={s:self.geom(v,s)[1][:3,:3].copy()for s in ['L','R']};current=v;rows=[]
  for k in range(1,31):
   t=k/30;t=t*t*(3-2*t);goal=current
   for side in ['R','L']:goal=self.solve_full(goal,side,starts[side]-distance*t*rot[side][:,2],rot[side])
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n])for n in names);N=max(frames//30,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,N+1):
    u=j/N;u=u*u*(3-2*u);q=current|{n:current[n]+u*(goal[n]-current[n])for n in names}
    if j==N or j%5==0:
     if max(self.collision(self.robot.fk(q),s).max()for s in ['L','R'])>.004:raise RuntimeError(('Axial open-pad withdrawal collision',k,j))
    rows.append((q,'LR_withdraw_open_pads'))
   current=goal
  return rows

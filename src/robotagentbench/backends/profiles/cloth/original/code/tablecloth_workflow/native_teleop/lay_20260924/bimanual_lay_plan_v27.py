"""Material-consistent pad normals (both measured material normals align to hand +Y), coordinated motion, with physical joint commands only."""
from bimanual_lay_plan_v24 import LayPlanner as BasePlanner,DATA
from pad_normal_ik_continuous_candidate import solve_normal
from scipy.spatial.transform import Rotation
import numpy as np

from bimanual_lay_plan_v23 import LayPlanner as RetainedPresentation

class LayPlanner(BasePlanner):
 forward_right_presentation=RetainedPresentation.forward_right_presentation
 flat_material_normal_sign=-1
 def both_cartesian(self,v,left,right,frames,phase):
  if phase not in ['LR_unfold_equal_height','LR_over_table','LR_lower_diagonal']:return super().both_cartesian(v,left,right,frames,phase)
  a,lh,_,_=self.geom(v,'L');b,rh,_,_=self.geom(v,'R');c0=(a+b)/2;c1=(np.asarray(left)+np.asarray(right))/2;d0=b-a;d1=np.asarray(right)-np.asarray(left);l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sn=np.linalg.norm(axis);angle=np.arctan2(sn,np.clip(u0@u1,-1,1))
  if sn<1e-8:
   axis=np.cross(u0,[0,0,1])
   if np.linalg.norm(axis)<1e-8:axis=np.cross(u0,[1,0,0])
  axis/=np.linalg.norm(axis);current=v;rows=[];normal0={'L':lh[:3,1],'R':rh[:3,1]};normal1={s:np.array([0.,0.,float(self.flat_material_normal_sign)])for s in ['L','R']};turn={}
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

"""Action-only unfold repair: rotate the inter-hand vector without squeezing it.
Native configuration and all collision/IK checks remain inherited unchanged.
"""
from bimanual_lay_plan_v12 import LayPlanner as BasePlanner,DATA
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(BasePlanner):
 def both_cartesian(self,v,left,right,frames,phase):
  if phase!='LR_unfold_equal_height':return super().both_cartesian(v,left,right,frames,phase)
  a=self.geom(v,'L')[0];b=self.geom(v,'R')[0];c0=(a+b)/2;c1=(np.asarray(left)+np.asarray(right))/2;d0=b-a;d1=np.asarray(right)-np.asarray(left);l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sn=np.linalg.norm(axis);angle=np.arctan2(sn,np.clip(u0@u1,-1,1))
  if sn<1e-8:
   axis=np.cross(u0,[0,0,1])
   if np.linalg.norm(axis)<1e-8:axis=np.cross(u0,[1,0,0])
  axis/=np.linalg.norm(axis);current=v;rows=[]
  for k in range(1,31):
   t=k/30;t=t*t*(3-2*t);center=c0+t*(c1-c0);direction=Rotation.from_rotvec(axis*angle*t).apply(u0);distance=l0+t*(l1-l0);targets={'L':center-direction*distance/2,'R':center+direction*distance/2};goal=current
   for side in ['R','L']:goal=self.solve(goal,side,targets[side])
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n])for n in names);N=max(frames//30,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,N+1):
    u=j/N;u=u*u*(3-2*u);q=current|{n:current[n]+u*(goal[n]-current[n])for n in names}
    if j==N or j%5==0:
     poses=self.robot.fk(q)
     if max(self.collision(poses,s).max()for s in ['L','R'])>.004:raise RuntimeError(('two-arm arc collision',phase,k,j))
    rows.append((q,phase))
   current=goal
  return rows

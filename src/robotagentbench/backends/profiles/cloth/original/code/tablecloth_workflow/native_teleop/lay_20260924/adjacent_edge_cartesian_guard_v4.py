"""Reach-aware continuation preserving actual captured-point span and pad normals.
Native joint targets only; this does not update any cloth state or constraint.
"""
from native_body1_collision_guard_v1 import AdjacentEdgePlanner as Base, DATA
from pad_normal_ik_continuous_candidate import solve_normal as fallback_normal
from scipy.spatial.transform import Rotation,Slerp
import numpy as np
def solve_normal(pl,v,side,target,normal,max_steps=240,allow_flip=False):
 # Follow the current IK branch first; all original native acceptance thresholds retained.
 names=pl.robot.arm_names(side);mid,h,_,_=pl.geom(v,side);local=(mid-h[:3,3])@h[:3,:3];n=np.asarray(normal,dtype=float);n/=np.linalg.norm(n);q=v.copy()
 for k in range(max_steps):
  poses,axes=pl.robot.fk(q,True);t=poses['gripper_'+side.lower()+'_base_link'];point=t[:3,:3]@local+t[:3,3];actual=t[:3,1];error=np.r_[target-point,.12*(n-actual)];jac=np.vstack([np.stack([np.cross(axes[j][1],point-axes[j][0])for j in names],axis=1),.12*np.stack([np.cross(axes[j][1],actual)for j in names],axis=1)])
  if np.linalg.norm(error)<.0002:break
  dq=jac.T@np.linalg.solve(jac@jac.T+.008**2*np.eye(6),error);dq*=min(1.,.08/max(abs(dq).max(),1e-12))
  for j,d in zip(names,dq):lo,hi=pl.robot.by_name[j]['limits'];q[j]=float(np.clip(q[j]+d,lo+1e-5,hi-1e-5))
 gm,gh,poses,out=pl.geom(q,side);pos=float(np.linalg.norm(gm-target));angle=float(np.rad2deg(np.arccos(np.clip(gh[:3,1]@n,-1,1))));coll=float(pl.collision(poses,side).max())
 if pos<.002 and angle<2 and coll<=.004 and out>=.005:return q,dict(position_error_m=pos,normal_angle_deg=angle,collision_residual_m=coll,outward_m=out,continuous_seed=True)
 return fallback_normal(pl,v,side,target,n,max_steps=max_steps,allow_flip=allow_flip)
class LayPlanner(Base):
 def solve_captured_normal(self,v,side,point,normal,mid_seed):
  mid=np.array(mid_seed,dtype=float);goal=v
  for _ in range(5):
   goal,info=solve_normal(self,goal,side,mid,np.array(normal),allow_flip=False)
   h=self.geom(goal,side)[1];actual=h[:3,:3]@self.captures[side]['point_local']+h[:3,3];error=point-actual
   if np.linalg.norm(error)<.0006:return goal
   mid+=error
  if np.linalg.norm(error)>.002:raise RuntimeError(('Captured-point IK residual',side,float(np.linalg.norm(error))))
  return goal
 def move_normal(self,v,targets,frames=320,phase='LR_adjacent_edge_unfold'):
  starts={s:self.geom(v,s)[:2]for s in ['L','R']};rotations={s:Slerp([0,1],Rotation.from_matrix(np.array([starts[s][1][:3,:3],targets[s][1]])))for s in ['L','R']};points=self.captured_points(v);end={s:targets[s][0]+targets[s][1]@(self.captures[s]['point_local']-self.captures[s]['mid_local'])for s in ['L','R']};c0=(points['L']+points['R'])/2;c1=(end['L']+end['R'])/2;d0=points['R']-points['L'];d1=end['R']-end['L'];l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sine=np.linalg.norm(axis);angle=np.arctan2(sine,np.clip(u0@u1,-1,1));axis=axis/max(sine,1e-10)if sine>1e-8 else np.array([1.,0,0]);current=v;rows=[];maxspan=0;max_cartesian_deviation=0.;max_normal_deviation=0.
  for k in range(1,41):
   a=k/40;a=a*a*(3-2*a);center=c0+a*(c1-c0);direction=Rotation.from_rotvec(axis*angle*a).apply(u0);distance=l0+a*(l1-l0);goal=current
   for side,sign in [('R',1),('L',-1)]:
    rot=rotations[side](a).as_matrix();cap=self.captures[side];point=center+sign*.5*distance*direction;mid=point-rot@(cap['point_local']-cap['mid_local']);goal=self.solve_captured_normal(goal,side,point,rot[:,1],mid)
   previous_points=self.captured_points(current);goal_points=self.captured_points(goal);previous_normals={s:self.geom(current,s)[1][:3,1]for s in ['L','R']};goal_normals={s:self.geom(goal,s)[1][:3,1]for s in ['L','R']}
   names=self.robot.arm_names('L')+self.robot.arm_names('R');N=max(frames//40,int(np.ceil(max(abs(goal[n]-current[n])for n in names)*1.5/.7*20)))
   for j in range(1,N+1):
    t=j/N;t=t*t*(3-2*t);q=current|{n:current[n]+t*(goal[n]-current[n])for n in names}
    pp_check=self.captured_points(q)
    for side in ['L','R']:
     expected=previous_points[side]+t*(goal_points[side]-previous_points[side]);deviation=float(np.linalg.norm(pp_check[side]-expected));max_cartesian_deviation=max(max_cartesian_deviation,deviation)
     normal=previous_normals[side]+t*(goal_normals[side]-previous_normals[side]);normal/=np.linalg.norm(normal);normal_error=float(np.rad2deg(np.arccos(np.clip(self.geom(q,side)[1][:3,1]@normal,-1,1))));max_normal_deviation=max(max_normal_deviation,normal_error)
     assert deviation<=.004 and normal_error<=2,('held-hand interpolation leaves intended segment',side,k,j,deviation,normal_error)
    if j==N or j%3==0:
     poses=self.robot.fk(q);assert max(self.collision(poses,s).max()for s in ['L','R'])<=.004,('native collision',k,j)
     pp=self.captured_points(q);span=float(np.linalg.norm(pp['R']-pp['L']));maxspan=max(maxspan,span);assert span<=self.rest_span*1.005,('captured material overextension',k,j,span,self.rest_span)
    rows.append((q,phase))
   current=goal
  self.last_span_audit=dict(rest_span=self.rest_span,maximum_predicted_span=maxspan,maximum_segment_cartesian_deviation_m=max_cartesian_deviation,maximum_segment_pad_normal_deviation_deg=max_normal_deviation,object_state_writes=0);return rows

"""Cartesian-projected native joint path, full rotation and all old guards."""
from adjacent_edge_full_rotation_guard_v9 import LayPlanner as Base
from scipy.spatial.transform import Rotation,Slerp
import numpy as np
class LayPlanner(Base):
 def move_normal(self,v,targets,frames=320,phase='LR_adjacent_edge_unfold'):
  starts={s:self.geom(v,s)[:2]for s in ['L','R']};rotations={s:Slerp([0,1],Rotation.from_matrix(np.array([starts[s][1][:3,:3],targets[s][1]])))for s in ['L','R']};points=self.captured_points(v);end={s:targets[s][0]+targets[s][1]@(self.captures[s]['point_local']-self.captures[s]['mid_local'])for s in ['L','R']};c0=(points['L']+points['R'])/2;c1=(end['L']+end['R'])/2;d0=points['R']-points['L'];d1=end['R']-end['L'];l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sine=np.linalg.norm(axis);angle=np.arctan2(sine,np.clip(u0@u1,-1,1));axis=axis/max(sine,1e-10)if sine>1e-8 else np.array([1.,0,0]);common_turn=Slerp([0,1],Rotation.from_matrix(np.array([np.eye(3),targets['R'][1]@starts['R'][1][:3,:3].T])));current=v;rows=[];maxspan=0;max_cartesian_deviation=0.;max_normal_deviation=0.
  for k in range(1,41):
   a=k/40;a=a*a*(3-2*a);center=c0+a*(c1-c0)+np.sin(np.pi*a)*np.array([.08,0,.04]);direction=common_turn(a).apply(u0);distance=l0+a*(l1-l0);goal=current
   for side,sign in [('R',1),('L',-1)]:
    rot=rotations[side](a).as_matrix();cap=self.captures[side];point=center+sign*.5*distance*direction;mid=point-rot@(cap['point_local']-cap['mid_local']);goal=self.solve_captured_full(goal,side,point,rot,mid)
   previous_points=self.captured_points(current);goal_points=self.captured_points(goal);previous_normals={s:self.geom(current,s)[1][:3,1]for s in ['L','R']};goal_normals={s:self.geom(goal,s)[1][:3,1]for s in ['L','R']};segment_turn={s:Slerp([0,1],Rotation.from_matrix(np.array([self.geom(current,s)[1][:3,:3],self.geom(goal,s)[1][:3,:3]])))for s in ['L','R']}
   names=self.robot.arm_names('L')+self.robot.arm_names('R');N=max(frames//40,int(np.ceil(max(abs(goal[n]-current[n])for n in names)*4.0/.7*20)))
   for j in range(1,N+1):
    t=j/N;t=t*t*(3-2*t);q=current|{n:current[n]+t*(goal[n]-current[n])for n in names}
    
    for side in ['L','R']:
     desired_point=previous_points[side]+t*(goal_points[side]-previous_points[side]);desired_rotation=segment_turn[side](t).as_matrix();q,projection_error=self.robot.solve(q,side,desired_point,self.captures[side]['point_local'],desired_rotation,max_steps=160)
     assert projection_error<=.002,('projected native hand path position',side,k,j,projection_error)
    prior=rows[-1][0]if rows else v
    assert max(abs(q[n]-prior[n])for n in names)<=.7/20+1e-6,('projected native joint rate',k,j)
    pp_check=self.captured_points(q)
    for side in ['L','R']:
     expected=previous_points[side]+t*(goal_points[side]-previous_points[side]);deviation=float(np.linalg.norm(pp_check[side]-expected));max_cartesian_deviation=max(max_cartesian_deviation,deviation)
     normal=previous_normals[side]+t*(goal_normals[side]-previous_normals[side]);normal/=np.linalg.norm(normal);normal_error=float(np.rad2deg(np.arccos(np.clip(self.geom(q,side)[1][:3,1]@normal,-1,1))));max_normal_deviation=max(max_normal_deviation,normal_error)
     full_error=float(np.rad2deg(Rotation.from_matrix(segment_turn[side](t).as_matrix().T@self.geom(q,side)[1][:3,:3]).magnitude()));assert full_error<=2,('held hand full orientation path',side,k,j,full_error)
     assert deviation<=.004 and normal_error<=2,('held-hand interpolation leaves intended segment',side,k,j,deviation,normal_error)
    if j==N or j%3==0:
     poses=self.robot.fk(q);assert max(self.collision(poses,s).max()for s in ['L','R'])<=.004,('native collision',k,j)
     pp=self.captured_points(q);span=float(np.linalg.norm(pp['R']-pp['L']));maxspan=max(maxspan,span);assert span<=self.rest_span*1.005,('captured material overextension',k,j,span,self.rest_span)
    rows.append((q,phase))
   current=goal
  self.last_span_audit=dict(rest_span=self.rest_span,maximum_predicted_span=maxspan,maximum_segment_cartesian_deviation_m=max_cartesian_deviation,maximum_segment_pad_normal_deviation_deg=max_normal_deviation,object_state_writes=0);return rows

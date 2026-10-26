"""Cartesian-projected native joint path, full rotation and all old guards."""
from adjacent_edge_continuous_pad_normal_v24 import LayPlanner as Base
from scipy.spatial.transform import Rotation,Slerp,RotationSpline
from scipy.interpolate import PchipInterpolator,CubicHermiteSpline
import numpy as np
class LayPlanner(Base):
 def move_normal(self,v,targets,frames=320,phase='LR_adjacent_edge_unfold'):
  starts={s:self.geom(v,s)[:2]for s in ['L','R']};rotations={s:Slerp([0,1],Rotation.from_matrix(np.array([starts[s][1][:3,:3],targets[s][1]])))for s in ['L','R']};points=self.captured_points(v);end={s:targets[s][0]+targets[s][1]@(self.captures[s]['point_local']-self.captures[s]['mid_local'])for s in ['L','R']};c0=(points['L']+points['R'])/2;c1=(end['L']+end['R'])/2;d0=points['R']-points['L'];d1=end['R']-end['L'];l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sine=np.linalg.norm(axis);angle=np.arctan2(sine,np.clip(u0@u1,-1,1));axis=axis/max(sine,1e-10)if sine>1e-8 else np.array([1.,0,0]);common_turn=Slerp([0,1],Rotation.from_matrix(np.array([np.eye(3),targets['R'][1]@starts['R'][1][:3,:3].T])));current=v;rows=[];maxspan=0;max_cartesian_deviation=0.;max_normal_deviation=0.;max_full_rotation_deviation=0.;knots=[v]
  for k in range(1,41):
   a=k/40;a=a*a*(3-2*a);center=c0+a*(c1-c0);direction=Rotation.from_rotvec(axis*angle*a).apply(u0);distance=l0+a*(l1-l0);goal=current
   for side,sign in [('R',1),('L',-1)]:
    rot=rotations[side](a).as_matrix();cap=self.captures[side];point=center+sign*.5*distance*direction;mid=point-rot@(cap['point_local']-cap['mid_local']);goal=self.solve_captured_full(goal,side,point,rot,mid)
   knots.append(goal);current=goal
  names=self.robot.arm_names('L')+self.robot.arm_names('R');qk=np.array([[q[n]for n in names]for q in knots]);steps=np.maximum(frames//40,np.ceil(np.max(abs(np.diff(qk,axis=0)),axis=1)*4/.7*20).astype(int));times=np.r_[0,np.cumsum(steps)]
  def curve(values):
   base=PchipInterpolator(times,values,axis=0);dv=base.derivative()(times);dv[0]=0;dv[-1]=0;return CubicHermiteSpline(times,values,dv,axis=0)
  joint_curve=curve(qk);pk={side:np.array([self.captured_points(q)[side]for q in knots])for side in ['L','R']};pc={side:curve(pk[side])for side in ['L','R']};rk={side:np.array([self.geom(q,side)[1][:3,:3]for q in knots])for side in ['L','R']};rc={side:Slerp(times,Rotation.from_matrix(rk[side]))for side in ['L','R']}
  for frame in range(1,int(times[-1])+1):
   q=v|dict(zip(names,joint_curve(frame)));segment=min(np.searchsorted(times,frame,side='right')-1,len(knots)-2);t=(frame-times[segment])/(times[segment+1]-times[segment])
   for side in ['L','R']:
    point=pc[side](frame);rot=rc[side](frame).as_matrix();q,err=self.robot.solve(q,side,point,self.captures[side]['point_local'],rot,max_steps=160);assert err<=.002,('continuous projection position',side,frame,err)
    expected=pk[side][segment]*(1-t)+pk[side][segment+1]*t;reference_rot=Slerp([0,1],Rotation.from_matrix(rk[side][segment:segment+2]))(t).as_matrix();achieved=self.captured_points(q)[side];deviation=float(np.linalg.norm(achieved-expected));full_ang=float(np.rad2deg(Rotation.from_matrix(reference_rot.T@self.geom(q,side)[1][:3,:3]).magnitude()));max_full_rotation_deviation=max(max_full_rotation_deviation,full_ang);ang=float(np.rad2deg(np.arccos(np.clip(reference_rot[:,1]@self.geom(q,side)[1][:3,1],-1,1))));max_cartesian_deviation=max(max_cartesian_deviation,deviation);max_normal_deviation=max(max_normal_deviation,ang);assert deviation<=.004 and ang<=2,('continuous curve deviates from guarded corridor',side,frame,deviation,ang)
   prior=rows[-1][0]if rows else v;pass # Final retiming below validates every emitted native command
   poses=self.robot.fk(q);assert max(self.collision(poses,side).max()for side in ['L','R'])<=.004,('native collision',frame)
   pp=self.captured_points(q);span=float(np.linalg.norm(pp['R']-pp['L']));maxspan=max(maxspan,span);assert span<=self.rest_span*1.005,('material overextension',frame,span,self.rest_span);rows.append((q,phase))
  rows,retiming=self.retime_projected_rows(rows,v);self.last_span_audit=dict(native_retiming=retiming,rest_span=self.rest_span,maximum_predicted_span=maxspan,maximum_segment_cartesian_deviation_m=max_cartesian_deviation,maximum_segment_pad_normal_deviation_deg=max_normal_deviation,guide_joint_velocity_continuous=True,orientation_interpolation="bounded segment slerp, no stationary knots",full_joint_velocity_continuity_unverified=True,maximum_full_rotation_deviation_deg=max_full_rotation_deviation,inplane_roll_is_free_objective=True,object_state_writes=0);return rows

 def retime_projected_rows(self,rows,start):
  names=self.robot.arm_names('L')+self.robot.arm_names('R');out=[];previous=start;added=0;worst=0.;maxspeed=0.
  for target,phase in rows:
   count=max(1,int(np.ceil(max(abs(target[n]-previous[n])for n in names)/(.7/20))));added+=count-1;pa=self.captured_points(previous);pb=self.captured_points(target);rot={s:Slerp([0,1],Rotation.from_matrix([self.geom(previous,s)[1][:3,:3],self.geom(target,s)[1][:3,:3]]))for s in ['L','R']}
   for j in range(1,count+1):
    t=j/count;q=previous|{n:previous[n]+t*(target[n]-previous[n])for n in names};pp=self.captured_points(q);poses=self.robot.fk(q);assert max(self.collision(poses,s).max()for s in ['L','R'])<=.004,('retimed native collision',len(out));assert np.linalg.norm(pp['R']-pp['L'])<=self.rest_span*1.005,('retimed span',len(out))
    for s in ['L','R']:
     err=float(np.linalg.norm(pp[s]-(pa[s]+t*(pb[s]-pa[s]))));angle=float(np.rad2deg(np.arccos(np.clip(rot[s](t).as_matrix()[:,1]@self.geom(q,s)[1][:3,1],-1,1))));worst=max(worst,err);assert err<=.004 and angle<=2,('retimed corridor',len(out),s,err,angle)
    prior=out[-1][0]if out else start;speed=max(abs(q[n]-prior[n])*20 for n in names);maxspeed=max(maxspeed,speed);assert speed<=.7+1e-5;out.append((q,phase))
   previous=target
  return out,dict(added_frames=added,maximum_actual_command_speed_rad_s=maxspeed,maximum_inserted_cartesian_deviation_m=worst,no_stationary_knots_added=True)

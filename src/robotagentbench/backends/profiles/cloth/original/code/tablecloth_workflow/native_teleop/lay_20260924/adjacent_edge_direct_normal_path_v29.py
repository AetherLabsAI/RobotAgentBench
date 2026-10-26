"""Direct incremental captured-point and pad-normal IK with one global easing.
No nested segment easing and no full-rotation re-projection of a free-roll path.
"""
from adjacent_edge_cartesian_guard_v4 import LayPlanner as Base
from adjacent_edge_rate_refined_normal_v27 import LayPlanner as Retimer
from scipy.spatial.transform import Rotation,Slerp
import numpy as np
class LayPlanner(Base):
 def move_normal(self,v,targets,frames=240,phase='LR_direct_normal_alignment'):
  pp=self.captured_points(v);end={s:targets[s][0]+targets[s][1]@(self.captures[s]['point_local']-self.captures[s]['mid_local'])for s in ['L','R']};c0=(pp['L']+pp['R'])/2;c1=(end['L']+end['R'])/2;d0=pp['R']-pp['L'];d1=end['R']-end['L'];l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sn=np.linalg.norm(axis);angle=np.arctan2(sn,np.clip(u0@u1,-1,1));axis=axis/max(sn,1e-10);rot={s:Slerp([0,1],Rotation.from_matrix([self.geom(v,s)[1][:3,:3],targets[s][1]]))for s in ['L','R']};current=v;rows=[];maxspan=0.;maxpos=0.;maxangle=0.
  for k in range(1,frames+1):
   t=k/frames;a=t*t*t*(10-15*t+6*t*t);center=c0+a*(c1-c0);direction=Rotation.from_rotvec(axis*angle*a).apply(u0);span=l0+a*(l1-l0);goal=current
   for side,sign in [('R',1),('L',-1)]:
    rr=rot[side](a).as_matrix();cap=self.captures[side];point=center+sign*.5*span*direction;mid=point-rr@(cap['point_local']-cap['mid_local']);goal=self.solve_captured_normal(goal,side,point,rr[:,1],mid);actual=self.captured_points(goal)[side];pos=float(np.linalg.norm(actual-point));ang=float(np.rad2deg(np.arccos(np.clip(self.geom(goal,side)[1][:3,1]@rr[:,1],-1,1))));maxpos=max(maxpos,pos);maxangle=max(maxangle,ang);assert pos<=.002 and ang<=2,(side,k,pos,ang)
   poses=self.robot.fk(goal);assert max(self.collision(poses,s).max()for s in ['L','R'])<=.004,('collision',k);pts=self.captured_points(goal);distance=float(np.linalg.norm(pts['R']-pts['L']));maxspan=max(maxspan,distance);assert distance<=self.rest_span*1.005,('span',k,distance,self.rest_span);rows.append((goal,phase));current=goal
  rows,retiming=Retimer.retime_projected_rows(self,rows,v);self.last_span_audit=dict(rest_span=self.rest_span,maximum_predicted_span=maxspan,maximum_position_error_m=maxpos,maximum_normal_error_deg=maxangle,native_retiming=retiming,one_global_quintic_easing=True,object_state_writes=0);return rows

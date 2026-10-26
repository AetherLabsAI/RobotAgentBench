"""Plan physical adjacent-edge manipulation from actual captured material patches.
All outputs are native joint commands. No cloth state writes or attachments.
"""
from bimanual_lay_plan_v30 import LayPlanner as Base,DATA
from scipy.spatial.transform import Rotation,Slerp
import numpy as np,json
class LayPlanner(Base):
 def configure(self,ob,audit):
  self.captures={}
  for side in ['L','R']:
   ids=audit['hands'][side.lower()]['indices'];assert len(ids)>=3
   uv=ob['uv'][ids];xyz=ob['cloth'][ids];u=uv-uv.mean(0);assert np.linalg.matrix_rank(u)==2
   jac=np.linalg.lstsq(u,xyz-xyz.mean(0),rcond=None)[0].T;uu,sv,vv=np.linalg.svd(jac,full_matrices=False);basis=uu@vv;frame=np.c_[basis,np.cross(basis[:,0],basis[:,1])];mid,h,_,_=self.geom(ob['actual'],side)
   self.captures[side]=dict(uv=uv.mean(0),point_local=(xyz.mean(0)-h[:3,3])@h[:3,:3],mid_local=(mid-h[:3,3])@h[:3,:3],frame_local=h[:3,:3].T@frame,strain=sv)
  a=self.captures['L']['uv'];b=self.captures['R']['uv'];self.rest_span=float(np.linalg.norm(b-a));self.material_edge=(b-a)/self.rest_span;inside=np.array([-self.material_edge[1],self.material_edge[0]]);toward=np.array([.171,.255])-(a+b)/2
  if inside@toward<0:inside=-inside
  self.material_inward=inside
 def desired(self,center,edge,inward,ratio=.97):
  edge=np.asarray(edge,dtype=float);edge/=np.linalg.norm(edge);inward=np.asarray(inward,dtype=float);inward-=edge*(edge@inward);inward/=np.linalg.norm(inward);basis=np.outer(edge,self.material_edge)+np.outer(inward,self.material_inward);frame=np.c_[basis,np.cross(basis[:,0],basis[:,1])];targets={}
  for s,sign in [('L',-1),('R',1)]:
   c=self.captures[s];rot=frame@c['frame_local'].T;point=np.asarray(center)+sign*.5*self.rest_span*ratio*edge;mid=point-rot@(c['point_local']-c['mid_local']);targets[s]=(mid,rot)
  return targets
 def captured_points(self,v):
  result={}
  for s,c in self.captures.items():
   h=self.geom(v,s)[1];result[s]=h[:3,:3]@c['point_local']+h[:3,3]
  return result
 def move(self,v,targets,frames=320,phase='LR_adjacent_edge_unfold'):
  starts={s:self.geom(v,s)[:2]for s in ['L','R']};rotations={s:Slerp([0,1],Rotation.from_matrix(np.array([starts[s][1][:3,:3],targets[s][1]])))for s in ['L','R']};points=self.captured_points(v);end={s:targets[s][0]+targets[s][1]@(self.captures[s]['point_local']-self.captures[s]['mid_local'])for s in ['L','R']};c0=(points['L']+points['R'])/2;c1=(end['L']+end['R'])/2;d0=points['R']-points['L'];d1=end['R']-end['L'];l0=np.linalg.norm(d0);l1=np.linalg.norm(d1);u0=d0/l0;u1=d1/l1;axis=np.cross(u0,u1);sine=np.linalg.norm(axis);angle=np.arctan2(sine,np.clip(u0@u1,-1,1));axis=axis/max(sine,1e-10)if sine>1e-8 else np.array([1.,0,0]);current=v;rows=[];maxspan=0
  for k in range(1,41):
   a=k/40;a=a*a*(3-2*a);center=c0+a*(c1-c0);direction=Rotation.from_rotvec(axis*angle*a).apply(u0);distance=l0+a*(l1-l0);goal=current
   for side,sign in [('R',1),('L',-1)]:
    rot=rotations[side](a).as_matrix();cap=self.captures[side];point=center+sign*.5*distance*direction;mid=point-rot@(cap['point_local']-cap['mid_local']);goal=self.solve_full(goal,side,mid,rot)
   names=self.robot.arm_names('L')+self.robot.arm_names('R');N=max(frames//40,int(np.ceil(max(abs(goal[n]-current[n])for n in names)*1.5/.7*20)))
   for j in range(1,N+1):
    t=j/N;t=t*t*(3-2*t);q=current|{n:current[n]+t*(goal[n]-current[n])for n in names}
    if j==N or j%3==0:
     poses=self.robot.fk(q);assert max(self.collision(poses,s).max()for s in ['L','R'])<=.004,('native collision',k,j)
     pp=self.captured_points(q);span=float(np.linalg.norm(pp['R']-pp['L']));maxspan=max(maxspan,span);assert span<=self.rest_span*1.005,('captured material overextension',span,self.rest_span)
    rows.append((q,phase))
   current=goal
  self.last_span_audit=dict(rest_span=self.rest_span,maximum_predicted_span=maxspan,object_state_writes=0);return rows

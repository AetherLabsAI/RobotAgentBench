"""Checked joint-space approach around real basket geometry, no object writes."""
from removal_pregrasp_route_v1 import RoutePlanner as Base
import numpy as np,json,time
from pathlib import Path
class ClosedRoutePlanner(Base):
 def plan(self,start,end,frames=480):
  names=self.robot.arm_names('R');a=np.array([start[n]for n in names]);b=np.array([end[n]for n in names]);bounds=np.array([self.robot.by_name[n]['limits']for n in names]);bounds[3]=[.25,1.75];rng=np.random.default_rng(169);checks=0
  def values(x):
   q=start|dict(zip(names,x));return self.robot.calibrate_loops(q)
  def valid(x):
   nonlocal checks
   checks+=1;return bool(np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]) and self.validate(values(x))is None)
  def edge(x,y,res=.035):
   N=max(1,int(np.ceil(np.max(abs(y-x))/res)))
   return all(valid(x+(y-x)*t/N)for t in range(1,N+1))
  assert valid(a)and valid(b),'Actual start or target violates collision guard'
  trees=[[(a,-1)],[(b,-1)]];path=None;begin=time.monotonic()
  for step in range(5000):
   which=step%2;T=trees[which];U=trees[1-which]
   if step%5==0:target=U[-1][0]
   else:
    u=rng.random();target=np.clip(a+(b-a)*u+rng.normal(size=7)*(.25 if step<600 else .65),bounds[:,0],bounds[:,1])
   idx=min(range(len(T)),key=lambda k:np.linalg.norm(T[k][0]-target));near=T[idx][0];delta=target-near;candidate=near+delta*min(1,.18/max(np.linalg.norm(delta),1e-8))
   if not edge(near,candidate):continue
   T.append((candidate,idx));j=min(range(len(U)),key=lambda k:np.linalg.norm(U[k][0]-candidate));other=U[j][0]
   if np.linalg.norm(other-candidate)<.6 and edge(other,candidate):
    def chain(tree,k):
     out=[]
     while k>=0:out.append(tree[k][0]);k=tree[k][1]
     return out[::-1]
    l=chain(T,len(T)-1);r=chain(U,j);path=l+r[::-1]if which==0 else r+l[::-1];break
   if step%200==0:print('RRT',step,len(T),len(U),checks,flush=True)
  if path is None:raise RuntimeError(('No collision-free native approach',checks))
  # Shortcut only after full dense native collision validation.
  result=[path[0]];i=0
  while i<len(path)-1:
   j=len(path)-1
   while j>i+1 and not edge(path[i],path[j],.015):j-=1
   result.append(path[j]);i=j
  distance=np.r_[0,np.cumsum([np.linalg.norm(y-x)for x,y in zip(result[:-1],result[1:])])];rows=[]
  for k in range(1,frames+1):
   u=k/frames;u=u*u*(3-2*u);s=distance[-1]*u;j=min(np.searchsorted(distance,s,side='right')-1,len(result)-2);t=(s-distance[j])/(distance[j+1]-distance[j]);x=result[j]*(1-t)+result[j+1]*t;assert valid(x);q=values(x);prev=rows[-1]if rows else start;assert max(abs(q[n]-prev[n])*20 for n in names)<=1.2;rows.append(q)
  self.audit=dict(frames=len(rows),planner='bidirectional_native_joint_collision_search',waypoints=[x.tolist()for x in result],collision_checks=checks,planning_seconds=time.monotonic()-begin,object_state_writes=0);return rows

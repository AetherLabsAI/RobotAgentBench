"""Run-local observed cloth approach search; joint commands only, no state writes.
Searches oblique approach frames and collision-checked joint routes. Physical
pad containment/closure/retention must still be confirmed by the extension.
"""
from bimanual_lay_plan_v10 import LayPlanner as BasePlanner,DATA
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(BasePlanner):
 def prepare_with_clearance(self,v,goal):
  try:return super().prepare_with_clearance(v,goal)
  except RuntimeError:pass
  opened=self.robot.calibrate_loops(v|{n:goal[n] for n in goal if '_gripper_l_'in n});prefix=self.interpolate(v,opened,'L',40,'L_open_clear_route');names=self.robot.arm_names('L');start=np.array([opened[n]for n in names]);end=np.array([goal[n]for n in names]);bounds=np.array([self.robot.by_name[n]['limits']for n in names]);rng=np.random.default_rng(81011)
  def valid(q):
   val=opened|dict(zip(names,q));_,_,poses,outward=self.geom(val,'L');return outward>=.005 and self.collision(poses,'L').max()<=.004
  def edge(a,b):
   count=max(1,int(np.ceil(np.max(np.abs(b-a))/.04)))
   return all(valid(a+(b-a)*(j/count))for j in range(1,count+1))
  if not valid(start) or not valid(end):raise RuntimeError('No collision-free joint-route endpoints')
  trees=[([start],[-1]),([end],[-1])]
  def extend(tree,target):
   nodes,parents=tree;dist=np.linalg.norm(np.asarray(nodes)-target,axis=1);i=int(dist.argmin());q=nodes[i]+(target-nodes[i])*min(1.,.22/max(dist[i],1e-9))
   if not edge(nodes[i],q):return None
   nodes.append(q);parents.append(i);return len(nodes)-1
  def rootpath(tree,i):
   path=[]
   while i>=0:path.append(tree[0][i]);i=tree[1][i]
   return path[::-1]
  route=None
  for k in range(1600):
   ai=k%2;bi=1-ai;a,b=trees[ai],trees[bi];sample=b[0][rng.integers(len(b[0]))]if rng.random()<.25 else np.clip(start+rng.uniform(-1.8,1.8,7),bounds[:,0],bounds[:,1]);ia=extend(a,sample)
   if ia is None:continue
   for _ in range(30):
    ib=extend(b,a[0][ia])
    if ib is None:break
    if np.linalg.norm(b[0][ib]-a[0][ia])<1e-7:
     pa=rootpath(a,ia);pb=rootpath(b,ib);route=pa+pb[::-1][1:]if ai==0 else pb+pa[::-1][1:];break
   if route is not None:break
  if route is None:raise RuntimeError('No joint route around observed cloth within1600samples')
  simple=[route[0]];i=0
  while i<len(route)-1:
   j=len(route)-1
   while j>i+1 and not edge(route[i],route[j]):j-=1
   simple.append(route[j]);i=j
  rows=prefix;current=opened
  for q in simple[1:]:
   target=opened|dict(zip(names,q));part=self.interpolate(current,target,'L',40,'L_prepare_joint_clear_route');rows+=part;current=part[-1][0]
  return rows
 def choose_left(self,v,cloth,uv,cloth_ids=None):
  self.cloth_tree=cKDTree(cloth)
  try:return self._search_observed(v,cloth,uv,cloth_ids)
  finally:self.cloth_tree=None
 def _search_observed(self,v,cloth,uv,cloth_ids):
  ids=np.arange(len(uv))if cloth_ids is None else np.flatnonzero(cloth_ids==1);opened=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-.35,'idx41_gripper_l_outer_joint1':.35});names=self.robot.arm_names('L');nominal=self.robot.nominal_values();mid,h,_,_=self.geom(opened,'L');local=(mid-h[:3,3])@h[:3,:3];attempts=[]
  seeds=[opened,opened|{n:nominal[n]for n in names},opened|dict(zip(names,-np.array([v[n]for n in self.robot.arm_names('R')])))]
  for desired in [[.323,.49111111],[.33725,.50527778]]:
   idx=int(ids[np.linalg.norm(uv[ids]-desired,axis=1).argmin()]);point=cloth[idx];near=ids[np.linalg.norm(uv[ids]-uv[idx],axis=1)<.022];_,_,basis=np.linalg.svd(cloth[near]-cloth[near].mean(0));normal=basis[-1];iu=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[-.019,0]),axis=1).argmin()]);iv=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[0,-.019]),axis=1).argmin()]);du=cloth[iu]-point;du/=np.linalg.norm(du);dv=cloth[iv]-point;dv-=du*(dv@du);dv/=np.linalg.norm(dv)
   for tilt in [0,25,-25,45,-45,65,-65]:
    candidates=[]
    for angle in [90,60,30,0,-30,-60,-90,120,150,180,-120,-150]:
     tangent=np.cos(np.deg2rad(angle))*du+np.sin(np.deg2rad(angle))*dv
     for sign in [1,-1]:
      y=normal*sign;y-=tangent*(y@tangent);y/=np.linalg.norm(y);d=np.cos(np.deg2rad(tilt))*tangent+np.sin(np.deg2rad(tilt))*y;y=np.cos(np.deg2rad(tilt))*y-np.sin(np.deg2rad(tilt))*tangent;rot=np.stack([np.cross(y,d),y,d],axis=1);target=point-.065*d
      for seed in seeds:
       goal,err=self.robot.solve(seed,'L',target,local,rot,max_steps=160);_,hh,poses,outward=self.geom(goal,'L');oe=np.rad2deg(Rotation.from_matrix(rot.T@hh[:3,:3]).magnitude())
       if err>.002 or oe>2 or outward<.005 or self.collision(poses,'L').max()>.004:continue
       bends=np.rad2deg(abs(np.array([goal[names[3]],goal[names[5]]])));score=float(np.linalg.norm([goal[n]-v[n]for n in names])+.02*np.maximum(0,bends-60).sum());info=dict(material_vertex=idx,uv=uv[idx].tolist(),observed_xyz=point.tolist(),locked_rotation=rot.tolist(),direction=d.tolist(),bend_deg=bends.tolist(),approach_tilt_deg=tilt,tangent_angle_deg=angle,cloth_avoidance_margin_m=.015,standoff_m=.065,physics_state_writes=0);candidates.append((score,goal,info));break
    for _,goal,info in sorted(candidates,key=lambda row:row[0]):
     try:return self.prepare_with_clearance(v,goal),info
     except RuntimeError as e:attempts.append(str(e))
  raise RuntimeError(('No collision-free observed-corner approach including oblique candidates',attempts))

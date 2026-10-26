"""Run-local action-only helpers for an observed free corner after a verified right grasp.
No object state writes. Returned joint commands require a fresh continuous physical trial.
"""
from pathlib import Path
import json,sys,numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/aether/workspace/agent_bench/experiments/physics_fluid')
from tablecloth_robot_kinematics import Robot
from tablecloth_scene_binding import station_xyyaw
DATA=Path('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905')
class LayPlanner:
 def __init__(self):
  meta=json.loads((DATA/'robot_native_collision.json').read_text());self.robot=Robot(meta,station_xyyaw('8.9'));self.cloth_tree=None;self.pad=np.array([.03285,0,0,1.]);self.shapes={}
  for s in meta['nonhand_collision_shapes']:
   if s['link'].startswith(('arm_l_','arm_r_')) or s['link']=='body_link2':
    v=np.asarray(s['vertices']);pts=np.r_[v,v[np.asarray(s['triangles'])].mean(1)];self.shapes.setdefault(s['link'],[]).append(pts)
  self.shapes={n:np.concatenate(v) for n,v in self.shapes.items()}
  with np.load(DATA/'robot_native_collision.npz') as d:
   for key in d.files:
    if key.startswith('points_gripper_'):
     v=np.unique(d[key],axis=0);h=ConvexHull(v);v=np.r_[v[h.vertices],v[h.simplices].mean(1)];_,idx=np.unique(np.round(v/.004).astype(int),axis=0,return_index=True);self.shapes[key[7:]]=v[idx]
  self.boxes=np.array([[[-2.439,-1.034,.26],[-2.419,-.716,.45]],[[-2.419,-1.034,.26],[-2.061,-1.013,.45]],[[-2.419,-.737,.26],[-2.061,-.716,.45]],[[-2.061,-1.034,.26],[-2.041,-.716,.45]]])
 def geom(self,v,side):
  poses,axes=self.robot.fk(v,True);s=side.lower();arm=self.robot.arm_names(side);mid=np.mean([(poses[f'gripper_{s}_{k}_link4']@self.pad)[:3] for k in ['inner','outer']],axis=0);outside=(axes[arm[2]][0][1]-axes[arm[0]][0][1])*(1 if side=='L' else -1)
  return mid,poses['gripper_'+s+'_base_link'],poses,float(outside)
 def collision(self,poses,side):
  moving=[n for n in self.shapes if n.startswith('gripper_'+side.lower()+'_') or n in ['arm_'+side.lower()+'_link'+k for k in ['3','5','6','7']]];other='r' if side=='L' else 'l'
  world={n:pts@poses[n][:3,:3].T+poses[n][:3,3] for n,pts in self.shapes.items()}
  obstacle=np.concatenate([pts for n,pts in world.items() if n=='body_link2' or n.startswith('gripper_'+other+'_') or n in ['arm_'+other+'_link'+k for k in ['3','5','6','7']]])
  tree=cKDTree(obstacle);out=[]
  for n in moving:
   points=world[n];q=np.abs(points[:,None,:]-self.boxes.mean(1))-((self.boxes[:,1]-self.boxes[:,0])/2);sdf=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0);out.extend(np.maximum(0,.012-sdf.min(0)).tolist());out.append(max(0,.012-float(tree.query(points)[0].min())))
   # The diagnostic scene uses CoffeeTable/Mesh convex hull; conservatively bound its top.
  for n in moving:
   points=world[n];lower=np.array([-2.35,-.7210436,0]);upper=np.array([-1.574649,.4789564,.45]);d=np.abs(points-(lower+upper)/2)-(upper-lower)/2
   sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0);out.append(max(0,.008-float(sdf.min())))
  if self.cloth_tree is not None and side=='L':
   for n in moving:out.append(max(0,.015-float(self.cloth_tree.query(world[n])[0].min())))
  return np.asarray(out)

 def solve(self,v,side,target,orientation=None,guide=None):
  arm=self.robot.arm_names(side);seed=np.array([v[n] for n in arm]);bounds=np.array([self.robot.by_name[n]['limits'] for n in arm]);guide=seed if guide is None else np.asarray(guide)
  # Native limits remain unchanged. Sixty degrees is a posture preference, not a fake physical limit.
  bounds[:,0]=np.maximum(bounds[:,0],seed-np.pi);bounds[:,1]=np.minimum(bounds[:,1],seed+np.pi)
  def fun(q):
   mid,h,poses,out=self.geom(v|dict(zip(arm,q)),side)
   return np.r_[12*(mid-target),np.zeros(3) if orientation is None else .8*Rotation.from_matrix(orientation.T@h[:3,:3]).as_rotvec(),8*self.collision(poses,side),2*max(0,.015-out),.001*(q-guide),.008*np.maximum(0,np.abs(q[[3,5]])-np.deg2rad(60))]
  opt=least_squares(fun,np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0],bounds[:,1]),max_nfev=120,ftol=1e-8,gtol=1e-8,xtol=1e-8);result=v|dict(zip(arm,opt.x));mid,h,poses,out=self.geom(result,side)
  metrics=dict(side=side,error_m=float(np.linalg.norm(mid-target)),outward_m=out,collision_residual_m=float(self.collision(poses,side).max()),orientation_deg=0 if orientation is None else float(np.rad2deg(Rotation.from_matrix(orientation.T@h[:3,:3]).magnitude())),bend_deg=np.rad2deg(np.abs(opt.x[[3,5]])).tolist())
  if metrics['error_m']>.002 or metrics['collision_residual_m']>.004 or metrics['outward_m']<.005 or metrics['orientation_deg']>2:raise RuntimeError(('IK',metrics))
  self.last_metrics=metrics
  return result
 def interpolate(self,v,goal,side,frames,phase):
  names=[n for n in goal if '_arm_'+side.lower()+'_' in n or '_gripper_'+side.lower()+'_' in n];rows=[]
  max_delta=max(abs(v[n]-goal[n]) for n in names if '_arm_' in n);frames=max(frames,int(np.ceil(max_delta*1.5/.7*20)))
  for i in range(1,frames+1):
   t=i/frames;t=t*t*(3-2*t);q=v|{n:v[n]+t*(goal[n]-v[n]) for n in names}
   # Only actuated opening is interpolated; follower targets are derived from native loop geometry.
   if i==frames or i%5==0:
    q=self.robot.calibrate_loops(q)
    if self.collision(self.robot.fk(q),side).max()>.004:raise RuntimeError(('transition collision',phase,i))
   rows.append((q,phase))
  return rows
 def cartesian(self,v,side,target,frames,phase,orientation=None):
  start=self.geom(v,side)[0];rows=[];current=v
  for k in range(1,21):
   t=k/20;t=t*t*(3-2*t);goal=self.solve(current,side,start+t*(np.asarray(target)-start),orientation)
   part=self.interpolate(current,goal,side,max(1,frames//20),phase);rows+=part;current=goal
  return rows
 def present_right(self,v):
  # A middle presentation gives the left hand access without moving base or waist.
  mid=self.geom(v,'R')[0];target=np.array([-2.035,-.47,1.12]);rows=self.cartesian(v,'R',target,240,'R_present_for_left');return rows+[(rows[-1][0],'R_wait_free_corner')]*60
 def _choose_left_static(self,v,cloth,uv,cloth_ids=None):
  ids=np.arange(len(uv)) if cloth_ids is None else np.flatnonzero(cloth_ids==1)
  options=[];attempts=[];opened=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-.35,'idx41_gripper_l_outer_joint1':.35})
  desireds=[[.323,.49111111],[.323,.48166667]]
  for desired in desireds:
   idx=int(ids[np.linalg.norm(uv[ids]-desired,axis=1).argmin()]);point=cloth[idx];near=ids[np.linalg.norm(uv[ids]-uv[idx],axis=1)<.022]
   _,_,basis=np.linalg.svd(cloth[near]-cloth[near].mean(0));normal=basis[-1]
   duv=np.array([-.019 if desired[0]>.17 else .019,0]);dvv=np.array([0,-.019 if desired[1]>.25 else .019]);iu=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+duv),axis=1).argmin()]);iv=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+dvv),axis=1).argmin()]);du=cloth[iu]-point;du/=np.linalg.norm(du);dv=cloth[iv]-point;dv-=du*(dv@du);dv/=np.linalg.norm(dv)
   for angle in [90,45,0,-45]:
    d=np.cos(np.deg2rad(angle))*du+np.sin(np.deg2rad(angle))*dv
    for sign in [1,-1]:
     y=normal*sign;y-=d*(y@d);y/=np.linalg.norm(y);rot=np.stack([np.cross(y,d),y,d],axis=1)
     for seedid in [0,1]:
      seed=opened.copy()
      if seedid:seed.update(zip(self.robot.arm_names('L'),-np.array([v[n] for n in self.robot.arm_names('R')])))
      try:
       goal=self.solve(seed,'L',point-.065*d,rot);bend=self.last_metrics['bend_deg'];score=np.linalg.norm([goal[n]-v[n]for n in self.robot.arm_names('L')])+.02*sum(max(0,x-60)for x in bend)
       options.append((score,goal,dict(material_vertex=idx,uv=uv[idx].tolist(),observed_xyz=point.tolist(),locked_rotation=rot.tolist(),direction=d.tolist(),bend_deg=bend,physics_state_writes=0)))
      except RuntimeError as e:attempts.append(str(e))
    if options:break
   if options:break
  if not options:raise RuntimeError(('no observed left pregrasp',attempts))
  for _,goal,info in sorted(options,key=lambda x:x[0]):
   try:rows=self.prepare_with_clearance(v,goal);info['cloth_avoidance_margin_m']=.015;info['standoff_m']=.065;return rows,info
   except RuntimeError as e:attempts.append(str(e))
  raise RuntimeError(('left transition blocked',attempts))
 def insert_left(self,v,point,info):
  direction=np.asarray(info['direction']);return self.cartesian(v,'L',np.asarray(point)+.008*direction,100,'L_straight_mouth_insertion',np.asarray(info['locked_rotation']))
 def close_left(self,v):
  start=abs(v['idx31_gripper_l_inner_joint1']);rows=[]
  # Fix the whole arm while closing, as for accepted right grasp133.
  for i in range(1,31):
   t=i/30;t=t*t*(3-2*t);opening=start+(.009-start)*t;q=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-opening,'idx41_gripper_l_outer_joint1':opening});rows.append((q,'L_physical_close'))
  return rows+[(rows[-1][0],'L_verify_grip')]*180
 def both_cartesian(self,v,left,right,frames,phase):
  starts={s:self.geom(v,s)[0] for s in ['L','R']};targets={'L':np.asarray(left),'R':np.asarray(right)};rows=[];current=v
  for k in range(1,31):
   t=k/30;t=t*t*(3-2*t);goal=current
   for side in ['R','L']:goal=self.solve(goal,side,starts[side]+t*(targets[side]-starts[side]))
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n]) for n in names);N=max(frames//30,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,N+1):
    a=j/N;a=a*a*(3-2*a);q=current|{n:current[n]+a*(goal[n]-current[n]) for n in names}
    if j==N:
     poses=self.robot.fk(q)
     if max(self.collision(poses,s).max() for s in ['L','R'])>.004:raise RuntimeError(('two-arm collision',phase,k))
    rows.append((q,phase))
   current=goal
  return rows

 def choose_left(self,v,cloth,uv,cloth_ids=None):
  self.cloth_tree=cKDTree(cloth)
  try:return self._choose_left_static(v,cloth,uv,cloth_ids)
  finally:self.cloth_tree=None
 def prepare_with_clearance(self,v,goal):
  try:return self.interpolate(v,goal,'L',200,'L_prepare_free_corner')
  except RuntimeError as original:
   target,h,_,_=self.geom(goal,'L');normal=h[:3,1];rotation=h[:3,:3]
   for offset in [normal*.12,-normal*.12,np.array([0,.15,.08])]:
    try:
     opened=v|{n:goal[n] for n in goal if '_gripper_l_' in n};waypoint=self.solve(opened,'L',target+offset,rotation)
     a=self.interpolate(v,waypoint,'L',150,'L_prepare_clear_route')
     b=self.cartesian(a[-1][0],'L',target,100,'L_prepare_free_corner',rotation)
     return a+b
    except RuntimeError:pass
   raise original
 def recenter_left(self,v,cloth,point,info):
  self.cloth_tree=cKDTree(cloth)
  try:return self.cartesian(v,'L',np.asarray(point)-.065*np.asarray(info['direction']),40,'L_recenter_before_insertion',np.asarray(info['locked_rotation']))
  finally:self.cloth_tree=None

 def micro_left(self,v,target,rotation):
  goal=self.solve(v,'L',target,rotation)
  return self.interpolate(v,goal,'L',10,'L_live_mouth_insertion')

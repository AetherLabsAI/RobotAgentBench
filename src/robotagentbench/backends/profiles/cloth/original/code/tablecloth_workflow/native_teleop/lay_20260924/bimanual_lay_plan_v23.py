"""Action-only forward right presentation and native pad insertion sweep screening."""
from bimanual_lay_plan_v19 import LayPlanner as BasePlanner,DATA
from scipy.spatial.transform import Rotation
import numpy as np,json
from pathlib import Path
from scipy.spatial import ConvexHull
class LayPlanner(BasePlanner):
 left_aperture=.35
 def __init__(self):
  super().__init__();native=json.loads((Path(__file__).parent/'runs/round12_self_clear_bimanual_0149/native_left_pad_cooked_convexes.json').read_text());self.left_pad_equations={s:[ConvexHull(np.asarray(c['vertices_body'])).equations for r in native[s] for c in r['convexes']]for s in ['inner','outer']};self.sweep_rejections=0
 def sweep_clear(self,opened,cloth,uv,idx,rot,d):
  mid,h,poses,_=self.geom(opened,'L');world_rotation=rot@h[:3,:3].T;point=cloth[idx];near=np.linalg.norm(cloth-point,axis=1)<.20;q=cloth[near];u=uv[near]
  for remain in np.arange(.065,-.0025,-.005):
   goal=point-remain*d
   for side in ['inner','outer']:
    t=poses['gripper_l_'+side+'_link4'];rotation=world_rotation@t[:3,:3];position=world_rotation@(t[:3,3]-mid)+goal;local=(q-position)@rotation
    for eq in self.left_pad_equations[side]:
     contacts=(local@eq[:,:3].T+eq[:,3]).max(axis=1)<.003
     if contacts.any() and (remain>=.012 or np.any(np.linalg.norm(u[contacts]-uv[idx],axis=1)>.04)):
      self.sweep_rejections+=1;return False
  return True
 def forward_right_presentation(self,v):
  rotation=Rotation.from_euler('z',40,degrees=True).as_matrix()@Rotation.from_euler('y',20,degrees=True).as_matrix()@np.array([[0,0,1],[0,-1,0],[1,0,0]])
  mid,h,_,_=self.geom(v,'R');local=(mid-h[:3,3])@h[:3,:3];names=self.robot.arm_names('R');nominal=self.robot.nominal_values();failures=[];options=[]
  for target in [[-2.05,-.4,1.12],[-2.05,-.4,1.03]]:
   for seed in [v,v|{n:nominal[n]for n in names}]+[v|{names[2]:float(np.clip(v[names[2]]+delta,*self.robot.by_name[names[2]]['limits']))}for delta in [-.8,-.4,.4,.8]]:
    goal,err=self.robot.solve(seed,'R',np.array(target),local,rotation,max_steps=200);_,hh,poses,out=self.geom(goal,'R');oe=np.rad2deg(Rotation.from_matrix(rotation.T@hh[:3,:3]).magnitude())
    if err>.002 or oe>2 or out<.005 or self.collision(poses,'R').max()>.004:continue
    bends=np.rad2deg(abs(np.array([goal[names[3]],goal[names[5]]])));score=float(np.maximum(0,bends-60).sum()+.01*np.linalg.norm([goal[n]-v[n]for n in names]));options.append((score,target,goal,bends))
  for _,target,goal,bends in sorted(options,key=lambda row:row[0]):
   try:
    rows=self.interpolate(v,goal,'R',360,'R_turn_cloth_for_left');self.last_presentation=dict(target=target,rotation=rotation.tolist(),bend_deg=bends.tolist(),frames=len(rows),object_state_writes=0);return rows
   except RuntimeError as e:failures.append(str(e))
  raise RuntimeError(('No physically clear forward right presentation',failures))
 def _search_observed(self,v,cloth,uv,cloth_ids):
  ids=np.arange(len(uv))if cloth_ids is None else np.flatnonzero(cloth_ids==1);opened=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-self.left_aperture,'idx41_gripper_l_outer_joint1':self.left_aperture});names=self.robot.arm_names('L');nominal=self.robot.nominal_values();mid,h,_,_=self.geom(opened,'L');local=(mid-h[:3,3])@h[:3,:3];attempts=[]
  seeds=[opened,opened|{n:nominal[n]for n in names},opened|dict(zip(names,-np.array([v[n]for n in self.robot.arm_names('R')])))]
  for desired in [[.323,.49111111],[.33725,.50527778]]:
   idx=int(ids[np.linalg.norm(uv[ids]-desired,axis=1).argmin()]);point=cloth[idx];near=ids[np.linalg.norm(uv[ids]-uv[idx],axis=1)<.022];_,_,basis=np.linalg.svd(cloth[near]-cloth[near].mean(0));normal=basis[-1];iu=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[-.019,0]),axis=1).argmin()]);iv=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[0,-.019]),axis=1).argmin()]);du=cloth[iu]-point;du/=np.linalg.norm(du);dv=cloth[iv]-point;dv-=du*(dv@du);dv/=np.linalg.norm(dv)
   for tilt in [0,25,-25,45,-45,65,-65]:
    candidates=[]
    for angle in [90,60,30,0,-30,-60,-90,120,150,180,-120,-150]:
     tangent=np.cos(np.deg2rad(angle))*du+np.sin(np.deg2rad(angle))*dv
     for sign in [1,-1]:
      y=normal*sign;y-=tangent*(y@tangent);y/=np.linalg.norm(y);d=np.cos(np.deg2rad(tilt))*tangent+np.sin(np.deg2rad(tilt))*y;y=np.cos(np.deg2rad(tilt))*y-np.sin(np.deg2rad(tilt))*tangent;rot=np.stack([np.cross(y,d),y,d],axis=1);target=point-.065*d
      if not self.sweep_clear(opened,cloth,uv,idx,rot,d):continue
      for seed in seeds:
       goal,err=self.robot.solve(seed,'L',target,local,rot,max_steps=160);_,hh,poses,outward=self.geom(goal,'L');oe=np.rad2deg(Rotation.from_matrix(rot.T@hh[:3,:3]).magnitude())
       if err>.002 or oe>2 or outward<.005 or self.collision(poses,'L').max()>.004:continue
       bends=np.rad2deg(abs(np.array([goal[names[3]],goal[names[5]]])));score=float(np.linalg.norm([goal[n]-v[n]for n in names])+.02*np.maximum(0,bends-60).sum());info=dict(material_vertex=idx,uv=uv[idx].tolist(),observed_xyz=point.tolist(),locked_rotation=rot.tolist(),direction=d.tolist(),bend_deg=bends.tolist(),approach_tilt_deg=tilt,tangent_angle_deg=angle,cloth_avoidance_margin_m=.015,standoff_m=.065,physics_state_writes=0,native_pad_sweep_checked=True,sweep_rejections=self.sweep_rejections);candidates.append((score,goal,info));break
    for _,goal,info in sorted(candidates,key=lambda row:row[0]):
     try:return self.prepare_with_clearance(v,goal),info
     except RuntimeError as e:attempts.append(str(e))
  raise RuntimeError(('No collision-free observed-corner approach including oblique candidates',attempts))

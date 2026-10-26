"""Parallel pad-face candidate: no oblique grasp substitute; same native gates."""
from bimanual_lay_plan_v30 import LayPlanner as BasePlanner,DATA
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(BasePlanner):
 def _search_observed(self,v,cloth,uv,cloth_ids):
  ids=np.arange(len(uv))if cloth_ids is None else np.flatnonzero(cloth_ids==1);opened=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-self.left_aperture,'idx41_gripper_l_outer_joint1':self.left_aperture});names=self.robot.arm_names('L');nominal=self.robot.nominal_values();mid,h,_,_=self.geom(opened,'L');local=(mid-h[:3,3])@h[:3,:3];attempts=[];all_options=[]
  seeds=[opened,opened|{n:nominal[n]for n in names},opened|dict(zip(names,-np.array([v[n]for n in self.robot.arm_names('R')])))]
  for desired in getattr(self,"recovery_desired_uv",[[.019,.019],[.019,.49111111],[.323,.019],[.323,.49111111]]):
   idx=int(ids[np.linalg.norm(uv[ids]-desired,axis=1).argmin()]);point=cloth[idx];near=ids[np.linalg.norm(uv[ids]-uv[idx],axis=1)<.022];_,_,basis=np.linalg.svd(cloth[near]-cloth[near].mean(0));normal=basis[-1];iu=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[-.019,0]),axis=1).argmin()]);iv=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[0,-.019]),axis=1).argmin()]);du=cloth[iu]-point;du/=np.linalg.norm(du);dv=cloth[iv]-point;dv-=du*(dv@du);dv/=np.linalg.norm(dv)
   for tilt in [0]:
    candidates=[]
    for angle in [90,60,30,0,-30,-60,-90,120,150,180,-120,-150]:
     tangent=np.cos(np.deg2rad(angle))*du+np.sin(np.deg2rad(angle))*dv
     for sign in [1,-1]:
      y=normal*sign;y-=tangent*(y@tangent);y/=np.linalg.norm(y);d=np.cos(np.deg2rad(tilt))*tangent+np.sin(np.deg2rad(tilt))*y;y=np.cos(np.deg2rad(tilt))*y-np.sin(np.deg2rad(tilt))*tangent;rot=np.stack([np.cross(y,d),y,d],axis=1);target=point-.065*d
      if not self.sweep_clear(opened,cloth,uv,idx,rot,d):continue
      for seed in seeds:
       goal,err=self.robot.solve(seed,'L',target,local,rot,max_steps=160);_,hh,poses,outward=self.geom(goal,'L');oe=np.rad2deg(Rotation.from_matrix(rot.T@hh[:3,:3]).magnitude())
       if err>.002 or oe>2 or outward<.005 or self.collision(poses,'L').max()>.004:continue
       bends=np.rad2deg(abs(np.array([goal[names[3]],goal[names[5]]])));score=float(np.linalg.norm([goal[n]-v[n]for n in names])+.15*np.maximum(0,bends-60).sum());info=dict(material_vertex=idx,uv=uv[idx].tolist(),observed_xyz=point.tolist(),locked_rotation=rot.tolist(),direction=d.tolist(),bend_deg=bends.tolist(),approach_tilt_deg=tilt,tangent_angle_deg=angle,cloth_avoidance_margin_m=.015,standoff_m=.065,physics_state_writes=0,native_pad_sweep_checked=True,sweep_rejections=self.sweep_rejections);candidates.append((score,goal,info));print('CLEAR_LEFT_ENDPOINT',desired,tilt,angle,bends.tolist(),flush=True)
       if max(bends)<=60:
        try:return self.prepare_with_clearance(v,goal),info
        except RuntimeError as e:attempts.append(str(e))
       break
    all_options.extend(candidates)
  for _,goal,info in sorted(all_options,key=lambda row:row[0]):
   try:return self.prepare_with_clearance(v,goal),info
   except RuntimeError as e:attempts.append(str(e))
  raise RuntimeError(('No collision-free observed-corner approach including oblique candidates',attempts))

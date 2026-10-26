"""Observed left adjacent-corner small-angle approach; unchanged contact/clearance gates."""
from native_body1_collision_guard_v1 import LayPlanner as BasePlanner,DATA
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import json
class LayPlanner(BasePlanner):
 def _search_observed(self,v,cloth,uv,cloth_ids):
  ids=np.arange(len(uv))if cloth_ids is None else np.flatnonzero(cloth_ids==1);opened=self.robot.calibrate_loops(v|{'idx31_gripper_l_inner_joint1':-self.left_aperture,'idx41_gripper_l_outer_joint1':self.left_aperture});names=self.robot.arm_names('L');nominal=self.robot.nominal_values();mid,h,_,_=self.geom(opened,'L');local=(mid-h[:3,3])@h[:3,:3];attempts=[];all_options=[];rejected=dict(sweep=0,ik=0,orientation=0,outward=0,collision=0,insertion=0)
  seeds=[opened,opened|{n:nominal[n]for n in names},opened|dict(zip(names,-np.array([v[n]for n in self.robot.arm_names('R')])))]
  seed_path=Path(__file__).parent/'runs/round12_parallel_padface_adjacent_0160/repair_request_005.json'
  old=json.loads(seed_path.read_text())['actions'][-1]['targets'];seeds.append(opened|{n:float(old[n])for n in names})
  for desired in getattr(self,"recovery_desired_uv",[[.019,.019],[.019,.49111111],[.323,.019],[.323,.49111111]]):
   idx=int(ids[np.linalg.norm(uv[ids]-desired,axis=1).argmin()]);point=cloth[idx];near=ids[np.linalg.norm(uv[ids]-uv[idx],axis=1)<.022];_,_,basis=np.linalg.svd(cloth[near]-cloth[near].mean(0));normal=basis[-1];iu=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[-.019,0]),axis=1).argmin()]);iv=int(ids[np.linalg.norm(uv[ids]-(uv[idx]+[0,-.019]),axis=1).argmin()]);du=cloth[iu]-point;du/=np.linalg.norm(du);dv=cloth[iv]-point;dv-=du*(dv@du);dv/=np.linalg.norm(dv)
   for tilt in [0,15,-15,25,-25,45,-45]:
    candidates=[]
    for angle in [90,60,30,0,-30,-60,-90,120,150,180,-120,-150]:
     tangent=np.cos(np.deg2rad(angle))*du+np.sin(np.deg2rad(angle))*dv
     for sign in [1,-1]:
      y=normal*sign;y-=tangent*(y@tangent);y/=np.linalg.norm(y);d=np.cos(np.deg2rad(tilt))*tangent+np.sin(np.deg2rad(tilt))*y;y=np.cos(np.deg2rad(tilt))*y-np.sin(np.deg2rad(tilt))*tangent;rot=np.stack([np.cross(y,d),y,d],axis=1);target=point-.065*d
      if not self.sweep_clear(opened,cloth,uv,idx,rot,d):rejected['sweep']+=1;continue
      for seed in seeds:
       goal,err=self.robot.solve(seed,'L',target,local,rot,max_steps=160);_,hh,poses,outward=self.geom(goal,'L');oe=np.rad2deg(Rotation.from_matrix(rot.T@hh[:3,:3]).magnitude())
       if err>.002:rejected['ik']+=1;continue
       if oe>2:rejected['orientation']+=1;continue
       if outward<.005:rejected['outward']+=1;continue
       if self.collision(poses,'L').max()>.004:rejected['collision']+=1;continue
       tree=self.cloth_tree;self.cloth_tree=None;current=goal;insert_ok=True
       try:
        for travel in [.018,.036,.054,.073]:
         nxt=self.solve_full(current,'L',target+travel*d,rot)
         self.interpolate(current,nxt,'L',16,'offline_complete_insertion_preflight');current=nxt
       except RuntimeError:insert_ok=False
       finally:self.cloth_tree=tree
       if not insert_ok:rejected['insertion']+=1;continue
       bends=np.rad2deg(abs(np.array([goal[names[3]],goal[names[5]]])));score=float(np.linalg.norm([goal[n]-v[n]for n in names])+.15*np.maximum(0,bends-60).sum());info=dict(material_vertex=idx,uv=uv[idx].tolist(),observed_xyz=point.tolist(),locked_rotation=rot.tolist(),direction=d.tolist(),bend_deg=bends.tolist(),approach_tilt_deg=tilt,tangent_angle_deg=angle,cloth_avoidance_margin_m=.015,standoff_m=.065,physics_state_writes=0,native_pad_sweep_checked=True,sweep_rejections=self.sweep_rejections);candidates.append((score,goal,info));print('CLEAR_LEFT_ENDPOINT',desired,tilt,angle,bends.tolist(),flush=True)
       if max(bends)<=100:
        try:return self.prepare_with_clearance(v,goal),info
        except RuntimeError as e:attempts.append(str(e))
       break
    all_options.extend(candidates)
   print('MEASURED_CANDIDATE_REJECTIONS',desired,rejected,flush=True)
  for _,goal,info in sorted(all_options,key=lambda row:row[0]):
   try:return self.prepare_with_clearance(v,goal),info
   except RuntimeError as e:attempts.append(str(e))
  raise RuntimeError(('No collision-free observed-corner approach with parallel face',dict(routes=attempts,rejected=rejected)))

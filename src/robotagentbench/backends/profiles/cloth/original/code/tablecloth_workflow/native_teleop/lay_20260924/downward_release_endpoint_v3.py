"""Read-only full-arm endpoint search from the current native joint observation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def find_endpoints(pl, command, max_results=3):
 robot=pl.robot;arm=robot.arm_names('R');start=np.array([command[n] for n in arm]);bounds=np.array([robot.by_name[n]['limits'] for n in arm]);bounds[3]=[.25,2.5]
 mid,h,_,_=pl.geom(command,'R');local=(mid-h[:3,3])@h[:3,:3];y=h[:3,1].copy();y[2]=0
 if np.linalg.norm(y)<1e-6:y=np.array([1.,0.,0.])
 y/=np.linalg.norm(y);down=np.array([0.,0.,-1.]);rng=np.random.default_rng(178);results=[];rejected=[]
 targets=[np.array([-2.28,-.90,.70]),np.array([-2.28,-.90,.75]),mid,np.array([-2.25,-.85,.80]),np.array([-2.32,-.82,.75])]
 for target in targets:
  for yaw in [90,0,-45,45]:
   yy=Rotation.from_euler('z',yaw,degrees=True).apply(y);rot=np.stack([np.cross(yy,down),yy,down],1)
   for seedid in range(5):
    seed=start if seedid==0 else np.clip(start+rng.normal(0,.75,7),bounds[:,0]+1e-7,bounds[:,1]-1e-7)
    def residual(x):
     hh=robot.fk(command|dict(zip(arm,x)))['gripper_r_base_link'];return np.r_[hh[:3,3]+hh[:3,:3]@local-target,.12*Rotation.from_matrix(rot@hh[:3,:3].T).as_rotvec(),1e-7*(x-start)]
    fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=300,ftol=1e-9,gtol=1e-9,xtol=1e-9)
    q=command|dict(zip(arm,fit.x.tolist()));m,hh,poses,out=pl.geom(q,'R');err=float(np.linalg.norm(m-target));angle=float(np.rad2deg(Rotation.from_matrix(rot@hh[:3,:3].T).magnitude()));bad=pl.validate(q)
    if err<.002 and angle<2 and bad is None and out>.005:
     results.append(dict(command=q,target=target.tolist(),yaw=yaw,seed=seedid,position_error_m=err,angle_error_deg=angle,delta_norm=float(np.linalg.norm(fit.x-start)),outside_m=out));break
    rejected.append(dict(target=target.tolist(),yaw=yaw,seed=seedid,error=err,angle=angle,collision=bad,outside_m=out))
   if len(results)>=max_results:return results,rejected
 return results,rejected

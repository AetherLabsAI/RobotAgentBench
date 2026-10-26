"""Preflight a measured cloth candidate before leaving the repeat-brush loop."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from removal_pregrasp_route_v3 import RoutePlanner

def endpoint_for_candidate(controller,candidate):
 robot=controller.robot;arm=controller.right_arm;shift=np.asarray(candidate['nominal_translation'])
 if np.linalg.norm(shift)>.16:return None,dict(retarget_range_m=float(np.linalg.norm(shift)),maximum_retarget_range_m=.16,range_passed=False)
 nominal=dict(zip(controller.names,map(float,controller.action_targets[controller.right_pre_end])));H=robot.fk(nominal)['gripper_r_base_link'];fitted,error=robot.solve(nominal,'R',H[:3,3]+shift,np.zeros(3),H[:3,:3],max_steps=600);bounds=np.array([robot.by_name[name]['limits']for name in arm]);bounds[arm.index('idx64_arm_r_joint4'),0]=.25
 def residual(x):
  T=robot.fk(nominal|dict(zip(arm,x)))['gripper_r_base_link'];return np.r_[T[:3,3]-H[:3,3]-shift,.065*Rotation.from_matrix(H[:3,:3]@T[:3,:3].T).as_rotvec()]
 fit=least_squares(residual,np.clip([fitted[name]for name in arm],bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=1000,gtol=1e-11,ftol=1e-11,xtol=1e-11);fitted.update(zip(arm,fit.x.tolist()));F=robot.fk(fitted)['gripper_r_base_link'];error=float(np.linalg.norm(F[:3,3]-H[:3,3]-shift));angle=float(np.rad2deg(Rotation.from_matrix(H[:3,:3]@F[:3,:3].T).magnitude()));start=controller.bundle_previous;endpoint=start|{n:fitted[n]for n in arm}|{n:nominal[n]for n in controller.names if 'gripper_r_'in n};endpoint=robot.calibrate_loops(endpoint);router=RoutePlanner();start_bad=router.validate(start);end_bad=router.validate(endpoint);audit=dict(position_error_m=error,rotation_error_deg=angle,start_collision=start_bad,endpoint_collision=end_bad)
 if error>.002 or angle>2 or start_bad or end_bad:return None,audit
 correction={n:fitted[n]-nominal[n]for n in arm};return (router,start,endpoint,correction),audit

def checked_candidate_route(controller,candidate):
 result,audit=endpoint_for_candidate(controller,candidate)
 if result is None:return None,audit
 router,start,end,correction=result
 try:route=router.plan(start,end,controller.right_pre_end-controller.right_start+1)
 except (AssertionError,RuntimeError)as exc:return None,audit|dict(route_failure=str(exc))
 return dict(route=route,correction=correction,audit=router.audit),audit|dict(full_route_passed=True)

"""Small reference-relative IK correction; never integrate redundant-joint drift."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
def solve_local_brush_target(robot,values,target,rotation):
 names=robot.arm_names('R');reference=np.array([values[n]for n in names]);bounds=np.array([robot.by_name[n]['limits']for n in names]);lo=np.maximum(bounds[:,0]+1e-6,reference-.35);hi=np.minimum(bounds[:,1]-1e-6,reference+.35)
 def residual(x):
  h=robot.fk(values|dict(zip(names,x)))['gripper_r_base_link'];return np.r_[h[:3,3]-target,.12*Rotation.from_matrix(rotation.T@h[:3,:3]).as_rotvec(),.002*(x-reference)]
 fit=least_squares(residual,np.clip(reference,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=120,xtol=1e-10,ftol=1e-10,gtol=1e-10);q=values|dict(zip(names,map(float,fit.x)));h=robot.fk(q)['gripper_r_base_link'];return q,float(np.linalg.norm(h[:3,3]-target))

from scipy.spatial.transform import Slerp
closed_q=np.array([values[n] for n in arm])
head=robot.fk(values)['head_link2'][:3,3]
eye_goal=head+np.array([.46,-.40,-.09])
neutral_q=np.deg2rad([-30,50,0,30,0,-25,15])
neutral_R=geom(values|dict(zip(arm,neutral_q)))[1][:3,:3]
# Find a reachable, bent-elbow and relaxed-wrist presentation pose.
goal_bounds=bounds.copy();goal_bounds[3]=np.deg2rad([15,50]);goal_bounds[5]=np.deg2rad([-45,-10])
def goal_res(q):
 mid,h,poses,out=geom(values|dict(zip(arm,q)))
 return np.r_[12*(mid-eye_goal),.035*Rotation.from_matrix(neutral_R.T@h[:3,:3]).as_rotvec(),.012*(q-neutral_q),5*collision_res(poses),5*max(0,.03-out)]
opt=least_squares(goal_res,np.clip(neutral_q,goal_bounds[:,0]+1e-7,goal_bounds[:,1]-1e-7),bounds=(goal_bounds[:,0],goal_bounds[:,1]),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
goal_q=opt.x;goal_mid,goal_h,goal_poses,goal_out=geom(values|dict(zip(arm,goal_q)))
if np.linalg.norm(goal_mid-eye_goal)>.002 or collision_res(goal_poses).max()>.012 or goal_out<.02:raise RuntimeError(('whole arm presentation goal',np.linalg.norm(goal_mid-eye_goal),np.rad2deg(goal_q).tolist()))
rotation_curve=Slerp([0,1],Rotation.from_matrix(np.stack([locked_R,goal_h[:3,:3]])))
# Lift first, with a small outward clearance; elbow and wrist move together.
lift_delta=np.array([0,0,max(.16,.57-corner[2])])
lift_delta[:2]=.30*(eye_goal[:2]-corner[:2])
for i in range(1,201):
 t=smooth(i/200);blend=.35*t;shape_guide=closed_q+blend*(goal_q-closed_q);shape_rotation=rotation_curve(blend).as_matrix()
 values=solve(values,corner+lift_delta*t,t<.05);append(values,'R_lift_outside_clear')
print('coordinated shoulder elbow wrist lift planned',flush=True)
start=geom(values)[0].copy();lift_q=np.array([values[n] for n in arm])
for i in range(1,601):
 t=smooth(i/600);coordinated_q=lift_q+t*(goal_q-lift_q)
 shape_guide=coordinated_q;shape_rotation=rotation_curve(.35+.65*t).as_matrix()
 values=solve(values,start+t*(eye_goal-start),False);mid,h,poses,outside=geom(values)
 if collision_res(poses).max()>.012 or outside<.015 or mid[2]<max(corner[2]+.12,.55):
  raise RuntimeError(('coordinated carry clearance',i,mid.tolist(),outside,float(collision_res(poses).max())))
 append(values,'R_extend_to_eye_height')
print('whole arm eye pose planned',np.rad2deg([values[n] for n in arm]).tolist(),flush=True)

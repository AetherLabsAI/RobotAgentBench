from pathlib import Path
exec(open(str(Path(__file__).parent/'probe_horizontal_collision.py')).read().split('rng=np.random')[0])
bounds[[0,2],0]=np.deg2rad(-160);bounds[[0,2],1]=np.deg2rad(160)
bounds[3]=np.deg2rad([.001,59]);bounds[5]=np.deg2rad([-59,-.001])
from profile_hand_clearance_exact import make_hand_clearance
hand_distance=make_hand_clearance()
pad=np.array([.03785,0,0,1.])
from profile_horizontal_clearance_exact import make_clearance
arm_collision_res=make_clearance(meta)
def collision_res(poses):return np.r_[arm_collision_res(poses),3*np.maximum(0,.012-hand_distance(poses))]
c=json.loads(Path(sys.argv[1]).read_text())[0];values=c['values'];corner=np.asarray(c['edge']);locked_R=np.asarray(c['mouth_rotation']);yaw=np.arctan2(locked_R[1,2],locked_R[0,2]);direction=locked_R[:,2].copy();direction/=np.linalg.norm(direction)
with np.load(P.parent/'right_basket_outward_long_20hz_0098.npz') as z:names=z['joint_names'].tolist();initial=dict(zip(names,z['joint_target'][0]))
rows=[];phases=[];audit=[]
def smooth(t):return t*t*(3-2*t)
def geom(v):
 poses,axes=robot.fk(v,True);h=poses['gripper_r_base_link'];mid=sum((poses['gripper_r_'+k+'_link4']@pad)[:3] for k in ['inner','outer'])/2
 return mid,h,poses,float(axes[arm[0]][0][1]-axes[arm[2]][0][1])
shape_guide=None
shape_rotation=None
def solve(v,target,lock=True):
 seed=np.array([v[n] for n in arm])
 def residual(q):
  mid,h,poses,out=geom(v|dict(zip(arm,q)))
  return np.r_[5*(mid-target),(1.5 if lock else .06)*Rotation.from_matrix((locked_R if lock or shape_rotation is None else shape_rotation).T@h[:3,:3]).as_rotvec(),3*max(0,.02-out),collision_res(poses),.0001*(q-seed),np.zeros(7) if lock or shape_guide is None else .015*(q-shape_guide)]
 step_bounds=bounds.copy()
 if len(rows)>=700:
  step_bounds[:,0]=np.maximum(step_bounds[:,0],seed-.07);step_bounds[:,1]=np.minimum(step_bounds[:,1],seed+.07)
 opt=least_squares(residual,np.clip(seed,step_bounds[:,0]+1e-8,step_bounds[:,1]-1e-8),bounds=(step_bounds[:,0],step_bounds[:,1]),max_nfev=160,gtol=1e-10,ftol=1e-10,xtol=1e-10)
 out=v|dict(zip(arm,opt.x));mid,h,poses,elbow=geom(out);err=float(np.linalg.norm(mid-target));rotation=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(locked_R.T@h[:3,:3]).as_rotvec())))
 if err>.002 or (lock and rotation>1) or elbow<.015 or collision_res(poses).max()>.012:raise RuntimeError(('pose',len(rows),err,rotation,elbow,np.rad2deg(opt.x).tolist()))
 return out

def append(v,phase):rows.append([v[n] if '_arm_r_' in n or '_gripper_r_' in n else initial[n] for n in names]);phases.append(phase)

for _ in range(800):append(values,"unused_tail_prefix")

for i in range(1,21):
 opening=.6+smooth(i/20)*(.012-.6);v=robot.calibrate_loops(values|{'idx71_gripper_r_inner_joint1':-opening,'idx81_gripper_r_outer_joint1':opening});values=v;append(values,'R_immediate_physical_close')
for _ in range(10):append(values,'R_verify_physical_grip')
corner=geom(values)[0].copy()
print('physical close planned with fixed arm targets, no withdrawal',flush=True)
exec((Path(__file__).parent/'whole_arm_short50_reach.py').read_text())
for _ in range(100):append(values,'R_eye_height_display')

a=np.asarray(rows);rates=np.max(np.abs(np.diff(a[:,[names.index(n) for n in arm]],axis=0)))*20
if rates>1.5:raise RuntimeError(('joint rate',rates,int(np.unravel_index(np.abs(np.diff(a[:,[names.index(n) for n in arm]],axis=0)).argmax(),(len(a)-1,7))[0])))
for i in range(0,len(rows),5):
 v=dict(zip(names,rows[i]));mid,h,poses,out=geom(v)
 if collision_res(poses).max()>.012 or (i>=700 and out<0):raise RuntimeError(('transition clearance or inward elbow',i,out))
 if False:
  delta=mid-pre;cross=float(np.linalg.norm(delta-direction*(delta@direction)));angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(locked_R.T@h[:3,:3]).as_rotvec())))
  if cross>.002 or angle>1:raise RuntimeError(('straightness',i,cross,angle))
  audit.append(dict(frame=i,lateral_error_m=cross,orientation_drift_deg=angle,camera_mount_outward_m=float(-(h[:3,:3]@np.array([.07593,-.009,.021]))[1])))
out=Path(sys.argv[2]);np.savez_compressed(out,joint_names=np.array(names),joint_target=a,phase=np.array(phases,dtype='U80'),time_s=np.arange(len(rows))/20)
out.with_suffix('.json').write_text(json.dumps(dict(schema='task8.single_exposed_corner_shallow_grasp.v1',pinch_point_link4_local_m=pad.tolist(),requested_single_layer=True,forward_extra_m=.020,closure_arm_targets_fixed=True,long_edge_heading_world=[1,0,0],forearm_pitch_deg=float(np.rad2deg(np.arcsin(direction[2]))),diagnostic_only=True,physical_success=False,body_fixed=True,whole_episode_initial_source=str(P.parent/'right_basket_outward_long_20hz_0098.npz'),camera_side='inward',eye_target=eye_goal.tolist(),corner_uv=c['uv'],corner_xyz=corner.tolist(),opening_normal_local=[0,0,1],closing_axis_local=[0,1,0],direction=direction.tolist(),locked_rotation=locked_R.tolist(),close_starts_frame=800,close_duration_s=1,pre_close_dwell_frames=0,arm_max_rate_rad_s=float(rates),approach_audit=audit),indent=2)+'\n');print(out,flush=True)

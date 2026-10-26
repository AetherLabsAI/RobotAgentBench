from pathlib import Path
exec(open(str(Path(__file__).parent/'probe_horizontal_collision.py')).read().split('rng=np.random')[0])
bounds[[0,2],0]=np.deg2rad(-160);bounds[[0,2],1]=np.deg2rad(160)
bounds[3]=np.deg2rad([.001,59]);bounds[5]=np.deg2rad([-59,-.001])
from profile_hand_clearance_exact import make_hand_clearance
hand_distance=make_hand_clearance()
pad=np.array([.03285,0,0,1.])
from profile_horizontal_clearance_exact import make_clearance
arm_collision_res=make_clearance(meta)
def collision_res(poses):return np.r_[arm_collision_res(poses),3*np.maximum(0,.012-hand_distance(poses))]
c=json.loads(Path(sys.argv[1]).read_text())[0];values=c['values'];corner=np.asarray(c['edge']);locked_R=np.asarray(c['mouth_rotation']);yaw=np.arctan2(locked_R[1,2],locked_R[0,2]);locked_R=Rotation.from_euler('z',-yaw).as_matrix()@locked_R;direction=locked_R[:,2].copy();direction/=np.linalg.norm(direction)
with np.load(P.parent/'right_basket_outward_long_20hz_0098.npz') as z:names=z['joint_names'].tolist();initial=dict(zip(names,z['joint_target'][0]))
rows=[];phases=[];audit=[];plan_warnings=[]
def plan_warning(value):
 plan_warnings.append(str(value))
 if len(plan_warnings)<5:print('DIAGNOSTIC PLANNER WARNING',value,flush=True)
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
 if err>.002 or (lock and rotation>1) or elbow<.015 or collision_res(poses).max()>.012:plan_warning(('pose',len(rows),err,rotation,elbow,np.rad2deg(opt.x).tolist()))
 return out

def append(v,phase):rows.append([v[n] if '_arm_r_' in n or '_gripper_r_' in n else initial[n] for n in names]);phases.append(phase)
corner=corner+.010*direction
pre=corner-.060*direction;values=solve(values,pre);pre_values=values.copy()
for _ in range(400):append(initial,'initial_fixed_body_settle')
from scipy.spatial.transform import Slerp
initial_mid,initial_hand,_,_=geom(initial)
orientation_curve=Slerp([0,1],Rotation.from_matrix(np.stack([initial_hand[:3,:3],locked_R])))
previous=initial.copy()
for i in range(1,301):
 a=smooth(i/300)
 waypoint=pre+np.array([0,0,.12])
 target=initial_mid+(waypoint-initial_mid)*smooth(min(1.,i/180)) if i<=180 else waypoint+(pre-waypoint)*smooth((i-180)/120)
 reference_q=np.array([initial[n]+a*(pre_values[n]-initial[n]) for n in arm])
 desired_R=geom(initial|dict(zip(arm,reference_q)))[1][:3,:3]
 v=initial.copy()
 for n in names:
  if '_gripper_r_' in n:v[n]=initial[n]+a*(pre_values[n]-initial[n])
 v=robot.calibrate_loops(v)
 prior=np.array([previous[n] for n in arm])
 envelope=np.array([initial[n] for n in arm])
 prep_bounds=bounds.copy();prep_bounds[:,0]=np.minimum(prep_bounds[:,0],envelope);prep_bounds[:,1]=np.maximum(prep_bounds[:,1],envelope)
 lo=np.maximum(prep_bounds[:,0],prior-.07);hi=np.minimum(prep_bounds[:,1],prior+.07)
 def prep_res(q):
  mid,h,poses,out=geom(v|dict(zip(arm,q)))
  return np.r_[8*(mid-target),(.02+1.48*a**8)*Rotation.from_matrix(desired_R.T@h[:3,:3]).as_rotvec(),collision_res(poses)*8, .001*(q-prior),.10*(q-reference_q),a**6*np.maximum(0,np.abs(q[[3,5]])-np.deg2rad(59)),3*max(0,.02*a-out),8*max(0,pre[1]-.005-mid[1])]
 opt=least_squares(prep_res,np.clip(prior,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=100,ftol=1e-8)
 v.update(zip(arm,opt.x));mid,h,poses,out=geom(v)
 if np.linalg.norm(mid-target)>.01 or mid[1]<pre[1]-.006 or collision_res(poses).max()>.012:plan_warning(('compact preparation',i,np.linalg.norm(mid-target),mid.tolist(),float(collision_res(poses).max()),np.rad2deg(opt.x).tolist(),opt.active_mask.tolist()))
 append(v,'R_prepare_inward_camera');previous=v
values=previous.copy()
# Freeze the entire tool orientation; all approach targets lie on ONE horizontal line.
for i in range(1,101):
 target=pre+smooth(i/100)*.060*direction;values=solve(values,target);append(values,'R_straight_mouth_insertion')
print('straight approach planned',flush=True)
# No dwell, no extra translation and no wrist turn between insertion and closure.
for i in range(1,21):
 opening=.35+smooth(i/20)*(.012-.35);v=robot.calibrate_loops(values|{'idx71_gripper_r_inner_joint1':-opening,'idx81_gripper_r_outer_joint1':opening});values=v;append(values,'R_immediate_physical_close')
for _ in range(10):append(values,'R_verify_physical_grip')
corner=geom(values)[0].copy()
print('physical close planned with fixed arm targets, no withdrawal',flush=True)
exec((Path(__file__).parent/'whole_arm_complete_review_reach.py').read_text())
for _ in range(100):append(values,'R_eye_height_display')

a=np.asarray(rows);rates=np.max(np.abs(np.diff(a[:,[names.index(n) for n in arm]],axis=0)))*20
if rates>1.5:plan_warning(('joint rate',rates,int(np.unravel_index(np.abs(np.diff(a[:,[names.index(n) for n in arm]],axis=0)).argmax(),(len(a)-1,7))[0])))
for i in range(0,len(rows),5):
 v=dict(zip(names,rows[i]));mid,h,poses,out=geom(v)
 if collision_res(poses).max()>.012 or (i>=700 and out<0):plan_warning(('transition clearance or inward elbow',i,out))
 if phases[i]=='R_straight_mouth_insertion':
  delta=mid-pre;cross=float(np.linalg.norm(delta-direction*(delta@direction)));angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(locked_R.T@h[:3,:3]).as_rotvec())))
  if cross>.002 or angle>1:plan_warning(('straightness',i,cross,angle))
  audit.append(dict(frame=i,lateral_error_m=cross,orientation_drift_deg=angle,camera_mount_outward_m=float(-(h[:3,:3]@np.array([.07593,-.009,.021]))[1])))
out=Path(sys.argv[2]);np.savez_compressed(out,joint_names=np.array(names),joint_target=a,phase=np.array(phases,dtype='U80'),time_s=np.arange(len(rows))/20)
out.with_suffix('.json').write_text(json.dumps(dict(schema='task8.full_motion_diagnostic.v1',plan_metric_failures=plan_warnings,pinch_point_link4_local_m=pad.tolist(),requested_single_layer=True,forward_extra_m=.010,closure_arm_targets_fixed=True,long_edge_heading_world=[1,0,0],forearm_pitch_deg=float(np.rad2deg(np.arcsin(direction[2]))),diagnostic_only=True,physical_success=False,body_fixed=True,whole_episode_initial_source=str(P.parent/'right_basket_outward_long_20hz_0098.npz'),camera_side='inward',eye_target=eye_goal.tolist(),corner_uv=c['uv'],corner_xyz=corner.tolist(),opening_normal_local=[0,0,1],closing_axis_local=[0,1,0],direction=direction.tolist(),locked_rotation=locked_R.tolist(),close_starts_frame=800,close_duration_s=1,pre_close_dwell_frames=0,arm_max_rate_rad_s=float(rates),approach_audit=audit),indent=2)+'\n');print(out,flush=True)

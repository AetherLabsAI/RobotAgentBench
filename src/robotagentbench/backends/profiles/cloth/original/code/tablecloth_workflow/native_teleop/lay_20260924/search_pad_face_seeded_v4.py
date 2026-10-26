from pathlib import Path
exec(open(str(Path(__file__).parent/'probe_horizontal_collision.py')).read().split('rng=np.random')[0])
from profile_hand_clearance_exact import make_hand_clearance
from profile_horizontal_clearance_exact import make_clearance
collision_res=make_clearance(meta)
hand_distance=make_hand_clearance()
pad=np.array([.03285,0,0,1.])
values=robot.calibrate_loops(values|{'idx71_gripper_r_inner_joint1':-.35,'idx81_gripper_r_outer_joint1':.35})
with np.load(Path(sys.argv[1])) as z:
 ids=z['cloth_id']==1;uv=z['uv'][ids];cloth=z['cloth_q'][0,ids]
bounds[:,0]=np.maximum(bounds[:,0],np.deg2rad(-160));bounds[:,1]=np.minimum(bounds[:,1],np.deg2rad(160))
bounds[3]=np.deg2rad([0.001,59]);bounds[5]=np.deg2rad([-59,-.001])
selected=[]
for target in [[.31825,.0283333333],[.323,.0283333333],[.32775,.0283333333]]:
 idx=int(np.linalg.norm(uv-np.array(target),axis=1).argmin())
 if np.linalg.norm(cloth[uv[:,1]>.303]-cloth[idx],axis=1).min()>=.04:selected.append(idx)
rng=np.random.default_rng(62);results=[]
with np.load(P.parent/'right_basket_outward_long_20hz_0098.npz') as original:
 initial_values=dict(zip(original['joint_names'].tolist(),original['joint_target'][0]))
seed_values=[json.loads((P/'runs/basket_shortstroke_full_0126/online_protocol/candidate_0.json').read_text())[0]['values'],initial_values]+[c['values'] for c in json.loads((P/'basket_parallel_candidates.json').read_text())[:3]]
prior_candidates=json.loads((P/'runs/continuous_lay_0166/online_protocol/candidates.json').read_text())
for idx in selected:
 rng=np.random.default_rng(62)
 goal=cloth[idx]
 patch=cloth[(uv[:,0]>.31)&(uv[:,1]<.04)];_,_,basis=np.linalg.svd(patch-patch.mean(0));normal=basis[-1];normal*=1 if normal[1]<0 else -1
 def residual(q):
  mid,gap,fore,cam,poses=metrics(q);h=poses['gripper_r_base_link'];_,axes=robot.fk(values|dict(zip(arm,q)),True);outside=axes[arm[0]][0][1]-axes[arm[2]][0][1]
  cv=robot.calibrate_loops(values|dict(zip(arm,q))|{'idx71_gripper_r_inner_joint1':-.012,'idx81_gripper_r_outer_joint1':.012})
  cp=robot.fk(cv)
  for name in cp:
   if name.startswith('gripper_r_'):
    cp[name]=cp[name].copy();cp[name][:3,3]+=.010*h[:3,2]
  return np.r_[200*(mid-goal),60*h[1,2],.7*h[2,2],80*max(0,.02-(h[:3,:3]@np.array([.07593,-.009,.021]))[1]),3.0*np.cross(h[:3,1],normal),3*max(0,-outside),collision_res(poses)*5,80*np.maximum(0,.012-hand_distance(poses)),80*np.maximum(0,.012-hand_distance(cp))]
 best=None
 local_seeds=[min(prior_candidates,key=lambda c:np.linalg.norm(np.asarray(c['uv'])-uv[idx]))['values']]+seed_values
 for k in range(2):
  seed=np.array([local_seeds[k][n] for n in arm]) if k<len(local_seeds) else rng.uniform(bounds[:,0]+1e-6,bounds[:,1]-1e-6)
  opt=least_squares(residual,np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0],bounds[:,1]),max_nfev=180)
  score=np.linalg.norm(residual(opt.x))
  if best is None or score<best[0]:best=(score,opt.x)
 mid,gap,fore,cam,poses=metrics(best[1]);h=poses['gripper_r_base_link'];row=dict(uv=uv[idx].tolist(),edge=goal.tolist(),position_error_m=float(np.linalg.norm(mid-goal)),yaw_deg=float(np.rad2deg(np.arctan2(h[1,2],h[0,2]))),pitch_deg=float(np.rad2deg(np.arcsin(h[2,2]))),mouth_rotation=h[:3,:3].tolist(),camera_mount_y_m=float((h[:3,:3]@np.array([.07593,-.009,.021]))[1]),collision_res_max=float(collision_res(poses).max()),hand_basket_clearance_m=float(hand_distance(poses).min()),opposite_panel_clearance_m=float(np.linalg.norm(cloth[uv[:,1]>.303]-goal,axis=1).min()),pinch_point_link4_local_m=pad.tolist(),values=values|dict(zip(arm,best[1].tolist())),score=best[0],pad_face_normal_error_deg=float(np.degrees(np.arccos(np.clip(abs(h[:3,1]@normal),0,1)))));row.update(position_error=row['position_error_m'],mouth_elevation_deg=row['pitch_deg'],camera_mount_outward_m=-row['camera_mount_y_m']);results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='values'}),flush=True)
Path(sys.argv[2]).write_text(json.dumps(sorted(results,key=lambda c:c['score']),indent=2)+'\n')

import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/aether/workspace/agent_bench/experiments/physics_fluid')
from tablecloth_robot_kinematics import Robot
from tablecloth_scene_binding import station_xyyaw
P=Path(__file__).resolve().parent
meta=json.loads(Path('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.json').read_text());robot=Robot(meta,station_xyyaw('8.9'))
c=json.loads((P/'front_iter256_grasp_candidates.json').read_text())[0];values=c['values'];arm=robot.arm_names('R');bounds=np.array([robot.by_name[n]['limits'] for n in arm]);bounds[[3,5]]=np.deg2rad([-59,59]);pad=np.array([.02285,0,0,1.])
camR=Rotation.from_quat([-.6989317357080854,-.6868383037850344,-.14219926877000738,-.1397388322373386]).as_matrix();camaxis=camR@np.array([0,0,-1.])
def metrics(q):
 v=values|dict(zip(arm,q));poses,axes=robot.fk(v,True);h=poses['gripper_r_base_link'];a=(poses['gripper_r_outer_link4']@pad)[:3];b=(poses['gripper_r_inner_link4']@pad)[:3];fore=axes[arm[6]][0]-axes[arm[4]][0];fore/=np.linalg.norm(fore);cam=h[:3,:3]@camaxis
 return (a+b)/2,(a-b)/np.linalg.norm(a-b),fore,cam,poses
shapes={}
for s in meta['nonhand_collision_shapes']:
 if s['link'] in ['arm_r_link3','arm_r_link5','arm_r_link6','arm_r_link7']:
  v=np.array(s['vertices']);shapes.setdefault(s['link'],[]).append(np.concatenate([v,v[np.array(s['triangles'])].mean(1)]))
shapes={n:np.concatenate(v) for n,v in shapes.items()}
boxes=[([-2.439,-1.034,.26],[-2.419,-.716,.45]),([-2.419,-1.034,.26],[-2.061,-1.013,.45]),([-2.419,-.737,.26],[-2.061,-.716,.45])]
from scipy.spatial import cKDTree
body_points=np.concatenate([np.concatenate([np.array(s['vertices']),np.array(s['vertices'])[np.array(s['triangles'])].mean(1)]) for s in meta['nonhand_collision_shapes'] if s['link']=='body_link2'])
def collision_res(poses):
 out=[]
 for n,pts in shapes.items():
  world=pts@poses[n][:3,:3].T+poses[n][:3,3]
  for aa,bb in boxes:
   a=np.array(aa);b=np.array(bb);q=np.abs(world-(a+b)/2)-(b-a)/2;d=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0);out.append(max(0,.015-d.min())*3)
 body=body_points@poses['body_link2'][:3,:3].T+poses['body_link2'][:3,3];tree=cKDTree(body)
 for n,pts in shapes.items():
  world=pts@poses[n][:3,:3].T+poses[n][:3,3];out.append(max(0,.012-tree.query(world)[0].min())*3)
 return np.asarray(out)
rng=np.random.default_rng(50);rows=[];goal=np.array(c['edge']);normal=np.array(c['normal'])
# Camera outward means away from the torso toward the robot right side (-world Y).
for mode in ['position_horizontal_normal_camera']:
 best=None
 for k in range(20):
  seed=np.array([values[n] for n in arm]) if k==0 else rng.uniform(bounds[:,0],bounds[:,1])
  def res(q):
   mid,gap,fore,cam,poses=metrics(q)
   parts=[(mid-goal)*3,np.array([fore[2]*.07]),collision_res(poses)]
   if 'normal' in mode:parts.append(.12*(gap-normal))
   if 'camera' in mode:parts.append(np.array([max(0,cam[1]+.3)*.12]))
   return np.concatenate(parts)
  opt=least_squares(res,seed,bounds=(bounds[:,0],bounds[:,1]),max_nfev=160,gtol=1e-8)
  score=np.linalg.norm(res(opt.x))
  print('candidate',k,score,flush=True)
  if best is None or score<best[0]:best=(score,opt.x)
 mid,gap,fore,cam,poses=metrics(best[1]);row=dict(mode=mode,residual=best[0],position_error_m=float(np.linalg.norm(mid-goal)),forearm_elevation_deg=float(np.rad2deg(np.arcsin(fore[2]))),gap_normal_error_deg=float(np.rad2deg(np.arccos(np.clip(gap@normal,-1,1)))),camera_world_axis=cam.tolist(),values=values|dict(zip(arm,best[1].tolist())),head_links={n:p[:3,3].tolist() for n,p in poses.items() if 'head' in n},q_deg=np.rad2deg(best[1]).tolist());rows.append(row);print(json.dumps(row),flush=True)
(P/'horizontal_protocol_collision_candidates.json').write_text(json.dumps(rows,indent=2)+'\n')

import numpy as np,json
from pathlib import Path
from scipy.spatial import ConvexHull
DATA=Path('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.npz')
def make_hand_clearance():
 from profile_distance import make_profile_distance
 distance=make_profile_distance();shapes={}
 with np.load(DATA) as data:
  for key in data.files:
   if not key.startswith('points_gripper_r_'):continue
   points=np.unique(data[key],axis=0);h=ConvexHull(points);points=np.concatenate([points[h.vertices],points[h.simplices].mean(1)]);_,idx=np.unique(np.round(points/.003).astype(int),axis=0,return_index=True);shapes[key[7:]]=points[idx]
 def distances(poses):
  result=[]
  for n,pts in shapes.items():
   q=pts@poses[n][:3,:3].T+poses[n][:3,3];result.append(distance(q))
  return np.array(result)
 return distances

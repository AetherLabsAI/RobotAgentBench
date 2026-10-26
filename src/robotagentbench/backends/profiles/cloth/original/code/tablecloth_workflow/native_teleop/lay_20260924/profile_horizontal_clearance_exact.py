import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import json
def make_clearance(meta):
    shapes={}
    for s in meta['nonhand_collision_shapes']:
     if s['link'] in ['arm_r_link3','arm_r_link5','arm_r_link6','arm_r_link7']:
      v=np.array(s['vertices']);shapes.setdefault(s['link'],[]).append(np.concatenate([v,v[np.array(s['triangles'])].mean(1)]))
    shapes={n:np.concatenate(v) for n,v in shapes.items()}
    from profile_distance import make_profile_distance
    point_distance=make_profile_distance()
    from scipy.spatial import cKDTree
    body_points=np.concatenate([np.concatenate([np.array(s['vertices']),np.array(s['vertices'])[np.array(s['triangles'])].mean(1)]) for s in meta['nonhand_collision_shapes'] if s['link']=='body_link2'])
    def collision_res(poses):
     out=[]
     for n,pts in shapes.items():
      world=pts@poses[n][:3,:3].T+poses[n][:3,3]
      out.append(max(0,.015-point_distance(world))*3)
     body=body_points@poses['body_link2'][:3,:3].T+poses['body_link2'][:3,3];tree=cKDTree(body)
     for n,pts in shapes.items():
      world=pts@poses[n][:3,:3].T+poses[n][:3,3];out.append(max(0,.012-tree.query(world)[0].min())*3)
     return np.asarray(out)
    return collision_res

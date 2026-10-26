"""Plan a broad two-layer contact using actual native jaw geometry, never attachment.
The cloth need not already be compressed before open jaws approach it.
Forty millimetres of table-edge clearance and physical post-close checks remain required.
"""
import numpy as np
from scipy.spatial import cKDTree

def select_pair(cloth,uv,cloth_ids,guard,bodies,body_names,pad_mid,table_min_x=-2.35,min_nodes_per_layer=10):
 old=np.flatnonzero(cloth_ids==0);lower=old[uv[old,1]<.235];upper=old[uv[old,1]>.275];distance,j=cKDTree(cloth[upper,:2]).query(cloth[lower,:2]);pairs=np.c_[lower,upper[j]];centers=cloth[pairs].mean(1)
 valid=(distance<.004)&np.all(cloth[pairs,0]<=table_min_x-.04,axis=1)&np.all(cloth[pairs,2]>.32,axis=1)&np.all(cloth[pairs,2]<.50,axis=1)&(abs(centers[:,1]+.57)<.16)
 candidates=np.flatnonzero(valid);candidates=candidates[np.argsort(abs(centers[candidates,1]-pad_mid[1]))];tested=[];best=None;qualified=[]
 for ci in candidates[:80]:
  pair=pairs[ci];center=centers[ci];shift=center-pad_mid;checks=[guard.check(cloth[i]-shift,bodies,body_names)for i in pair]
  if not all(c['geometry_allows_lift']and min(c['signed_plane_distances_m'].values())>=.002 for c in checks):continue
  indices=[];counts=[]
  for layer in [lower,upper]:
   local=layer[np.linalg.norm(cloth[layer]-center,axis=1)<.04];inside=[]
   for i in local:
    c=guard.check(cloth[i]-shift,bodies,body_names)
    if c['geometry_allows_lift']and min(c['signed_plane_distances_m'].values())>=.001:inside.append(int(i))
   indices.append(inside);counts.append(len(inside))
  row=dict(pair_indices=pair.tolist(),pair_uv=uv[pair].tolist(),pair_xyz=cloth[pair].tolist(),target_pad_mid=center.tolist(),nominal_translation=shift.tolist(),minimum_layer_overhang_m=float((table_min_x-cloth[pair,0]).min()),layer_vertical_separation_m=float(abs(cloth[pair[0],2]-cloth[pair[1],2])),native_open_gap_m=[c['cooked_gap_m']for c in checks],nodes_inside_open_mouth_by_layer=counts,inside_node_ids=indices,force_confirmed=False)
  tested.append({k:v for k,v in row.items()if k!='inside_node_ids'})
  if min(counts)<min_nodes_per_layer:continue
  score=min(counts)-20*abs(float(center[1]-pad_mid[1]));qualified.append((score,row))
  if best is None or score>best[0]:best=(score,row)
 return dict(ranked_candidates=[row for score,row in sorted(qualified,key=lambda x:-x[0])],selected=best[1]if best else None,projected_pairs_both_over40mm=len(candidates),tested_candidates=tested,minimum_nodes_per_layer=min_nodes_per_layer,overhang_m=.04,actual_cloth_positions_used=True,native_cooked_pad_geometry_used=True,physical_grasp_unverified=True,object_state_writes=0)

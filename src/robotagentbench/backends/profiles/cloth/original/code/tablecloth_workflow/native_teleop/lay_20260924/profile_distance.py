from pathlib import Path
import json,numpy as np
from scipy.spatial import ConvexHull
def make_profile_distance():
 rs=json.loads((Path(__file__).parent/'basket_profile_split_colliders.json').read_text())['colliders'];bounds=np.array([r['world_bounds_m']for r in rs]);parts=[]
 for r in rs:
  if r['name']=='floor':parts.append(None);continue
  axis=int(r['name'].split('_')[1]);v=np.array(r['world_vertices_m']);q=np.unique(v[:,[axis,2]],axis=0);h=ConvexHull(q);a=q[h.vertices];b=np.roll(a,-1,axis=0);parts.append((axis,a,b-a,((b-a)**2).sum(1),h.equations))
 def distance(q):
  lo=q.min(0)-.02;hi=q.max(0)+.02;near=np.flatnonzero(np.all((bounds[:,1]>=lo)&(bounds[:,0]<=hi),axis=1));best=.02
  for j in near:
   lower,upper=bounds[j];v=q[np.all((q>=lower-.02)&(q<=upper+.02),axis=1)]
   if not len(v):continue
   part=parts[j]
   if part is None:
    d=abs(v-(lower+upper)/2)-(upper-lower)/2;s=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0)
   else:
    axis,a,edge,lsq,eq=part;delta=v[:,None,[axis,2]]-a[None,:,:];t=np.clip((delta*edge[None,:,:]).sum(2)/lsq[None,:],0,1);d2=np.linalg.norm(delta-t[:,:,None]*edge[None,:,:],axis=2).min(1);inside=(v[:,[axis,2]]@eq[:,:2].T+eq[:,2]).max(1)<=0;d2=np.where(inside,-d2,d2);dl=np.maximum(lower[1-axis]-v[:,1-axis],v[:,1-axis]-upper[1-axis]);s=np.hypot(np.maximum(d2,0),np.maximum(dl,0))+np.minimum(np.maximum(d2,dl),0)
   best=min(best,float(s.min()))
  return best
 return distance

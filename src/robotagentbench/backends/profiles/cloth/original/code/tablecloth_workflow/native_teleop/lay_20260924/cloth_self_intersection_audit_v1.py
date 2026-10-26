"""Read-only native cloth self-intersection audit; no state mutation."""
import numpy as np
from scipy.spatial import cKDTree
def audit_self_intersections(points,faces):
 q=np.asarray(points,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
 tri=q[f];cent=tri.mean(1);radius=np.linalg.norm(tri-cent[:,None],axis=2).max();pairs=cKDTree(cent).query_pairs(float(2*radius+1e-5),output_type='ndarray');a,b=pairs.T;mask=~(f[a,:,None]==f[b,None,:]).any((1,2));pairs=pairs[mask];a,b=pairs.T;mask=np.all(np.maximum(tri[a].min(1),tri[b].min(1))<=np.minimum(tri[a].max(1),tri[b].max(1))+1e-7,axis=1);pairs=pairs[mask];hits=np.zeros(len(pairs),bool);depth=np.zeros(len(pairs))
 for flip in [False,True]:
  a,b=pairs[:,int(flip)],pairs[:,int(not flip)];t=tri[b];e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
  for k in range(3):
   o=tri[a,k];d=tri[a,(k+1)%3]-o;h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h);nz=abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=nz);s=o-t[:,0];u=inv*np.einsum('ij,ij->i',s,h);v=inv*np.einsum('ij,ij->i',d,np.cross(s,e1));tt=inv*np.einsum('ij,ij->i',e2,np.cross(s,e1));hit=nz&(u>=1e-5)&(v>=1e-5)&(u+v<=1-1e-5)&(tt>1e-5)&(tt<1-1e-5)
   normal=np.cross(e1,e2);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12);sd0=np.einsum('ij,ij->i',o-t[:,0],normal);sd1=np.einsum('ij,ij->i',o+d-t[:,0],normal);dep=np.minimum(abs(sd0),abs(sd1));depth=np.maximum(depth,np.where(hit,dep,0));hits|=hit
 found=pairs[hits];row=dict(strict_nonadjacent_triangle_intersection_pairs=len(found),candidate_pairs=len(pairs),max_edge_piercing_depth=float(depth.max()) if len(depth) else 0.,pairs_over_50micron=int((depth>.00005).sum()),pairs=found.tolist(),coplanar_overlap_not_tested=True,method='readonly segment triangle intersection; shared-vertex pairs excluded')
 return row

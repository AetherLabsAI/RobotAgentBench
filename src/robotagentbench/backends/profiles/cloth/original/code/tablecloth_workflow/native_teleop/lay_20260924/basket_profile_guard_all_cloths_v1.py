import numpy as np
class BasketWallGuard:
 def __init__(self,records,cloth_ids,faces):
  self.ids=np.arange(len(cloth_ids));f=faces[np.all(np.isin(faces,self.ids),axis=1)];self.edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);self.shapes=[(r['name'],np.array(r['world_bounds_m']),np.array(r['planes']))for r in records]
 def check(self,q,tolerance=.0005):
  points=q[self.ids];aa=q[self.edges[:,0]];bb=q[self.edges[:,1]];elo=np.minimum(aa,bb);ehi=np.maximum(aa,bb);rows=[]
  qlo=points.min(0);qhi=points.max(0)
  for name,(lo,hi),planes in self.shapes:
   # Exact broad phase: segments between vertices stay inside their total AABB.
   if np.any(qhi<lo) or np.any(qlo>hi):
    rows.append(dict(name=name,penetrating_vertices=0,max_depth_m=0.,crossing_edges=0));continue
   A=planes[:,:3];b=planes[:,3];vm=np.all((points>=lo)&(points<=hi),axis=1);d=-(points[vm]@A.T+b).max(1)if vm.any()else np.array([0.]);em=np.all((ehi>=lo)&(elo<=hi),axis=1);a=aa[em];v=bb[em]-a;av=v@A.T;dist=a@A.T+b+tolerance;safe=np.where(abs(av)>1e-12,av,1e-12);t=-dist/safe;en=np.maximum(0,np.max(np.where(av< -1e-12,t,-np.inf),axis=1));ex=np.minimum(1,np.min(np.where(av>1e-12,t,np.inf),axis=1));hit=(ex>en+1e-6)&~np.any((abs(av)<=1e-12)&(dist>0),axis=1)
   rows.append(dict(name=name,penetrating_vertices=int((d>tolerance).sum()),max_depth_m=float(max(0,d.max())),crossing_edges=int(hit.sum())))
  return dict(tolerance_m=tolerance,valid=not any(r['penetrating_vertices']or r['crossing_edges']for r in rows),boxes=rows)

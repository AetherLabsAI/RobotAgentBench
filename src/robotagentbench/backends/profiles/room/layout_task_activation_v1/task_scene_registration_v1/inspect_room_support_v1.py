import json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
root=Path(__file__).resolve().parent
s=Usd.Stage.Open(str(root/'room_demo_v2/room.usda'));xf=UsdGeom.XformCache()
for path in ['/World/RoomEpisode/Sofa_baseball_cap','/World/RoomEpisode/Sofa_power_bank','/World/Assets/Sofa']:
 for p in Usd.PrimRange(s.GetPrimAtPath(path)):
  if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI):
   print(str(p.GetPath()),{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physxCollision:'))})
triangles=[]
for p in Usd.PrimRange(s.GetPrimAtPath('/World/Assets/Sofa')):
 if not p.IsA(UsdGeom.Mesh):continue
 mesh=UsdGeom.Mesh(p);v=np.array(mesh.GetPointsAttr().Get());v=(np.c_[v,np.ones(len(v))]@np.array(xf.GetLocalToWorldTransform(p)))[:,:3]
 indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());o=0
 for n in mesh.GetFaceVertexCountsAttr().Get():
  f=indices[o:o+n];o+=n
  for j in range(1,n-1):triangles.append(v[[f[0],f[j],f[j+1]]])
t=np.array(triangles)
for x,y in [(-.1295,-.5827),(-.176,-1.0167),(-.2198,-1.3956),(-.1899,-1.6963),(-.133,-.1655),(-.016,-1.4488)]:
 a,b,c=t[:,0],t[:,1],t[:,2];v0=b-a;v1=c-a;v2=np.array([x,y])-a[:,:2]
 den=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0];valid=abs(den)>1e-12
 u=np.zeros(len(t));w=u.copy();u[valid]=(v2[valid,0]*v1[valid,1]-v2[valid,1]*v1[valid,0])/den[valid];w[valid]=(v0[valid,0]*v2[valid,1]-v0[valid,1]*v2[valid,0])/den[valid]
 z=a[:,2]+u*v0[:,2]+w*v1[:,2];hit=valid&(u>=0)&(w>=0)&(u+w<=1)&(z<.65)
 print('SUPPORT',x,y,sorted(set(np.round(z[hit],5).tolist())))

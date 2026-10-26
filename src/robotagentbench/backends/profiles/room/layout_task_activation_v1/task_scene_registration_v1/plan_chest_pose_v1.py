"""User-requested folded empty-arm initial pose; kinematics only until native test.
Only arm targets change; no joint limit/drive/collision or torso changes.
"""
import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from pxr import Usd,UsdGeom,UsdPhysics
ROOT=Path(__file__).resolve().parent
src=ROOT/'coffee_initialized_rooms_v1/layout04/room.usda'
s=Usd.Stage.Open(str(src));cache=UsdGeom.XformCache()
def world(p):return np.asarray(cache.GetLocalToWorldTransform(s.GetPrimAtPath(p))).T
def frame(p,i):
 m=np.eye(4);q=p.GetAttribute('physics:localRot'+str(i)).Get()
 m[:3,:3]=R.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
 m[:3,3]=p.GetAttribute('physics:localPos'+str(i)).Get();return m
parents={}
for p in Usd.PrimRange(s.GetPrimAtPath('/World/Robot')):
 if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
 a,b=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
 if len(a)==len(b)==1:
  if str(b[0]) in parents:continue  # gripper loops are not in the arm-to-centre chain
  parents[str(b[0])]=(p,str(a[0]))
report=dict(status='kinematic_candidate_not_collision_validated',targets_rad={},arms=[],source=str(src))
for side,sign,height in [('l',1,1.25),('r',-1,1.08)]:
 hand='/World/Robot/gripper_'+side+'_center_link';body=hand;chain=[]
 while body in parents:
  p,parent=parents[body];fixed=p.IsA(UsdPhysics.FixedJoint);angular=p.IsA(UsdPhysics.RevoluteJoint)
  axis=None if fixed else 'XYZ'.index(p.GetAttribute('physics:axis').Get())
  a,b=frame(p,0),frame(p,1);rel=np.linalg.inv(world(parent)@a)@world(body)@b
  q=0 if fixed else R.from_matrix(rel[:3,:3]).as_rotvec()[axis] if angular else rel[axis,3]
  limits=None if fixed else np.array([p.GetAttribute('physics:'+v+'Limit').Get() for v in ('lower','upper')],float)
  if angular:limits=np.radians(limits)
  chain.append(dict(name=p.GetName(),a=a,bi=np.linalg.inv(b),axis=axis,angular=angular,q=q,limits=limits));body=parent
 assert body=='/World/Robot/base_link';chain.reverse()
 moving=[c for c in chain if c['axis'] is not None];frozen=np.array([c['q'] for c in moving]);limits=np.array([c['limits'] for c in moving])
 def fk(q):
  t=np.eye(4);n=0;points=[]
  for c in chain:
   t=t@c['a']
   if '_arm_'+side+'_joint' in c['name']:points.append(t[:3,3].copy())
   if c['axis'] is not None:
    m=np.eye(4);axis=c['axis'];v=q[n];n+=1
    if c['angular']:
     vec=np.zeros(3);vec[axis]=v;m[:3,:3]=R.from_rotvec(vec).as_matrix()
    else:m[axis,3]=v
    t=t@m
   t=t@c['bi']
  return t,np.array(points)
 assert np.max(np.abs(world('/World/Robot/base_link')@fk(frozen)[0]-world(hand)))<1e-5
 target=np.array([.49,-sign*.065,height])
 elbow=np.array([.34,sign*.29,1.00])
 def residual(v):
  t,pts=fk(np.r_[frozen[:2],v])
  return np.r_[30*(t[:3,3]-target),5*(pts[3]-elbow),20*np.maximum(np.abs(pts[:,1])-.33,0),.02*(v-frozen[2:])]
 rng=np.random.default_rng(512)
 candidates=[]
 for seed in [frozen[2:],*rng.uniform(limits[2:,0],limits[2:,1],size=(12,7))]:
  r=least_squares(residual,np.clip(seed,limits[2:,0],limits[2:,1]),bounds=(limits[2:,0],limits[2:,1]),max_nfev=500)
  candidates.append(r)
 best=min(candidates,key=lambda x:np.linalg.norm(x.fun));q=np.r_[frozen[:2],best.x];t,pts=fk(q)
 assert np.linalg.norm(t[:3,3]-target)<.015
 report['targets_rad'].update({c['name']:float(v) for c,v in zip(moving[2:],best.x)})
 report['arms'].append(dict(side=side,hand_base_m=t[:3,3].tolist(),elbow_base_m=pts[3].tolist(),pivots=pts.tolist(),target=target.tolist()))
out=ROOT/'chest_pose_candidate_v1.json'
assert not out.exists(),'Do not overwrite a pose candidate'
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

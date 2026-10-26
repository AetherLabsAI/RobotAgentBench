"""Plan free right-hand approach from actual native joints; objects are read-only."""
from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation,Slerp
from scipy.spatial import ConvexHull
from native_body1_collision_guard_v1 import LayPlanner
class RoutePlanner(LayPlanner):
 def __init__(self):
  super().__init__();self.pieces=[(x['name'],np.array(x['world_bounds_m']),np.unique(np.array(x['planes']),axis=0))for x in json.loads((Path(__file__).parent/'basket_profile_split_colliders.json').read_text())['colliders']]
 def validate(self,v):
  poses=self.robot.fk(v)
  if self.collision(poses,'R').max()>.004:return 'native/self/torso clearance'
  for n,pts in self.shapes.items():
   if not n.startswith(('arm_r_','gripper_r_')):continue
   h=poses[n];w=pts@h[:3,:3].T+h[:3,3]
   for name,bb,eq in self.pieces:
    if np.any(w.max(0)<=bb[0])or np.any(w.min(0)>=bb[1]):continue
    if np.any((w@eq[:,:3].T+eq[:,3]).max(1)<-.001):return n+' vs '+name
  return None
 def plan(self,start,end,frames=480):
  names=self.robot.arm_names('R');a=self.robot.fk(start)['gripper_r_base_link'];b=self.robot.fk(end)['gripper_r_base_link'];turn=Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]));failures=[]
  for bulge in [[0,0,0],[0,0,.06],[.04,0,.12],[-.04,0,.18]]:
   rows=[];prior=start;failed=None
   for k in range(1,41):
    u=k/40;u=u*u*(3-2*u);pos=a[:3,3]*(1-u)+b[:3,3]*u+np.sin(np.pi*u)*np.asarray(bulge);rot=turn(u).as_matrix();q,e=self.robot.solve(prior,'R',pos,np.zeros(3),rot,max_steps=400);h=self.robot.fk(q)['gripper_r_base_link'];ang=np.rad2deg(Rotation.from_matrix(rot.T@h[:3,:3]).magnitude())
    if e>.002 or ang>2:failed=('IK',k,e,ang);break
    N=frames//40
    for j in range(1,N+1):
     t=j/N;t=t*t*(3-2*t);v=prior|{n:prior[n]+t*(q[n]-prior[n])for n in names};bad=self.validate(v)
     if bad:failed=('collision',k,j,bad);break
     prev=rows[-1]if rows else start
     if max(abs(v[n]-prev[n])*20 for n in names)>1.2:failed=('rate',k,j);break
     rows.append(v)
    if failed:break
    prior=q
   if not failed:
    self.audit=dict(frames=len(rows),bulge_m=bulge,endpoint_position_error_m=float(e),endpoint_rotation_error_deg=float(ang),failed_candidates=failures,object_state_writes=0);return rows
   failures.append(failed)
  raise RuntimeError(('No fully checked right approach route',failures))
if __name__=='__main__':
 p=Path(__file__).parent;pl=RoutePlanner();z=np.load(p/'runs/same_physics_removal_0167/action.npz');v=dict(zip(z['joint_names'],z['joint_target'][2284]));a=np.load(p.parent/'right_high_side_release_0130.npz');end=dict(zip(a['joint_names'],a['joint_target'][2764]));row=next(json.loads(s)for s in(p/'same_physics_removal_0167.log').read_text().splitlines()if s.startswith('{"event": "right_joint_pregrasp_retarget"'));end.update(row['endpoint_arm']);rows=pl.plan(v,end);(p/'removal_pregrasp_route_preflight.json').write_text(json.dumps(pl.audit,indent=2));print(pl.audit,flush=True)

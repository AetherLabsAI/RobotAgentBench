"""Author a new post-brew fixture proposal and re-solve its approach/extraction.

No physics writes here. Failed proposals keep their sampled machine pose.
"""
import hashlib,json,math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from preflight import create_model,configure,interpolate

def retarget(plan,rows,source,out):
 seed=int(hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:8],16)
 rng=np.random.default_rng(seed);offset=[float(rng.uniform(-.08,.08)),float(rng.uniform(-.08,.08)),0.];yaw=float(rng.uniform(-10,10))
 from scene_pose import MACHINE_ORIGIN
 assert np.allclose(plan.get('machine_offset_m',[0,0,0]),0) and plan.get('machine_yaw_deg',0)==0
 pivot=MACHINE_ORIGIN.copy()
 rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix();D=np.eye(4);D[:3,:3]=rot;D[:3,3]=pivot+offset-rot@pivot
 def pos(v):return (D@np.r_[v,1])[:3].tolist()
 g=plan['geometry'];g['cup_origin']=pos(g['cup_origin']);g['cup_rotation']=(rot@np.array(g['cup_rotation'])).tolist();g['cup_target']=pos(g['cup_target']);g['outlet']=pos(g['outlet']);g['machine_front']=(rot@np.array(g['machine_front'])).tolist();g['cup_target_rotation']=(rot@np.array(g['cup_target_rotation'])).tolist()
 plan.update(machine_offset_m=offset,machine_yaw_deg=yaw,retrieval_pose_randomization=dict(seed=seed,machine_offset_m=offset,machine_yaw_deg=yaw,source=str(source),scope='New explicitly authored post-brew fixture before physics; cup follows machine; not a physical relocation.'))
 (out/'pose_proposal.json').write_text(json.dumps(plan['retrieval_pose_randomization'],indent=2))
 m,c=create_model(plan);c.goal=np.array(g['cup_target']);grasp=[x for x in rows if x['phase']=='regrasp'][-1];c.grasp_tcp=D@m.tcp(grasp['q'])
 current=dict(rows[0],q=plan['initial'],opening=.18);t=m.tcp(current['q']);current.update(tcp=t[:3,3].tolist(),rotation=t[:3,:3].tolist());new=[dict(current,phase='removal_ready')]
 # Keep every closure/lift waypoint; compress other paths to 5-command anchors.
 anchors=[row for i,row in enumerate(rows) if i==len(rows)-1 or i%5==0 or row['phase'] in ['regrasp','pickup_lift','pickup_hold'] or rows[i+1]['phase']!=row['phase']]
 for i,row in enumerate(anchors):
  tcp=D@m.tcp(row['q']);target=tcp[:3,3];targetR=tcp[:3,:3];q=np.array(current['q'])
  def residual(v):
   t=m.tcp(v);configure(m,c,dict(row,q=list(v)));slack=c.check(row['phase'],details=False)[0]
   return np.r_[(t[:3,3]-target)*15,Rotation.from_matrix(targetR@t[:3,:3].T).as_rotvec(),(v-q)*.005,np.minimum(slack-.0003,0)*160]
  a=least_squares(residual,q,bounds=(m.lo,m.hi),max_nfev=140);t=m.tcp(a.x);pe=np.linalg.norm(t[:3,3]-target);re=np.linalg.norm(Rotation.from_matrix(targetR@t[:3,:3].T).as_rotvec());configure(m,c,dict(row,q=list(a.x)));slack,v=c.check(row['phase'])
  if pe>.0015 or re>.03 or v:
   report=dict(stage='random_machine_retarget',anchor=i,phase=row['phase'],position_error=float(pe),rotation_error=float(re),violations=v);(out/'planning_failure.json').write_text(json.dumps(report,indent=2));raise RuntimeError(report)
  nr=dict(row,q=a.x.tolist(),tcp=t[:3,3].tolist(),rotation=t[:3,:3].tolist());n=max(1 if row['phase']=='regrasp' else 5,math.ceil(max(abs(a.x-q))/.012))
  for u in np.linspace(0,1,n+1)[1:]:
   rr=interpolate(current,nr,u);tt=m.tcp(rr['q']);rr.update(tcp=tt[:3,3].tolist(),rotation=tt[:3,:3].tolist());new.append(rr)
  current=nr;print('RANDOM_MACHINE',i,row['phase'],flush=True)
 plan['steps']=new
 return new

"""Offline/action-planning candidate: constrain pad normal, leave in-plane roll free.
Produces joint targets only; native limits and existing collision gates are unchanged.
"""
import numpy as np

def solve_normal(pl,v,side,target,normal,max_steps=240,allow_flip=True):
 names=pl.robot.arm_names(side);mid,h,_,_=pl.geom(v,side);local=(mid-h[:3,3])@h[:3,:3];normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal);nom=pl.robot.nominal_values();seeds=[v,v|{n:nom[n]for n in names}]
 for delta in [-.6,.6]:seeds.append(v|{names[2]:float(np.clip(v[names[2]]+delta,*pl.robot.by_name[names[2]]['limits']))})
 options=[]
 for sign in ([1,-1]if allow_flip else[1]):
  n=sign*normal
  for seed in seeds:
   q=seed.copy()
   for k in range(max_steps):
    poses,axes=pl.robot.fk(q,True);t=poses['gripper_'+side.lower()+'_base_link'];point=t[:3,:3]@local+t[:3,3];actual=t[:3,1];error=np.r_[target-point,.12*(n-actual)];jac=np.vstack([np.stack([np.cross(axes[j][1],point-axes[j][0])for j in names],axis=1),.12*np.stack([np.cross(axes[j][1],actual)for j in names],axis=1)])
    if np.linalg.norm(error)<.0002:break
    dq=jac.T@np.linalg.solve(jac@jac.T+.008**2*np.eye(6),error);dq*=min(1.,.1/max(abs(dq).max(),1e-12))
    for j,d in zip(names,dq):lo,hi=pl.robot.by_name[j]['limits'];q[j]=float(np.clip(q[j]+d,lo+1e-5,hi-1e-5))
   gm,gh,poses,out=pl.geom(q,side);position=float(np.linalg.norm(gm-target));angle=float(np.rad2deg(np.arccos(np.clip(gh[:3,1]@n,-1,1))));collision=float(pl.collision(poses,side).max());bends=np.rad2deg(abs(np.array([q[names[3]],q[names[5]]])));score=float(np.maximum(0,bends-60).sum()+np.linalg.norm([q[j]-v[j]for j in names]))
   if position<.002 and angle<2 and collision<=.004 and out>=.005:options.append((score,q,dict(position_error_m=position,normal_angle_deg=angle,collision_residual_m=collision,outward_m=out,bends=bends.tolist(),normal=n.tolist())))
 if not options:raise RuntimeError('No native-clear pad-normal IK for '+side+' at '+str(list(target)))
 _,q,info=min(options,key=lambda x:x[0]);return q,info

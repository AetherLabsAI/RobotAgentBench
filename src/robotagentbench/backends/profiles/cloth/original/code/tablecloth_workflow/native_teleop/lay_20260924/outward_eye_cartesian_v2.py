"""Smooth outward/upward right-arm arc; contact and branch guards unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
def plan(pl,v,goal):
 m0,h0,_,_=pl.geom(v,'R');m1,h1,_,_=pl.geom(goal,'R');turn=Slerp([0,1],Rotation.from_matrix(np.array([h0[:3,:3],h1[:3,:3]])));cur=v;rows=[];peak=0.;audit=[];names=pl.robot.arm_names('R')
 for k in range(1,101):
  t=k/100;t=t*t*(3-2*t);mid=m0+t*(m1-m0)+np.sin(np.pi*t)*np.array([.06,0,.05]);q=pl.solve_full(cur,'R',mid,turn(t).as_matrix());part=pl.interpolate(cur,q,'R',2,'R_continuous_outward_arc_eye_presentation');rows+=part;cur=part[-1][0];audit.append(pl.last_cartesian_segment)
  for val,_ in part:peak=max(peak,max(float(abs(np.rad2deg(val[names[j]])))for j in [3,5]))
 if peak>60:raise RuntimeError(('Outward arc exceeds60deg elbow/wrist preference',peak))
 return rows,dict(x_bulge_m=.06,z_bulge_m=.05,max_commanded_elbow_wrist_deg=peak,max_cartesian_segment_deviation_m=max(a['max_position_deviation_m']for a in audit),max_cartesian_segment_rotation_deg=max(a['max_rotation_deviation_deg']for a in audit),physical_replay_required=True)

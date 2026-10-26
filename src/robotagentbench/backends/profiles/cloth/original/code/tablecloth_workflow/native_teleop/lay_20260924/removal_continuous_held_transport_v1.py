"""Native contact-only closed transport and gravity release; no object state writes."""
import numpy as np
from removal_closed_outward_route_v1 import ClosedRoutePlanner
from downward_release_endpoint_v3 import find_endpoints
from bounded_full_pose_guard_v1 import LayPlanner

def plan_transport(start,held_frames=340,open_frames=30,withdraw_frames=100):
 route_planner=ClosedRoutePlanner();ik=LayPlanner();candidates,rejected=find_endpoints(route_planner,start,max_results=6);errors=[]
 for candidate in candidates:
  end=candidate['command']
  if not .25<=end['idx64_arm_r_joint4']<=1.75:continue
  try:
   held=route_planner.plan(start,end,held_frames);mids=np.array([route_planner.geom(q,'R')[0]for q in held]);assert mids[:,2].min()>.42,'held hand lower than safe measured corridor';opened=[];base=held[-1];initial=abs(base['idx81_gripper_r_outer_joint1'])
   for k in range(1,open_frames+1):
    t=k/open_frames;t=t*t*(3-2*t);opening=initial+(.75-initial)*t;q=ik.robot.calibrate_loops(base|{'idx71_gripper_r_inner_joint1':-opening,'idx81_gripper_r_outer_joint1':opening});assert route_planner.validate(q)is None,('open',k,route_planner.validate(q));opened.append(q)
   v=opened[-1];mid,h,_,_=ik.geom(v,'R');delta=-h[:3,2]*.14;assert delta[2]>.13;withdraw=[]
   for k in range(1,withdraw_frames+1):
    t=k/withdraw_frames;t=t*t*(3-2*t);v=ik.solve_full(v,'R',mid+t*delta,h[:3,:3]);assert route_planner.validate(v)is None,('withdraw',k,route_planner.validate(v));withdraw.append(v)
   allrows=held+opened+withdraw;names=ik.robot.arm_names('R');rate=max(max(abs(q[n]-prev[n])*20 for n in names)for prev,q in zip([start]+allrows[:-1],allrows));assert rate<=1.2,('native arm rate',rate)
   return dict(held=held,opened=opened,withdraw=withdraw,audit=dict(endpoint={k:v for k,v in candidate.items()if k!='command'},route=route_planner.audit,maximum_held_elbow_deg=float(max(q['idx64_arm_r_joint4']for q in held)*180/np.pi),maximum_arm_rate_rad_s=float(rate),held_mid_bounds=[mids.min(0).tolist(),mids.max(0).tolist()],rejected_paths=errors,full_lift_transport_release_withdraw_checked=True,object_state_writes=0,physics_state_restore=False,physics_unchanged=True))
  except (RuntimeError,AssertionError)as exc:errors.append(dict(target=candidate['target'],yaw=candidate['yaw'],error=str(exc)))
 raise RuntimeError(('No fully checked closed transport and release path',errors))

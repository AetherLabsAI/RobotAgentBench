"""Retimes native joint commands; all original collision/rate gates still apply."""
from live_repair_actions_v1 import Repair as Base
import numpy as np,json
class Repair(Base):
 def submit(self,index,observation,actions):
  cols=self.planner.robot.arm_names('L')+self.planner.robot.arm_names('R');out=[];previous=observation['command'];audit=[];start=0
  while start<len(actions):
   end=start+1
   while end<len(actions) and actions[end][1]==actions[start][1]:end+=1
   segment=actions[start:end];phase=segment[0][1];same=all(max(abs(q[n]-segment[0][0][n])for n in q)<1e-9 for q,_ in segment)
   if same:
    kept=segment[:min(len(segment),20)]
    # Preserve the explicit one-second closure-stability measurement interval.
   elif phase.startswith('LR_'):
    kept=segment # Preserve previously verified dual-held material transport speed.
   else:
    kept=[];j=0
    while j<len(segment):
     candidate=min(j+1,len(segment)-1)
     if max(abs(segment[candidate][0][n]-previous[n])for n in cols)>.7/20: candidate=j
     kept.append(segment[candidate]);previous=segment[candidate][0];j=candidate+1
   if kept:previous=kept[-1][0];out+=kept
   audit.append(dict(phase=phase,before_frames=len(segment),after_frames=len(kept),stationary=same));start=end
  (self.ep/f'retiming_{index:03d}.json').write_text(json.dumps(dict(phases=audit,original_frames=len(actions),executed_frames=len(out),record_hz=20,physics_hz=60,object_state_writes=0,rate_gate_rad_s=.7),indent=2));super().submit(index,observation,out)

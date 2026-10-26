"""Native open-pad roll removes the lower horizontal support before axial withdrawal."""
from bimanual_lay_plan_v29 import LayPlanner as BasePlanner, DATA
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(BasePlanner):
 def tip_open_pads(self,v,frames=120):
  starts={s:self.geom(v,s)[:2]for s in ['L','R']};current=v;rows=[]
  for k in range(1,21):
   a=k/20;a=a*a*(3-2*a);goal=current
   for side in ['L','R']:
    mid,h=starts[side];rotation=h[:3,:3]@Rotation.from_euler('z',(90 if side=='L'else -90)*a,degrees=True).as_matrix();goal=self.solve_full(goal,side,mid,rotation)
   names=[n for n in goal if '_arm_'in n];delta=max(abs(goal[n]-current[n])for n in names);count=max(frames//20,int(np.ceil(delta*1.5/.7*20)))
   for j in range(1,count+1):
    a=j/count;a=a*a*(3-2*a);q=current|{n:current[n]+a*(goal[n]-current[n])for n in names}
    if j==count or j%3==0:
     if max(self.collision(self.robot.fk(q),s).max()for s in ['L','R'])>.004:raise RuntimeError(('Open-pad tipping native collision',k,j))
    rows.append((q,'LR_tip_open_pads_to_release'))
   current=goal
  return rows

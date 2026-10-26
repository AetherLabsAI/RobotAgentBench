"""Read-only planned native arm motion; positive-up pad normal toward left access."""
from parallel_left_corner_plan_v1 import LayPlanner as Base,DATA
from adjacent_edge_lay_plan_v3 import solve_normal
from scipy.spatial.transform import Rotation
import numpy as np
class LayPlanner(Base):
 recovery_desired_uv=[[.038,.03777778],[.0475,.0472222],[.3135,.48166667]]
 def forward_right_presentation(self,v):
  start,h,_,_=self.geom(v,'R');n0=h[:3,1].copy();n1=np.array([0.,0.,1.]);axis=np.cross(n0,n1);angle=np.arctan2(np.linalg.norm(axis),n0@n1);axis/=max(np.linalg.norm(axis),1e-12);target=np.array([-2.03,-.30,1.08]);current=v;rows=[]
  for k in range(1,31):
   a=k/30;a=a*a*(3-2*a);goal,info=solve_normal(self,current,'R',start+a*(target-start),Rotation.from_rotvec(axis*angle*a).apply(n0),allow_flip=False);part=self.interpolate(current,goal,'R',8,'R_present_for_parallel_face_left_grasp');rows+=part;current=part[-1][0]
  self.last_presentation=dict(target=target.tolist(),target_gap_axis=n1.tolist(),frames=len(rows),object_state_writes=0);return rows

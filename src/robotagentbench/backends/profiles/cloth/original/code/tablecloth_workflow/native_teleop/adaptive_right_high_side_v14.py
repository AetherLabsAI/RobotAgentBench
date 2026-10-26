"""Higher-lift diagnostic: require actual cloth/table clearance before transfer."""
from adaptive_right_two_layer_v14 import AdaptiveRightBasket as Base
import numpy as np,json
class AdaptiveRightBasket(Base):
 def on_frame(self,source_frame,physical_frame,cloth):
  following=super().on_frame(source_frame,physical_frame,cloth)
  phase=self.phases[source_frame]
  if phase=='R_hold_lifted_cloth' and self.phases[source_frame+1]!=phase:
   q=cloth[self.cloth_ids==0];lo,hi=self.table_bounds;over=np.all((q[:,:2]>=lo[:2])&(q[:,:2]<=hi[:2]),axis=1);clear=float(q[over,2].min()-hi[2]) if over.any() else None
   row=dict(event='right_free_cloth_table_clearance',physical_frame=physical_frame,old_cloth_min_z_m=float(q[:,2].min()),nodes_over_table=int(over.sum()),minimum_over_table_clearance_m=clear,required_clearance_m=.01)
   self.right_trace.append(row);print(json.dumps(row),flush=True)
   if clear is not None and clear<.01:
    self.failure=f'Cloth free segment not clear of tabletop by1cm before basket transfer: {clear:.5f}m';return None
  return following

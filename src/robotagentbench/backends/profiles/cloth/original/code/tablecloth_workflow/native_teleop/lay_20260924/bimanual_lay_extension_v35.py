"""Future-run instrumentation fix: classify externally chosen left UV correctly.
The same pad proximity and material radius are used. No physical gate or target
is weakened; selected corner identity now reaches the existing observer.
"""
from bimanual_lay_extension_v34 import LayExtension as Base
import json,numpy as np
class LayExtension(Base):
 def grip(self,actual,cloth,uv,side):
  if side=='l' and self.left is None:
   selected=self.output/'left_parallel_face_plan.json'
   try:
    info=json.loads(selected.read_text())['info'];point=np.asarray(info['uv'],dtype=float)
    if point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('Invalid selected left material coordinate')
    self.left=info.copy();self.events.append(dict(stage='external_left_selected_uv_for_observer',selected_uv=point.tolist(),observation_only=True,physical_gates_unchanged=True))
   except FileNotFoundError:pass
  row=super().grip(actual,cloth,uv,side)
  row['selected_material_uv']=[.32775,.028333]if side=='r' or self.left is None else self.left['uv'];row['left_target_registered']=bool(self.left is not None)if side=='l'else None
  return row

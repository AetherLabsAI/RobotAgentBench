"""Continuous-motion diagnostic: retain physical eye grip and plan immediately."""
from bimanual_lay_extension_v37 import LayExtension as Base
class LayExtension(Base):
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage=='right_eye' and not self.early_right_reinforced:
   self.stage='right_eye_reinforced'
  return super().append_at_end(command,actual,cloth,uv,recent)

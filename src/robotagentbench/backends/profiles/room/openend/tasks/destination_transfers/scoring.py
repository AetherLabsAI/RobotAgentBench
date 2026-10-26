"""Approved tray rule: lifted while following 2 + current table-zone final 3."""
import math

class TrayScore:
    def __init__(self):
        self.operation_complete=False;self.final=False;self.ended=False;self.time=None
    def observe(self,*,time_s,held,clear_original_support,raised,inside,supported,on_floor,active,final_sample=True):
        if self.ended:raise ValueError('Ended task')
        if not math.isfinite(time_s) or (self.time is not None and time_s<=self.time):raise ValueError('Fresh monotonic time required')
        for x in (held,clear_original_support,raised,inside,supported,on_floor,active,final_sample):
            if type(x) is not bool:raise ValueError('Measured boolean facts required')
        self.time=time_s
        if active and held and clear_original_support and raised and not on_floor:self.operation_complete=True
        if final_sample:self.final=bool(self.operation_complete and inside and supported and not on_floor)
        return self.summary()
    def summary(self):
        return dict(score=2*int(self.operation_complete)+3*int(self.final),maximum_score=5,
            operation_score=2*int(self.operation_complete),operation_complete=self.operation_complete,
            final_state_score=3*int(self.final),success=self.final,ended=self.ended,
            process_complete=self.operation_complete,full_physical_trajectory_verified=False)
    def pause(self):pass
    def finalize(self,reason):self.ended=True;return dict(self.summary(),termination_reason=reason)

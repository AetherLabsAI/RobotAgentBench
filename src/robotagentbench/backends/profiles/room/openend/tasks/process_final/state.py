"""One-way process milestones, reversible location-only final credit."""
class DestinationState:
    def __init__(self,process_points,final_points):
        self.process_points=process_points;self.final_points=final_points
        self.placed=False;self.met=False;self.complete=False

    def observe(self,*,process_complete,located,released):
        if any(type(v) is not bool for v in (process_complete,located,released)):
            raise ValueError('Typed current destination evidence required')
        # Preserve an existing placement PROCESS milestone, where applicable.
        # Its final-state successor does not require release or a timed hold.
        if process_complete and located and released:self.placed=True
        self.complete=process_complete and (self.placed or self.process_points==0)
        self.met=bool(self.complete and located)
        return self.summary()

    def summary(self):
        process=self.process_points if self.placed else 0
        final=self.final_points if self.met else 0
        return dict(score=process+final,raw_score=process+final,
            maximum_score=self.process_points+self.final_points,
            operation_score=process,operation_complete=self.complete,
            final_state_points=final,current_final=self.met,success=self.met,
            hold_required_s=0,requires_release_for_final=False)

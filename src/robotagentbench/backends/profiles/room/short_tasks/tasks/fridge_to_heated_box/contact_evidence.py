"""Task-owned fail-closed diagnostics for native contact identity failures."""
import json
from .contact_lifecycle import ContactLifecycle

def capture(report, decode, points=False):
    rows=[]
    for h in report[0]:
        row={k:int(getattr(h,k)) for k in ('stage_id','actor0','actor1','collider0','collider1','contact_data_offset','num_contact_data')}
        row['type']=str(h.type)
        row['paths']={k:str(decode(getattr(h,k))) for k in ('actor0','actor1','collider0','collider1')}
        if points:
            start,count=row['contact_data_offset'],row['num_contact_data']
            row['points']=[{k:list(getattr(p,k)) for k in ('position','normal','impulse')}
                           for p in report[1][start:start+count]]
        rows.append(row)
    return rows

class ContactEvidence:
    def __init__(self, source, decode, clock, output, fridge_root):
        self.source,self.decode,self.clock,self.output=source,decode,clock,output
        self.lifecycle=ContactLifecycle(source,decode,clock,fridge_root)
        self.previous=None
    def __getattr__(self,name):return getattr(self.source,name)
    def note_transition(self,before,after):
        self.lifecycle.note_transition(before,after)
    def read(self, report, plate_up):
        try:
            normalized,events=self.lifecycle.normalize(report)
            result=self.source.read(normalized,plate_up)
        except Exception as exc:
            row=dict(time_s=self.clock.current_time,error=str(exc),
                     current=capture(report,self.decode,True),previous=self.previous)
            (self.output/'contact_identity_failure.json').write_text(json.dumps(row,indent=2)+'\n')
            raise
        if events:
            with (self.output/'contact_lifecycle_events.jsonl').open('a') as f:
                for event in events:f.write(json.dumps(event)+'\n')
        self.lifecycle.commit(normalized)
        self.previous=dict(time_s=self.clock.current_time,headers=capture(report,self.decode))
        return result

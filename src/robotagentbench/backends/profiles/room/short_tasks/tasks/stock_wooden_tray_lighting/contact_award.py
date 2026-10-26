"""User-approved v2 contact stage on trusted physical facts, never model output."""
from dataclasses import dataclass
from math import isfinite

VERSION='robustness_target_contact_plus_one.v2'
POINTS=1
BASE_MAXIMUM={'shelf_move':4,'shelf_light':4,'coffee_bag':6,
              'coffee_machine':6,'coffee_combined':6,'button_light':5}
TASK_IDS=dict(stock_wooden_tray_interference='shelf_move',stock_wooden_tray_lighting='shelf_light',
    coffee_bag_interference='coffee_bag',coffee_machine_push_interference='coffee_machine',
    coffee_bag_then_machine_interference='coffee_combined',coffee_button_lighting_interference='button_light')

def under(path,root):return path==root or path.startswith(root.rstrip('/')+'/')
def gripper(path):
    return path.startswith('/World/Robot/gripper_l_') or path.startswith('/World/Robot/gripper_r_')

@dataclass(frozen=True)
class ContactEvidence:
    episode_id:str
    sample_index:int
    time_s:float
    native_pairs:tuple
    complete:bool
    admitted:bool
    relocation_complete:bool

class TargetContactAward:
    def __init__(self,episode_id,task,target):
        if task not in BASE_MAXIMUM or not episode_id or not target.startswith('/'):
            raise ValueError('Exact episode, task and native target identity required')
        self.episode_id,self.task,self.target=episode_id,task,target
        self.last_index=-1;self.last_time=-1.;self.award=None;self.gap_samples=0
    def observe(self,f):
        if not isinstance(f,ContactEvidence) or f.episode_id!=self.episode_id:
            raise ValueError('Trusted same-episode contact evidence required')
        if type(f.sample_index) is not int or f.sample_index<=self.last_index:
            raise ValueError('Fresh increasing native sample required')
        if not isfinite(f.time_s) or f.time_s<0 or f.time_s<=self.last_time:
            raise ValueError('Fresh increasing native time required')
        if any(type(v) is not bool for v in (f.complete,f.admitted,f.relocation_complete)):
            raise ValueError('Explicit native evidence flags required')
        if not isinstance(f.native_pairs,tuple) or any(len(p)!=2 or any(
                not isinstance(v,str) or not v.startswith('/') for v in p) for p in f.native_pairs):
            raise ValueError('Resolved native collider pair identities required')
        self.last_index,self.last_time=f.sample_index,f.time_s
        if not f.complete:self.gap_samples+=1
        gate=f.admitted and (self.task not in ('shelf_move','coffee_bag','coffee_combined') or f.relocation_complete)
        # Missing reports never earn a point, even if stale cached pairs exist.
        if self.award is None and gate and f.complete:
            pair=next((p for p in f.native_pairs if any(under(p[i],self.target) and gripper(p[1-i]) for i in (0,1))),None)
            if pair:self.award=dict(points=POINTS,sample_index=f.sample_index,time_s=f.time_s,pair=list(pair))
        return self.snapshot()
    def snapshot(self):
        return dict(rule_version=VERSION,points=POINTS if self.award else 0,maximum=POINTS,
            awarded=self.award is not None,event=self.award,target=self.target,gap_samples=self.gap_samples,
            maximum_score=BASE_MAXIMUM[self.task]+POINTS)

def observe_stream(award,stream,*,sample_index,time_s,admitted,relocation_complete):
    """Trusted native callback only, using this physical substep's reducer."""
    status=stream.evidence_status()
    unknown=getattr(stream,'unknown_pairs',set())
    pairs=tuple(tuple(row[0]) for key,row in stream.active.items() if key not in unknown and row[-1])
    complete=stream.complete_for_roots(award.target) if hasattr(stream,'complete_for_roots') else status['complete']
    return award.observe(ContactEvidence(award.episode_id,sample_index,float(time_s),pairs,
        bool(complete),bool(admitted),bool(relocation_complete)))

def apply_generic(plan,rows,values,result_factory):
    """One task-wide point, including tasks with more than one target port."""
    candidates=[]
    for row in rows:
        contact=values[row['instance_id']].get('target_contact',{})
        if contact.get('awarded'):
            if contact.get('rule_version')!=VERSION or contact.get('points')!=1:
                raise ValueError('Wrong native contact award version or value')
            event=contact['event']
            candidates.append((event['time_s'],row['instance_id'],contact))
    winner=min(candidates,key=lambda x:(x[0],x[1])) if candidates else None
    for row in rows:
        award=bool(winner and winner[1]==row['instance_id'])
        row['target_contact']=dict(points=int(award),maximum=1,awarded=award,
            event=winner[2]['event'] if award else None,rule_version=VERSION)
        if award:
            row['process']+=1
            row['component_total']=max(0.,row['process']+row['current_final']-row['deductions'])
    out=result_factory(plan,rows)
    out['scoring_revision']=VERSION
    out['target_contact']=dict(points=int(winner is not None),maximum=1,
        awarded=winner is not None,event=winner[2]['event'] if winner else None,rule_version=VERSION)
    return out

def add_to_score(base_score,award,*,base_deductions=0):
    """Apply once before the original floor, so old penalties still count.

    Callers supply original earned points minus original deductions. A floored
    old total must not be used here: that could erase an existing penalty.
    """
    if not all(isfinite(x) for x in (base_score,base_deductions)) or base_deductions<0:
        raise ValueError('Finite original earned points and deductions required')
    return max(0.,base_score+award.snapshot()['points']-base_deductions)

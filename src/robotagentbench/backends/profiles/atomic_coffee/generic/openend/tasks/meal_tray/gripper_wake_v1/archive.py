"""Bounded last-real-contact archive for first-wake cache hydration only.

Not a new sensor, score or per-task idle check. Raw pair events are retained
once, removed by LOST, and never replayed as scoring frames. A consumer's
contact cache is primed separately before reading the current native report.
"""
from collections import namedtuple
from types import SimpleNamespace as NS
import math

Vec3=namedtuple('Vec3','x y z')
FIELDS=('stage_id','actor0','actor1','collider0','collider1')


class ContactArchive:
    def __init__(self,resolve,event_types):
        self.resolve=resolve;self.found,self.persist,self.lost=event_types
        self.active={};self.time=None;self.seen=set();self.paths={}
    def observe(self,report,time_s):
        if self.time is not None and time_s<self.time:raise ValueError('Backwards archive time')
        if not math.isfinite(time_s):raise ValueError('Invalid archive time')
        if not isinstance(report,tuple) or len(report)<2 or report[0] is None or report[1] is None:
            raise RuntimeError('Missing native report')
        headers,data=report[:2];seen=set()
        for h in headers:
            key=(int(h.stage_id),*sorted((int(h.collider0),int(h.collider1))))
            seen.add(key)
            if h.type==self.lost:self.active.pop(key,None);continue
            if h.type not in (self.found,self.persist):raise RuntimeError('Unknown native event')
            start,count=int(h.contact_data_offset),int(h.num_contact_data)
            if count<=0 or start<0 or start+count>len(data):raise RuntimeError('Incomplete contact')
            points=[]
            for i in range(start,start+count):
                p=data[i]
                vectors={n:Vec3(*(float(getattr(getattr(p,n),a)) for a in ('x','y','z')))
                         for n in ('normal','position','impulse')}
                if not all(math.isfinite(v) for row in vectors.values() for v in row):
                    raise RuntimeError('Nonfinite contact')
                points.append(NS(**vectors))
            fields={n:int(getattr(h,n)) for n in FIELDS}
            self.active[key]=(dict(fields,type=h.type),tuple(points))
        self.time=time_s;self.seen=seen
    def snapshot_for(self,reader):
        headers=[];data=[]
        sleeping=getattr(reader,'sleeping',None) or getattr(reader,'is_sleeping',None)
        if sleeping is None:raise TypeError('Unsupported contact cache')
        for key,(fields,points) in self.active.items():
            paths,_,classification=reader.identity.get(fields['stage_id'],fields['collider0'],fields['collider1'])
            relevant=any(classification) if isinstance(classification,tuple) else bool(classification)
            if not relevant:continue
            if key not in self.seen and not all(sleeping(fields['stage_id'],fields[a]) is True for a in ('actor0','actor1')):
                raise RuntimeError('Missing awake contact while dormant: '+str(paths))
            headers.append(NS(**fields,contact_data_offset=len(data),num_contact_data=len(points)))
            data.extend(points)
        return (headers,data)
    def prime(self,reader,up=None):
        cache=getattr(reader,'stream',reader)
        report=self.snapshot_for(cache)
        # RoomPutContacts is an explicit adapter around the same active cache.
        owner=cache.source if hasattr(cache,'source') else cache
        if not isinstance(owner.active,dict):raise TypeError('Unsupported contact storage')
        owner.active.clear()
        if hasattr(reader,'ingest'):reader.ingest(report)
        elif up is not None:reader.read(report,up)
        else:raise TypeError('Orientation required for placement cache')
        # No scorer sees this priming return value. The next measured substep
        # uses its original current report; missing cached pairs stay sleeping.

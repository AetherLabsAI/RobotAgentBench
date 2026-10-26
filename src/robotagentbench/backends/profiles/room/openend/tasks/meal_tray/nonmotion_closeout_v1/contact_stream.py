"""Preserve legacy loads; normalize observed zero-load FOUND -> LOST pairs only.

Real evidence: shared_entry.p6bbd_3l, initial substep 19. PhysX reported a
zero-impulse candidate followed by LOST for the SAME ordered actor/collider
pair. It cannot establish load/grasp/impact, and LOST removes cached support.
All loaded, reordered, overlapping-active or unknown duplicates fail closed.
No thresholds, impulses, contacts, bodies, parameters or score facts are made up.

OPENEND-09: every merge (this wrapper's and the core reader's) is APPENDED to
the active contact log (openend06 contact_log.install) with raw headers and
points, run id, physics tick, control frame, rule and source. Without an
active log a fallback log of the same format is opened in the room output.
zero_load_transient_contacts.json now holds counters only; no event is ever
overwritten. Merge decisions and returned facts are unchanged.
"""
import json
import math
from pathlib import Path

# Label for the unchanged decision below (accepts FOUND/PERSIST chains ending in a 0-point LOST).
WRAPPER_RULE='room_zero_load_found_or_persist_chain_then_zero_point_lost.v1'
WRAPPER_SOURCE='openend/tasks/meal_tray/nonmotion_closeout_v1/contact_stream.py:normalize_zero_load_transients'
CORE_SOURCE='api-protocol/evaluation/tasks/microwave_put/physical_contacts.py:PutContactStream.read'
LOG_MODULE='openend.tasks.meal_tray.openend06_vision_mailbox_v1.contact_log'

def normalize_zero_load_transients(source,report):
    if not isinstance(report,tuple) or len(report)<2 or report[0] is None or report[1] is None:
        return report,[]
    headers,data=report[:2];groups={}
    for i,h in enumerate(headers):
        paths,pair,(payload,wall,stand)=source.identity.get(h.stage_id,h.collider0,h.collider1)
        if payload or wall is not None or stand:groups.setdefault((int(h.stage_id),pair),[]).append((i,h,paths))
    removed=set();events=[]
    for key,rows in groups.items():
        if len(rows)==1:continue
        _,last,paths=rows[-1]
        if last.type!=source.lost or int(last.num_contact_data)!=0:continue
        actors=(int(last.actor0),int(last.actor1));colliders=(int(last.collider0),int(last.collider1))
        safe=True;points=0
        for i,h,_ in rows[:-1]:
            if h.type not in (source.found,source.persist) or (int(h.actor0),int(h.actor1))!=actors or (int(h.collider0),int(h.collider1))!=colliders:
                safe=False;break
            start,count=int(h.contact_data_offset),int(h.num_contact_data)
            if start<0 or count<=0 or start+count>len(data):safe=False;break
            for index in range(start,start+count):
                p=data[index]
                impulse=tuple(p.impulse);normal=tuple(p.normal);position=tuple(p.position)
                if (any(not math.isfinite(x) for x in (*impulse,*normal,*position))
                        or any(x!=0. for x in impulse) or not any(x!=0. for x in normal)):
                    safe=False;break
            if not safe:break
            points+=count
        if safe:
            removed.update(i for i,_,_ in rows[:-1])
            from importlib import import_module
            raw_header=import_module(LOG_MODULE).raw_header
            events.append(dict(stage_id=key[0],pair=list(key[1]),zero_load_points=points,
                               removed_headers=len(rows)-1,terminal_event='LOST',
                               headers=[raw_header(h,data,paths=p,index=i) for i,h,p in rows]))
    if not removed:return report,[]
    return ([h for i,h in enumerate(headers) if i not in removed],data,*report[2:]),events

class RoomPutContacts:
    def __init__(self,source,output):
        self.source=source;self.output=Path(output);self.calls=0;self.normalized_pairs=0
        self.fallback_log=None;self.audit_unavailable=0;self.merges=0;self.last_merge_seq=None
    def __getattr__(self,name):return getattr(self.source,name)
    def _audit(self):
        from importlib import import_module
        module=import_module(LOG_MODULE);log=module.active()
        if log is not None:return log
        if self.fallback_log is None:
            try:self.fallback_log=module.ContactEventLog(self.output,path_of=str,step=None,
                    provenance='room_wrapper_fallback_no_audit_context')
            except OSError:return None
        return self.fallback_log
    def _record(self,method,**kw):
        # Evidence failures never change the contact decision; they are counted, not hidden.
        try:
            log=self._audit();seq=None if log is None else getattr(log,method)(**kw)
        except Exception:seq=None
        if seq is None:self.audit_unavailable+=1
        return seq
    def _merge(self,**kw):
        self.merges+=1;seq=self._record('record_merge',**kw)
        if seq is not None:self.last_merge_seq=seq
    def read(self,report,plate_up):
        self.calls+=1
        normalized,events=normalize_zero_load_transients(self.source,report)
        core_before=len(getattr(self.source,'zero_load_transients',()))
        try:
            result=self.source.read(normalized,plate_up)
        except RuntimeError as exc:
            if 'duplicate contact header' not in str(exc):raise
            headers,data=report[:2];rows=[]
            for h in headers:
                paths,pair,selected=self.source.identity.get(h.stage_id,h.collider0,h.collider1)
                if not selected[0] and selected[1] is None and not selected[2]:continue
                start,count=int(h.contact_data_offset),int(h.num_contact_data)
                row={k:int(getattr(h,k)) for k in ('stage_id','actor0','actor1','collider0','collider1','contact_data_offset','num_contact_data')}
                row.update(type=str(h.type),paths=paths,pair=pair,selected=selected,
                    points=[{k:list(getattr(data[i],k)) for k in ('position','normal','impulse')} for i in range(start,start+count)])
                rows.append(row)
            file=self.output/'duplicate_contact_evidence.json'
            if not file.exists():file.write_text(json.dumps(dict(call=self.calls,error=str(exc),plate_up=list(plate_up),rows=rows),indent=2))
            self._record('record_reader_error',source=CORE_SOURCE,error=exc,headers=rows,read_call=self.calls)
            raise
        for e in events:
            self._merge(source=WRAPPER_SOURCE,rule=WRAPPER_RULE,
                read_call=self.calls,headers=e['headers'],
                detail=dict({k:v for k,v in e.items() if k!='headers'},physical_values_modified=False))
        for rec in getattr(self.source,'zero_load_transients',())[core_before:]:
            self._merge(source=CORE_SOURCE,rule=rec['rule'],
                read_call=rec['read_call'],headers=rec['headers'],
                detail={k:v for k,v in rec.items() if k not in ('rule','read_call','headers')})
        if self.merges>self.normalized_pairs:
            self.normalized_pairs=self.merges
            log=self._audit()
            (self.output/'zero_load_transient_contacts.json').write_text(json.dumps(dict(
                calls=self.calls,merges=self.merges,events_appended_to=None if log is None else str(log.file),
                last_merge_seq=self.last_merge_seq,audit_unavailable=self.audit_unavailable,
                physical_values_modified=False,loaded_contacts_removed=False),indent=2))
        return result

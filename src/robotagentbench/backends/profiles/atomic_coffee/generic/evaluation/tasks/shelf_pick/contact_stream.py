"""Reduce PhysX contacts by existing object identity; no simulation writes."""
import math
import warnings
from arena_api.contact_identity import ContactIdentityCache
from .initial_state import TARGET, SUPPORT, ROBOT

FLOOR='/World/Background/Shell/Floor'
PADS={h:tuple(ROBOT+f'/gripper_{h}_{side}_link4' for side in ('inner','outer')) for h in ('l','r')}


def under(path,root):
    return path == root or path.startswith(root.rstrip('/')+'/')


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


class ContactStream:
    """Use observed pairs; cache absent reports only for sleeping/static actors.

    Zero-impulse pairs cannot prove load-bearing support or opposing grip.
    Multiple colliders/pads on one neighbor reduce to one object identity.
    ``targets`` lists every acceptable target instance; the caller names the
    currently active one when summarizing, so an alternate target is treated
    as a neighbor until it is the one being manipulated.
    """
    def __init__(self,resolve,sleeping,event_types,neighbors,support_top_m,face_tolerance_m=.003,floor_paths=(FLOOR,), *, shelf_paths=(), support_face=None, targets=(TARGET,), support=SUPPORT, object_supports=None, support_tops=None):
        self.sleeping=sleeping
        self.found,self.persist,self.lost=event_types
        self.targets=tuple(targets)
        self.support=support
        self.neighbors=tuple(neighbors)
        if not self.targets or self.targets[0] in self.neighbors or len(set(self.neighbors)) != len(self.neighbors):
            raise ValueError('Invalid neighbor identities')
        self.objects=tuple(dict.fromkeys(self.targets+self.neighbors))
        self.floor_paths=tuple(floor_paths)
        if not self.floor_paths:raise ValueError("Missing floor identities")
        self.shelf_paths=tuple(shelf_paths)
        self.support_face=support_face
        if self.shelf_paths and (self.support not in self.shelf_paths or support_face is None):
            raise ValueError("Shelf contacts require bound original support geometry")
        self.top=support_top_m
        # Each object is judged against the surface it actually stands on.
        self.object_supports=dict(object_supports or {})
        self.support_tops=dict(support_tops or {})
        self.tolerance=face_tolerance_m
        self.identity=ContactIdentityCache(resolve,self.classify)
        self.active={}
        # Legacy APIs remain strict. The room port explicitly opts into
        # nonfatal gaps and must withhold scoring while evidence is unknown.
        self.allow_missing_awake=True
        self.report_missing=False
        self.unresolved_report_gap=False
        self.unknown_pairs=set()
        self.gap_events=0
        self.gap_steps=0

    def _unknown(self,key):
        if key not in self.unknown_pairs:
            self.gap_events+=1
            warnings.warn('Contact evidence gap; run continues, affected shelf '
                          'scoring withheld: '+str(key),RuntimeWarning)
        self.unknown_pairs.add(key)
        if key in self.active:
            paths,stage,actors,_=self.active[key]
            self.active[key]=(paths,stage,actors,())

    def classify(self,paths):
        return any(under(p,root) for p in paths for root in (ROBOT,*self.objects))

    def ingest(self,report):
        if not isinstance(report,tuple) or len(report)<2 or report[0] is None or report[1] is None:
            if not self.report_missing:
                warnings.warn('Missing current PhysX contact report; continuing '
                              'with unknown contact evidence.',RuntimeWarning)
            self.report_missing=True
            self.unresolved_report_gap=True
            for key in self.active:self._unknown(key)
            self.gap_steps+=1
            return
        self.report_missing=False
        self.unknown_pairs.intersection_update(self.active)
        headers,data=report[:2]
        seen=set()
        for h in headers:
            paths,key,relevant=self.identity.get(h.stage_id,h.collider0,h.collider1)
            if not relevant:continue
            seen.add(key)
            if h.type == self.lost:
                self.active.pop(key,None)
                self.unknown_pairs.discard(key)
                continue
            if h.type not in (self.found,self.persist):
                raise RuntimeError('Unknown contact event')
            count,offset=int(h.num_contact_data),int(h.contact_data_offset)
            if count <= 0 or offset < 0 or offset+count > len(data):
                self.active[key]=(paths,int(h.stage_id),(int(h.actor0),int(h.actor1)),())
                self._unknown(key)
                continue
            samples=[]
            for i in range(offset,offset+count):
                v=data[i]
                imp=tuple(getattr(v.impulse,k) for k in ('x','y','z'))
                normal=tuple(getattr(v.normal,k) for k in ('x','y','z'))
                point=tuple(getattr(v.position,k) for k in ('x','y','z'))
                if not all(math.isfinite(x) for x in imp+normal+point):
                    raise RuntimeError('Nonfinite physical contact')
                if dot(imp,imp)>1e-18:
                    samples.append((normal,point,imp))
            self.active[key]=(paths,int(h.stage_id),(int(h.actor0),int(h.actor1)),tuple(samples))
            self.unknown_pairs.discard(key)
        newly_missing=[]
        for key,(paths,stage,actors,samples) in self.active.items():
            if key not in seen and not all(self.sleeping(stage,a) is True for a in actors):
                self._unknown(key)
        if self.unknown_pairs:self.gap_steps+=1

    def evidence_status(self):
        return dict(complete=not self.report_missing and not self.unresolved_report_gap and not self.unknown_pairs,
                    report_missing=self.report_missing,
                    unresolved_report_gap=self.unresolved_report_gap,
                    unknown_pairs=[list(p) for p in sorted(self.unknown_pairs)],
                    gap_events=self.gap_events,gap_steps=self.gap_steps)

    def complete_for_roots(self,*roots):
        return not self.report_missing and not self.unresolved_report_gap and not any(any(under(p,r) for p in key for r in roots)
                                                   for key in self.unknown_pairs)

    def summarize(self,active_target=None,target_center_world=None):
        target=active_target or self.targets[0]
        if target not in self.targets:raise ValueError('Active target is not an acceptable target')
        grips={obj:{h:[[0.,0.,0.],[0.,0.,0.]] for h in PADS} for obj in self.objects}
        shelf_details=[]
        shelf_contact=False
        neighbor_contacts=set()
        robot_contacts=set()
        supported={obj:False for obj in grips}
        support_contact={obj:False for obj in grips}
        on_floor={obj:False for obj in grips}
        support=False
        for paths,_,_,samples in self.active.values():
            if not samples:continue
            for index, subject in enumerate(paths):
                other=paths[1-index]
                if not (under(subject,target) or under(subject,ROBOT)) or not any(under(other,p) for p in self.shelf_paths):continue
                sign=1 if index==0 else -1
                for normal,point,impulse in samples:
                    oriented=tuple(sign*x for x in normal)
                    exempt=False; support_facts={}
                    if under(subject,target) and under(other,self.support):
                        exempt,support_facts=self.support_face.classify(point,oriented,target_center_world)
                    shelf_contact |= not exempt
                    shelf_details.append(dict(subject=subject,component=other,point_world_m=point,
                        normal_on_subject=oriented,impulse_on_subject=tuple(sign*x for x in impulse),
                        exempt=exempt,reason='normal_original_support' if exempt else 'shelf_collision',support_facts=support_facts))
            for obj,hands in grips.items():
                index=0 if under(paths[0],obj) else (1 if under(paths[1],obj) else None)
                if index is None:continue
                other=paths[1-index]
                sign=1 if index == 0 else -1
                for hand,pads in PADS.items():
                    for pad_index,pad in enumerate(pads):
                        if under(other,pad):
                            for normal,_,_ in samples:
                                for axis in range(3):hands[hand][pad_index][axis]+=sign*normal[axis]
                if under(other,ROBOT):robot_contacts.add(obj)
                own=self.object_supports.get(obj,self.support)
                if under(other,own):
                    support_contact[obj]=True
                    own_top=self.support_tops.get(own,self.top)
                    supported[obj] |= any(sign*n[2]>.5 and abs(p[2]-own_top)<=self.tolerance for n,p,_ in samples)
                if any(under(other,p) for p in self.floor_paths):
                    on_floor[obj]=True  # any measured floor impact of the active target is fatal
                if obj != target and (under(other,ROBOT) or under(other,target)):neighbor_contacts.add(obj)
        support=supported[target]
        bilateral={obj:{h:dot(*pads)<0 for h,pads in hands.items()} for obj,hands in grips.items()}
        return dict(bilateral_contacts=bilateral,neighbor_contacts=frozenset(neighbor_contacts),
                    shelf_contact=bool(shelf_contact),shelf_contacts=shelf_details,
                    target_supported=bool(support),target_support_contact=bool(support_contact[target]),
                    target_on_floor=bool(on_floor[target]),active_pair_count=len(self.active),
                    stock_supported=supported,robot_contacts=frozenset(robot_contacts),
                    active_target=target,object_support_contact=support_contact,object_on_floor=on_floor,
                    contact_evidence=self.evidence_status())

    def read(self,report, *, target_center_world=None, active_target=None):
        self.ingest(report)
        return self.summarize(active_target,target_center_world)

"""Sequential object logic on trusted dense facts. No native physics claim."""
from dataclasses import dataclass, replace
import math
from openend.tasks.all_objects_integration.selective_read import SuspensibleEvidence
from arena_api.policy_execution import PHYSICS_DT
from evaluation.tasks.tidy_sofa import process_rules as old
from evaluation.tasks.tidy_sofa.specs import spec, DRAWER_OPEN_M, DRAWER_CLOSED_M
from evaluation.tasks.tidy_sofa.hang_hat import process as hat
from evaluation.tasks.trash_disposal import definitions as trash
from evaluation.tasks.basket_place import process_rules as basket
from .definitions import definition, pickup_spec, placement_max
from .binding import Binding, ContactSelection, path

@dataclass(frozen=True)
class Facts:
    episode_id: str
    instance_id: str
    target: str
    destination: str
    index: int
    time_s: float
    height_m: float
    height_reference: str
    held: bool
    robot_contact: bool
    finger_contact: bool
    supports: frozenset
    on_floor: bool
    in_zone: bool
    rim_only: bool
    fully_inside: bool
    supporting_hooks: frozenset
    drawer_travel_m: float
    furniture_contacts: frozenset
    scenery_on_floor: frozenset
    external_contact: bool
    active: bool
    complete: bool
    dt_s: float = PHYSICS_DT

def roots(contacts, furniture):
    return frozenset((furniture,)) if any(p==furniture or p.startswith(furniture+'/')
        for p in contacts) else frozenset()

class Pickup(old.BaseScorer):
    def pause(self):
        self.hold_start=None; self.hold_elapsed=0.
        self.contact_started.clear(); self.contact_ticks.clear()

    def observe(self,f,*,active,pickup_phase):
        self.last=f
        if not active: self.pause(); return False
        if pickup_phase: self._deductions(f)
        else: self.contact_started.clear(); self.contact_ticks.clear()
        if f.target_on_floor: self._hold(f,False); return False
        if f.target_grasped: self._award(f,'grasp')
        ready=self._hold(f, f.target_grasped and not f.target_supported and f.lift_m>=self.spec.lift_m)
        if ready: self._award(f,'lift')
        return ready

class Placement:
    def __init__(self,d,episode,*,allow_packed_support=False):
        if type(allow_packed_support) is not bool:raise ValueError('Explicit packed-support policy')
        self.allow_packed_support=allow_packed_support
        self.d=d; self.awarded={}; self.events=[]; self.since=None; self.elapsed=0.
        self.penalties=(old.BaseScorer(episode,PHYSICS_DT,spec(d.placement_api))
                        if d.mode in ('surface','drawer') else None)
    def pause(self):
        self.since=None; self.elapsed=0.
        if self.penalties:
            self.penalties.contact_started.clear(); self.penalties.contact_ticks.clear()
    def award(self,f,name,points):
        if name not in self.awarded:
            self.awarded[name]=points
            self.events.append(dict(stage=name,points=points,time_s=f.time_s,index=f.index))
    def observe(self,f,b,*,enabled,final_sample):
        if not enabled:
            self.pause();self.current_final=False
            for key in (('place',) if self.d.mode=='drawer' else ('stable','settle','disposed')):
                self.awarded.pop(key,None)
            return False
        if self.penalties:
            self.penalties._deductions(old.PhysicalFrame(b.episode_id,f.index,f.time_s,
                f.held,f.robot_contact,False,False,0.,f.on_floor,
                furniture_contacts=roots(f.furniture_contacts,b.destination_furniture)))
        released=not f.robot_contact and not f.held
        # OpenEnd may pack onto another item already inside the actual cavity.
        # Surface/drawer and callers retaining the legacy policy stay exact.
        supported=(bool(f.supports) if self.d.mode=='bag' and self.allow_packed_support
                   else f.supports==frozenset((b.destination,)))
        mode=self.d.mode
        if mode=='bag':
            good=supported and f.in_zone and not f.rim_only and not f.on_floor
            hold=basket.HOLD_SECONDS
            if good and released: self.award(f,'place',basket.PLACE_POINTS)
            good=good and 'place' in self.awarded
            stable=('stable',basket.STABLE_POINTS)
        elif mode=='surface':
            source=spec(self.d.placement_api)
            good=supported and f.in_zone and not f.on_floor
            hold=source.hold_s
            if good and released: self.award(f,'place',source.points['place'])
            good=good and 'place' in self.awarded
            stable=('settle',source.points['settle'])
        elif mode=='hang':
            # Hook IDs must be load-bearing witnesses in the bound destination.
            allowed=f.supports <= {b.destination}
            good=bool(f.supporting_hooks) and allowed and not f.on_floor
            hold=hat.HOLD_S
            if good and released: self.award(f,'hang',hat.POINTS['hang'])
            good=good and 'hang' in self.awarded
            stable=('stable',hat.POINTS['stable'])
        elif mode=='trash':
            good=f.fully_inside
            hold=trash.HOLD_SECONDS
            stable=('disposed',trash.MAXIMUM_SCORE)
        else:
            source=spec(self.d.placement_api)
            if not f.on_floor and f.drawer_travel_m>=DRAWER_OPEN_M:
                self.award(f,'open',source.points['open'])
            if not f.on_floor and 'open' in self.awarded and supported and f.in_zone:
                self.drawer_entered=True
            if not f.on_floor and getattr(self,'drawer_entered',False) and f.in_zone and f.drawer_travel_m<=DRAWER_CLOSED_M:
                self.award(f,'close',source.points['close'])
            # Preserve legacy drawer final criteria; native in_zone must prove
            # actual drawer interior, not an AABB-only proximity test.
            good=not f.on_floor and 'close' in self.awarded and f.in_zone and supported
            hold=source.hold_s
            stable=('place',source.points['place'])
        self.current_final=bool(good)
        self.since=None;self.elapsed=0.
        if stable:
            # Original final-stage points, current destination only. No release
            # or dwell prerequisite is added to this final-state channel.
            if good:self.awarded[stable[0]]=stable[1]
            else:self.awarded.pop(stable[0],None)
        return bool(good and final_sample)
    def summary(self):
        deduction=0 if self.penalties is None else self.penalties.collision_deduction
        return dict(earned=sum(self.awarded.values()),raw_score=sum(self.awarded.values())-deduction,
            collision_deduction=deduction,stages=dict(self.awarded),events=list(self.events),
            current_final=getattr(self,'current_final',False),
            operation_complete=(self.d.mode=='trash' or
                {'bag':'place','surface':'place','hang':'hang','drawer':'close'}.get(self.d.mode) in self.awarded),
            continuous_s=self.elapsed)

class Transfer(SuspensibleEvidence):
    def __init__(self,b, *, workspace_rates=False,allow_packed_support=False):
        if type(workspace_rates) is not bool:raise ValueError('Explicit workspace rate mode')
        if type(allow_packed_support) is not bool:raise ValueError('Explicit packed-support policy')
        self.allow_packed_support=allow_packed_support
        self.workspace_rates=workspace_rates
        if not isinstance(b,Binding): raise ValueError('Scene binding required')
        self.b=b; self.d=definition(b.key)
        self.pick=Pickup(b.episode_id,PHYSICS_DT,pickup_spec(self.d))
        self.place=Placement(self.d,b.episode_id,allow_packed_support=allow_packed_support)
        self.selection=ContactSelection()
        self.last=None; self.permit=False; self.ended=False; self.reason=None; self.finalized=False
    def validate(self,f):
        b=self.b
        if not isinstance(f,Facts): raise ValueError('Typed reader facts required')
        if (f.episode_id,f.instance_id,f.target,f.destination)!=(b.episode_id,b.instance_id,b.target,b.destination):
            raise ValueError('Wrong episode/task/object/destination')
        if type(f.index) is not int or f.index!=(0 if self.last is None else self.last.index+1):
            raise ValueError('Missing/repeated physical sample')
        for v in (f.time_s,f.height_m,f.drawer_travel_m,f.dt_s):
            if type(v) not in (int,float) or not math.isfinite(v): raise ValueError('Invalid physical measurement')
        if f.time_s<0 or f.dt_s not in (PHYSICS_DT,1/60) or (f.active and f.dt_s!=PHYSICS_DT and not self.workspace_rates):
            raise ValueError('Incorrect physical rate')
        if self.last and abs(f.time_s-self.evidence_time()-f.dt_s)>1e-6: raise ValueError('Evidence gap')
        if f.height_reference!=self.d.height_reference: raise ValueError('Height reference changed')
        for name in ('held','robot_contact','finger_contact','on_floor','in_zone','rim_only',
                     'fully_inside','external_contact','active','complete'):
            if type(getattr(f,name)) is not bool: raise ValueError('Missing boolean '+name)
        if not f.complete or (f.finger_contact and not f.robot_contact):
            raise ValueError('Incomplete/contradictory facts')
        for values in (f.supports,f.furniture_contacts,f.scenery_on_floor):
            if not isinstance(values,frozenset): raise ValueError('Resolved contact identities required')
            for p in values: path(p)
        if not isinstance(f.supporting_hooks,frozenset) or not f.supporting_hooks<=hat.ALLOWED_HOOKS:
            raise ValueError('Unknown hook')
        if self.d.mode!='hang' and f.supporting_hooks: raise ValueError('Unexpected hook facts')
        if self.last is None and (f.time_s!=b.initial_time_s or f.height_m!=b.initial_height_m or
                f.held or f.robot_contact or f.on_floor or f.supports!=b.initial_supports):
            raise ValueError('Initial evidence must match accepted scene')
    def observe(self,f,*,final_sample=True):
        if self.ended: raise RuntimeError('Ended task')
        if type(final_sample) is not bool: raise ValueError('Explicit final substep flag')
        self.validate(f); b=self.b
        # Release at a destination remains a placement attempt. Losing it to
        # elsewhere requires a new valid carry, but historical points survive.
        at_destination=(b.destination in f.supports or
            (self.d.mode=='trash' and f.fully_inside) or
            (self.d.mode=='hang' and bool(f.supporting_hooks)))
        if self.d.mode=='bag' and self.allow_packed_support:
            at_destination=bool(f.supports) and f.in_zone and not f.rim_only and not f.on_floor
        # Free flight alone is not a confirmed drop (especially for disposal).
        lost=f.on_floor or (not f.held and bool(f.supports) and not at_destination)
        pickup_phase=not self.permit or lost
        already_complete=getattr(self,'pickup_complete',False)
        ready=False if already_complete else self.pick.observe(old.PhysicalFrame(b.episode_id,f.index,f.time_s,
            f.held,f.robot_contact,bool(f.supports) or f.external_contact,False,
            f.height_m-b.initial_height_m,f.on_floor,
            furniture_contacts=roots(f.furniture_contacts,b.source_furniture),
            scenery_on_floor=f.scenery_on_floor & b.scenery),active=f.active,pickup_phase=pickup_phase)
        if lost: self.permit=False
        elif ready: self.permit=True
        self.pickup_complete=already_complete or ready
        done=self.place.observe(f,b,enabled=f.active and self.pickup_complete,final_sample=final_sample)
        self.selection.observe(finger_contact=f.finger_contact,supports=f.supports,
            external_contact=f.external_contact,on_floor=f.on_floor)
        self.last=f
        self.current_success=done
        return self.summary()
    def finalize(self,reason):
        if self.last is None: raise ValueError('No evidence')
        if self.finalized:
            if reason!=self.reason: raise ValueError('Conflicting settlement')
            return self.summary()
        if self.reason=='success':
            if reason!='success': raise ValueError('Already successful')
        elif reason not in ('no_score_increase_timeout','client_closed'):
            raise ValueError('Only room termination')
        self.reason=reason; self.ended=True; self.finalized=True
        return self.summary()
    def summary(self):
        p,q=self.pick.summary(),self.place.summary()
        raw=p['raw_score']+q['raw_score']
        return dict(key=self.b.key,instance_id=self.b.instance_id,score=max(0,raw),raw_score=raw,
            maximum_score=self.pick.spec.maximum_score+placement_max(self.d),
            success=getattr(self,'current_success',False),ended=self.ended,reason=self.reason,finalized=self.finalized,
            operation_complete=getattr(self,'pickup_complete',False) and q['operation_complete'],
            pickup=p,placement=q,selection=self.selection.request(ended=self.ended),
            timeout_owner='room',component_timeouts_enforced=False,
            registered=False,evidence_status='rule_core_not_native_reader')

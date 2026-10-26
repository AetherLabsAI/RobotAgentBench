"""One scene-owned remote, continuous dense evidence, pickup then place.

Scoring logic shared by CPU fixtures and the native room port. Active samples
use the existing PHYSICS_DT; inactive navigation samples carry their real dt.
The room enforces the shared deadline BEFORE admitting a batch. No own timer.
"""
from dataclasses import dataclass
import math
from openend.tasks.all_objects_integration.selective_read import SuspensibleEvidence
from openend.tasks.process_final.state import DestinationState
from arena_api.policy_execution import PHYSICS_DT
from .binding import RemoteBinding, Snapshot, Pose
from .components import RecoverablePick, RecoverablePlace, pick, place
from .geometry import fully_inside


@dataclass(frozen=True)
class Facts:
    episode_id: str
    target_path: str
    destination_path: str
    sample_index: int
    time_s: float
    pose: Pose
    held: bool
    robot_contact: bool
    supports: frozenset
    on_floor: bool
    footprint: tuple
    tabletop: tuple
    exclusions: tuple
    furniture_contacts: frozenset = frozenset()
    scenery_on_floor: frozenset = frozenset()
    active: bool = True
    complete: bool = True
    physics_dt_s: float = PHYSICS_DT
    external_contact: bool = False


class RemoteTransfer(SuspensibleEvidence):
    def __init__(self,binding, *, workspace_rates=False):
        if type(workspace_rates) is not bool:raise ValueError('Explicit workspace rate mode')
        self.workspace_rates=workspace_rates
        if not isinstance(binding,RemoteBinding):
            raise ValueError('Scene-owned remote binding required')
        if binding.current != binding.initial:
            raise ValueError('Create the composite once before streaming observations')
        self.binding=binding
        self.pick=RecoverablePick(binding.episode_id,PHYSICS_DT,binding.target_path)
        self.place=RecoverablePlace(binding.episode_id,PHYSICS_DT)
        self.destination_state=DestinationState(2,2)
        self.last=None;self.permit_place=False;self.phase='pickup';self.ended=False
        self.reason=None;self.finalized=False

    def _validate(self,f):
        b=self.binding
        if not isinstance(f,Facts) or not isinstance(f.pose,Pose):raise ValueError('Typed physical facts required')
        if (f.episode_id,f.target_path,f.destination_path)!=(b.episode_id,b.target_path,b.destination_path):
            raise ValueError('Wrong episode/object/destination binding')
        if type(f.sample_index) is not int or f.sample_index!=(0 if self.last is None else self.last.sample_index+1):
            raise ValueError('Duplicate or missing sample')
        if type(f.time_s) not in (int,float) or not math.isfinite(f.time_s) or f.time_s<0:
            raise ValueError('Invalid simulation time')
        if type(f.physics_dt_s) not in (int,float) or f.physics_dt_s not in (PHYSICS_DT,1/60):raise ValueError('Unapproved measured sample interval')
        if f.active and f.physics_dt_s!=PHYSICS_DT and not self.workspace_rates:raise ValueError('Active scoring requires original native rate')
        if self.last and abs(f.time_s-self.evidence_time()-f.physics_dt_s)>1e-6:
            raise ValueError('Physical evidence gap')
        for name in ('held','robot_contact','on_floor','active','complete','external_contact'):
            if type(getattr(f,name)) is not bool:raise ValueError('Missing physical boolean')
        if not f.complete:raise ValueError('Incomplete evidence; never score as failure')
        # Holding is measured position following; contact remains independent.
        for group in (f.supports,f.furniture_contacts,f.scenery_on_floor):
            if not isinstance(group,frozenset) or any(not isinstance(p,str) or not p.startswith('/') for p in group):
                raise ValueError('Resolved physical body identities required')
        if self.last is None:
            if f.time_s!=b.initial.simulation_time_s or f.pose!=b.initial.pose:
                raise ValueError('First frame must match scene initial receipt')
            if f.held or f.robot_contact or f.on_floor or f.supports != b.initial_supports:
                raise ValueError('Unified scene must supply unheld supported remote')
        return fully_inside(f.footprint,f.tabletop,f.exclusions)

    def observe(self,f,*,final_sample=True):
        if self.ended:raise RuntimeError('Task ended; no further scoring')
        in_zone=self._validate(f)  # all input validation before any mutation
        had_pick=self.pick.ever_complete
        ready=self.pick.observe_one(pick.PhysicalFrame(f.episode_id,f.sample_index,f.time_s,
            f.target_path,f.held,not f.supports and not f.external_contact,f.on_floor),f.active,validated=True)
        supported=f.supports==frozenset((self.binding.destination_path,))
        placed=not f.robot_contact and supported and in_zone and not f.on_floor
        if f.on_floor:self.permit_place=False
        elif ready:self.permit_place=True
        elif not f.held and not (supported and in_zone):self.permit_place=False
        # Physical floor still reaches the place component as recovery evidence
        # once pickup completed; not a terminating event or an implicit award.
        enabled=f.active and (had_pick or self.pick.ever_complete)
        pf=place.PhysicalFrame(f.episode_id,f.sample_index,f.time_s,
            f.held,f.robot_contact,supported,in_zone and self.permit_place,0.,f.on_floor,
            furniture_contacts=f.furniture_contacts,scenery_on_floor=f.scenery_on_floor)
        # Keep original contact deductions, not its historical terminal hold.
        if enabled:self.place._deductions(pf)
        self.destination_state.observe(process_complete=bool(self.pick.ever_complete),
            located=bool(supported and in_zone and not f.on_floor),
            released=bool(not f.robot_contact and not f.held))
        done=self.destination_state.met
        self.last=f
        if f.sample_index>0:
            self.binding.observe(Snapshot(f.episode_id,f.target_path,f.sample_index,f.time_s,f.pose))
        if done:self.phase='complete'
        elif f.on_floor:self.phase='recover_pickup'
        elif self.permit_place:self.phase='place' if placed else 'carrying'
        else:self.phase='pickup'
        return self.summary()

    def finalize(self,reason):
        if self.last is None:raise ValueError('Cannot finalize absent evidence')
        if self.finalized:
            if reason!=self.reason:raise ValueError('Conflicting settlement')
            return self.summary()
        if self.reason=='success':
            if reason!='success':raise ValueError('Successful task already ended')
        elif reason not in ('no_score_increase_timeout','client_closed'):
            raise ValueError('Only the room may stop an unfinished task')
        if not self.pick.ever_complete:self.pick.hold_deduction=pick.HOLD_PENALTY
        self.reason=reason;self.ended=True;self.finalized=True
        return self.summary()

    def summary(self):
        a,b=self.pick.summary(),self.destination_state.summary()
        b['collision_deduction']=self.place.collision_deduction
        b['raw_score']-=self.place.collision_deduction
        raw=a['raw_score']+b['raw_score']
        return dict(score=max(0,raw),raw_score=raw,maximum_score=a['maximum_score']+b['maximum_score'],
            success=self.destination_state.met,failure=None,ended=self.ended,finalized=self.finalized,
            operation_complete=self.destination_state.complete,
            reason=self.reason,phase=self.phase,pickup=a,placement=b,
            timeout_owner='room',component_timeouts_enforced=False,
            evidence_status='external_facts_not_native_verified')

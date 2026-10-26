"""Continuous power bank pickup/carry/bag placement. No resets or physics writes."""
from dataclasses import dataclass
import math
from openend.tasks.all_objects_integration.selective_read import SuspensibleEvidence
from openend.tasks.process_final.state import DestinationState
from arena_api.policy_execution import PHYSICS_DT
from .task_binding import PowerBankBinding, Pose, Snapshot
from .components import RecoverablePick, BoundBasket, pick, place


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
    inside_cavity: bool
    rim_only: bool
    external_contact: bool = False
    active: bool = True
    complete: bool = True
    physics_dt_s: float = PHYSICS_DT
    furniture_contacts: frozenset = frozenset()


class PowerBankTransfer(SuspensibleEvidence):
    def __init__(self, binding, *, workspace_rates=False, allow_packed_support=False):
        if type(workspace_rates) is not bool:raise ValueError('Explicit workspace rate mode')
        if type(allow_packed_support) is not bool:raise ValueError('Explicit packed-support policy')
        self.allow_packed_support = allow_packed_support
        self.workspace_rates=workspace_rates
        if not isinstance(binding, PowerBankBinding) or binding.current != binding.initial:
            raise ValueError('Fresh scene binding required')
        self.binding = binding
        self.pick = RecoverablePick(binding.episode_id, PHYSICS_DT, binding.target)
        self.place = BoundBasket(binding.target, binding.bag)
        self.destination_state = DestinationState(2,2)
        self.last = None
        self.permit_place = False
        self.phase = 'pickup'
        self.ended = self.finalized = False
        self.reason = None

    def _validate(self, f):
        b = self.binding
        if not isinstance(f, Facts) or not isinstance(f.pose, Pose):
            raise ValueError('Typed physical evidence required')
        if (f.episode_id, f.target_path, f.destination_path) != (b.episode_id, b.target, b.bag):
            raise ValueError('Wrong episode/object/bag')
        if type(f.sample_index) is not int or f.sample_index != (0 if self.last is None else self.last.sample_index + 1):
            raise ValueError('Missing or duplicate sample')
        if type(f.time_s) not in (int, float) or not math.isfinite(f.time_s) or f.time_s < 0:
            raise ValueError('Invalid physical clock')
        if type(f.physics_dt_s) not in (int, float) or f.physics_dt_s not in (PHYSICS_DT, 1/60):
            raise ValueError('Unapproved physical interval')
        if f.active and f.physics_dt_s != PHYSICS_DT and not self.workspace_rates:
            raise ValueError('Active scoring requires 240 Hz')
        if self.last and abs(f.time_s-self.evidence_time()-f.physics_dt_s)>1e-6:
            raise ValueError('Physical evidence gap')
        for name in ('held', 'robot_contact', 'on_floor', 'inside_cavity', 'rim_only', 'external_contact', 'active', 'complete'):
            if type(getattr(f, name)) is not bool: raise ValueError('Missing boolean ' + name)
        if not f.complete or (f.held and not f.robot_contact):
            raise ValueError('Incomplete/contradictory evidence')
        if not isinstance(f.supports, frozenset) or any(not isinstance(p, str) or not p.startswith('/') for p in f.supports):
            raise ValueError('Resolved support identities required')
        if not isinstance(f.furniture_contacts, frozenset) or any(
                not isinstance(p, str) or not p.startswith('/') for p in f.furniture_contacts):
            raise ValueError('Resolved furniture evidence required')
        if self.last is None:
            if (f.time_s != b.initial.simulation_time_s or f.pose != b.initial.pose
                    or f.held or f.robot_contact or f.on_floor or f.supports != b.initial_supports):
                raise ValueError('First sample must match supported scene-owned initial object')

    def observe(self, f, *, final_sample=True):
        if self.ended: raise RuntimeError('Task already ended')
        self._validate(f)
        supported = (bool(f.supports) if self.allow_packed_support
                     else f.supports == frozenset((self.binding.bag,)))
        entered = f.inside_cavity and not f.rim_only
        # Carry/place starts after the completed pickup. A lost/dropped target
        # re-enters recovery pickup, without changing the original furniture.
        pickup_phase = (not self.permit_place or f.on_floor
                        or (not f.held and not (supported and entered)))
        source = self.binding.source.rstrip('/')
        contacts = frozenset((self.binding.source,)) if any(
            p == source or p.startswith(source + '/') for p in f.furniture_contacts
        ) else frozenset()
        ready = self.pick.observe_one(pick.PhysicalFrame(
            episode_id=f.episode_id, sample_index=f.sample_index, time_s=f.time_s,
            target_grasped=f.held, target_pad_contact=f.robot_contact,
            target_supported=bool(f.supports) or f.external_contact, target_in_zone=False,
            lift_m=f.pose.position_m[2] - self.binding.initial.pose.position_m[2],
            target_on_floor=f.on_floor, furniture_contacts=contacts), f.active,
            pickup_phase=pickup_phase)
        if f.on_floor: self.permit_place = False
        elif ready: self.permit_place = True
        elif not f.held and not (supported and entered): self.permit_place = False
        # Start the old placement component only from a REAL completed carry,
        # not a fabricated held frame at the scene's unheld initialization.
        # Destination credit is evaluated below from this real current frame.
        # Do not feed the old irreversible stable-hold component after handoff.
        self.last = f
        if f.sample_index > 0:
            self.binding.observe(Snapshot(f.episode_id, f.target_path, f.sample_index, f.time_s, f.pose))
        self.destination_state.observe(process_complete=bool(self.pick.ever_complete),
            located=bool(supported and entered and not f.on_floor),
            released=bool(not f.robot_contact and not f.held))
        if self.destination_state.met:self.phase='complete'
        elif f.on_floor: self.phase = 'recover_pickup'
        elif self.permit_place: self.phase = 'place' if supported and entered else 'carrying'
        else: self.phase = 'pickup'
        return self.summary()

    def finalize(self, reason):
        if self.last is None: raise ValueError('No evidence')
        if self.finalized:
            if reason != self.reason: raise ValueError('Conflicting finalization')
            return self.summary()
        if self.reason == 'success':
            if reason != 'success': raise ValueError('Success already settled')
        elif reason not in ('no_score_increase_timeout', 'client_closed'):
            raise ValueError('Only shared room termination allowed')
        self.reason = reason; self.ended = self.finalized = True
        return self.summary()

    def summary(self):
        a, b = self.pick.summary(), self.destination_state.summary()
        raw = a['raw_score'] + b['raw_score']
        return dict(score=max(0, raw), raw_score=raw, maximum_score=9,
            success=self.destination_state.met, failure=None, ended=self.ended,
            operation_complete=self.destination_state.complete,
            finalized=self.finalized, reason=self.reason, phase=self.phase,
            pickup=a, placement=b, timeout_owner='room', component_timeouts_enforced=False,
            evidence_status='external_facts_not_native_verified')

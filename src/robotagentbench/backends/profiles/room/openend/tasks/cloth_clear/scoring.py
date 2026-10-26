"""Per-instance approved 1+1+1 rules. Facts are not native verification.

No simulation writes, object spawning, local timeout or immediate drop failure.
Source/destination identities come from the unified scene, not name heuristics.
"""
from dataclasses import dataclass
import math
from openend.tasks.remote_transfer.geometry import fully_inside


@dataclass(frozen=True)
class Binding:
    episode_id: str
    instance_id: str
    target_path: str
    cloth_path: str
    table_path: str

    def __post_init__(self):
        paths = (self.target_path, self.cloth_path, self.table_path)
        if not self.episode_id or not self.instance_id or len(set(paths)) != 3:
            raise ValueError('Distinct scene-bound roles required')
        if any(not isinstance(p, str) or not p.startswith('/World/') for p in paths):
            raise ValueError('Resolved scene paths required')


@dataclass(frozen=True)
class Facts:
    binding: Binding
    sample_index: int
    time_s: float
    physics_dt_s: float
    held: bool
    robot_contact: bool
    supports: frozenset
    on_floor: bool
    external_contact: bool
    stable: bool
    footprint: tuple
    tabletop: tuple
    cloth: tuple
    exclusions: tuple  # fixed wooden tray and other approved occupied footprints
    complete: bool = True
    support_contact: bool = False  # even zero-load pairs prevent false unsupported lift


class ClothClear:
    def __init__(self, binding, *, terminate_on_success=True):
        if not isinstance(binding, Binding):
            raise ValueError('Explicit binding required')
        self.binding = binding
        self.last = None
        self.clock = None
        self.awards = set()
        self.hold_start = None
        self.permit_place = False
        self.ended = False
        self.reason = None
        self.finalized = False
        self.terminate_on_success = terminate_on_success
        self.current_placed = False

    def _clock_check(self, time_s, dt):
        if type(time_s) not in (int, float) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Finite simulation time required')
        if dt not in (1 / 60, 1 / 240):
            raise ValueError('Measured workspace substep required')
        if self.clock is not None and abs(time_s - self.clock - dt) > 1e-6:
            raise ValueError('Missing or duplicate physical substep')

    def observe(self, f, *, final_sample=True):
        if self.ended:
            raise RuntimeError('Task ended')
        if not isinstance(f, Facts) or f.binding != self.binding:
            raise ValueError('Wrong task/episode/object roles')
        self._clock_check(f.time_s, f.physics_dt_s)
        expected = 0 if self.last is None else self.last.sample_index + 1
        if type(f.sample_index) is not int or f.sample_index != expected:
            raise ValueError('Missing or duplicate measured sample')
        for key in ('held', 'robot_contact', 'on_floor', 'external_contact', 'stable', 'complete', 'support_contact'):
            if type(getattr(f, key)) is not bool:
                raise ValueError('Complete typed physical facts required')
        if not f.complete:
            raise ValueError('Missing evidence')
        if not isinstance(f.supports, frozenset) or any(not isinstance(p, str) or not p.startswith('/World/') for p in f.supports):
            raise ValueError('Resolved support identities required')
        inside_initial = fully_inside(f.footprint, f.cloth, ())
        destination = fully_inside(f.footprint, f.tabletop, (f.cloth,) + f.exclusions)
        if self.last is None:
            if f.held or f.robot_contact or f.on_floor or f.external_contact or not inside_initial or f.supports != frozenset((self.binding.cloth_path,)):
                raise ValueError('Unified scene must initialize unheld, cloth-supported targets')
        else:
            had_grasp = 'grasp' in self.awards
            if f.held and not f.on_floor:
                self.awards.add('grasp')
            if had_grasp and f.held and not f.supports and not f.support_contact and not f.external_contact and not f.on_floor:
                self.awards.add('lift')
                self.permit_place = True
            resting_at_destination = destination and f.supports == frozenset((self.binding.table_path,))
            placed = resting_at_destination and not f.on_floor
            if f.on_floor or (not f.held and not resting_at_destination):
                self.permit_place = False
            self.current_placed = bool(placed and 'lift' in self.awards)
            # Final credit is a current snapshot, never a historical award.
            if self.current_placed:self.awards.add('place')
            else:self.awards.discard('place')
        self.last = f
        self.clock = f.time_s
        return self.summary()

    def suspend_substep(self, now, dt):
        if self.ended:
            return
        if self.last is None:
            raise ValueError('Initial evidence required before sleeping')
        self._clock_check(now, dt)
        self.clock = now
        self.hold_start = None
        self.current_placed = False
        self.awards.discard('place')
        # We cannot claim continuous transfer across an unobserved interval.
        self.permit_place = 'place' in self.awards

    def finalize(self, reason):
        if self.last is None:
            raise ValueError('Cannot settle absent evidence')
        if self.finalized:
            if self.reason != reason:
                raise ValueError('Conflicting settlement')
            return self.summary()
        if reason == 'success':
            if self.reason != 'success':
                raise ValueError('No observed success')
        elif reason not in ('no_score_increase_timeout', 'client_closed') or self.reason == 'success':
            raise ValueError('Only shared room termination is supported')
        self.reason, self.ended, self.finalized = reason, True, True
        return self.summary()

    def summary(self):
        return dict(score=len(self.awards), raw_score=len(self.awards), maximum_score=3,
                    success=self.current_placed, ended=self.ended, finalized=self.finalized,
                    operation_complete='lift' in self.awards,final_state_points=int(self.current_placed),
                    reason=self.reason, failure=None, awards=sorted(self.awards), timeout_owner='room',
                    current_placed=self.current_placed,
                    evidence_status='external_facts_not_native_verified')

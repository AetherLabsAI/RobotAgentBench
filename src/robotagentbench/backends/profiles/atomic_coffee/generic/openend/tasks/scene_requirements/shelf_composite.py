"""Read-only dense-fact composition. No reset, actions, USD or device writes.

Original scorers are imported unchanged. This is not a registered HTTP runtime
or a native contact reader; a unified-scene adapter must supply trusted facts.
"""
from copy import deepcopy
from openend.tasks.process_final.state import DestinationState
from dataclasses import dataclass, replace
import math
from arena_api.physics_defaults import PHYSICS_DT
from evaluation.tasks.shelf_pick.process_rules import ProcessScorer, PhysicalFrame, TIME_LIMIT_S
from evaluation.tasks.coffee_table_place.process_rules import PlacementProcess, HeldFacts
from evaluation.tasks.coffee_table_place.registration import TIMEOUT_S
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS, TASK_KIND, normalize_parameters


def _path(value):
    return (isinstance(value, str) and value.startswith('/World/')
            and not value.endswith('/') and '//' not in value
            and not any(x in ('.', '..') for x in value.split('/')))


@dataclass(frozen=True)
class Binding:
    episode_id: str
    task_instance_id: str
    product: str
    candidates: tuple[str, ...]
    source_support: str
    destination: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.episode_id, self.task_instance_id)):
            raise ValueError('Episode/task instance identity required')
        normalize_parameters(dict(product=self.product))
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError('Scene must provide concrete eligible product bodies')
        paths = (*self.candidates, self.source_support, self.destination)
        if not all(_path(p) for p in paths) or len(set(paths)) != len(paths):
            raise ValueError('Distinct canonical source/target/destination bindings required')
        if any(a.startswith(b+'/') or b.startswith(a+'/') for i,a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError('Object roles overlap')


@dataclass(frozen=True)
class Frame:
    binding: Binding
    pick: PhysicalFrame
    place: HeldFacts
    physics_dt_s: float = PHYSICS_DT


class SnackTrayComposite:
    def __init__(self, binding, *, room_managed=False):
        if not isinstance(binding, Binding):
            raise ValueError('Unified-scene binding required')
        self.binding = binding
        self.room_managed = room_managed
        self.phase = 'pick'
        self.pick = ProcessScorer(binding.episode_id, PHYSICS_DT, PRODUCTS[binding.product].pick_api)
        self.place = None
        self.completed_pick = None
        self.last = None
        self.place_start = None
        self.committed_target = None
        self.success = self.ended = False
        self.reason = None
        self.failure = None
        self.transitions = []
        self.destination_state=DestinationState(0,3)

    def _validate(self, frame):
        if self.ended:
            raise ValueError('Terminal task; cannot score again')
        if not isinstance(frame, Frame) or frame.binding != self.binding:
            raise ValueError('Wrong scene/episode/task/product binding')
        p, h = frame.pick, frame.place
        if not isinstance(p, PhysicalFrame) or not isinstance(h, HeldFacts):
            raise ValueError('Both original typed facts required')
        expected = 0 if self.last is None else self.last.pick.sample_index + 1
        if type(p.sample_index) is not int or p.sample_index != expected or type(h.step) is not int or h.step != expected:
            raise ValueError('Missing, repeated or misaligned physical sample')
        if type(frame.physics_dt_s) not in (int,float) or not any(abs(frame.physics_dt_s-dt)<1e-10 for dt in (1/60,1/240)):
            raise ValueError('Actual approved native dt required')
        expected_time=0. if self.last is None else self.last.pick.time_s+frame.physics_dt_s
        for t in (p.time_s, h.time_s):
            if isinstance(t, bool) or not isinstance(t,(float,int)) or not math.isfinite(t) or abs(t-expected_time)>1e-6:
                raise ValueError('Expected unchanged dense native evidence clock')
        if p.episode_id != self.binding.episode_id or not p.complete:
            raise ValueError('Wrong episode or incomplete evidence')
        for name in ('target_grasped', 'target_separated_from_support', 'target_on_floor',
                     'target_outside_workspace', 'complete', 'shelf_contact'):
            if type(getattr(p, name)) is not bool:
                raise ValueError('Missing pick physical boolean')
        if type(p.lift_m) not in (float, int) or not math.isfinite(p.lift_m):
            raise ValueError('Finite measured lift required')
        for values in (p.neighbor_contacts, p.wrong_drinks_grasped):
            if not isinstance(values, frozenset) or not all(_path(v) for v in values):
                raise ValueError('Resolved contact identities required')
        if p.active_target in p.neighbor_contacts | p.wrong_drinks_grasped:
            raise ValueError('Target cannot also be a wrong object')
        if p.active_target not in self.binding.candidates:
            raise ValueError('Facts must describe a scene-bound requested product instance')
        if self.committed_target is not None and p.active_target != self.committed_target:
            raise ValueError('Target instance changed after grasp')
        if (p.target_grasped, p.target_on_floor, p.target_outside_workspace) != (h.grasped, h.on_floor, h.outside_workspace):
            raise ValueError('Pick/place facts disagree about the same physical object')
        for name in ('grasped','supported_by_destination','in_destination_region','on_floor','outside_workspace'):
            if type(getattr(h,name)) is not bool:
                raise ValueError('Missing physical boolean')
        for value, length in ((h.position_in_destination_m,3), (h.quaternion_wxyz,4), (h.quaternion_in_destination_wxyz,4)):
            if not isinstance(value,tuple) or len(value)!=length or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
                raise ValueError('Finite measured position/orientation required')
            if length==4 and abs(sum(v*v for v in value)-1)>1e-5:
                raise ValueError('Measured quaternion must be normalized')
        if type(h.height_m) not in (float,int) or not math.isfinite(h.height_m):
            raise ValueError('Finite measured height required')

    def observe(self, frame):
        self._validate(frame)
        # Source validation can raise: do not consume the input or partially score it.
        candidate = deepcopy(self)
        candidate._advance(frame)
        self.__dict__.update(candidate.__dict__)
        return self.summary()

    def _advance(self, frame):
        # Evidence clock follows the scene's real substep, not a solver override.
        self.pick.dt=frame.physics_dt_s
        # User-approved openended policy: outside-workspace is not a failure.
        # Keep the raw input unchanged in self.last; only the original scorers'
        # policy inputs disable that predicate. This is NOT a measured-inside claim.
        p=replace(frame.pick,target_outside_workspace=False)
        h=replace(frame.place,outside_workspace=False)
        if self.phase == 'pick':
            result=self.pick.observe_batch((p,))
            self.committed_target=self.pick.committed_target
            if result['failure']:
                self.failure=dict(phase='pick',reason=result['failure'])
                self.reason=result['failure'];self.ended=True
                self.pick.finalize(self.reason)
            elif result['success']:
                self.completed_pick=self.pick.finalize('success')
                self.phase='carry_and_place'
                self.place_start=p.sample_index
                self.place_start_time=p.time_s
                self.place=PlacementProcess(self.binding.episode_id,
                    task_id=PRODUCTS[self.binding.product].place_api,carries_payload=False)
                # Seed the original drop detector with the ACTUAL held boundary
                # frame, so a fall at the very next sample is not lost. No reward
                # or motion is invented and no scene state is reset.
                self.place.observe(replace(h,step=0,time_s=0.))
                self.transitions.append(dict(sample=p.sample_index,time_s=p.time_s,
                    from_phase='pick',to_phase=self.phase,target=self.committed_target,
                    scene_reset=False,initialization_called=False))
            elif not self.room_managed and p.time_s>=TIME_LIMIT_S:
                self.pick.finalize('time_limit');self.reason='time_limit';self.ended=True
        if self.completed_pick is not None:
            self.destination_state.observe(process_complete=bool(self.completed_pick['success']),
                located=bool(h.supported_by_destination and h.in_destination_region),released=not h.grasped)
            self.success=self.destination_state.met
        self.last=frame

    def pause(self):
        """Keep earned stages/identity; never bridge an unobserved hold."""
        if self.ended:return
        self.pick.hold_start=None
        self.pick.hold_elapsed=0.
        if self.place is not None and self.place.window is not None:
            self.place.window.clear()
            self.place.detector.support_since=None

    def finalize(self, reason='client_closed'):
        if self.ended:
            if reason!=self.reason:raise ValueError('Contradictory finalization')
            return self.summary()
        if reason!='client_closed' or self.last is None:
            raise ValueError('Cannot invent success, timeout or missing evidence')
        if self.place is None:self.pick.finalize(reason)
        else:self.place.summary(final=True)
        self.reason=reason;self.ended=True
        return self.summary()

    def summary(self):
        pick=self.completed_pick or (self.pick.summary() if self.last else None)
        place=self.destination_state.summary() if self.completed_pick else None
        return deepcopy(dict(schema='snack_tray.openended.composition.v2',task_kind=TASK_KIND,
            task_parameters=dict(product=self.binding.product),task_instance_id=self.binding.task_instance_id,
            phase=self.phase,committed_target=self.committed_target,
            score=(pick['score'] if pick else 0)+(place['score'] if place else 0),maximum_score=7,
            score_policy='original_phase_scores_with_user_approved_openended_exclusions',pick=pick,place=place,
            outside_workspace_failure_enabled=False,initial_state_gate_enabled=False,
            outside_workspace_raw=(self.last.pick.target_outside_workspace if self.last else None),
            success=self.success,ended=self.ended,reason=self.reason,failure=self.failure,
            operation_complete=bool(self.completed_pick and self.completed_pick['success']),
            transitions=self.transitions,initialization_owner='unified_scene',
            runtime_registered=False,physics_verified=False,
            evidence_source='external_typed_facts_native_reader_not_connected',
            phase_budgets_s=dict(pick=TIME_LIMIT_S,carry_and_place=TIMEOUT_S),
            global_timeout_policy='shared_room_no_progress' if self.room_managed else 'not_added',
            leave_resume_policy='pause_reset_unfinished_hold' if self.room_managed else 'dense_evidence_required'))

"""Task-local cutlery/salad/drink composition; no native reader or scene writes.

Original salad and shelf scorers remain untouched. Their evidence clocks count
only measured selected samples; pause/missing operations break pending holds.
Final state uses the independent current 3-point predicate, even with no pickup.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

from evaluation.tasks.refrigerator_take.process_rules import Frame as FridgeFrame
from evaluation.tasks.refrigerator_take.process_rules import ProcessScorer as FridgeScorer
from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame, ProcessScorer as ShelfScorer
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
from .final_state import Binding, PlacementFrame, FinalState, path

KINDS = ('cutlery_to_meal_tray', 'salad_to_meal_tray', 'drink_to_meal_tray')
DRINKS = tuple(k for k, p in PRODUCTS.items() if p.tier == 'tier3')
OPERATION_MAXIMUM = dict(zip(KINDS, (2, 8, 4)))


@dataclass(frozen=True)
class CutleryFacts:
    grasped: bool
    separated_from_source: bool
    measurements_complete: bool


@dataclass(frozen=True)
class OperationFrame:
    binding: Binding
    step: int
    time_s: float
    facts: object
    physics_dt_s: float | None = None


class AdditionalMealTask:
    @property
    def ended(self):
        return self.termination is not None

    def __init__(self, binding, *, product=None):
        if not isinstance(binding, Binding) or binding.kind not in KINDS:
            raise ValueError('Concrete additional meal-task binding required')
        if binding.kind == 'drink_to_meal_tray':
            if product not in DRINKS:
                raise ValueError('Explicit supported drink product required; snacks are not drinks')
        elif product is not None:
            raise ValueError('Product parameter belongs only to the drink task')
        self.binding, self.product = binding, product
        self.final_state = FinalState(binding)
        self.scorer = None
        self.last_step = None
        self.operation_step = -1
        self.operation_time_s = None
        self.previous_operation_control = None
        self.last_physical_index = None
        self.operation_result = None
        self.operation_closed = False
        self.closed_by_final = False
        self.cutlery_grasp = self.cutlery_lift = False
        self.paused = False
        self.termination = None

    def _break_hold(self):
        if isinstance(self.scorer, ShelfScorer):
            self.scorer.hold_start = None
            self.scorer.hold_elapsed = 0.
        elif isinstance(self.scorer, FridgeScorer):
            self.scorer._hold_start = None
            self.scorer._hold_s = 0.
        self.previous_operation_control = None

    def _validate_operation(self, evidence, placement):
        if (not isinstance(evidence, OperationFrame) or evidence.binding != self.binding
                or type(evidence.step) is not int or evidence.step != placement.step
                or type(evidence.time_s) not in (int, float)
                or evidence.time_s != placement.time_s):
            raise ValueError('Both score channels must describe the same frame and object')
        f = evidence.facts
        if self.binding.kind == 'cutlery_to_meal_tray':
            if not isinstance(f, CutleryFacts) or evidence.physics_dt_s is not None:
                raise ValueError('Typed cutlery measurements required')
            if any(type(v) is not bool for v in (f.grasped, f.separated_from_source, f.measurements_complete)):
                raise ValueError('Cutlery facts must be measured booleans')
            if not f.measurements_complete:
                raise ValueError('Incomplete cutlery evidence')
        elif self.binding.kind == 'salad_to_meal_tray':
            if (not isinstance(f, FridgeFrame) or evidence.physics_dt_s is not None
                    or f.episode_id != self.binding.episode_id or f.target_prim != self.binding.target
                    or type(f.step_index) is not int or f.step_index != evidence.step
                    or type(f.time_s) not in (int, float) or f.time_s != evidence.time_s):
                raise ValueError('Original fridge facts must match the bound salad and frame')
            # Validate even if this operation track has already closed.
            for name in ('target_held', 'target_fully_out', 'non_operational_collision',
                         'hold_conditions_throughout_interval', 'measurements_complete'):
                if type(getattr(f, name)) is not bool:
                    raise ValueError('Invalid fridge boolean')
            if not f.measurements_complete or not isinstance(f.door_angles_deg, tuple) or not f.door_angles_deg:
                raise ValueError('Complete fridge door evidence required')
            if any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in (f.target_door_angle_deg, *f.door_angles_deg)):
                raise ValueError('Finite measured fridge angles required')
        else:
            dt = evidence.physics_dt_s
            if type(dt) not in (float, int) or dt not in (1/60, 1/240):
                raise ValueError('Actual approved workspace physical dt required')
            contiguous = self.previous_operation_control == placement.step-1
            count = round(.05/dt) if contiguous else 1
            if not isinstance(f, tuple) or len(f) != count:
                raise ValueError('Complete dense selected interval, or one new-segment sample, required')
            for i, sample in enumerate(f):
                expected_time = placement.time_s-(count-1-i)*dt
                if (not isinstance(sample, PhysicalFrame) or sample.episode_id != self.binding.episode_id
                        or sample.active_target != self.binding.target
                        or type(sample.sample_index) is not int or sample.sample_index < 0
                        or type(sample.time_s) not in (float, int) or not math.isfinite(sample.time_s)
                        or abs(sample.time_s-expected_time) > 1e-6
                        or (i == 0 and contiguous and self.last_physical_index is not None
                            and sample.sample_index != self.last_physical_index+1)
                        or (i and sample.sample_index != f[i-1].sample_index+1)):
                    raise ValueError('Wrong drink identity or missing/reordered dense sample')
                for name in ('target_grasped', 'target_separated_from_support', 'target_on_floor',
                             'target_outside_workspace', 'complete', 'shelf_contact'):
                    if type(getattr(sample, name)) is not bool:
                        raise ValueError('Invalid shelf measurement boolean')
                if not sample.complete or type(sample.lift_m) not in (int, float) or not math.isfinite(sample.lift_m):
                    raise ValueError('Incomplete/nonfinite shelf measurements')
                for contacts in (sample.neighbor_contacts, sample.wrong_drinks_grasped):
                    if (not isinstance(contacts, frozenset) or not all(path(p) for p in contacts)
                            or self.binding.target in contacts):
                        raise ValueError('Invalid resolved contact identities')

    def _advance_operation(self, evidence):
        f = evidence.facts
        if self.previous_operation_control != evidence.step-1:
            self._break_hold()
        if isinstance(f, CutleryFacts):
            self.cutlery_grasp |= f.grasped
            # A floating/unheld item cannot earn either operation point.
            self.cutlery_lift |= self.cutlery_grasp and f.grasped and f.separated_from_source
            self.operation_step += 1
            if self.cutlery_lift:
                self.operation_closed = True
        elif isinstance(f, FridgeFrame):
            if self.scorer is None:
                # The original scorer still validates a real closed/unheld zero
                # frame. This flag does not bypass room-level native acceptance.
                self.scorer = FridgeScorer(self.binding.episode_id, self.binding.target,
                                          initial_state_accepted=True)
            self.operation_step += 1
            result = self.scorer.observe(replace(f, step_index=self.operation_step,
                                                time_s=self.operation_step*.05))
            if result.success:
                self.operation_result = asdict(self.scorer.finalize())
                self.operation_closed = True
        else:
            if self.scorer is None:
                self.scorer = ShelfScorer(self.binding.episode_id, evidence.physics_dt_s,
                                         PRODUCTS[self.product].pick_api)
            self.scorer.dt = evidence.physics_dt_s  # observed dt, never a solver setter
            batch = []
            for sample in f:
                self.operation_step += 1
                self.operation_time_s = (0. if self.operation_time_s is None else
                                         self.operation_time_s+evidence.physics_dt_s)
                # Same openended exclusion as the existing shelf composition:
                # transporting outside the old shelf workspace is not a failure.
                batch.append(replace(sample, sample_index=self.operation_step,
                                     time_s=self.operation_time_s, target_outside_workspace=False))
            result = self.scorer.observe_batch(tuple(batch))
            self.last_physical_index = f[-1].sample_index
            if result['success'] or result['failure']:
                self.operation_result = self.scorer.finalize(result['reason'])
                self.operation_closed = True
        self.previous_operation_control = evidence.step

    def _settle_operations(self):
        if self.operation_closed:
            return
        if isinstance(self.scorer, FridgeScorer):
            self.operation_result = asdict(self.scorer.finalize())
        elif isinstance(self.scorer, ShelfScorer):
            self.operation_result = self.scorer.finalize(self.scorer.reason or 'client_closed')
        self.operation_closed = True

    def observe(self, placement, operation=None):
        if self.termination is not None:
            raise ValueError('Room already settled this task')
        if not isinstance(placement, PlacementFrame):
            raise ValueError('Typed current final-state evidence required')
        expected = 0 if self.last_step is None else self.last_step+1
        if placement.step != expected and not (self.paused and self.last_step is not None and placement.step > self.last_step):
            raise ValueError('Missing or repeated control frame without an explicit pause')
        candidate = deepcopy(self)
        candidate.final_state.observe(placement)
        if operation is not None:
            candidate._validate_operation(operation, placement)
        if operation is not None and not candidate.operation_closed:
            # Cutlery has no legacy putdown penalty. Actual same-frame grasp
            # and separation remain eligible even when final placement is met.
            candidate._advance_operation(operation)
        candidate.final_state.qualify_process(bool(candidate.summary()['operation_complete']))
        if operation is None:
            candidate._break_hold()
        candidate.last_step = placement.step
        candidate.paused = False
        self.__dict__.update(candidate.__dict__)
        return self.summary()

    def pause(self):
        self.final_state.invalidate()
        self._break_hold()
        self.paused = True

    def finalize(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError('Explicit room settlement reason required')
        if self.termination is not None:
            if reason != self.termination:
                raise ValueError('Contradictory repeated settlement')
            return self.summary()
        if reason == 'success' and not self.final_state.met:
            raise ValueError('No current tray-success evidence')
        self._settle_operations()
        self.termination = reason
        return self.summary()

    def summary(self):
        if self.binding.kind == 'cutlery_to_meal_tray':
            result = dict(score=int(self.cutlery_grasp)+int(self.cutlery_lift),
                          success=self.cutlery_lift, grasp_awarded=self.cutlery_grasp,
                          lift_awarded=self.cutlery_lift)
        elif self.operation_result is not None:
            result = deepcopy(self.operation_result)
        elif isinstance(self.scorer, FridgeScorer):
            result = asdict(self.scorer.result())
        elif isinstance(self.scorer, ShelfScorer):
            result = self.scorer.summary()
        else:
            result = dict(score=0, success=False)
        final = self.final_state.summary()
        return dict(schema='openend.meal_tray.additional_score.v1', task_type=self.binding.kind,
                    product=self.product, score=result['score']+final['score'],
                    maximum_score=OPERATION_MAXIMUM[self.binding.kind]+3,
                    operation_score=result['score'], operation_complete=result['success'],
                    operations=result, operation_closed=self.operation_closed,
                    operation_closed_by_final=self.closed_by_final,
                    final_state=final, success=final['condition_met'],
                    ended=self.termination is not None, termination=self.termination,
                    timeout_owner='room_120s_no_score_increase', scene_reset=False,
                    runtime_registered=False, native_physics_verified=False)

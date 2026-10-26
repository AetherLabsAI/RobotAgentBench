"""One actual payload across Task9 take and put, without scene resets.

Consumes trusted measurements only. No Isaac imports, actions, helper control,
initial-pose gates or shared registration changes. Existing subtask rules own
their thresholds, hold durations and deductions. The room owns its watchdog.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

from arena_api.task_runtime import TaskEvidence
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1 import (
    RefrigeratorScorer, MicrowaveProgressTracker)
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1.fridge_rules import Frame as FridgeFrame
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1.fridge_rules import MAXIMUM_SCORE as TAKE_MAXIMUM
from evaluation.tasks.microwave_put.process_rules import PutFacts
from evaluation.tasks.microwave_put.scoring import PutScorer, ScoringFrame
from evaluation.tasks.microwave_put.standards import V1

VARIANTS = {'beef_box': ('26-1', '27-1'), 'leftover_box': ('26-2', '27-2')}
CONTROL_DT = .05  # approved existing 20-Hz room control contract


@dataclass(frozen=True)
class Sample:
    episode_id: str
    target_prim: str
    step_index: int
    time_s: float
    fridge: FridgeFrame
    microwave: PutFacts
    inner_wall_collision: bool
    put_hold_interrupted: bool
    source: str = 'isaac_physics'


class Composite:
    def __init__(self, episode_id, target_prim, payload_variant):
        if payload_variant not in VARIANTS: raise ValueError('Unsupported Task9 variant')
        self.episode_id, self.target_prim, self.variant = episode_id, target_prim, payload_variant
        self.fridge = RefrigeratorScorer(episode_id, target_prim)
        self.take_id, self.put_id = VARIANTS[payload_variant]
        self.put_progress = MicrowaveProgressTracker(episode_id=episode_id, task_id=self.put_id,
            payload_variant=payload_variant, payload_instance=target_prim, control_dt_s=CONTROL_DT)
        self.put = PutScorer(episode_id=episode_id, task_id=self.put_id,
                             payload_variant=payload_variant, control_dt_s=CONTROL_DT)
        self.phase = 'refrigerator'
        self.last = None
        self.phase_start = 0
        self.paused = False
        self.closed = False
        self.transitions = []
        self.take_result = None
        self.put_result = None
        self.success = False

    @property
    def requested_physics_hz(self):
        return 60 if self.phase == 'refrigerator' else 240

    def _validate(self, f):
        if not isinstance(f, Sample) or f.source != 'isaac_physics': raise ValueError('Trusted sample required')
        if (f.episode_id, f.target_prim) != (self.episode_id, self.target_prim): raise ValueError('Wrong instance or episode')
        if type(f.step_index) is not int or f.step_index < 0: raise ValueError('Invalid step')
        if type(f.time_s) not in (float,int) or not math.isfinite(f.time_s) or f.time_s < 0: raise ValueError('Invalid time')
        if not isinstance(f.fridge, FridgeFrame) or not isinstance(f.microwave, PutFacts): raise ValueError('Measured facts required')
        if (f.fridge.episode_id,f.fridge.target_prim,f.fridge.step_index,f.fridge.time_s) != (
                f.episode_id,f.target_prim,f.step_index,f.time_s): raise ValueError('Facts from different frames')
        self.put_progress._validate(TaskEvidence(self.episode_id,self.put_id,f.step_index,f.time_s,f.microwave))
        if f.fridge.target_held != f.microwave.held_by_robot: raise ValueError('Conflicting target grasp facts')
        if type(f.inner_wall_collision) is not bool or type(f.put_hold_interrupted) is not bool: raise ValueError('Interval flags required')
        if self.last is None:
            if f.step_index != 0: raise ValueError('Measured baseline required')
        else:
            if f.step_index != self.last.step_index+1: raise ValueError('Missing selected frame')
            dt=f.time_s-self.last.time_s
            if not math.isfinite(dt) or dt < CONTROL_DT-1e-6: raise ValueError('Stale physical time')
            if not self.paused and not math.isclose(dt,CONTROL_DT,abs_tol=1e-6): raise ValueError('Missing physical interval')

    def observe(self, f):
        if self.last is not None and f == self.last: return self.summary()
        if self.closed or self.success: raise RuntimeError('Composite already ended')
        self._validate(f)
        # Rejected evidence must not partially consume stages, rewards or time.
        candidate=deepcopy(self)
        candidate._advance(f)
        self.__dict__.update(candidate.__dict__)
        return self.summary()

    def _advance(self, f):
        if self.phase == 'refrigerator':
            local=replace(f.fridge,time_s=f.step_index*CONTROL_DT,
                          hold_conditions_throughout_interval=(f.fridge.hold_conditions_throughout_interval and not self.paused))
            self.take_result=self.fridge.observe(local)
            if self.take_result.success:
                self.take_result=self.fridge.finalize()
                self.phase='microwave';self.phase_start=f.step_index
                self.transitions.append(dict(from_phase='refrigerator',to_phase='microwave',
                    global_step=f.step_index,simulation_time_s=f.time_s,target_prim=self.target_prim,scene_reset=False))
                # Same real frame becomes an unscored baseline, not a reset.
                self._put_frame(f,baseline=True)
        else:
            self._put_frame(f,baseline=False)
        self.last=f;self.paused=False

    def _put_frame(self,f,*,baseline):
        i=f.step_index-self.phase_start
        raw=TaskEvidence(self.episode_id,self.put_id,i,i*CONTROL_DT,f.microwave)
        progress=self.put_progress.update(raw)
        scored=TaskEvidence(self.episode_id,self.put_id,i,i*CONTROL_DT,
            ScoringFrame(progress,False if baseline else f.inner_wall_collision,
                         False if baseline else f.put_hold_interrupted or self.paused))
        self.put_result=self.put.evaluate(scored)
        self.success=self.put.check_success(scored,self.put_result)

    def pause(self):
        if self.closed or self.success: return
        self.paused=True  # next measured interval resets unfinished hold, not rewards

    def finalize(self):
        if self.closed: return self.summary()
        if self.last is None: raise RuntimeError('No measured baseline')
        if self.phase=='refrigerator': self.take_result=self.fridge.finalize()
        self.closed=True
        return self.summary()

    def summary(self):
        take=asdict(self.take_result) if self.take_result is not None else None
        put=asdict(self.put_result) if self.put_result is not None else None
        return dict(payload_variant=self.variant,target_prim=self.target_prim,phase=self.phase,
            source_api_ids=[self.take_id,self.put_id],score=(take['score'] if take else 0)+(put['score'] if put else 0),
            maximum_score=TAKE_MAXIMUM+V1.maximum_score,refrigerator=take,microwave=put,success=self.success,
            ended=self.closed or self.success,requested_physics_hz=self.requested_physics_hz,control_hz=20,
            transitions=deepcopy(self.transitions),native_reader_connected=False,room_registered=False)

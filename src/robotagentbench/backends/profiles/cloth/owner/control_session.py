"""Cloth-only lifecycle for the unchanged common SessionHost/NativeDriver.

One measured 60Hz scene, 20Hz controls and the user-approved shared 120s
aggregate-score deadline. No World, action generation, reset or pose setter.
Other tasks' existing episode defaults and score implementations are untouched.
"""
from copy import deepcopy
import math
from types import MappingProxyType


class ClothControlSession:
    def __init__(self,*,composition,clock,read_native_facts,instruction,initial_acceptance):
        if (clock.hz!=60 or not callable(read_native_facts)or composition.latest is None
                or initial_acceptance.get('native_initial_state_accepted')is not True
                or initial_acceptance.get('source_profile_verified')is not True):
            raise ValueError('Fresh native-accepted cloth scene and exact clock required')
        if composition.latest['evidence_step']!=0 or clock.current_time!=0 or not instruction:
            raise ValueError('Attach once at the admitted initial boundary')
        self.composition,self.clock,self.read_facts=composition,clock,read_native_facts
        self.episode_id=composition.episode_id;self.plan=MappingProxyType(dict(instruction=instruction))
        self.started=True;self.closed=self.faulted=self.paused=False
        self.pending=self.terminal=self.error=None;self.control_index=0
        self.last_time=self.progress_time=0.;self.previous_total=composition.latest['scores']['total']
        self.substeps=0

    def _fault(self,exc):
        self.faulted=True
        if self.error is None:self.error=dict(type=type(exc).__name__,message=str(exc))

    def begin_control(self,robot=None,*,command_vw=None,base_velocity=None):
        if self.closed or self.faulted or self.pending is not None or self.terminal is not None:raise RuntimeError('No live control interval')
        if self.clock.current_time!=self.last_time:raise RuntimeError('Physics advanced while awaiting an action')
        self.pending=dict(physics_hz=60,substeps=3,control_hz=20,start_time_s=self.last_time)
        self.substeps=0;return deepcopy(self.pending)

    def after_physics(self,helper_report):
        if self.pending is None or self.closed or self.faulted:raise RuntimeError('No admitted physical substep')
        if not isinstance(helper_report,dict)or any(helper_report.get(k)for k in('events','docked_object_count','tray_dock_enabled')):
            raise ValueError('No assisted cloth motion is permitted')
        self.substeps+=1
        if self.substeps>3:raise RuntimeError('Too many physical substeps')
        expected=self.pending['start_time_s']+self.substeps/60
        if not math.isclose(self.clock.current_time,expected,rel_tol=0,abs_tol=1e-9):raise RuntimeError('Native 60Hz clock differs from interval')
        current=self.composition.observe(self.read_facts())
        if not math.isclose(current['evidence_time_s'],self.clock.current_time,rel_tol=0,abs_tol=1e-9):raise RuntimeError('Scoring evidence from another physical frame')

    def finish_control(self,*,completed_substeps):
        if self.pending is None or self.closed or self.faulted or completed_substeps!=3 or self.substeps!=3:raise RuntimeError('Incomplete native control interval')
        now=self.clock.current_time
        if not math.isclose(now-self.pending['start_time_s'],.05,rel_tol=0,abs_tol=1e-9):raise RuntimeError('Expected one 20Hz control step')
        self.pending=None;self.control_index+=1;self.last_time=now
        current=self.composition.latest;total=current['scores']['total']
        if total>self.previous_total:self.progress_time=now
        self.previous_total=total
        # Same boundary rule as the existing room: score, success, then timeout.
        if current['success']:self._finish('success')
        elif now-self.progress_time>=120-1e-9:self._finish('no_new_net_progress_timeout')
        return self.result()

    def _finish(self,reason):
        if self.pending is not None:raise RuntimeError('Cannot settle a partial native interval')
        if self.terminal is None:
            final=self.composition.finalize(reason)
            self.terminal=dict(status='finished',evaluation_valid=True,scoring_available=True,
                episode_id=self.episode_id,task_name=self.composition.task_name,step=self.control_index,
                success=bool(final['success']),scores=deepcopy(final['scores']),termination_reason=reason,
                evidence_time_s=self.last_time,shared_no_progress_timeout_s=120,details=final)

    def result(self):
        if self.faulted:
            return dict(status='error',evaluation_valid=False,scoring_available=False,score=None,success=None,
                episode_id=self.episode_id,task_name=self.composition.task_name,step=self.control_index,error=deepcopy(self.error))
        if self.terminal is not None:return deepcopy(self.terminal)
        return dict(status='running',evaluation_valid=True,scoring_available=True,success=False,
            episode_id=self.episode_id,task_name=self.composition.task_name,step=self.control_index,
            scores=deepcopy(self.composition.latest['scores']),termination_reason=None,
            evidence_time_s=self.last_time,no_progress_time_s=self.last_time-self.progress_time)

    def close(self):
        if self.closed:return
        if self.pending is not None:self._fault(RuntimeError('Closed with an incomplete native interval'))
        if not self.faulted and self.terminal is None:self._finish('client_closed')
        self.closed=True

"""Selected-at-start task registry, one current task, shared no-progress timeout.

Pure orchestration. No poses/actions/assets/physics parameters are modified.
Approved policy: 120 simulation seconds without a net score increase; most
front-facing sector hit wins. Exact angular ties use distance, then stable ID.
The host supplies measured localization, approved per-task rates and native facts.
Per measured control frame: select -> optional current-task record_score ->
finish_frame. Score on the exact deadline frame is evaluated before timeout.
An observation arriving AFTER the deadline cannot retrospectively revive it.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from sector_activation_v1 import sector_polygon, angle_error


@dataclass(frozen=True)
class TaskRegistration:
    instance_id: str
    kind: str
    target_xy: tuple
    target_polygon: tuple
    physics_hz: int

    def __post_init__(self):
        if not self.instance_id or not self.kind:
            raise ValueError('Task instance and type required')
        if self.physics_hz not in (60,240):
            raise ValueError('Unapproved task physics rate')
        if len(self.target_xy)!=2 or len(self.target_polygon)<3 or any(len(p)!=2 for p in self.target_polygon):
            raise ValueError('Actual object target and polygon required')
        if not all(math.isfinite(v) for p in (self.target_xy,*self.target_polygon) for v in p):
            raise ValueError('Finite object coordinates required')


class MultiTaskDispatcher:
    def __init__(self, registrations, *, sector_radius, sector_half_angle,
                 no_progress_timeout_s=120., selection_order='front_then_distance'):
        if not isinstance(no_progress_timeout_s,(int,float)) or isinstance(no_progress_timeout_s,bool) or not math.isfinite(no_progress_timeout_s) or no_progress_timeout_s<=0:
            raise ValueError('Positive no-progress timeout required')
        if selection_order != 'front_then_distance':
            raise ValueError('Approved selection is most frontal first')
        if not math.isfinite(sector_radius) or sector_radius<=0 or not 0<=sector_half_angle<=180:
            raise ValueError('Existing calibrated sector required')
        self.registrations=tuple(registrations)
        if any(not isinstance(r,TaskRegistration) for r in self.registrations):raise TypeError('Typed registrations required')
        if len({r.instance_id for r in self.registrations})!=len(self.registrations):raise ValueError('Duplicate instance')
        self.timeout=float(no_progress_timeout_s);self.order=selection_order
        self.radius=sector_radius;self.half=sector_half_angle
        self.started_at=None;self.last_time=None;self.current=None;self.ended=False
        self.last_progress_time=None;self.termination_reason=None
        self.pending=False;self.scored=False
        self.switches=[];self.progress=[]
        self._scores={r.instance_id:dict(score=0,started=False,success=False) for r in self.registrations}

    @property
    def scores(self):
        return deepcopy(self._scores)

    @property
    def total_score(self):
        return math.fsum(r['score'] for r in self._scores.values())

    def _timer(self):
        idle=self.last_time-self.last_progress_time
        return dict(elapsed_s=self.last_time-self.started_at,no_progress_s=idle,
                    no_progress_timeout_s=self.timeout,remaining_s=max(0.,self.timeout-idle),
                    last_progress_time_s=self.last_progress_time,total_score=self.total_score,
                    timed_out=self.ended,termination_reason=self.termination_reason,control_hz=20)

    def _expire(self):
        self.ended=True
        self.termination_reason='no_score_increase_timeout'
        self.current=None

    def start(self, *, accepted_scene, simulation_time_s):
        if self.started_at is not None:raise RuntimeError('Reentry cannot restart the room')
        if accepted_scene is not True or isinstance(simulation_time_s,bool) or not math.isfinite(simulation_time_s):raise ValueError('Host must accept actual scene first')
        self.started_at=self.last_time=self.last_progress_time=float(simulation_time_s)

    def select(self, robot, *, simulation_time_s):
        if self.started_at is None or self.ended:raise RuntimeError('Room not started or terminal')
        if self.pending:raise RuntimeError('Finish the previous scoring frame first')
        now=float(simulation_time_s)
        if isinstance(simulation_time_s,bool) or not math.isfinite(now) or now<self.last_time:raise ValueError('Global simulation time cannot reset')
        if len(robot['xy'])!=2 or not all(math.isfinite(v) for v in (*robot['xy'],robot['yaw'])):raise ValueError('Measured robot localization required')
        if hasattr(self,'last_frame_time') and now<=self.last_frame_time:raise ValueError('Repeated scoring frame')
        self.last_time=now
        self.last_frame_time=now
        expired=now-self.last_progress_time>self.timeout+1e-9
        candidates=[]
        if not expired:
            for task in self.registrations:
                if not sector_polygon(robot,self.radius,self.half,task.target_polygon):continue
                dx,dy=(task.target_xy[i]-robot['xy'][i] for i in (0,1))
                distance=math.hypot(dx,dy)
                heading=0. if distance<=1e-9 else angle_error(robot['yaw'],math.degrees(math.atan2(dy,dx)))
                candidates.append(dict(instance_id=task.instance_id,distance_m=distance,
                                       heading_error_deg=heading,physics_hz=task.physics_hz))
        candidates.sort(key=lambda r:(r['heading_error_deg'],r['distance_m'],r['instance_id']))
        selected=candidates[0]['instance_id'] if candidates else None
        transition=None
        if selected!=self.current:
            transition=dict(from_task=self.current,to_task=selected,time_s=now)
            self.switches.append(transition)
        self.current=selected
        self.pending=not expired;self.scored=False
        if expired:self._expire()
        return dict(current_task=selected,candidates=candidates,transition=transition,
                    requested_physics_hz=candidates[0]['physics_hz'] if candidates else 60,
                    **self._timer())

    def record_score(self, instance_id, result):
        if self.ended or not self.pending or self.scored:raise RuntimeError('One score observation per live frame')
        if instance_id!=self.current or instance_id not in self._scores:raise ValueError('Only current registered task may score')
        if (not isinstance(result,dict) or not isinstance(result.get('score'),(int,float))
                or isinstance(result['score'],bool) or not math.isfinite(result['score'])):
            raise ValueError('Finite task result required')
        before=self.total_score
        updated=deepcopy(result)
        after=math.fsum(updated['score'] if k==instance_id else r['score'] for k,r in self._scores.items())
        if not math.isfinite(after):raise ValueError('Non-finite total score')
        self._scores[instance_id]=updated
        self.scored=True
        # Net sum across ALL registered tasks, never the displayed current-task
        # score. Switching from a 1-point task to a 5-point task is not progress.
        if after>before:
            self.last_progress_time=self.last_time
            self.progress.append(dict(instance_id=instance_id,time_s=self.last_time,
                                      previous_total_score=before,total_score=after,delta=after-before))
        return self._timer()

    def complete_physics_frame(self, *, simulation_time_s):
        """Native host stamps the end of the held 20Hz action before scoring."""
        if not self.pending or self.scored or self.ended:raise RuntimeError('No unscored action')
        now=float(simulation_time_s)
        if not math.isfinite(now) or abs(now-self.last_frame_time-.05)>1e-6:
            raise ValueError('Expected one real 20Hz control interval')
        if getattr(self,'physical_completion',None)==self.last_frame_time:raise RuntimeError('Already completed')
        self.physical_completion=self.last_frame_time
        self.last_time=now
        if now-self.last_progress_time>self.timeout+1e-9:
            self.pending=False;self._expire()
        return self._timer()

    def finish_frame(self):
        if self.ended or not self.pending:raise RuntimeError('No unfinished live frame')
        self.pending=False
        if self.last_time-self.last_progress_time>=self.timeout-1e-9:self._expire()
        return dict(current_task=self.current,scoreboard=self.scores,**self._timer())

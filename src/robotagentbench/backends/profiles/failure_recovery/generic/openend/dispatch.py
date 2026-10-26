"""Registered tasks, an awake set and one display focus; shared idle watchdog.

Pure orchestration. No poses/actions/assets/physics parameters are modified.
Latest policy: nearest sector appliance plus overlapping 10 s object touch
windows. All awake tasks may score, with net progress settled atomically.
Legacy front_then_distance/locked-contact requests remain readable for old
fixtures; new room requests use distance_then_front and timed contact evidence.
The native host supplies an independent single-frontal-workspace rate request.
Per-task rates remain only for historical callers without that workspace input.
Per measured control frame: select -> optional awake-task record_scores ->
finish_frame. Score on the exact deadline frame is evaluated before timeout.
An observation arriving AFTER the deadline cannot retrospectively revive it.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
import math
from .sector import sector_polygon, angle_error


@dataclass(frozen=True)
class TaskRegistration:
    instance_id: str
    kind: str
    target_xy: tuple
    target_polygon: tuple
    physics_hz: int
    selection_mode: str = 'sector'

    def __post_init__(self):
        if self.selection_mode not in ('sector','contact'):raise ValueError('Unknown selection mode')
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
        if selection_order not in ('front_then_distance','distance_then_front'):
            raise ValueError('Unknown task selection order')
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

    def update_geometry(self,instance_id,target_xy,target_polygon):
        """Host-only measured localization update, never a score/timer reset."""
        if self.pending or self.ended:raise RuntimeError('Geometry updates only between actions')
        matches=[r for r in self.registrations if r.instance_id==instance_id]
        if len(matches)!=1:raise ValueError('Unknown task instance')
        changed=replace(matches[0],target_xy=tuple(target_xy),target_polygon=tuple(tuple(p) for p in target_polygon))
        self.registrations=tuple(changed if r.instance_id==instance_id else r for r in self.registrations)

    def select(self, robot, *, simulation_time_s, activity=None, workspace=None):
        if self.started_at is None or self.ended:raise RuntimeError('Room not started or terminal')
        if self.pending:raise RuntimeError('Finish the previous scoring frame first')
        now=float(simulation_time_s)
        if isinstance(simulation_time_s,bool) or not math.isfinite(now) or now<self.last_time:raise ValueError('Global simulation time cannot reset')
        if len(robot['xy'])!=2 or not all(math.isfinite(v) for v in (*robot['xy'],robot['yaw'])):raise ValueError('Measured robot localization required')
        if hasattr(self,'last_frame_time') and now<=self.last_frame_time:raise ValueError('Repeated scoring frame')
        activity={} if activity is None else activity
        if workspace is not None:
            if workspace.get('physics_rate_owner')!='single_frontal_workspace' or workspace.get('requested_physics_hz') not in (60,240):
                raise ValueError('Measured independent workspace selection required')
        required={r.instance_id for r in self.registrations if r.selection_mode=='contact'}
        if not isinstance(activity,dict) or set(activity)!=required:raise ValueError('Current contact-task activation required')
        for a in activity.values():
            if (not isinstance(a,dict) or set(a) not in ({'active','locked'},{'active','locked','touching','last_touch_s'})
                    or any(type(a[k]) is not bool for k in ('active','locked')) or (a['locked'] and not a['active'])):
                raise ValueError('Invalid task selection request')
            if 'touching' in a:
                if (type(a['touching']) is not bool or (a['touching'] and not a['active']) or a['locked']
                        or (a['last_touch_s'] is not None and (type(a['last_touch_s']) not in (int,float)
                            or not math.isfinite(a['last_touch_s']) or not 0<=a['last_touch_s']<=now))
                        or (a['active'] and a['last_touch_s'] is None)):
                    raise ValueError('Invalid timed contact wake evidence')
        self.last_time=now
        self.last_frame_time=now
        expired=now-self.last_progress_time>self.timeout+1e-9
        candidates=[];prewarm_hz=60
        if not expired:
            for task in self.registrations:
                hit=sector_polygon(robot,self.radius,self.half,task.target_polygon)
                a=activity.get(task.instance_id)
                if task.selection_mode=='contact':
                    if hit and self.order=='front_then_distance':prewarm_hz=max(prewarm_hz,task.physics_hz)
                    if not a['active']:continue
                    priority=0 if a['locked'] else (1 if a.get('touching',True) else 2)
                    reason='carrying_task_lock' if a['locked'] else ('finger_contact' if a.get('touching',True) else 'post_contact_10s')
                else:
                    if not hit:continue
                    if workspace is not None and task.kind!=workspace['current_workspace']:continue
                    priority=3;reason='sector'
                dx,dy=(task.target_xy[i]-robot['xy'][i] for i in (0,1))
                distance=math.hypot(dx,dy)
                heading=0. if distance<=1e-9 else angle_error(robot['yaw'],math.degrees(math.atan2(dy,dx)))
                candidates.append(dict(instance_id=task.instance_id,distance_m=distance,
                                       heading_error_deg=heading,physics_hz=task.physics_hz,priority=priority,reason=reason,
                                       last_touch_s=None if a is None else a.get('last_touch_s')))
        # Latest policy: nearest sector-hit appliance. Portable objects still
        # require real finger contact, and an existing carry lock takes priority.
        candidates.sort(key=lambda r:(r['priority'],
            -r['last_touch_s'] if r['last_touch_s'] is not None else 0.,
            r['distance_m'] if self.order=='distance_then_front' and r['reason']=='sector' else r['heading_error_deg'],
            r['heading_error_deg'] if self.order=='distance_then_front' and r['reason']=='sector' else r['distance_m'],r['instance_id']))
        selected=candidates[0]['instance_id'] if candidates else None
        # New room policy: overlapping object touch windows score independently;
        # at most one sector appliance (the nearest) joins that active set.
        if self.order=='distance_then_front':
            active=[r['instance_id'] for r in candidates if r['reason']!='sector']
            appliances=[r for r in candidates if r['reason']=='sector']
            if appliances:active.append(appliances[0]['instance_id'])
        else:active=[] if selected is None else [selected]
        self.active_tasks=tuple(active)
        transition=None
        if selected!=self.current:
            transition=dict(from_task=self.current,to_task=selected,time_s=now)
            self.switches.append(transition)
        self.current=selected
        self.pending=not expired;self.scored=False
        if expired:self._expire()
        requested_hz=(workspace['requested_physics_hz'] if workspace is not None else
                      max([prewarm_hz]+[r['physics_hz'] for r in candidates if r['instance_id'] in active]))
        result=dict(current_task=selected,active_tasks=active,candidates=candidates,transition=transition,
                    selection_reason=candidates[0]['reason'] if candidates else None,
                    requested_physics_hz=requested_hz,
                    **self._timer())
        # Legacy fixtures retain their historical rate policy. The native host
        # supplies independent workspace evidence, which is authoritative.
        if workspace is not None:result.update(workspace)
        return result

    def record_score(self, instance_id, result):
        # Legacy single-task caller. The room uses one atomic score batch.
        if self.ended or not self.pending or self.scored:raise RuntimeError('One score observation per live frame')
        if instance_id!=self.current:raise ValueError('Only the current display task may use single-score submission')
        return self.record_scores({instance_id:result})

    def record_scores(self, results):
        if self.ended or not self.pending or self.scored:raise RuntimeError('One score observation per live frame')
        if not isinstance(results,dict) or not results or not set(results)<=set(self.active_tasks):
            raise ValueError('Only awake registered tasks may score')
        for result in results.values():
            if (not isinstance(result,dict) or not isinstance(result.get('score'),(int,float))
                    or isinstance(result['score'],bool) or not math.isfinite(result['score'])):
                raise ValueError('Finite task result required')
        before=self.total_score
        updated=deepcopy(results)
        after=math.fsum(updated[k]['score'] if k in updated else r['score'] for k,r in self._scores.items())
        if not math.isfinite(after):raise ValueError('Non-finite total score')
        self._scores.update(updated)
        self.scored=True
        # Net sum across ALL registered tasks, never the displayed current-task
        # score. Switching from a 1-point task to a 5-point task is not progress.
        if after>before:
            self.last_progress_time=self.last_time
            self.progress.append(dict(instance_ids=sorted(results),time_s=self.last_time,
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
        return dict(current_task=self.current,active_tasks=[] if self.ended else list(self.active_tasks),scoreboard=self.scores,**self._timer())

    def settle_terminal_scores(self, scores):
        """Once-only final deductions; cannot revive or refresh an ended room."""
        if not self.ended or getattr(self,'settled',False):
            raise RuntimeError('Terminal settlement exactly once')
        if set(scores)!=set(self._scores):raise ValueError('All registered tasks required')
        for result in scores.values():
            score=result.get('score')
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
                raise ValueError('Finite final scores required')
        self._scores=deepcopy(scores);self.settled=True
        return dict(current_task=None,scoreboard=self.scores,**self._timer())

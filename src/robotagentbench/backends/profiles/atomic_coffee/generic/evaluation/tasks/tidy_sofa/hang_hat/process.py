"""17-1 approved arithmetic on trusted physical evidence; no scene or controller.

The task Isaac reader must prove support, release and dropping. A contact,
closed-gripper command, proximity or model-supplied label is not such proof.
No draft collision deductions, height, retreat, speed or settling thresholds.
"""
from dataclasses import dataclass
from copy import deepcopy
import math

from arena_api.task_runtime import TaskEvidence, TaskScore

TASK_ID='17-1'
VERSION='arena.hang_hat.score.v2'
TIME_LIMIT_S=180.
HOLD_S=1.
POINTS={'hang':2.,'stable':2.}
ALLOWED_HOOKS=frozenset(f'Branch{i:02}' for i in range(1,9))


@dataclass(frozen=True)
class HangFacts:
    """Reader-owned facts for the current control sample, never action fields.

    supporting_hooks: normalized branch identities with actual load-bearing
    support of the cap, not merely overlapping or touching colliders.
    robot_contact: ANY robot link touching the cap, including non-pad links.
    foreign_support: cap supported by something outside the coat rack.
    Rack co-support is allowed, but a loaded hook is still mandatory.
    ground_contact: any active cap-floor contact, even with zero impulse.
    dropped: reader-confirmed fall/drop, not simply absent gripper contact.
    continuity: no interruption of qualifying support/release SINCE the last
    sample. Required to prevent a substep loss hidden between observations.
    """
    supporting_hooks:frozenset[str]
    robot_contact:bool
    foreign_support:bool
    dropped:bool
    continuity:bool
    complete:bool=True
    ground_contact:bool=False


class HangScoringHooks:
    """Composable WorkerTask scoring hooks; this is NOT an Isaac reader/runtime."""
    def __init__(self,episode_id,control_dt_s):
        if not isinstance(episode_id,str) or not episode_id.strip():raise ValueError('Missing episode')
        if isinstance(control_dt_s,bool) or not math.isfinite(control_dt_s) or control_dt_s<=0:
            raise ValueError('Invalid control interval')
        self.episode_id=episode_id;self.dt=control_dt_s
        self.last=None;self.last_score=None;self.start=None;self.held_s=0.
        self.awarded={};self.events=[];self.success=False;self.failure=None;self.final_reason=None

    def _validate(self,e):
        if not isinstance(e,TaskEvidence) or e.source!='isaac_physics':raise ValueError('Trusted physical evidence required')
        if e.task_id!=TASK_ID or e.episode_id!=self.episode_id:raise ValueError('Cross-task/episode evidence')
        expected=0 if self.last is None else self.last.step_index+1
        if type(e.step_index) is not int or e.step_index!=expected:raise ValueError('Duplicate or missing control sample')
        if isinstance(e.time_s,bool) or not isinstance(e.time_s,(int,float)) or not math.isfinite(e.time_s) or e.time_s<0:
            raise ValueError('Invalid simulation time')
        if self.last is not None and abs(e.time_s-self.last.time_s-self.dt)>1e-6:
            raise ValueError('Control evidence time gap')
        f=e.data
        if not isinstance(f,HangFacts) or not isinstance(f.supporting_hooks,frozenset):raise ValueError('Missing physical facts')
        if not f.supporting_hooks<=ALLOWED_HOOKS:raise ValueError('Unknown supporting branch')
        for k in ('robot_contact','foreign_support','dropped','continuity','complete','ground_contact'):
            if type(getattr(f,k)) is not bool:raise ValueError('Missing physical boolean '+k)
        if not f.complete:raise ValueError('Incomplete evidence is not a valid zero-score result')

    def _award(self,e,name):
        if name not in self.awarded:
            self.awarded[name]=e.time_s
            self.events.append(dict(stage=name,points=POINTS[name],time_s=e.time_s,step=e.step_index))

    def _score(self):
        return TaskScore(sum(POINTS[k] for k in self.awarded),sum(POINTS.values()),dict(
            schema=VERSION,stages={k:dict(points=v,awarded_at_s=self.awarded.get(k)) for k,v in POINTS.items()},
            hold_elapsed_s=self.held_s,hold_required_s=HOLD_S,success=self.success,failure=self.failure,
            final_reason=self.final_reason,events=deepcopy(self.events),collision_deduction=0.,
            collision_rule='not_defined_for_this_task'))

    def evaluate(self,evidence):
        if self.final_reason is not None or self.success or self.failure:raise RuntimeError('Episode already terminal')
        self._validate(evidence);f=evidence.data
        # Real failure beats simultaneous success; never award a dropped cap.
        if f.dropped:
            self.failure='hat_dropped';self.start=None;self.held_s=0.
        else:
            hung=bool(f.supporting_hooks) and not f.robot_contact and not f.foreign_support and not f.ground_contact
            if hung:
                self._award(evidence,'hang')
                if self.start is None or not f.continuity:self.start=evidence.time_s
                self.held_s=evidence.time_s-self.start
                if self.held_s+1e-9>=HOLD_S:
                    self._award(evidence,'stable');self.success=True
            else:self.start=None;self.held_s=0.
        self.last=evidence;self.last_score=self._score()
        return self.last_score

    def _current(self,evidence,score):
        if evidence is not self.last or score is not self.last_score:
            raise ValueError('Predicates must share exactly the evaluated evidence and score')

    def check_success(self,evidence,score):
        self._current(evidence,score);return self.success and self.failure is None

    def check_failure(self,evidence,score):
        self._current(evidence,score);return self.failure

    def finalize(self,evidence,score,outcome):
        self._current(evidence,score)
        expected=self.failure or ('success' if self.success else None)
        if expected and outcome.reason!=expected:raise ValueError('Finalization contradicts physical evidence')
        if not expected and outcome.reason not in ('time_limit','client_closed'):
            raise ValueError('No evidence for terminal verdict')
        if outcome.success!=(self.success and self.failure is None):raise ValueError('Contradictory success')
        if self.final_reason and self.final_reason!=outcome.reason:raise ValueError('Conflicting repeated finalization')
        self.final_reason=outcome.reason
        return self._score()

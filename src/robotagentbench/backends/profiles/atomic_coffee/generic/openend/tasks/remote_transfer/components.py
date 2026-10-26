"""Task-local adaptations; imported legacy modules/constants remain untouched.

Only approved semantic changes: bind existing scene identity, recover instead of
physical-failure termination, no component timeout. Scene/physics are never read
or changed here. Existing scoring values/holds/deductions are reused.
"""
from evaluation.tasks.small_table_pick import process_rules as pick
from evaluation.tasks.tidy_sofa import process_rules as place
from evaluation.tasks.tidy_sofa.specs import spec


class RecoverablePick(pick.ProcessScorer):
    def __init__(self, episode_id, dt, target):
        super().__init__(episode_id, dt, '11-2')
        self.target = target  # instance binding, not a source-global mutation
        self.ever_complete = False

    def pause(self):
        self.hold_start = None
        self.hold_elapsed = 0.
        self.carrying = False

    def observe_one(self, f, active, *, validated=False):
        if self.finalized:raise RuntimeError('Finalized pickup')
        if not validated:self._validate((f,))
        self.last = f
        if self.origin is None:self.origin=f.time_s
        if not active or f.target_on_floor:
            self.pause()
            return False
        if f.target_held and not self.grasp_awarded:
            self.grasp_awarded=True;self._event(f,'correct_grasp',pick.GRASP_POINTS)
        self.carrying=f.target_held and f.target_separated_from_table
        if not self.carrying:
            self.pause();return False
        if not self.lift_awarded:
            self.lift_awarded=True;self._event(f,'lifted',pick.LIFT_POINTS)
        if self.hold_start is None:self.hold_start=f.time_s
        self.hold_elapsed=f.time_s-self.hold_start
        ready=self.hold_elapsed+1e-9>=pick.HOLD_S
        if ready and not self.ever_complete:
            self.ever_complete=True;self._event(f,'stable_hold',0)
        return ready

    def summary(self):
        return dict(super().summary(), success=self.ever_complete, failure=None,
                    time_limit_s=None, timeout_owner='room',
                    rule_adapter='openend_remote_recoverable_11_2')


class RecoverablePlace(place.PlaceScorer):
    def __init__(self, episode_id, dt):
        super().__init__(episode_id,dt,spec('10-3'))
        self.recovery=None

    def pause(self):
        self.hold_start=None;self.hold_elapsed=0.;self.released_since=None
        self.contact_started.clear();self.contact_ticks.clear()

    def _fail(self,f,reason):
        # The source's branch still stops this frame; the next frame is recoverable.
        self.recovery=reason;self.hold_start=None;self.hold_elapsed=0.

    def observe_one(self,f,enabled, *, validated=False,final_sample=True):
        if self.finalized or self.terminal:raise RuntimeError('Terminal place component')
        if not validated:self._validate_batch((f,))
        self.last=f
        if not enabled:
            self.pause();return False
        self.recovery=None
        self._deductions(f)
        done=bool(self._observe(f))
        if done and not final_sample:
            # Native batches may reach 1 s then lose support before the control
            # observation ends. Only its final measured sample may finish.
            self.awarded.pop('settle',None)
            self.events=[e for e in self.events if e['kind']!='settle']
            done=False
        if done:self.success=True;self.terminal=True;self.reason='success'
        return done

    def summary(self):
        return dict(super().summary(), recovery=self.recovery,
                    timeout_owner='room', rule_adapter='openend_remote_recoverable_10_3')

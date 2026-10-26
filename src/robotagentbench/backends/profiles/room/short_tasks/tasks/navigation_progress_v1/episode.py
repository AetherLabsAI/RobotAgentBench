"""Phase-aware deadline; changing phase never resets the progress watermark."""
from copy import deepcopy
import math
from short_tasks.common.episode import SingleTaskEpisode
from .rules import VERSION,POINTS,SPACING_M

class NavigationEpisode(SingleTaskEpisode):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.base_maximum=self.maximum;self.timeout_s=60.;self.phase='operation'
    def _snapshot(self,snapshot,*,settlement=False):
        nav=snapshot.get('navigation_progress')
        if not isinstance(nav,dict) or nav.get('version')!=VERSION:raise ValueError('Bound navigation evidence required')
        if nav.get('phase') not in ('operation','navigation') or nav.get('timeout_s')!={'operation':60.,'navigation':120.}[nav['phase']]:
            raise ValueError('Explicit phase deadline required')
        maximum=nav['maximum'];points=nav['points']
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in (maximum,points)) or points>maximum+1e-9:
            raise ValueError('Invalid navigation budget')
        budget=earned=0
        for row in nav['routes'].values():
            n=row['budget_steps'];e=row['earned_steps'];d=row['start_distance_m']
            if type(n) is not int or type(e) is not int or not math.isfinite(d) or d<0 or not 0<=e<=n or n!=math.floor((d+1e-9)/SPACING_M):
                raise ValueError('Route budget not bound to initial distance')
            budget+=n;earned+=e
        if not math.isclose(budget*POINTS,maximum,abs_tol=1e-9) or not math.isclose(earned*POINTS,points,abs_tol=1e-9):
            raise ValueError('Navigation arithmetic mismatch')
        old=self.latest.get('navigation_progress') if self.latest else None
        if old:
            for key,row in old['routes'].items():
                new=nav['routes'].get(key)
                if new is None or any(new[k]!=row[k] for k in ('budget_steps','start_distance_m')) or new['earned_steps']<row['earned_steps']:
                    raise ValueError('A bound route cannot reset or grow its budget')
        expected=self.base_maximum+maximum
        previous=self.maximum;self.maximum=expected
        try:return super()._snapshot(snapshot,settlement=settlement)
        finally:self.maximum=previous
    def observe(self,snapshot,*,simulation_time_s,finalizer=None):
        if self.terminal is not None:raise RuntimeError('A finalized episode cannot accept another action')
        self._time(simulation_time_s)
        if simulation_time_s<=self.last_time:raise ValueError('Evidence must advance the simulation clock')
        scores=self._snapshot(snapshot)
        if finalizer is not None and not callable(finalizer):raise ValueError('Evaluator finalizer must be callable')
        now=float(simulation_time_s);nav=snapshot['navigation_progress']
        expired=now>self.progress_time+self.timeout_s+1e-9
        previous=(self.last_time,self.latest,self.best_net_score,self.progress_time,self.maximum,self.timeout_s,self.phase)
        self.last_time,self.latest=now,deepcopy(snapshot)
        self.maximum=scores['maximum']
        try:
            # Previous phase's expired clock cannot be revived by a late transition.
            if expired:return self.finish('no_new_net_progress_timeout',finalizer=finalizer)
            self.timeout_s=nav['timeout_s'];self.phase=nav['phase']
            if scores['total']>self.best_net_score+1e-9:
                self.best_net_score=float(scores['total']);self.progress_time=now
            if snapshot['success']:return self.finish('success',finalizer=finalizer)
            if now>=self.progress_time+self.timeout_s-1e-9:
                return self.finish('no_new_net_progress_timeout',finalizer=finalizer)
            return self.result()
        except BaseException:
            self.last_time,self.latest,self.best_net_score,self.progress_time,self.maximum,self.timeout_s,self.phase=previous
            raise
    def result(self):
        out=super().result()
        if self.terminal is None and self.latest:
            out['navigation_progress']=deepcopy(self.latest['navigation_progress'])
        return out
    def timer(self):
        return dict(timeout_s=self.timeout_s,phase=self.phase,last_new_progress_s=self.progress_time,
            best_net_score=self.best_net_score,no_new_progress_s=self.last_time-self.progress_time,
            remaining_s=max(0.,self.progress_time+self.timeout_s-self.last_time),
            policy='episode_high_watermark_phase_60_120_v1')

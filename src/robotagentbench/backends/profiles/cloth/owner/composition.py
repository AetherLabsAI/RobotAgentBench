"""Continuous 5+5 cloth composition; original per-cloth numeric rules unchanged."""
from copy import deepcopy
from approved_rules import Binding,Facts,ClothProcessScore


class EligibleComponent(ClothProcessScore):
    def observe(self,f,*,active=True):
        if type(active)is not bool:raise ValueError('Trusted boolean stage eligibility required')
        if active or self.last is None:return super().observe(f)
        if self.termination is not None:raise RuntimeError('Finalized component')
        self._validate(f)
        if self.last==f:return deepcopy(self.cached)
        # Still record real inactive physical evidence and irreversible drops.
        # Do not falsify native contacts or reward actions before their phase.
        if f.on_floor:self.dropped=True
        self.last=f;self.hold_start=None;self.placed=self.released=False
        self.cached=self._summary();return deepcopy(self.cached)


class ClothComposition:
    """Both real objects stay live throughout; no reset at the stage boundary.

    The already existing scene/dispatcher owns control timeouts. This module
    owns no World, physics loop, independent timeout, action or scene mutation.
    """
    def __init__(self,*,task_name,episode_id,old_binding=None,new_binding=None):
        if task_name not in ('collect_tablecloth','spread_tablecloth','long_task_18'):raise ValueError('Wrong cloth family')
        expected={'collect_tablecloth':(True,False),'spread_tablecloth':(False,True),'long_task_18':(True,True)}[task_name]
        if (old_binding is not None,new_binding is not None)!=expected:raise ValueError('Exact required cloth identities must be bound')
        self.task_name,self.episode_id=task_name,episode_id
        self.components={}
        for label,binding in [('old',old_binding),('new',new_binding)]:
            if binding is not None:
                if binding.episode_id!=episode_id or binding.task_name!=('collect_tablecloth'if label=='old'else'spread_tablecloth'):raise ValueError('Wrong episode/task binding')
                self.components[label]=EligibleComponent(binding)
        if old_binding is not None and new_binding is not None and old_binding.target_path==new_binding.target_path:
            raise ValueError('Replacing a cloth requires two distinct physical objects')
        self.new_enabled=task_name=='spread_tablecloth';self.latest=self.terminal=None

    def observe(self,facts):
        if self.terminal is not None:raise RuntimeError('Episode finalized')
        if set(facts)!=set(self.components)or any(not isinstance(x,Facts)for x in facts.values()):raise ValueError('All exact measured cloth facts required')
        if len({(f.sample_index,f.time_s)for f in facts.values()})!=1:raise ValueError('Both cloths must use the same native physical frame')
        rows={}
        if 'old'in self.components:
            rows['old']=self.components['old'].observe(facts['old'])
            if rows['old']['success']and not self.new_enabled and 'new'in facts:
                clean=facts['new']
                # Enforce the approved collect-before-spread order. A clean
                # cloth already carried away cannot inherit early milestones
                # merely because the old cloth happens to finish this frame.
                self.new_enabled=bool(clean.source_supported and not clean.held and not clean.finger_contact
                    and not clean.whole_clear_source and not clean.on_floor)
        if 'new'in self.components:rows['new']=self.components['new'].observe(facts['new'],active=self.new_enabled)
        self.latest=self._summary(rows);return deepcopy(self.latest)

    def _summary(self,rows):
        scores={k:sum(r['scores'][k]for r in rows.values())for k in ('process','current_final','deductions','total','maximum')}
        return dict(task_name=self.task_name,episode_id=self.episode_id,scores=scores,
            success=all(r['success']for r in rows.values()),components=deepcopy(rows),new_phase_enabled=self.new_enabled,
            evidence_step=next(iter(rows.values()))['evidence_step'],evidence_time_s=next(iter(rows.values()))['evidence_time_s'],
            shared_no_progress_timeout_s=120,owner='scene_session_shared_progress_clock',
            native_admission=False,termination=None)

    def finalize(self,reason):
        if self.terminal is not None:
            if self.terminal['termination']!=reason:raise ValueError('Conflicting settlement')
            return deepcopy(self.terminal)
        if self.latest is None:raise ValueError('No measured evidence')
        if reason not in ('success','client_closed','no_new_net_progress_timeout'):raise ValueError('Unsupported termination')
        if (reason=='success')!=self.latest['success']:raise ValueError('Termination disagrees with physical state')
        settled={}
        for label,component in self.components.items():
            # Do not finalize a still-successful old component as failed merely
            # because another component is unfinished. The aggregate alone
            # determines whole-scenario success; each keeps its current result.
            settled[label]=component.finalize('success'if component.released else reason)
        self.terminal=self._summary(settled);self.terminal['termination']=reason
        return deepcopy(self.terminal)

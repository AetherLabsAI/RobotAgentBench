"""Atomic whole-group scoring; historical points are not current completion."""
from copy import deepcopy
from .scoring import Binding, Facts, ClothClear


class ClothClearGroup:
    def __init__(self, bindings):
        bindings = tuple(bindings)
        if not bindings or any(not isinstance(b, Binding) for b in bindings):
            raise ValueError('Explicit nonempty group bindings required')
        if (len({b.target_path for b in bindings}) != len(bindings)
                or len({b.instance_id for b in bindings}) != len(bindings)
                or len({(b.episode_id, b.cloth_path, b.table_path) for b in bindings}) != 1):
            raise ValueError('Unique targets in one episode/support group required')
        self.items = {b.target_path: ClothClear(b, terminate_on_success=False) for b in bindings}
        self.ended = self.finalized = False
        self.reason = None

    def observe(self, batch, *, final_sample=True):
        if self.ended:
            raise RuntimeError('Group ended')
        batch = tuple(batch)
        if any(not isinstance(f, Facts) for f in batch):
            raise ValueError('Typed group evidence required')
        if (len(batch) != len(self.items) or {f.binding.target_path for f in batch} != set(self.items)
                or len({(f.sample_index, f.time_s, f.physics_dt_s) for f in batch}) != 1):
            raise ValueError('All targets at the same physical boundary required')
        # Roll back the entire group if even the final member has invalid facts.
        before = deepcopy(self.items)
        try:
            for f in batch:
                self.items[f.binding.target_path].observe(f, final_sample=final_sample)
        except Exception:
            self.items = before
            raise
        return self.summary()

    def suspend_substep(self, now, dt):
        if self.ended:
            return
        before = deepcopy(self.items)
        try:
            for s in self.items.values():
                s.suspend_substep(now, dt)
        except Exception:
            self.items = before
            raise

    def pause(self):
        if not self.ended:
            for s in self.items.values():
                s.hold_start = None
                s.current_placed = False
                s.awards.discard('place')

    def finalize(self, reason):
        if any(s.last is None for s in self.items.values()):
            raise ValueError('No initial evidence')
        if self.finalized:
            if reason != self.reason:
                raise ValueError('Conflicting group settlement')
            return self.summary()
        if reason == 'success':
            if self.reason != 'success':
                raise ValueError('No observed whole-group success')
        elif reason not in ('no_score_increase_timeout', 'client_closed') or self.reason == 'success':
            raise ValueError('Only shared termination is allowed')
        self.reason, self.ended, self.finalized = reason, True, True
        return self.summary()

    def summary(self):
        items = {p: s.summary() for p, s in self.items.items()}
        score = sum(s['score'] for s in items.values())
        return dict(score=score, raw_score=score, maximum_score=3*len(items), items=items,
                    success=all(s['current_placed'] for s in items.values()), ended=self.ended, reason=self.reason,
                    operation_complete=all(s['operation_complete'] for s in items.values()),
                    finalized=self.finalized, failure=None, timeout_owner='room',
                    evidence_status='external_facts_not_native_verified')

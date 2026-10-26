"""Read-only simulation-time deadline; never changes score, stage, or physics.

Completion receipts must come from the trusted environment actor, not model
arguments. Episode-local identifiers prevent a retired actor refreshing every
frame. Lighting and robot motion cannot produce such receipts.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

TASK_EVENTS = {
    'stock_wooden_tray_lighting': frozenset(),
    'stock_wooden_tray_interference': frozenset({'shelf_relocation'}),
    'coffee_bag_interference': frozenset({'bag_relocation'}),
    'coffee_machine_push_interference': frozenset({'machine_push'}),
    'coffee_bag_then_machine_interference': frozenset({'bag_relocation', 'machine_push'}),
    'coffee_button_lighting_interference': frozenset(),
}
TIMEOUT_S = 120.0
EPS = 1e-6


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative simulation time/score required')
    return float(value)


@dataclass(frozen=True)
class Completion:
    kind: str
    time_s: float
    completed: bool
    released: bool
    supported: bool
    actor_retired: bool


class Deadline:
    def __init__(self, task, start_time_s, initial_score=0.0):
        if task not in TASK_EVENTS:
            raise ValueError('Timing revision is limited to the approved six tasks')
        self.task = task
        self.start = self.last = self.refresh = finite(start_time_s)
        self.best = finite(initial_score)
        self.seen = {}
        self.events = []
        self.expired = False

    def observe(self, time_s, score, completions=()):
        now, score = finite(time_s), finite(score)
        if self.expired:
            raise RuntimeError('Expired episodes cannot be revived')
        if now <= self.last:
            raise ValueError('Exactly one advancing control observation required')
        # Validate the whole batch before mutating any timing state.
        fresh = {}
        for event in completions:
            if not isinstance(event, Completion) or event.kind not in TASK_EVENTS[self.task]:
                raise ValueError('Undeclared manual interference event')
            at = finite(event.time_s)
            if not all(v is True for v in (event.completed, event.released,
                                          event.supported, event.actor_retired)):
                raise ValueError('Incomplete/unreleased/unsupported interference')
            if at < self.start - EPS or at > now + EPS:
                raise ValueError('Interference outside this episode/observation')
            previous = self.seen.get(event.kind, fresh.get(event.kind))
            if previous is not None:
                if previous != event:
                    raise ValueError('Conflicting duplicate completion receipt')
                continue
            if at <= self.last - EPS:
                raise ValueError('Stale completion cannot refresh time retroactively')
            fresh[event.kind] = event
        # Event timestamps are physical substep times, not the later control
        # frame. Process in time order so even a skipped interval cannot revive
        # an expired deadline using a newer event.
        refresh = self.refresh
        accepted = []
        for event in sorted(fresh.values(), key=lambda e: e.time_s):
            if event.time_s > refresh + TIMEOUT_S + EPS:
                break
            refresh = max(refresh, event.time_s)
            accepted.append(event)
        expired_before = now > refresh + TIMEOUT_S + EPS
        if not expired_before and score > self.best:
            self.best = score
            refresh = now
            self.events.append(dict(reason='new_score', time_s=now, score=score))
        for event in accepted:
            self.seen[event.kind] = event
            self.events.append(dict(reason='manual_interference_completed',
                                    kind=event.kind, time_s=event.time_s))
        self.last, self.refresh = now, refresh
        self.expired = expired_before or now >= refresh + TIMEOUT_S - EPS
        return self.expired

    def snapshot(self):
        return dict(policy='robustness_no_progress_120_manual_completion_v1',
                    timeout_s=TIMEOUT_S, start_time_s=self.start,
                    evidence_time_s=self.last, last_refresh_time_s=self.refresh,
                    best_net_score=self.best,
                    no_new_progress_s=self.last-self.refresh,
                    remaining_s=max(0.0, self.refresh+TIMEOUT_S-self.last),
                    expired=self.expired, events=deepcopy(self.events),
                    simulation_clock_reset=False, score_injected=False)

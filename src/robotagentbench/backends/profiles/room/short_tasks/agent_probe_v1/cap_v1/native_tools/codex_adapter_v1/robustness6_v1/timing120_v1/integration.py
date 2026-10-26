"""Timing-only adapters; read existing native evidence, never command bodies."""
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

from .policy import Completion, Deadline, TASK_EVENTS, TIMEOUT_S

POLICY = 'robustness_no_progress_120_manual_completion_v1'


def attach(plan):
    if plan['task_name'] not in TASK_EVENTS:
        return
    plan['no_progress_timeout_s'] = TIMEOUT_S
    plan['robustness_timing_policy'] = POLICY
    if plan.get('execution_mode') == 'atomic_coffee_robustness_v1':
        plan['score_policy']['time_limit_s'] = None


def configure_process(process):
    # A per-episode immutable Spec copy: all points, success/failure predicates,
    # deductions, geometric thresholds, stage transitions and holds stay intact.
    process.spec = replace(process.spec, stage_budget_s=MappingProxyType({}), time_limit_s=None)
    process._used = {}
    process._charging = None


def shelf_completion(actor):
    if actor is None:
        return ()
    if actor.closed or not actor.armed:
        return ()
    if actor.audit.phase.value != 'done':
        return ()
    rows = [r for r in actor.backend.evidence['events'] if r['phase'] == 'done']
    if len(rows) != 1:
        raise ValueError('One native shelf completion record required')
    row = rows[0]
    if not (row['contacts_complete'] and row['destination_valid'] and row['actor_clear']):
        raise ValueError('Unverified shelf relocation completion')
    return (Completion('shelf_relocation', row['simulation_time_s'], True,
                       row['human_released'], row['target_supported'],
                       actor.backend.human.appearance_state == 'retired'),)


def bag_completion(actor):
    if not (actor.completed and actor.retired):
        return None
    row = actor.completion_evidence
    if not row or not row['acquired'] or row['constraint_active'] or not row['in_training_domain']:
        raise ValueError('Unverified native bag relocation completion')
    return Completion('bag_relocation', row['time_s'], True, row['released'],
                      row['supported'], actor.retired)


def machine_completion(actor, reader, now):
    if not (actor.completed and actor.retired):
        return None
    # Reuse source contact normals/threshold and the same substep stream. A
    # retired palm alone is not evidence that the machine is back on the table.
    from evaluation.tasks.coffee_machine.specs import MACHINE_BODY, COUNTER
    from evaluation.tasks.coffee_machine.isaac_evidence import UP_NORMAL_Z, under
    supported = False
    released = True
    for paths, samples in reader.contacts.pairs():
        i = 0 if under(paths[0], MACHINE_BODY) else 1 if under(paths[1], MACHINE_BODY) else None
        if i is None:
            continue
        other = paths[1-i]
        if under(other, actor.human.root):
            released = False
        sign = 1 if i == 0 else -1
        if under(other, COUNTER) and any(sign*n[2] > UP_NORMAL_Z for n, _ in samples):
            supported = True
    if not (supported and released):
        return None
    return Completion('machine_push', now, True, released, supported, True)


class TimedWorker:
    def configure_timing(self, variant):
        self.timing_task = {
            'bag_only': 'coffee_bag_interference',
            'machine_only': 'coffee_machine_push_interference',
            'bag_then_machine': 'coffee_bag_then_machine_interference',
            'button_lighting': 'coffee_button_lighting_interference',
        }[variant]
        self.timing_variant = variant
        self.timing_deadline = None
        self.timing_completions = {}
        configure_process(self.process)

    def capture_timing_completions(self):
        if self.timing_deadline is None or self.timing_variant == 'button_lighting':
            return
        now = float(self.world.current_time)
        if self.reader.latest['time_s'] != now:
            raise ValueError('Stale native interference evidence')
        actors = ({'bag_relocation': self.actor.bag, 'machine_push': self.actor.push}
                  if self.timing_variant == 'bag_then_machine' else
                  {'bag_relocation' if self.timing_variant == 'bag_only' else 'machine_push': self.actor})
        for kind, actor in actors.items():
            if kind in self.timing_completions:
                continue
            event = (bag_completion(actor) if kind == 'bag_relocation'
                     else machine_completion(actor, self.reader, now))
            if event is not None:
                self.timing_completions[kind] = event

    def evaluate(self, evidence):
        from arena_api.task_runtime import TaskScore
        result = super().evaluate(evidence)
        if self.timing_deadline is None:
            if evidence.step_index != 0:
                raise ValueError('Timing must start at admitted initial evidence')
            self.timing_deadline = Deadline(self.timing_task, evidence.time_s, result.score)
        else:
            self.timing_deadline.observe(evidence.time_s, result.score, self.timing_completions.values())
        return TaskScore(result.score, result.maximum_score,
                         dict(result.details, robustness_timer=self.timing_deadline.snapshot()))

    def check_failure(self, evidence, score):
        original = super().check_failure(evidence, score)
        if original:
            return original
        if super().check_success(evidence, score):
            return None
        return 'no_new_net_progress_timeout' if self.timing_deadline.expired else None


def lifecycle_type():
    from arena_api.task_lifecycle import TaskLifecycle
    from arena_api.episode_clock import ControlClock

    class SimulationClock(ControlClock):
        """Preserve source 20-Hz integrity checks, with no absolute episode cap."""
        def __init__(self, dt):
            super().__init__(1, dt)
            self.max_steps = self.limit_s = None

        @property
        def expired(self):
            return False  # The task's progress deadline is the sole time limit.

    class RobustLifecycle(TaskLifecycle):
        def __init__(self, task, context, registration, dt):
            super().__init__(task, context, registration, dt)
            self.clock = SimulationClock(dt)

    return RobustLifecycle


def shelf_episode_type():
    from short_tasks.common.episode import SingleTaskEpisode

    class RobustShelfEpisode(SingleTaskEpisode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.deadline = Deadline(self.task_name, self.start_time)
            self.actor = None

        def bind_actor(self, actor):
            self.actor = actor

        def observe(self, snapshot, *, simulation_time_s, finalizer=None):
            if self.terminal is not None:
                raise RuntimeError('Finalized episode cannot accept actions')
            scores = self._snapshot(snapshot)
            previous = deepcopy((self.last_time, self.latest, self.best_net_score,
                                 self.progress_time, self.deadline))
            try:
                expired = self.deadline.observe(simulation_time_s, scores['total'], shelf_completion(self.actor))
                self.last_time, self.latest = float(simulation_time_s), deepcopy(snapshot)
                self.best_net_score, self.progress_time = self.deadline.best, self.deadline.refresh
                if snapshot['success']:
                    return self.finish('success', finalizer=finalizer)
                if expired:
                    return self.finish('no_new_net_progress_timeout', finalizer=finalizer)
                return self.result()
            except BaseException:
                (self.last_time, self.latest, self.best_net_score,
                 self.progress_time, self.deadline) = previous
                raise

        def timer(self):
            return self.deadline.snapshot()

        def result(self):
            result = super().result()
            result['robustness_timer'] = self.timer()
            return result

    return RobustShelfEpisode

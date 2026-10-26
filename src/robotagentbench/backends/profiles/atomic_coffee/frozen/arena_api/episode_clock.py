"""Shared simulation-time budget; no task predicates, wall-clock or Isaac imports."""
import math


class ControlClock:
    schema = 'arena.episode.time_limit.v1'

    def __init__(self, max_steps, control_dt_s):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('Expected positive integer control-step budget')
        if isinstance(control_dt_s, bool) or not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError('Expected finite positive control dt')
        self.max_steps, self.control_dt_s = max_steps, float(control_dt_s)
        self.limit_s = max_steps * self.control_dt_s
        self.start_s = self.last_s = None
        self.steps = 0

    @staticmethod
    def _time(value):
        if isinstance(value, bool):
            raise ValueError('Invalid simulation time')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid simulation time')
        return value

    def start(self, time_s):
        if self.start_s is not None:
            raise RuntimeError('Episode clock already started')
        self.start_s = self.last_s = self._time(time_s)
        return self.summary()

    def require_step_allowed(self):
        if self.start_s is None:
            raise RuntimeError('Initial acceptance must precede task timing')
        if self.expired:
            raise RuntimeError('Episode time limit reached; no further action allowed')

    def observe(self, time_s):
        self.require_step_allowed()
        now = self._time(time_s)
        # Check consecutive intervals, not accumulated float32 dt roundoff.
        if not math.isclose(now-self.last_s, self.control_dt_s, rel_tol=0, abs_tol=1e-6):
            raise ValueError('Missing, repeated, or misaligned control-time observation')
        self.steps += 1
        self.last_s = now
        return self.summary()

    @property
    def expired(self):
        return self.steps >= self.max_steps

    def summary(self):
        return dict(schema=self.schema, time_basis='simulation_time',
                    limit_sim_s=self.limit_s, control_hz=1/self.control_dt_s,
                    max_control_steps=self.max_steps, executed_control_steps=self.steps,
                    start_sim_time_s=self.start_s, end_sim_time_s=self.last_s,
                    elapsed_sim_s=None if self.start_s is None else self.last_s-self.start_s,
                    time_limit_reached=self.expired, terminate_episode=self.expired,
                    termination_reason='time_limit' if self.expired else None,
                    initial_settle_included=False, hold_tail_included=True,
                    success_judgment_included=False)

"""Deterministic illumination schedule driven only by simulation time."""
import math
import random

BRIGHT_MULTIPLIER = 1.0
DARK_MULTIPLIER = 0.25
INTERVAL_S = (2.0, 3.0)


class LightingCycle:
    def __init__(self, seed):
        if type(seed) is not int:
            raise ValueError('Explicit integer episode seed required')
        self.seed = seed
        self.rng = random.Random(seed)
        self.started = False
        self.bright = True
        self.multiplier = BRIGHT_MULTIPLIER
        self.last_time = self.deadline = None
        self.transitions = []

    def arm(self, now):
        if self.started:
            raise RuntimeError('Lighting cycle starts once after initial admission')
        self._time(now)
        self.started = True
        self.last_time = float(now)
        self.deadline = float(now) + self.rng.uniform(*INTERVAL_S)

    @staticmethod
    def _time(now):
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
            raise ValueError('Finite nonnegative simulation time required')

    def tick(self, now):
        self._time(now)
        now = float(now)
        if not self.started:
            return []
        if now < self.last_time:
            raise ValueError('Lighting simulation time moved backwards')
        self.last_time = now
        changed = []
        while now >= self.deadline:
            scheduled = self.deadline
            self.bright = not self.bright
            self.multiplier = BRIGHT_MULTIPLIER if self.bright else DARK_MULTIPLIER
            self.deadline += self.rng.uniform(*INTERVAL_S)
            row = dict(scheduled_time_s=scheduled, observed_time_s=now,
                       state='bright' if self.bright else 'dark',
                       multiplier=self.multiplier, next_change_s=self.deadline)
            self.transitions.append(row)
            changed.append(dict(row))
        return changed

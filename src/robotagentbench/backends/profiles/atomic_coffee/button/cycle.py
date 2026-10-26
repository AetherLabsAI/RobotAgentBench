"""Seeded schedule; only admitted simulation time advances the disturbance."""
import math
import random

MULTIPLIER = (0.15, 1.8)
TEMPERATURE_K = (2700.0, 7500.0)
INTERVAL_S = (1.0, 3.0)
FLASH_S = (0.2, 0.5)
FLASH_MULTIPLIER = (5.0, 10.0)  # User-confirmed 2026-10-14; environment remains 0.15–1.8.
MODES = ('ambient', 'side', 'camera_flash')


def valid_time(now):
    if isinstance(now, bool) or not isinstance(now, (float, int)) or not math.isfinite(now) or now < 0:
        raise ValueError('Finite nonnegative simulation time required')
    return float(now)


class LightingCycle:
    def __init__(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Explicit nonnegative integer seed required')
        self.seed = seed
        self.rng = random.Random(seed)
        # Independent stream: strengthening a flash does not alter any existing
        # ambient samples, colors, event deadlines or flash duration.
        self.flash_rng = random.Random(str(seed) + ':camera_flash')
        self.started = self.closed = False
        self.last_time = self.deadline = self.flash_end = None
        self.index = 0
        self.state = dict(mode='normal', multiplier=1.0, temperature_k=None, side=None,
                          flash=False, flash_multiplier=None)
        self.events = []

    def arm(self, now):
        now = valid_time(now)
        if self.started or self.closed:
            raise RuntimeError('Cycle can only be armed once after initial acceptance')
        self.started = True
        self.last_time = now
        self.deadline = now + self.rng.uniform(*INTERVAL_S)

    def tick(self, now):
        now = valid_time(now)
        if self.closed:
            raise RuntimeError('Cycle closed')
        if not self.started:
            return []
        if now < self.last_time:
            raise ValueError('Simulation clock moved backwards')
        self.last_time = now
        changed = []
        # Consume all deadlines in chronological order, independent of call cadence.
        while True:
            end = self.flash_end if self.flash_end is not None else math.inf
            when = min(self.deadline, end)
            if when > now:
                break
            if end <= self.deadline:
                self.flash_end = None
                self.state = dict(self.state, mode='ambient', flash=False, side=None, flash_multiplier=None)
                kind = 'flash_ended'
            else:
                mode = MODES[self.index % len(MODES)]
                self.index += 1
                self.state = dict(mode=mode, multiplier=self.rng.uniform(*MULTIPLIER),
                                  temperature_k=self.rng.uniform(*TEMPERATURE_K),
                                  side=self.rng.choice((-1, 1)), flash=mode == 'camera_flash',
                                  flash_multiplier=None)
                if self.state['flash']:
                    self.state['flash_multiplier'] = self.flash_rng.uniform(*FLASH_MULTIPLIER)
                    self.flash_end = when + self.rng.uniform(*FLASH_S)
                self.deadline = when + self.rng.uniform(*INTERVAL_S)
                kind = 'light_changed'
            row = dict(kind=kind, scheduled_time_s=when, observed_time_s=now,
                       state=dict(self.state), next_change_s=self.deadline, flash_end_s=self.flash_end)
            self.events.append(row)
            changed.append(row)
        return changed

    def close(self):
        self.closed = True

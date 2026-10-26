"""Sample-driven camera capture; initialization is separate from steady sampling.

This module has no Isaac imports so freshness and timing gates can be tested
without a GPU. Callbacks read real render-product stamps in the worker.
"""
import math
from fractions import Fraction


class SampledCameraCapture:
    def __init__(self, world, read_frames, ready, read_stamps, *, capture_frame,
                 warmup_limit=60, warmup_minimum=4):
        if not 1 <= warmup_minimum <= warmup_limit:
            raise ValueError('Invalid camera warmup budget')
        self.world = world
        self.read_frames = read_frames
        self.ready = ready
        self.read_stamps = read_stamps
        self.capture_frame = capture_frame  # must wait for rendering completion, not only enqueue an update
        self.warmup_limit = warmup_limit
        self.warmup_minimum = warmup_minimum
        self.previous_reference = None
        self.last_evidence = None
        self.failed = False

    def _validate(self, frames, stamps, expected_time):
        if not self.ready(frames) or not frames or set(stamps) != set(frames):
            raise RuntimeError('Incomplete current camera observation')
        references = set()
        for name, stamp in stamps.items():
            numerator, denominator = stamp['numerator'], stamp['denominator']
            if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
                raise RuntimeError('Invalid render-product reference time')
            references.add(Fraction(numerator, denominator))
            actual = float(stamp['simulation_time_s'])
            if not math.isfinite(actual) or abs(actual - expected_time) > 1e-6:
                raise RuntimeError(f'Stale camera {name}: rendered {actual}, expected {expected_time}')
        if len(references) != 1:
            raise RuntimeError('Camera products belong to different render frames')
        reference = next(iter(references))
        if self.previous_reference is not None and reference <= self.previous_reference:
            raise RuntimeError('Camera render frame did not advance')
        return reference

    def capture(self):
        if self.failed:
            raise RuntimeError('Camera capture is failed; reset worker before continuing')
        expected_time = float(self.world.current_time)
        if not math.isfinite(expected_time):
            raise RuntimeError('Invalid physics clock')
        first = self.previous_reference is None
        budget = self.warmup_limit if first else 1
        try:
            for updates in range(1, budget + 1):
                self.capture_frame()
                after = float(self.world.current_time)
                if not math.isfinite(after) or abs(after - expected_time) > 1e-10:
                    raise RuntimeError('Camera rendering unexpectedly advanced physics')
                if first and updates < self.warmup_minimum:
                    continue
                try:
                    frames = self.read_frames()
                    stamps = self.read_stamps()
                    reference = self._validate(frames, stamps, expected_time)
                except (KeyError, RuntimeError, ValueError, TypeError):
                    if first and updates < budget:
                        continue
                    raise
                self.previous_reference = reference
                self.last_evidence = dict(simulation_time_s=expected_time,
                    capture_requests=updates, initialization_warmup=first, cameras=stamps)
                return frames
            raise RuntimeError('Camera initialization exhausted its render budget')
        except BaseException:
            self.failed = True
            raise

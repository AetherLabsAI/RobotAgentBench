"""Task 10.2's approved 3-point arithmetic, separate from Isaac detection.

Only an environment-owned reader may supply these facts. They are not model
outputs or a public API schema. A drop flag covers the entire interval since
the preceding observation (including a fall and recovery between samples).
No distances, contact thresholds, hold durations or task termination are
invented here. isaac_detection.py supplies the task-local physical facts.
"""
from dataclasses import asdict, dataclass
import math

TASK_ID = '10.2'
SCHEMA = 'arena.dining_place.process.v3'
TARGET_POINTS = 3
PLATE_DROP_PENALTY = 0.5
PAYLOAD_DROP_PENALTY = 1.0
MINIMUM_SCORE = 0


@dataclass(frozen=True)
class ProcessFrame:
    episode_id: str
    sample_index: int
    simulation_time_s: float
    plate_in_target_region: bool
    plate_supported_by_target: bool
    plate_dropped_since_last_sample: bool
    payload_dropped_since_last_sample: bool
    measurements_complete: bool

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError('An episode identity is required')
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError('sample_index must be a nonnegative integer')
        if (isinstance(self.simulation_time_s, bool)
                or not isinstance(self.simulation_time_s, (int, float))
                or not math.isfinite(self.simulation_time_s) or self.simulation_time_s < 0):
            raise ValueError('A finite nonnegative simulation time is required')
        for field in ('plate_in_target_region', 'plate_supported_by_target',
                      'plate_dropped_since_last_sample', 'payload_dropped_since_last_sample',
                      'measurements_complete'):
            if type(getattr(self, field)) is not bool:
                raise ValueError('Missing/invalid physical fact: '+field)
        if not self.measurements_complete:
            raise ValueError('Incomplete evidence must not be treated as no drop/no contact')


@dataclass(frozen=True)
class ScoreResult:
    episode_id: str
    final_sample_index: int
    target_reached: bool
    target_points: int
    plate_drop_penalty: float
    payload_drop_penalty: float
    raw_score: float
    score: float
    minimum_score: int
    maximum_score: int = TARGET_POINTS
    schema: str = SCHEMA

    def to_dict(self):
        return asdict(self)


class ProcessScorer:
    """Latch each kind of drop once; award destination points only at end.

Observe never terminates an episode, including when a drop lands on target.
The caller explicitly ends the episode with finalize(). Use a new instance
after every reset. sample_index enumerates reader observations, NOT necessarily
individual physical steps; interval drop evidence must cover intervening steps.
"""
    def __init__(self, episode_id):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('An episode identity is required')
        self.episode_id = episode_id
        self._last = None
        self._plate_dropped = False
        self._payload_dropped = False
        self._result = None

    def observe(self, frame: ProcessFrame):
        if not isinstance(frame, ProcessFrame):
            raise TypeError('Expected an environment-owned ProcessFrame')
        if self._result is not None:
            raise RuntimeError('Finalized episode; create a new scorer after reset')
        expected_index = 0 if self._last is None else self._last.sample_index+1
        if frame.episode_id != self.episode_id or frame.sample_index != expected_index:
            raise ValueError('Wrong episode or missing/duplicate/out-of-order observation')
        if self._last is not None and frame.simulation_time_s <= self._last.simulation_time_s:
            raise ValueError('Simulation time must increase within an episode')
        # Validate everything before changing history.
        self._plate_dropped |= frame.plate_dropped_since_last_sample
        self._payload_dropped |= frame.payload_dropped_since_last_sample
        self._last = frame

    def finalize(self):
        """User-approved 0..3 points; retain raw deductions for audit.

        This does not determine when an episode should end or a success flag.
        """
        if self._result is not None:
            return self._result
        self._result = self.summary()
        return self._result

    def summary(self):
        """Current score without closing the episode or erasing drop history."""
        if self._result is not None:
            return self._result
        if self._last is None:
            raise ValueError('No final observation; cannot score an empty episode')
        reached = self._last.plate_in_target_region and self._last.plate_supported_by_target
        points = TARGET_POINTS if reached else 0
        plate_penalty = PLATE_DROP_PENALTY if self._plate_dropped else 0
        payload_penalty = PAYLOAD_DROP_PENALTY if self._payload_dropped else 0
        raw = points-plate_penalty-payload_penalty
        score = max(MINIMUM_SCORE, raw)
        return ScoreResult(self.episode_id, self._last.sample_index, reached,
                                   points, plate_penalty, payload_penalty, raw, score, MINIMUM_SCORE)

"""Approved 14-1 rules; pure measured-fact scoring, no simulator control.

Facts MUST come from a validated physics reader, not commands, source success
flags, a target centre inside an AABB, or disappearance of contact reports.
The runtime supplies the approved indirect soft-tote geometry/motion evidence.
"""
from dataclasses import asdict, dataclass, replace
import math

API_ID = '14-1'
TARGET = '/World/Assets/Umbrella'
DESTINATION = '/World/Task73/Tote'
PLACE_POINTS = 2
STABLE_POINTS = 2
HOLD_SECONDS = 1.0
TIME_LIMIT_SECONDS = 20.0
MAXIMUM_SCORE = PLACE_POINTS + STABLE_POINTS
_TIME_EPS = 1e-9  # Arithmetic roundoff only, not a physical tolerance.


@dataclass(frozen=True)
class PlacementFacts:
    step_index: int
    time_s: float  # Actual simulation clock; reset/settle excluded via start_s.
    target_path: str
    destination_path: str
    inside_cavity: bool  # Inserted into actual deformed cavity; top may protrude.
    supported_by_basket: bool  # Approved indirect geometry/motion support, not measured soft force.
    released: bool  # Measured release; a commanded open target is insufficient.
    held_by_robot: bool  # Either hand, including finger links, not just end pads.
    supported_elsewhere: bool  # E.g. countertop/floor still carrying the target.
    rim_only: bool  # Merely lying/hanging on outer rim, not inserted placement.
    uninterrupted_since_previous: bool  # All placement conditions throughout interval.

    @property
    def placed(self):
        return (self.inside_cavity and self.supported_by_basket and self.released
                and not self.held_by_robot and not self.supported_elsewhere
                and not self.rim_only)


@dataclass(frozen=True)
class PlacementResult:
    step_index: int
    time_s: float
    elapsed_s: float
    place_awarded: bool
    stable_awarded: bool
    placed_now: bool
    continuous_s: float
    score: int
    success: bool
    reason: str | None
    new_events: tuple[str, ...]
    maximum_score: int = MAXIMUM_SCORE

    def to_dict(self):
        return asdict(self)


class BasketPlacementScorer:
    """One episode, sequential control evidence. Initial frame must be held.

    Same-frame identical retries return the cached result. Contradictory,
    missing, stale or skipped evidence fails closed and does not advance state.
    One-time placement credit survives a recoverable drop; continuous time does
    not. Full success terminates immediately, before timeout on the same frame.
    """
    def __init__(self, *, public_clock=False, target=TARGET):
        # In the registered runtime the shared ControlClock owns deadlines,
        # avoiding a second accumulated float32 time check. Pure standalone
        # scoring retains its original strict timestamp deadline.
        self._public_clock = public_clock
        self.target = target
        self._last = self._result = None
        self._start = self._since = None
        self._place = self._stable = False
        self._closed = False

    def _validate(self, facts):
        if not isinstance(facts, PlacementFacts):
            raise ValueError('Explicit measured PlacementFacts required')
        if type(facts.step_index) is not int or facts.step_index < 0:
            raise ValueError('Invalid step index')
        if (type(facts.time_s) not in (int, float)
                or not math.isfinite(facts.time_s) or facts.time_s < 0):
            raise ValueError('Invalid simulation time')
        if facts.target_path != self.target or facts.destination_path != DESTINATION:
            raise ValueError('Wrong target or destination instance')
        for name in ('inside_cavity', 'supported_by_basket', 'released',
                     'held_by_robot', 'supported_elsewhere', 'rim_only',
                     'uninterrupted_since_previous'):
            if type(getattr(facts, name)) is not bool:
                raise ValueError('Missing or non-boolean physical evidence: ' + name)
        if facts.released and facts.held_by_robot:
            raise ValueError('Contradictory release and holding evidence')

    def observe(self, facts):
        self._validate(facts)
        if self._last is not None and facts == self._last:
            return self._result
        if self._closed or (self._result and self._result.reason):
            raise ValueError('Episode already ended')
        if self._last is None:
            if (facts.step_index != 0 or not facts.held_by_robot or facts.released
                    or facts.uninterrupted_since_previous):
                raise ValueError('Start with validated held initial frame at step zero')
            start = facts.time_s
        else:
            start = self._start
            if (facts.step_index != self._last.step_index + 1
                    or facts.time_s <= self._last.time_s):
                raise ValueError('Missing, repeated, or stale control frame')
            if (facts.uninterrupted_since_previous
                    and not (facts.placed and self._last.placed)):
                raise ValueError('Contradictory continuous-interval evidence')
        elapsed = facts.time_s - start
        if not self._public_clock and elapsed > TIME_LIMIT_SECONDS + _TIME_EPS:
            raise ValueError('Evidence exceeds task deadline; do not award late actions')

        # All checks above are transactional: errors cannot change counters.
        self._start = start
        events = []
        if facts.placed:
            if not self._place:
                self._place = True
                events.append('placed_and_released')
            if self._since is None or not facts.uninterrupted_since_previous:
                self._since = facts.time_s
            continuous = facts.time_s - self._since
            if continuous + _TIME_EPS >= HOLD_SECONDS and not self._stable:
                self._stable = True
                events.append('stable_in_basket')
        else:
            self._since = None
            continuous = 0.0
        success = facts.placed and continuous + _TIME_EPS >= HOLD_SECONDS
        reason = ('success' if success else
                  'time_limit' if not self._public_clock and elapsed + _TIME_EPS >= TIME_LIMIT_SECONDS else None)
        self._result = PlacementResult(facts.step_index, facts.time_s, elapsed,
            self._place, self._stable, facts.placed, continuous,
            PLACE_POINTS * self._place + STABLE_POINTS * self._stable,
            success, reason, tuple(events))
        self._last = facts
        return self._result

    def finalize(self, reason):
        """No extra rewards/penalties. Source exhaustion is NOT success."""
        if self._result is None:
            raise ValueError('No validated initial evidence')
        if reason not in ('success', 'time_limit', 'client_closed', 'trajectory_exhausted'):
            raise ValueError('Unknown normal finalization reason')
        if reason == 'success' and not self._result.success:
            raise ValueError('Cannot force success at finalization')
        if not self._closed:
            self._result = replace(self._result, reason=self._result.reason or reason)
            self._closed = True
        return self._result

"""Unscored microwave-put progress from trusted, current physics facts.

No Isaac control, geometry thresholds, reward values or task registration here.
The future reader must measure these predicates for the bound payload instance;
gripper commands, object proximity and model statements are not evidence.
"""
from dataclasses import asdict, dataclass
import math

from arena_api.task_runtime import TaskEvidence
from arena_api.episode_clock import ControlClock
from .variants import variant_for


@dataclass(frozen=True)
class PutFacts:
    payload_variant: str
    payload_instance: str
    door_open: bool
    door_closed: bool
    payload_inside: bool
    payload_supported: bool
    held_by_robot: bool
    held_by_right: bool
    # Physical reader can distinguish "not proven held" from "proven released".
    # None preserves old synthetic callers; production supplies an explicit bit.
    released: bool | None = None


@dataclass(frozen=True)
class StageEvent:
    stage: str
    step_index: int
    time_s: float


@dataclass(frozen=True)
class PutProgress:
    episode_id: str
    task_id: str
    step_index: int
    time_s: float
    events: tuple[StageEvent, ...]
    final_candidate: bool
    final_hold_s: float

    def to_dict(self):
        return asdict(self)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class PutProgressTracker:
    """One episode's ordered milestones; NOT a registered WorkerTask/scorer.

    control_dt_s must come from the existing controller contract, not a new
    task frequency. Missing frames are invalid evidence, not inferred success.
    Closing and placement can be observed in the same control frame after an
    earlier open observation. Milestones describe observations, not substep
    event timestamps. Historical milestones and current final state are separate.
    """
    def __init__(self, *, episode_id, task_id, payload_variant,
                 payload_instance, control_dt_s, scene_managed_initial=False):
        if type(scene_managed_initial) is not bool:
            raise ValueError("scene_managed_initial must be boolean")
        self.scene_managed_initial = scene_managed_initial
        if any(not isinstance(v, str) or not v for v in
               (episode_id, task_id, payload_instance)):
            raise ValueError("Nonempty episode/task/instance bindings required")
        variant_for(payload_variant)
        if not finite_number(control_dt_s) or control_dt_s <= 0:
            raise ValueError("Existing positive control interval required")
        self.episode_id = episode_id
        self.task_id = task_id
        self.variant = payload_variant
        self.instance = payload_instance
        self.control_dt_s = float(control_dt_s)
        self._last = None
        self._current = None
        self._events = ()
        self._hold_since = None

    def _validate(self, evidence):
        if not isinstance(evidence, TaskEvidence) or evidence.source != "isaac_physics":
            raise ValueError("Trusted physics TaskEvidence required")
        if (evidence.episode_id != self.episode_id or evidence.task_id != self.task_id):
            raise ValueError("Evidence belongs to another episode/task")
        if type(evidence.step_index) is not int or evidence.step_index < 0:
            raise ValueError("Invalid control index")
        if not finite_number(evidence.time_s) or evidence.time_s < 0:
            raise ValueError("Invalid simulation time")
        f = evidence.data
        if not isinstance(f, PutFacts):
            raise ValueError("Complete measured PutFacts required")
        if f.payload_variant != self.variant or f.payload_instance != self.instance:
            raise ValueError("Wrong payload variant/instance")
        predicates = (f.door_open, f.door_closed, f.payload_inside,
                      f.payload_supported, f.held_by_robot, f.held_by_right)
        if any(type(v) is not bool for v in predicates):
            raise ValueError("Missing or non-boolean physics predicate")
        if f.door_open and f.door_closed:
            raise ValueError("Contradictory door predicates")
        if f.held_by_right and not f.held_by_robot:
            raise ValueError("Contradictory grasp predicates")
        if f.released is not None and type(f.released) is not bool:
            raise ValueError("Invalid release predicate")
        if self.scene_managed_initial and type(f.released) is not bool:
            raise ValueError("OpenEnded requires measured release evidence")
        if f.released and f.held_by_robot:
            raise ValueError("Held object cannot be confirmed released")
        return f

    def update(self, evidence):
        f = self._validate(evidence)
        if self._last is None:
            if evidence.step_index != 0:
                raise ValueError("Start with the validated initial control frame")
            if not self.scene_managed_initial and (not f.door_closed or not f.held_by_right):
                raise ValueError("Initial state requires closed door and actual right-hand grasp")
            if self.scene_managed_initial:
                # Record the real room state, without changing it or awarding
                # milestones for actions that predate this observation window.
                self._last = evidence
                self._current = PutProgress(self.episode_id, self.task_id,
                    evidence.step_index, evidence.time_s, (), False, 0.0)
                return self._current
        else:
            if evidence.step_index == self._last.step_index:
                if evidence != self._last:
                    raise ValueError("Repeated control frame changed evidence")
                return self._current
            if evidence.step_index != self._last.step_index + 1:
                raise ValueError("Missing or stale control-frame evidence")
            # Reuse the existing public interval checker, including float32
            # clock precision. One interval is checked; no task budget is reset.
            interval = ControlClock(1, self.control_dt_s)
            interval.start(self._last.time_s)
            try:
                interval.observe(evidence.time_s)
            except ValueError as exc:
                raise ValueError("Evidence clock differs from the existing control interval") from exc

        events = list(self._events)
        reached = {event.stage for event in events}
        opened_before = "opened" in reached
        previous = self._last.data if self._last is not None else None
        def released(value):
            return not value.held_by_robot and value.released is not False

        previously_released_inside = bool(previous and previous.payload_inside
            and previous.payload_supported and released(previous))

        def record(stage):
            if stage not in reached:
                events.append(StageEvent(stage, evidence.step_index, evidence.time_s))
                reached.add(stage)

        if f.door_open:
            record("opened")
        released_inside = f.payload_inside and f.payload_supported and released(f)
        # Require a newly observed placement/release after an earlier open.
        # A box already inside must not earn retroactive credit when the door
        # is opened later. Permit release+close within one control interval
        # only if its previous boundary still showed the door open.
        during_open_interval = f.door_open or bool(previous and previous.door_open)
        if (opened_before and released_inside and not previously_released_inside
                and during_open_interval):
            record("placed_and_released")
        if "placed_and_released" in reached and f.door_closed and released_inside:
            record("closed_after_placement")
        final = "closed_after_placement" in reached and f.door_closed and released_inside
        hold_since = self._hold_since if final else None
        if final and hold_since is None:
            hold_since = evidence.time_s
        held = evidence.time_s - hold_since if final else 0.0
        result = PutProgress(self.episode_id, self.task_id, evidence.step_index,
                             evidence.time_s, tuple(events), final, held)
        self._events = tuple(events)
        self._hold_since = hold_since
        self._last = evidence
        self._current = result
        return result

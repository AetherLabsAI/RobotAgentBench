"""OpenEnd wiping scoring candidate: grasp 1 + cleanup 0..4 + return 1.

Adapted from verified migration task st18-1. Physical evidence, sequencing,
brush-only deductions and success criteria are preserved. Other objects are
not monitored. A native confirmed brush grasp activates the task.
Room adapter uses measured 60/240 Hz timestamps; room owns timeout.
"""
from collections import deque
from dataclasses import dataclass
import math

# New scoring schema: never silently rescore the original four-point runs.
RULE_VERSION = 'arena.wipe_table.score.v2'
RULE_REVISION = 'openend-continuous-wipe-grasp-gated-no-global-watch-v3'
SOURCE_RULE_VERSION = 'arena.wipe_table.score.v1'
TASK_ID = 'wipe_table'  # OpenEnd family key, not a standalone HTTP API ID.
MAXIMUM_SCORE = 6
MAXIMUM_WIPE_POINTS = 4.0
LIFT_M = 0.01
DROP_MIN_HEIGHT_M = 0.01  # same height that defines 'lifted'
CLEAN_COVERAGE = 0.05
RETURN_RADIUS_M = 0.05
RETURN_YAW_RAD = math.radians(15)
REST_WINDOW_S = 0.5
REST_SPAN_M = 0.002
TIME_LIMIT_S = 240.0  # stage 2 registration budget (4800 steps); per-stage limits = 2 x longest real duration


@dataclass(frozen=True)
class WipeFrame:
    """One physics substep of trusted measurements."""
    episode_id: str
    sample_index: int
    time_s: float
    tool_held: bool
    tool_robot_contact: bool
    face_height_m: float          # lowest wiping-face corner minus table top
    tool_table_contact: bool
    tool_supported: bool
    tool_on_floor: bool
    tool_xy: tuple
    tool_yaw_rad: float
    complete: bool = True


@dataclass(frozen=True)
class StepFacts:
    """One control step: its physics frames and the stain state after the step."""
    frames: tuple
    coverage: float
    initial_coverage: float
    face_contact: bool            # any substep: face <= 5 mm and tool-table contact
    newly_passed_cells: int


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wipe_points_for_coverage(initial_coverage, coverage):
    """Linearly reward removal up to the unchanged 5% clean threshold."""
    if (not _finite(initial_coverage) or not CLEAN_COVERAGE < initial_coverage <= 1
            or not _finite(coverage) or not 0 <= coverage <= 1):
        raise ValueError('Invalid stain coverage evidence')
    if coverage <= CLEAN_COVERAGE + 1e-12:
        return MAXIMUM_WIPE_POINTS
    fraction = (initial_coverage - coverage) / (initial_coverage - CLEAN_COVERAGE)
    return MAXIMUM_WIPE_POINTS * min(1.0, max(0.0, fraction))


class GraspGate:
    """Latch trusted physical grasp detection, never contact/closure commands."""
    def __init__(self):
        self.started = False

    def observe(self, held):
        if type(held) is not bool:
            raise ValueError('Confirmed physical grasp boolean required')
        self.started = self.started or held
        return self.started


class ProcessScorer:
    def __init__(self, episode_id, physics_dt_s, start_xy, start_yaw_rad):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Missing episode identity')
        if not _finite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError('Invalid physical sampling interval')
        self.episode_id, self.dt = episode_id, physics_dt_s
        self.start_xy = tuple(float(v) for v in start_xy)
        self.start_yaw = float(start_yaw_rad)
        self.last = None
        self.grasp_gate = GraspGate()
        self.stages = dict(grasp=False, wiping=False, clean=False, returned=False)
        self.returned_now = False
        self.deductions = dict(tool_on_floor=False, dropped_on_table=False)
        self.drop_pending = False
        self.drop_height_m = 0.0
        self.was_touching = False
        self.rest = deque()
        self.suspended_until = None
        self.coverage = None
        self.initial_coverage = None
        self.wipe_points = 0.0
        self.reason = None
        self.finalized = False
        self.events = []

    def _validate(self, facts):
        if not isinstance(facts, StepFacts) or not isinstance(facts.frames, tuple) or not facts.frames:
            raise ValueError('Nonempty step facts required')
        if not _finite(facts.coverage) or not 0 <= facts.coverage <= 1:
            raise ValueError('Invalid coverage')
        if (not _finite(facts.initial_coverage)
                or not CLEAN_COVERAGE < facts.initial_coverage <= 1):
            raise ValueError('Initial coverage must describe an initially dirty surface')
        if self.initial_coverage is not None and abs(facts.initial_coverage - self.initial_coverage) > 1e-12:
            raise ValueError('Initial coverage changed within the episode')
        if type(facts.face_contact) is not bool:
            raise ValueError('Missing wiping-face contact evidence')
        previous = self.last
        previous_time = self.suspended_until if self.suspended_until is not None else (None if previous is None else previous.time_s)
        for frame in facts.frames:
            if not isinstance(frame, WipeFrame) or frame.episode_id != self.episode_id:
                raise ValueError('Wrong episode or physical frame')
            expected = 0 if previous is None else previous.sample_index + 1
            if type(frame.sample_index) is not int or frame.sample_index != expected:
                raise ValueError('Missing, duplicate or out-of-order physical sample')
            if not _finite(frame.time_s) or (previous_time is not None and abs(frame.time_s - previous_time - self.dt) > 1e-6):
                raise ValueError('Physical evidence gap or changed sampling interval')
            for name in ('tool_held', 'tool_robot_contact', 'tool_table_contact', 'tool_supported',
                         'tool_on_floor', 'complete'):
                if type(getattr(frame, name)) is not bool:
                    raise ValueError('Missing physical boolean: ' + name)
            if not frame.complete:
                raise ValueError('Incomplete evidence is not a zero-score episode')
            previous = frame
            previous_time = frame.time_s

    def configure_clock(self, dt):
        # Measured room clock only; this does not configure the simulator.
        if not any(abs(dt - 1 / hz) < 1e-12 for hz in (60, 240)):
            raise ValueError('Unsupported measured OpenEnd clock')
        self.dt = dt

    @property
    def ended(self):
        return self.reason is not None

    def pause(self):
        self.rest.clear()
        self.returned_now = False

    def suspend_substep(self, now, dt):
        self.configure_clock(dt)
        previous = self.suspended_until if self.suspended_until is not None else self.last.time_s
        if abs(now - previous - dt) > 1e-6:
            raise ValueError('Missing dormant physical substep')
        self.suspended_until = now
        self.pause()

    def _event(self, frame, kind, points):
        self.events.append(dict(kind=kind, points=points, sample_index=frame.sample_index, time_s=frame.time_s))

    def _at_rest_in_start(self):
        if not self.rest or self.rest[-1][0] - self.rest[0][0] < REST_WINDOW_S - 1e-9:
            return False
        if not all(ok for _, ok, _ in self.rest):
            return False
        xs = [p[0] for _, _, p in self.rest]
        ys = [p[1] for _, _, p in self.rest]
        return max(max(xs) - min(xs), max(ys) - min(ys)) <= REST_SPAN_M

    def observe_step(self, facts):
        if self.finalized or self.reason is not None:
            raise RuntimeError('Terminal episode; create a new scorer for reset')
        self._validate(facts)
        self.suspended_until = None
        if self.initial_coverage is None:
            self.initial_coverage = facts.initial_coverage
        for frame in facts.frames:
            self.last = frame
            if self.reason is not None:
                continue
            if not self.grasp_gate.observe(frame.tool_held):
                continue  # Unstarted: no progress, deductions or return history.
            if frame.tool_on_floor:
                if not self.deductions['tool_on_floor']:
                    self.deductions['tool_on_floor'] = True
                    self._event(frame, 'tool_on_floor', -1)
                self.reason = 'tool_on_floor'
                continue
            held = frame.tool_held
            if held and frame.face_height_m >= LIFT_M and not self.stages['grasp']:
                self.stages['grasp'] = True
                self._event(frame, 'grasp_lift', 1)
            # 'Lost hold' = the robot no longer touches the tool at all (raw
            # contact, not the 0.1 s grasp confirmation, so grasp-detector
            # flicker while wiping or carrying is never a drop).
            touching = frame.tool_robot_contact
            if self.stages['grasp']:
                if self.was_touching and not touching and not frame.tool_table_contact:
                    self.drop_pending = True
                    self.drop_height_m = frame.face_height_m
                if touching:
                    self.drop_pending = False
                elif self.drop_pending and not frame.tool_table_contact:
                    self.drop_height_m = max(self.drop_height_m, frame.face_height_m)
                elif self.drop_pending and frame.tool_table_contact:
                    self.drop_pending = False
                    if self.drop_height_m >= DROP_MIN_HEIGHT_M and not self.deductions['dropped_on_table']:
                        self.deductions['dropped_on_table'] = True
                        self._event(frame, 'dropped_on_table', -1)
            self.was_touching = touching
            dx = frame.tool_xy[0] - self.start_xy[0]
            dy = frame.tool_xy[1] - self.start_xy[1]
            in_region = (math.hypot(dx, dy) <= RETURN_RADIUS_M
                         and abs(wrap(frame.tool_yaw_rad - self.start_yaw)) <= RETURN_YAW_RAD)
            released_supported = (not held and not frame.tool_robot_contact and frame.tool_supported)
            self.rest.append((frame.time_s, in_region and released_supported, frame.tool_xy))
            while len(self.rest) > 1 and self.rest[1][0] <= frame.time_s - REST_WINDOW_S + 1e-9:
                self.rest.popleft()
        self.coverage = facts.coverage
        if self.reason is None:
            frame = self.last
            if (self.stages['grasp'] and not self.stages['wiping'] and facts.face_contact
                    and facts.coverage < facts.initial_coverage - 1e-12):
                self.stages['wiping'] = True
                self._event(frame, 'wiping_started', 0)
            if self.stages['wiping'] and facts.face_contact:
                # Absolute progress, not a sum of per-frame reductions. Stopping
                # on a spot, revisiting clean cells or duplicate coverage cannot
                # farm points. The unchanged stain engine supplies coverage.
                candidate = wipe_points_for_coverage(self.initial_coverage, facts.coverage)
                if candidate > self.wipe_points:
                    delta = candidate - self.wipe_points
                    self.wipe_points = candidate
                    self._event(frame, 'wipe_progress', delta)
            if self.stages['wiping'] and not self.stages['clean'] and facts.coverage <= CLEAN_COVERAGE + 1e-12:
                self.stages['clean'] = True
                self._event(frame, 'stain_clean', 0)
            self.returned_now = self.stages['clean'] and self._at_rest_in_start()
            if self.returned_now and not self.stages['returned']:
                self.stages['returned'] = True
                self._event(frame, 'tool_returned', 1)
            if self.stages['clean'] and self.returned_now:
                self.reason = 'success'
        return self.summary()

    def summary(self):
        earned = int(self.stages['grasp']) + self.wipe_points + int(self.stages['returned'])
        deducted = sum(self.deductions.values())
        raw = earned - deducted
        return dict(schema=RULE_VERSION, rule_revision=RULE_REVISION, episode_id=self.episode_id, task_id=TASK_ID,
                    maximum_score=MAXIMUM_SCORE, score=max(0, raw), raw_score=raw,
                    earned_points=earned, deducted_points=deducted,
                    started=self.grasp_gate.started,
                    activation='native_confirmed_brush_grasp',
                    stages=dict(self.stages), deductions=dict(self.deductions),
                    coverage=self.coverage, initial_coverage=self.initial_coverage,
                    wipe_points=self.wipe_points, maximum_wipe_points=MAXIMUM_WIPE_POINTS,
                    returned_now=self.returned_now,
                    drop_pending=self.drop_pending,
                    success=self.reason == 'success',
                    failure='tool_on_floor' if self.reason == 'tool_on_floor' else None,
                    reason=self.reason, finalized=self.finalized, time_limit_s=TIME_LIMIT_S)

    def finalize(self, reason):
        if self.last is None:
            raise ValueError('Cannot settle without valid evidence')
        if self.finalized:
            if reason != self.reason:
                raise ValueError('Contradictory repeated finalization')
            return self.summary()
        if self.reason is not None and reason != self.reason:
            raise ValueError('Finalization disagrees with measured terminal event')
        if self.reason is None and reason not in ('client_closed', 'time_limit'):
            raise ValueError('No evidence for requested terminal result')
        self.reason = reason
        self.finalized = True
        return self.summary()

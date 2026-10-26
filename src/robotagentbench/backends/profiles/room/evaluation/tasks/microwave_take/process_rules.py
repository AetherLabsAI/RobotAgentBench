"""Task 10.1 process points, driven ONLY by trusted per-physics-step evidence.

This module does not collect evidence, open USD, control physics, register an
HTTP endpoint or determine task success. source/complete/acceptance fields are
contracts with an environment reader, NOT authentication of a model payload.
Never expose ProcessFrame or the scorer to model-supplied evaluation facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping

from .initial_state import BINDINGS, MICROWAVE, ROBOT, PLATE, PAYLOAD, TASK_ID, bindings_for

SCHEMA = "arena.microwave_take.process.v4"
STAGE_POINTS = (("open", 3), ("grasp", 1), ("take_out", 3), ("payload_retained", 2), ("close", 1))
OPEN_DEG = 60.0
HOLD_SECONDS = 5.0
HOLD_FAILURE_PENALTY = 3  # user-approved; duration and detection unchanged
COLLISION_POINTS = 1
# User-approved 2026-09-11: keep every event, deduct at most 1 per episode.
COLLISION_CAP = 1
IMPULSE_EPSILON_NS = 1e-9  # numerical zero, not an impact-severity threshold
# Existing box colliders; panel/door/stand are NOT interior wall surfaces.
# axis, high face: normal points into the usable cavity.
INNER_WALL_FACES = {
    MICROWAVE + "/Asset/body/col_left": (0, True),
    MICROWAVE + "/Asset/body/col_right": (0, False),
    MICROWAVE + "/Asset/body/col_back": (1, False),
    MICROWAVE + "/Asset/body/col_top": (2, False),
    MICROWAVE + "/Asset/body/col_bottom": (2, True),
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Expected a finite real measurement")
    return float(value)


def _vec3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Expected three measured coordinates")
    return tuple(_number(v) for v in value)


def front_clearance_m(vertices_microwave_m, opening_front_y_m):
    """Signed clearance of COMPLETE surface vertices past the front plane.

    Approved oven opens toward local -Y. Positive means every supplied point
    is outside. The reader must supply all relevant visual/collision geometry,
    transformed using measured poses; a centre point is NOT sufficient.
    """
    plane = _number(opening_front_y_m)
    vertices = [_vec3(v) for v in vertices_microwave_m]
    if not vertices:
        raise ValueError("Missing complete object geometry")
    return plane - max(v[1] for v in vertices)


@dataclass(frozen=True)
class WallBox:
    collider: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    def __post_init__(self):
        if self.collider not in INNER_WALL_FACES:
            raise ValueError("Unreviewed microwave wall collider")
        lo, hi = _vec3(self.minimum_m), _vec3(self.maximum_m)
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("Invalid wall geometry")
        object.__setattr__(self, "minimum_m", lo)
        object.__setattr__(self, "maximum_m", hi)

    def inner_face(self, point):
        """Closest box face, not 'any contact with the wall's collider'.

        Contact coordinates are in the rigid microwave frame, in metres.
        This handles solver contact offsets without making an exterior face
        an interior hit. Edge ties include the interior edge. Axis-aligned
        boxes must be verified by the geometry reader before constructing us.
        """
        p = _vec3(point)
        distances = [abs(p[i]-v[i]) for i in range(3)
                     for v in (self.minimum_m, self.maximum_m)]
        axis, high = INNER_WALL_FACES[self.collider]
        return distances[axis*2+int(high)] <= min(distances)+1e-9


@dataclass(frozen=True)
class ContactPoint:
    collider0: str
    collider1: str
    point_microwave_m: tuple[float, float, float]
    impulse_norm_ns: float

    def validate(self):
        for path in (self.collider0, self.collider1):
            if not isinstance(path, str) or not path.startswith("/") or ".." in path.split("/"):
                raise ValueError("Invalid measured collider path")
        _vec3(self.point_microwave_m)
        if _number(self.impulse_norm_ns) < 0:
            raise ValueError("Impulse norm cannot be negative")


def _under(path, root):
    return path == root or path.startswith(root + "/")


def interior_contact(point, walls, payload_root=PAYLOAD):
    """Exclude normal door/grip/support contacts; include robot cameras too."""
    point.validate()
    return _interior_contact_validated(point, walls, payload_root)


def _interior_contact_validated(point, walls, payload_root=PAYLOAD):
    """Internal only: observe() has already validated this immutable frame."""
    for wall, actor in ((point.collider0, point.collider1), (point.collider1, point.collider0)):
        if wall in walls and any(_under(actor, root) for root in (ROBOT, PLATE, payload_root)):
            return walls[wall].inner_face(point.point_microwave_m)
    return False


@dataclass(frozen=True)
class ProcessFrame:
    """Compact physical facts, NOT targets, stage labels, or model predictions.

    Clearances use full surfaces and the closed-frame opening plane. Holding
    requires actual support/contact/carry evidence, not gripper commands or
    proximity. payload_on_plate includes physical support plus containment on
    the plate. dropped_since_last_frame is interval evidence, not just a final
    pose. Contacts may be preclassified to all robot/plate/payload vs inner-wall
    candidates, but must retain zero-impulse persistent contacts. An empty tuple
    means CONFIRMED no relevant contacts; None/False completeness is never
    'no collision'. Carry/support facts still use their own complete point set.
    """
    episode_id: str
    step_index: int
    time_s: float
    door_angle_deg: float
    plate_front_clearance_m: float
    payload_front_clearance_m: float
    plate_held_by_robot: bool
    payload_on_plate: bool
    dropped_since_last_frame: bool
    contacts: tuple[ContactPoint, ...] | None
    measurements_complete: bool
    source: str
    task_id: str
    bindings: Mapping[str, str] = field(default_factory=dict)
    plate_grasped_by_robot: bool | None = None
    plate_lifted_from_support: bool | None = None


class ProcessScorer:
    """Ordered one-time rewards plus capped, continuous-contact deductions.

    closed_door_deg=2 is a provisional implementation tolerance (same current
    reset tolerance), NOT a separately approved final-success rule. Wall
    geometry is copied from a trusted reader and fixed for this episode.
    Call once for every physical step, including accepted initial frame 0.
    """
    def __init__(self, episode_id: str, physics_dt_s: float, walls: tuple[WallBox, ...],
                 *, initial_state_accepted: bool, hold_failure_penalty: int,
                 closed_door_deg: float = 2.0, bindings=None):
        self.bindings = dict(BINDINGS if bindings is None else bindings)
        if self.bindings not in (bindings_for('beef_box'), bindings_for('leftover_box')):
            raise ValueError('Unknown task instance bindings')
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Missing episode identity")
        if initial_state_accepted is not True:
            raise ValueError("Validate the physical initial state before scoring")
        self.dt = _number(physics_dt_s)
        self.closed_deg = _number(closed_door_deg)
        if self.dt <= 0 or not 0 <= self.closed_deg < OPEN_DEG:
            raise ValueError("Invalid timing or closed-door tolerance")
        if type(hold_failure_penalty) is not int or not 1 <= hold_failure_penalty <= 10:
            raise ValueError("Explicit hold-failure deduction must be an integer in 1..10")
        self.hold_failure_penalty = hold_failure_penalty
        if not isinstance(walls, tuple) or any(not isinstance(w, WallBox) for w in walls):
            raise ValueError("Expected measured wall geometry")
        self.walls = {w.collider: w for w in walls}
        if len(walls) != len(INNER_WALL_FACES) or set(self.walls) != set(INNER_WALL_FACES):
            raise ValueError("All five measured interior walls are required")
        self.episode_id = episode_id
        self.last_step = -1
        self.start_time = None
        self.last_time = None
        self.awards = {}
        self.events = []
        self.collision_episodes = 0
        self.collision_active = False
        self.hold_start = None
        self.hold_elapsed = 0.0
        self.hold_satisfied = False
        self.hold_deduction = 0
        self.finalized = False
        self.error = None

    def _validate(self, f):
        if not isinstance(f, ProcessFrame):
            raise ValueError("Missing physical process frame")
        if f.episode_id != self.episode_id or f.task_id != TASK_ID:
            raise ValueError("Wrong episode/task")
        if f.source != "isaac_physics" or dict(f.bindings) != self.bindings:
            raise ValueError("Wrong source or instance bindings")
        if f.measurements_complete is not True:
            raise ValueError("Incomplete geometry, holding or contact evidence")
        if type(f.step_index) is not int or f.step_index != self.last_step+1:
            raise ValueError("Missing, repeated or reordered physics step")
        t = _number(f.time_s)
        if t < 0:
            raise ValueError("Negative simulation time")
        if self.start_time is not None and not math.isclose(
                t-self.start_time, f.step_index*self.dt, rel_tol=0., abs_tol=1e-6):
            raise ValueError("Simulation clock does not match physics steps")
        for v in (f.door_angle_deg, f.plate_front_clearance_m, f.payload_front_clearance_m):
            _number(v)
        for v in (f.plate_held_by_robot, f.payload_on_plate, f.dropped_since_last_frame,
                  f.plate_grasped_by_robot, f.plate_lifted_from_support):
            if type(v) is not bool:
                raise ValueError("Unknown physical holding/drop fact")
        if not isinstance(f.contacts, tuple) or any(not isinstance(p, ContactPoint) for p in f.contacts):
            raise ValueError("Missing or malformed contact stream")
        for point in f.contacts:
            point.validate()
        if f.step_index == 0 and (abs(f.door_angle_deg) > self.closed_deg or
                f.plate_front_clearance_m > 0 or f.payload_front_clearance_m > 0 or
                f.plate_held_by_robot or f.plate_grasped_by_robot or f.plate_lifted_from_support or
                not f.payload_on_plate or f.dropped_since_last_frame):
            raise ValueError("Initial frame is not the accepted closed, loaded microwave")

    def _award(self, name, f):
        self.awards[name] = dict(step_index=f.step_index, time_s=f.time_s,
                                points=dict(STAGE_POINTS)[name])
        self.events.append(dict(kind="stage", stage=name, **self.awards[name]))

    def observe(self, frame: ProcessFrame):
        if self.finalized:
            raise RuntimeError("Process score finalized; no further observations allowed")
        if self.error is not None:
            raise RuntimeError("Scoring evidence is invalid; start a new validated episode")
        try:
            self._validate(frame)
        except (ValueError, TypeError) as exc:
            self.error = str(exc)
            self.hold_start, self.hold_elapsed = None, 0.0
            self.hold_satisfied = False
            raise ValueError(self.error) from exc
        f = frame
        self.last_step, self.last_time = f.step_index, f.time_s
        if self.start_time is None:
            self.start_time = f.time_s
            return self.summary()  # reset is not an action and earns no points

        self._update_collisions(f)
        return self._update_stages(f)

    def _update_collisions(self, f):
        # Detailed profile: evaluate current physics-step points.
        eligible = [p for p in f.contacts if _interior_contact_validated(p, self.walls, self.bindings['payload'])]
        impact = any(p.impulse_norm_ns > IMPULSE_EPSILON_NS for p in eligible)
        if impact and not self.collision_active:
            self.collision_episodes += 1
            self.collision_active = True
            self.events.append(dict(kind="interior_collision", step_index=f.step_index,
                time_s=f.time_s, episode_index=self.collision_episodes,
                deduction=COLLISION_POINTS if self.collision_episodes <= COLLISION_CAP else 0,
                colliders=sorted({p.collider0 for p in eligible} | {p.collider1 for p in eligible})))
        if not eligible:
            self.collision_active = False

    def _update_stages(self, f):
        outside = f.plate_front_clearance_m > 0 and f.payload_front_clearance_m > 0
        carrying = outside and f.plate_held_by_robot and f.payload_on_plate and not f.dropped_since_last_frame
        closed = abs(f.door_angle_deg) <= self.closed_deg
        # At most one new stage per measured frame. Reclosing the initial door,
        # repeated grasps or repeated door cycles cannot farm stage points.
        if "open" not in self.awards:
            if f.door_angle_deg >= OPEN_DEG:
                self._award("open", f)
        elif "grasp" not in self.awards:
            if f.plate_grasped_by_robot:
                self._award("grasp", f)
        elif "take_out" not in self.awards:
            if f.plate_front_clearance_m > 0 and f.plate_held_by_robot and f.plate_lifted_from_support:
                self._award("take_out", f)
        elif "close" not in self.awards:
            if outside and closed:
                self._award("close", f)

        # Payload retention is a separate quality item, not a prerequisite for
        # the 3-point plate extraction or the later close action. It can be
        # recorded on the extraction frame; never award it while still inside.
        if "take_out" in self.awards and "payload_retained" not in self.awards and carrying:
            self._award("payload_retained", f)

        if "close" in self.awards and closed and carrying:
            if self.hold_start is None:
                self.hold_start = f.time_s
            self.hold_elapsed = f.time_s-self.hold_start
        else:
            self.hold_start, self.hold_elapsed = None, 0.0
        # No bonus for holding. Both objects must still be present throughout
        # the CURRENT continuous window. Loss after an earlier good window
        # clears satisfaction too; regrasp requires a fresh uninterrupted 5 s.
        self.hold_satisfied = self.hold_start is not None and self.hold_elapsed+1e-9 >= HOLD_SECONDS
        return self.summary()

    def finalize(self):
        """Settle the hold deduction once at episode end, never prematurely.

        Before the close stage, this post-close check is not applicable; lost
        stage points already reflect incomplete progress. This is a score,
        not a success/failure verdict, and never stops the robot by itself.
        """
        if self.error is not None or self.start_time is None:
            raise ValueError("Cannot finalize missing/invalid physical evidence")
        if not self.finalized:
            if "close" in self.awards and not self.hold_satisfied:
                self.hold_deduction = self.hold_failure_penalty
                self.events.append(dict(kind="post_close_hold_failure", deduction=self.hold_deduction,
                    step_index=self.last_step, time_s=self.last_time,
                    continuous_hold_s=self.hold_elapsed, required_s=HOLD_SECONDS))
            self.finalized = True
        return self.summary()

    def summary(self):
        earned = sum(row["points"] for row in self.awards.values())
        penalty = min(COLLISION_CAP, self.collision_episodes*COLLISION_POINTS)
        valid = self.error is None and self.start_time is not None
        return dict(schema=SCHEMA, task_id=TASK_ID, episode_id=self.episode_id,
            status="invalid_evidence" if self.error else ("finalized" if self.finalized else
                ("scoring" if valid else "awaiting_initial_frame")),
            error=self.error, last_step=self.last_step, earned_points=earned,
            collision_deduction=penalty, collision_episodes=self.collision_episodes,
            collision_deduction_cap=COLLISION_CAP,
            hold_deduction=self.hold_deduction,
            process_score=max(0, earned-penalty-self.hold_deduction) if valid else None, maximum_score=10,
            stages={name:dict(self.awards[name]) if name in self.awards else None for name, _ in STAGE_POINTS},
            hold_elapsed_s=self.hold_elapsed, hold_required_s=HOLD_SECONDS,
            stage_sequence_completed="close" in self.awards,
            hold_requirement_status=("not_applicable" if "close" not in self.awards else
                ("satisfied" if self.hold_satisfied else ("failed" if self.finalized else "pending"))),
            hold_failure_penalty_points=self.hold_failure_penalty,
            # Historical stage rewards are not revoked on later failure. The
            # separate success module must check actual terminal conditions.
            task_success=None, terminate_episode=False,
            evidence_collected_by_this_module=False, runtime_api_connected=False,
            physical_reader_validated=False)

    def event_log(self):
        from copy import deepcopy
        return deepcopy(self.events)

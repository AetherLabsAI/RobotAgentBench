"""Editable code-as-policy motion helper; NOT a trusted safety boundary.

Adapted from Inspect Robots / YAM's interpolation and online IK control logic.
See UPSTREAM.md for pinned sources and MIT notices.
Input: approved robot-only MuJoCo model, named measured joints, a target SE(3).
Output: named joint targets through a separately trusted observation/action port.
No Isaac imports, scene access, task rules, hidden contact or object pose reads.
"""
from dataclasses import dataclass, field
from typing import Mapping, Protocol
import math

import numpy as np


def vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return result.copy()


def pose(value):
    result = vector(value, (4, 4), "pose")
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("pose must be a proper rigid transform")
    return result


def errors(actual, target):
    actual, target = pose(actual), pose(target)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    cosine = (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2
    return position, float(np.arccos(np.clip(cosine, -1, 1)))


def interpolate(start, target, fraction):
    """Linear xyz and Euler dimensions, without wrapping, with identity reference.

    Persistent MoveToController uses the episode-start orientation reference.
    Named targets preserve intentional Euler winding; matrix targets cannot.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction outside [0,1]")
    a, b = pose_values(start, np.eye(3)), pose_values(target, np.eye(3))
    return values_pose(a + (b - a) * fraction, np.eye(3))


LABELS = ("x", "y", "z", "yaw", "pitch", "roll")


def pose_values(value, reference):
    value = pose(value)
    relative = value[:3, :3] @ reference.T
    pitch = float(np.arcsin(np.clip(relative[2, 0], -1., 1.)))
    if abs(pitch) >= np.pi / 2 - 1e-6:
        raise ValueError("Euler pitch singularity")
    yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
    roll = float(np.arctan2(relative[2, 1], relative[2, 2]))
    return np.array([*value[:3, 3], yaw, pitch, roll])


def values_pose(value, reference):
    x, y, z, yaw, pitch, roll = vector(value, (6,), "Cartesian target")
    cy, sy, cp, sp, cr, sr = (np.cos(yaw), np.sin(yaw), np.cos(pitch),
                             np.sin(pitch), np.cos(roll), np.sin(roll))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    # Upstream convention: positive pitch is a rotation about -y.
    ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx @ reference
    result[:3, 3] = [x, y, z]
    return pose(result)


@dataclass
class Observation:
    episode_id: str
    step: int
    joints: Mapping[str, float]
    gripper: float | None = None  # normalized measured opening coordinate, NOT metric aperture


class RobotPort(Protocol):
    """Implemented by a trusted bridge, never by direct host/Isaac access.

    command_joints must enforce episode/step freshness atomically, action limits,
    the configured control period, and joint allowlists OUTSIDE editable code.
    It returns a NEW measured observation after exactly one control interval.
    Optional gripper is a normalized explicit opening target (0 closed, 1 open).
    It need not vary linearly with physical jaw separation. The trusted bridge owns
    its published native-channel mapping, never inferred from joint counts.
    No automatic gripper correction, task skill, force change or physics reset.
    """
    def observe(self) -> Observation: ...
    def command_joints(self, targets: Mapping[str, float], *, episode_id: str,
                       expected_step: int, gripper: float | None = None) -> Observation: ...


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    reason: str = ""


class MinkIK:
    """Fixed-base, scalar-joint robot model. All non-arm DOFs are frozen in IK.

    model must be an approved robot-only MjModel, with no task/world geometry.
    This class does kinematics only: it never calls mj_step or executes physics.
    Real robot model conversion/joint mapping must be independently validated.
    """
    def __init__(self, model, site_name, arm_joints, *, max_iters=200,
                 position_tolerance=0.001, angle_tolerance=0.01):
        import mujoco
        import mink
        self.model = model
        self.names = tuple(model.joint(i).name for i in range(model.njnt))
        if (model.nq != model.nv or len(self.names) != model.nq
                or not all(self.names) or len(set(self.names)) != len(self.names)
                or any(int(t) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                       for t in model.jnt_type)):
            raise ValueError("v1 requires named scalar joints and a fixed base")
        self.arm_joints = tuple(arm_joints)
        if (not self.arm_joints or len(set(self.arm_joints)) != len(self.arm_joints)
                or not set(self.arm_joints) <= set(self.names)):
            raise ValueError("invalid arm joint list")
        if type(max_iters) is not int or max_iters < 1:
            raise ValueError("max_iters must be a positive integer")
        if any(not np.isfinite(x) or x <= 0 for x in (position_tolerance, angle_tolerance)):
            raise ValueError("invalid IK tolerances")
        self.indices = np.array([int(model.joint(n).qposadr[0]) for n in self.arm_joints])
        self.frozen = np.array([i for i in range(model.nq) if i not in self.indices], dtype=int)
        self.low = np.full(model.nq, -np.inf)
        self.high = np.full(model.nq, np.inf)
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                address = int(model.jnt_qposadr[i])
                self.low[address], self.high[address] = model.jnt_range[i]
        # Validate site name before any solve or motion.
        model.site(site_name)
        self.site_name = site_name
        self.configuration = mink.Configuration(model)
        self.task = mink.FrameTask(frame_name=site_name, frame_type="site",
                                   position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.constraints = ([mink.DofFreezingTask(model=model, dof_indices=self.frozen.tolist())]
                            if len(self.frozen) else [])
        self.max_iters = max_iters
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.command_step_limits = None
        self.command_dt = None

    def set_command_limits(self, step_limits, dt):
        """Limits for ONE sent command, not each inner IK iteration."""
        limits = vector(step_limits, (len(self.indices),), "command step limits")
        if np.any(limits <= 0) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive command limits and dt required")
        self.command_step_limits = limits
        self.command_dt = float(dt)

    def _solver_limits(self, seed):
        import mink
        if self.command_step_limits is None:
            return None  # Standalone raw FK/IK tests retain the i2rt contract.
        indices = self.indices
        projection = np.eye(self.model.nv)[indices]
        lower = seed[indices] - self.command_step_limits
        upper = seed[indices] + self.command_step_limits

        class CommandStepLimit(mink.Limit):
            """Fixed-origin total-displacement box using Mink's Limit API.

            VelocityLimit alone restarts at each internal IK iteration. This
            additional box keeps their sum inside one physical command period.
            Valid for our validated scalar hinge/slide model only.
            """
            def compute_qp_inequalities(self, configuration, dt):
                q = configuration.q[indices]
                return mink.Constraint(np.concatenate((projection, -projection)),
                                       np.concatenate((upper - q, q - lower)))

        return [mink.ConfigurationLimit(self.model),
                mink.VelocityLimit(self.model, dict(zip(self.arm_joints,
                    self.command_step_limits / self.command_dt))), CommandStepLimit()]

    def read_q(self, observation):
        if not isinstance(observation.episode_id, str) or not observation.episode_id:
            raise ValueError("missing episode id")
        if type(observation.step) is not int or observation.step < 0:
            raise ValueError("invalid observation step")
        q = np.zeros(self.model.nq)
        for name in self.names:
            q[int(self.model.joint(name).qposadr[0])] = observation.joints[name]
        # Sensor values are not commands. Physical limit overshoot must remain
        # visible; FK is well-defined there. Command bounds are checked separately.
        return vector(q, (self.model.nq,), "measured joints")

    def validate_q(self, q):
        q = vector(q, (self.model.nq,), "joints")
        if np.any(q < self.low - 1e-8) or np.any(q > self.high + 1e-8):
            raise ValueError("joint outside model limits")
        return q

    def fk(self, q):
        self.configuration.update(vector(q, (self.model.nq,), "FK joints"))
        return self.configuration.get_transform_frame_to_world(self.site_name, "site").as_matrix()

    def solve(self, target, seed):
        import mink
        target, seed = pose(target), self.validate_q(seed)
        self.configuration.update(seed)
        self.task.set_target(mink.SE3.from_matrix(target))
        limits = self._solver_limits(seed)
        dt = .01 if self.command_dt is None else self.command_dt
        for _ in range(self.max_iters):
            current = self.configuration.get_transform_frame_to_world(self.site_name, "site").as_matrix()
            p_error, r_error = errors(current, target)
            if p_error <= self.position_tolerance and r_error <= self.angle_tolerance:
                q = self.validate_q(self.configuration.q)
                if len(self.frozen) and not np.allclose(q[self.frozen], seed[self.frozen], atol=1e-8, rtol=0):
                    return IKResult(False, seed, "frozen_joint_changed")
                return IKResult(True, q)
            try:
                velocity = mink.solve_ik(self.configuration, [self.task], dt,
                                         solver="quadprog", damping=1e-4,
                                         constraints=self.constraints, limits=limits)
            except mink.NoSolutionFound:
                return IKResult(False, seed, "qp_no_solution")
            if not np.all(np.isfinite(velocity)):
                return IKResult(False, seed, "nonfinite_solution")
            self.configuration.integrate_inplace(velocity, dt)
        # Match i2rt/YAM: return the last iterate, even without convergence.
        # The controller verifies FK and limits; it never clips this candidate.
        return IKResult(False, self.configuration.q.copy(), "iteration_limit")

    def reference_solve(self, target, seed):
        """YAM-style raw iterative IK, then caller applies range/rate clamps.

        G1 non-arm coordinates remain frozen at measured values. Do not apply
        the Arena command-displacement QP box or reject approximate finite IK.
        This computes kinematics only; it never advances MuJoCo physics.
        """
        import mink
        self.configuration.update(vector(seed, (self.model.nq,), "IK seed"))
        self.task.set_target(mink.SE3.from_matrix(pose(target)))
        for _ in range(self.max_iters):
            try:
                velocity = mink.solve_ik(self.configuration, [self.task], .01,
                    solver="quadprog", damping=1e-4, constraints=self.constraints, limits=[])
            except mink.NoSolutionFound:
                return IKResult(False, self.configuration.q.copy(), "qp_no_solution")
            if not np.all(np.isfinite(velocity)):
                return IKResult(False, self.configuration.q.copy(), "nonfinite_solution")
            self.configuration.integrate_inplace(velocity, .01)
            current = self.configuration.get_transform_frame_to_world(self.site_name, "site").as_matrix()
            p, a = errors(current, target)
            if p <= self.position_tolerance and a <= self.angle_tolerance:
                return IKResult(True, self.configuration.q.copy())
        return IKResult(False, self.configuration.q.copy(), "iteration_limit")


@dataclass
class MoveConfig:
    # Must match the trusted bridge's fixed control frequency.
    control_hz: float = 20.0
    linear_speed: float = 0.04  # m/s
    angular_speed: float = 0.25  # rad/s
    # Per-arm-joint SI units/s (hinge rad/s, slide m/s); explicit, no guessing.
    joint_speeds: Mapping[str, float] = field(default_factory=dict)
    # Public robot-specific contract; do not copy YAM geometry/joint limits.
    eef_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    resync_thresholds: Mapping[str, float] = field(default_factory=dict)
    oscillation_deadbands: Mapping[str, float] = field(default_factory=dict)
    gripper_speed: float = 1.0  # normalized aperture/s, still bounded by bridge
    gripper_tolerance: float = 0.02
    position_tolerance: float = 0.002
    angle_tolerance: float = 0.02
    max_steps: int = 300
    stagnation_seconds: float = 1.0
    stagnation_min_improvement: float = 0.05  # fraction of tracking tolerance
    oscillation_reversals: int = 2
    oscillation_window: int = 6
    oscillation_hold_steps: int = 10

    def validate(self, ik):
        for name in ("control_hz", "linear_speed", "angular_speed", "position_tolerance", "angle_tolerance",
                     "gripper_speed", "gripper_tolerance", "stagnation_seconds", "stagnation_min_improvement"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 10000:
            raise ValueError("invalid max_steps")
        for name in ("oscillation_reversals", "oscillation_window", "oscillation_hold_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if set(self.eef_bounds) != set(LABELS):
            raise ValueError("explicit bounds required for all six Cartesian dimensions")
        for name in LABELS:
            lo, hi = vector(self.eef_bounds[name], (2,), name + " bounds")
            if lo > hi:
                raise ValueError("reversed Cartesian bounds")
        lo, hi = self.eef_bounds["pitch"]
        if lo <= -np.pi / 2 + 1e-6 or hi >= np.pi / 2 - 1e-6:
            raise ValueError("pitch bounds must exclude Euler singularities")
        self.joint_values(self.resync_thresholds, ik, "resync_thresholds")
        self.joint_values(self.oscillation_deadbands, ik, "oscillation_deadbands")
        speeds = vector([self.joint_speeds[n] for n in ik.arm_joints], (len(ik.arm_joints),), "joint_speeds")
        if np.any(speeds <= 0):
            raise ValueError("joint speeds must be positive")
        return speeds / self.control_hz

    @staticmethod
    def joint_values(mapping, ik, name):
        values = vector([mapping[n] for n in ik.arm_joints], (len(ik.arm_joints),), name)
        if np.any(values <= 0):
            raise ValueError(f"{name} must be positive")
        return values


@dataclass
class MoveResult:
    status: str
    reason: str
    steps_executed: int
    position_error_m: float
    angle_error_rad: float
    actual_pose: np.ndarray
    gripper_error: float | None = None
    nonconverged_steps: int = 0
    resync_steps: int = 0
    held_steps: int = 0
    rate_limited_steps: int = 0
    progress_fraction: float = 0.0
    arm_reached: bool = False
    gripper_target_reached: bool | None = None
    command_recovery_steps: int = 0
    execution_mode: str = "replace"
    command_position_error_m: float = 0.0
    command_angle_error_rad: float = 0.0
    max_joint_tracking_error: float = 0.0
    gripper_command: float | None = None
    gripper_command_complete: bool | None = None
    stagnation_steps: int = 0


class TrackingMoveToController:
    """Single-arm adaptation of upstream's tool + embodiment split.

    Retain one instance per episode/arm: orientation reference and previous
    command must survive separate move_to calls. A new episode
    resets them. There is no scene access, path collision check or grasp logic.
    """
    def __init__(self, port: RobotPort, ik: MinkIK, *, config: MoveConfig):
        self.port, self.ik, self.config = port, ik, config
        config.validate(ik)
        self.episode = None
        self.last_step = None
        self.reference = None
        self.previous = None
        self.goal = None
        self.goal_updates = None
        self.recovering = False
        self.previous_gripper = None

    def _start(self, observation, q):
        if self.episode != observation.episode_id:
            self.episode = observation.episode_id
            self.reference = pose(self.ik.fk(q))[:3, :3].copy()
            self.previous = q[self.ik.indices].copy()
            self.goal = self.goal_updates = None
            self.recovering = False
            self.previous_gripper = None
        elif self.last_step is not None and observation.step < self.last_step:
            raise ValueError("observation step regressed in same episode")
        self.last_step = observation.step

    def _joint_target(self, target, measured, step_limits):
        """Accept a complete IK solution or reject it; NEVER clip its joints.

        The caller reduces the Cartesian increment on rejection. Checking FK
        independently avoids trusting a solver's success flag or last iterate.
        No measured-state resync may bypass the last-command slew limit.
        """
        ik, cfg = self.ik, self.config
        # Keep non-commanded DOFs at CURRENT measured state for this IK solve.
        seed = measured.copy()
        seed[ik.indices] = self.previous
        ik.set_command_limits(step_limits, 1. / cfg.control_hz)
        answer = ik.solve(target, seed)
        if answer.reason == "qp_no_solution":
            return None, "ik_no_solution", True
        candidate = np.asarray(answer.q, dtype=float)
        if candidate.shape != measured.shape:
            raise ValueError("invalid IK joint shape")
        if answer.reason == "nonfinite_solution" or not np.all(np.isfinite(candidate)):
            return None, "ik_nonfinite", True
        if len(ik.frozen) and not np.allclose(candidate[ik.frozen], seed[ik.frozen], atol=1e-8, rtol=0):
            raise ValueError("IK changed a non-commanded joint")
        if np.any(candidate < ik.low) or np.any(candidate > ik.high):
            return None, "ik_joint_bounds", True
        p_error, r_error = errors(ik.fk(candidate), target)
        if p_error > ik.position_tolerance or r_error > ik.angle_tolerance:
            if np.any(np.abs(candidate[ik.indices] - self.previous) >= step_limits - 1e-8):
                return None, "joint_rate_limit", True
            return None, "ik_residual", True
        if np.any(np.abs(candidate[ik.indices] - self.previous) > step_limits + 1e-12):
            return None, "joint_rate_limit", not answer.success
        return candidate[ik.indices].copy(), "accepted", not answer.success

    def move_to(self, targets, *, mode="replace"):
        """Named xyz/yaw/pitch/roll/gripper targets; omitted dimensions hold.

        Also accepts SE(3) matrices for compatibility. The reference is fixed
        at first observation of the episode, not reset at each move. This does
        not pre-solve the path: later errors can follow already executed motion.
        The trusted port must hold on return/error/disconnect and enforce limits.
        replace (default) replaces the goal, withdrawing a lagging old command
        toward measured joints at the SAME command slew limit before planning.
        continue keeps the previous resolved goal, including omitted dimensions;
        pass {} or unchanged target dimensions. It cannot silently retarget.
        Recovery is joint-space command withdrawal, NOT collision-free planning
        or a guarantee of physical stopping. It preserves gripper drive targets.
        """
        if mode not in ("replace", "continue"):
            raise ValueError("mode must be replace or continue")
        ik, cfg = self.ik, self.config
        step_limits = cfg.validate(ik)
        latest = self.port.observe()
        measured = ik.read_q(latest)
        self._start(latest, measured)
        current = pose_values(ik.fk(measured), self.reference)
        labels = LABELS
        if latest.gripper is not None:
            if not np.isfinite(latest.gripper) or not 0 <= latest.gripper <= 1:
                raise ValueError("invalid measured gripper aperture")
            current = np.append(current, latest.gripper)
            labels += ("gripper",)
        if isinstance(targets, Mapping):
            if set(targets) - set(labels):
                raise ValueError("unknown target dimension or gripper observation unavailable")
            updates = dict(targets)
        else:
            updates = dict(zip(LABELS, pose_values(targets, self.reference)))
        target = current.copy()
        low = np.array([cfg.eef_bounds[n][0] for n in LABELS] + ([0.] if len(labels) == 7 else []))
        high = np.array([cfg.eef_bounds[n][1] for n in LABELS] + ([1.] if len(labels) == 7 else []))
        if np.any(current < low - 1e-8) or np.any(current > high + 1e-8):
            raise ValueError("measured Cartesian state outside published bounds")
        current = np.clip(current, low, high)
        target = current.copy()
        for name, value in updates.items():
            i = labels.index(name)
            target[i] = vector([value], (1,), name)[0]
            if target[i] < low[i] or target[i] > high[i]:
                raise ValueError("target outside published Cartesian bounds: " + name)
        if mode == "continue":
            if self.goal is None or self.goal.shape != target.shape:
                raise ValueError("no previous goal in this episode; use replace")
            for name in updates:
                if not np.isclose(target[labels.index(name)], self.goal[labels.index(name)],
                                  atol=1e-12, rtol=0):
                    raise ValueError("continue cannot change the goal; use replace")
            target = self.goal.copy()
            updates = self.goal_updates.copy()
            if np.any(target < low) or np.any(target > high):
                raise ValueError("previous goal outside current published bounds")
        # Start the command path at the previous sent command, not the lagging
        # measured pose. Omitted GOALS above still hold measured coordinates.
        # This avoids restarting behind an in-flight command on a later call.
        measured_start = current.copy()
        path_start_q = measured.copy()
        path_start_q[ik.indices] = self.previous
        current[:6] = pose_values(ik.fk(path_start_q), self.reference)
        if np.any(current < low - 1e-8) or np.any(current > high + 1e-8):
            raise ValueError("commanded Cartesian state outside published bounds")
        # Upstream's range backstop: no step exceeds 5% of a dimension range.
        speeds = np.array([cfg.linear_speed] * 3 + [cfg.angular_speed] * 3 +
                          ([cfg.gripper_speed] if len(labels) == 7 else []))
        limits = np.minimum(speeds / cfg.control_hz, (high - low) * .05)
        active = limits > 0
        if len(labels) == 7:
            active[6] = False  # arm path clock never controls gripper progress
        gripper_command = (float(latest.gripper) if self.previous_gripper is None
                           else self.previous_gripper) if "gripper" in updates else None
        gripper_step = limits[6] if "gripper" in updates else None
        def gripper_complete():
            return bool(gripper_command is None or abs(gripper_command - target[6]) <= 1e-12)
        ratio = float(np.max(np.abs(target[active] - current[active]) / limits[active], initial=0))
        target_pose = values_pose(target[:6], self.reference)
        nonconverged = resyncs = holds = rate_limited = recovery_steps = 0
        progress = 0.0
        stagnant = 0
        best_lag = math.inf
        stagnation_limit = max(1, math.ceil(cfg.stagnation_seconds * cfg.control_hz))

        def tracking_errors():
            prior_q = measured.copy()
            prior_q[ik.indices] = self.previous
            p_lag, r_lag = errors(ik.fk(measured), ik.fk(prior_q))
            return p_lag, r_lag, np.abs(self.previous - measured[ik.indices])

        def result(status, reason, count):
            actual = pose(ik.fk(measured))
            p_error, r_error = errors(actual, target_pose)
            g_error = (abs(float(latest.gripper) - target[6])
                       if "gripper" in updates else None)
            p_lag, r_lag, joint_lag = tracking_errors()
            return MoveResult(status, reason, count, p_error, r_error, actual,
                              g_error, nonconverged, resyncs, holds, rate_limited, progress,
                              bool(p_error <= cfg.position_tolerance and r_error <= cfg.angle_tolerance),
                              None if g_error is None else bool(g_error <= cfg.gripper_tolerance),
                              recovery_steps, mode, p_lag, r_lag, float(np.max(joint_lag)),
                              gripper_command, gripper_complete() if "gripper" in updates else None,
                              stagnant)

        def lagging():
            p_lag, r_lag, joint_lag = tracking_errors()
            return (p_lag > cfg.position_tolerance or r_lag > cfg.angle_tolerance or
                np.any(joint_lag >
                       cfg.joint_values(cfg.resync_thresholds, ik, "resync_thresholds")))

        def send_command(command, gripper=None):
            nonlocal latest, measured
            kwargs = dict(episode_id=latest.episode_id, expected_step=latest.step)
            if gripper is not None:
                kwargs["gripper"] = float(gripper)
            updated = self.port.command_joints(dict(zip(ik.arm_joints, map(float, command))), **kwargs)
            if updated.episode_id != latest.episode_id or updated.step != latest.step + 1:
                raise RuntimeError("invalid bridge response: episode/step changed")
            measured = ik.read_q(updated)
            if len(labels) == 7 and (updated.gripper is None or not np.isfinite(updated.gripper)
                                    or not 0 <= updated.gripper <= 1):
                raise RuntimeError("invalid bridge gripper observation")
            self.previous = command.copy()
            if gripper is not None:
                self.previous_gripper = float(gripper)
            latest, self.last_step = updated, updated.step

        # Same relative headroom and 10-second per-call cap as Toolset.
        needs_recovery = lagging() if mode == "replace" else self.recovering
        preflight_ratio = (float(np.max(np.abs(target[active] - measured_start[active]) /
                           limits[active], initial=0)) if needs_recovery else ratio)
        if gripper_command is not None:
            preflight_ratio = max(preflight_ratio, abs(target[6] - gripper_command) / gripper_step)
        headed_ratio = preflight_ratio / (1. - 1e-6)
        if not np.isfinite(headed_ratio) or headed_ratio > min(cfg.max_steps, math.ceil(10 * cfg.control_hz)):
            return result("rejected", "move_too_long", 0)
        steps = max(1, math.ceil(ratio / (1. - 1e-6)))
        budget = min(cfg.max_steps, math.ceil(10 * cfg.control_hz))
        if mode == "replace":
            self.goal, self.goal_updates = target.copy(), updates.copy()
            self.recovering = needs_recovery
        if self.recovering:
            # No IK candidate is clipped here. Explicitly withdraw the old
            # drive command in joint space, using live feedback each interval.
            # This also works when the plant cannot reach that old command.
            while lagging() and recovery_steps < budget:
                delta = measured[ik.indices] - self.previous
                command = self.previous + np.clip(delta, -step_limits, step_limits)
                send_command(command)  # NEVER open/re-close a held gripper here.
                recovery_steps += 1
            if lagging():
                return result("partial", "command_recovery_pending", recovery_steps)
            self.recovering = False
            # Replan from the withdrawn command and refreshed measured state.
            path_start_q = measured.copy()
            path_start_q[ik.indices] = self.previous
            current[:6] = pose_values(ik.fk(path_start_q), self.reference)
            if len(labels) == 7:
                current[6] = latest.gripper
            if np.any(current < low - 1e-8) or np.any(current > high + 1e-8):
                return result("partial", "recovery_outside_cartesian_bounds", recovery_steps)
            ratio = float(np.max(np.abs(target[active] - current[active]) / limits[active], initial=0))
            steps = max(1, math.ceil(ratio / (1. - 1e-6)))
            if steps > budget:
                return result("partial", "move_too_long_after_recovery", recovery_steps)
        accepted_waypoint = current.copy()
        infeasible_steps = 0
        reason = "step_budget_exhausted"
        for count in range(recovery_steps + 1, budget + 1):
            # Pause the waypoint clock while the robot catches up. Do not snap
            # the reference command to measured joints (which bypasses slew).
            is_lagging = lagging()
            command = None
            waypoint = accepted_waypoint.copy()
            step_approximate = step_limited = False
            if is_lagging:
                reason = "tracking_lag"
                p_lag, r_lag, joint_lag = tracking_errors()
                lag_score = max(p_lag / cfg.position_tolerance, r_lag / cfg.angle_tolerance,
                    float(np.max(joint_lag / cfg.joint_values(cfg.resync_thresholds, ik, "resync_thresholds"))))
                if best_lag - lag_score >= cfg.stagnation_min_improvement:
                    best_lag, stagnant = lag_score, 0
                else:
                    stagnant += 1
            else:
                best_lag, stagnant = math.inf, 0
                increment = min(1.0 / steps, 1.0 - progress)
                # Backtrack in Cartesian path space, re-solve, and send the
                # entire verified solution only when every joint can follow.
                for _ in range(12):
                    fraction = min(1.0, progress + increment)
                    if 1.0 - fraction < 1e-12:
                        fraction = 1.0
                    trial = np.clip(current + (target - current) * fraction, low, high)
                    candidate, why, approximate = self._joint_target(
                        values_pose(trial[:6], self.reference), measured, step_limits)
                    step_approximate |= approximate
                    step_limited |= why == "joint_rate_limit"
                    if candidate is not None:
                        command, waypoint, progress = candidate, trial, fraction
                        accepted_waypoint = waypoint.copy()
                        reason = "step_budget_exhausted"
                        break
                    reason = why
                    increment *= .5
            if command is None:
                command = self.previous.copy()
                holds += 1
                infeasible_steps = 0 if is_lagging else infeasible_steps + 1
            else:
                infeasible_steps = 0
            nonconverged += int(step_approximate)
            rate_limited += int(step_limited)
            # Separate clock: advance from the last SENT aperture, not measured
            # aperture or Cartesian progress. Loaded fingers need not reach it.
            if gripper_command is not None:
                delta = target[6] - gripper_command
                gripper_command = (float(target[6]) if abs(delta) <= gripper_step + 1e-12 else
                                   gripper_command + math.copysign(gripper_step, delta))
            send_command(command, gripper_command)
            final = result("partial", reason, count)
            if (progress == 1.0 and final.position_error_m <= cfg.position_tolerance
                    and final.angle_error_rad <= cfg.angle_tolerance and gripper_complete()):
                # Like YAM settling, do not require a loaded gripper to attain
                # its free-space aperture. This does NOT establish a grasp.
                final.status = "reached"
                final.reason = ("arm_reached_gripper_command_sent" if final.gripper_target_reached is False
                                else "measured_target_reached")
                return final
            if stagnant >= stagnation_limit and gripper_complete():
                final.reason = "tracking_stalled"
                return final  # feedback, NOT proof of collision or unreachable IK
            if infeasible_steps >= 10:
                final.reason = "path_not_followable:" + reason
                return final
        return result("partial", reason, budget)


class MoveToController(TrackingMoveToController):
    """Reference-style ordered playout, default no per-waypoint settling.

    See UPSTREAM.md. The v4 tracking implementation above is retained only for
    regression comparison. This class never calls its move_to/_joint_target.
    G1 adaptation: final sent-command slew remains bounded even after resync.
    Arrival is measured, not inferred from exhausting the action chunk.
    """
    def _start(self, observation, q):
        changed = self.episode != observation.episode_id
        super()._start(observation, q)
        if changed:
            self.previous = np.clip(self.previous, self.ik.low[self.ik.indices], self.ik.high[self.ik.indices])
            self._holds = 0
            self._reversals = []
            self._delta = None

    def move_to(self, targets, *, mode="replace"):
        if mode not in ("replace", "continue"):
            raise ValueError("mode must be replace or continue")
        ik, cfg = self.ik, self.config
        step_limits = cfg.validate(ik)
        latest = self.port.observe()
        measured = ik.read_q(latest)
        self._start(latest, measured)
        start = pose_values(ik.fk(measured), self.reference)
        labels = LABELS
        if latest.gripper is not None:
            if not np.isfinite(latest.gripper) or not 0 <= latest.gripper <= 1:
                raise ValueError("invalid measured gripper aperture")
            labels += ("gripper",)
            start = np.append(start, latest.gripper)
        updates = dict(targets) if isinstance(targets, Mapping) else dict(zip(LABELS, pose_values(targets, self.reference)))
        if set(updates) - set(labels):
            raise ValueError("unknown target dimension or gripper observation unavailable")
        target = start.copy()
        low = np.array([cfg.eef_bounds[n][0] for n in LABELS] + ([0.] if len(labels)==7 else []))
        high = np.array([cfg.eef_bounds[n][1] for n in LABELS] + ([1.] if len(labels)==7 else []))
        for name, value in updates.items():
            i = labels.index(name)
            target[i] = vector([value], (1,), name)[0]
            if not low[i] <= target[i] <= high[i]:
                raise ValueError("target outside published Cartesian bounds: " + name)
        if mode == "continue":
            if self.goal is None or self.goal.shape != target.shape:
                raise ValueError("no previous goal in this episode; use replace")
            if any(abs(target[labels.index(n)]-self.goal[labels.index(n)]) > 1e-12 for n in updates):
                raise ValueError("continue cannot change the goal; use replace")
            target, updates = self.goal.copy(), self.goal_updates.copy()
        if np.any(target < low) or np.any(target > high):
            raise ValueError("goal outside current published bounds")
        # Explicit gripper starts at last command, not aperture under load;
        # omission does not send or reopen a held gripper (Arena bridge contract).
        grip = None
        if "gripper" in updates:
            grip = float(latest.gripper if self.previous_gripper is None else self.previous_gripper)
            start[6] = grip
        speeds = np.array([cfg.linear_speed]*3 + [cfg.angular_speed]*3 + ([cfg.gripper_speed] if len(labels)==7 else []))
        limits = np.minimum(speeds/cfg.control_hz, (high-low)*.05)
        distances = np.abs(target-start)
        if len(labels)==7 and grip is None:
            distances[6] = 0
        if np.any((limits==0) & (distances>1e-12)):
            raise ValueError("motion along a fixed dimension")
        ratio = float(np.max(np.divide(distances, limits, out=np.zeros_like(distances), where=limits>0), initial=0))
        headed = ratio/(1.-1e-6)
        budget = min(cfg.max_steps, math.ceil(10*cfg.control_hz))
        target_pose = values_pose(target[:6], self.reference)
        holds = rates = resyncs = nonconverged = executed = 0
        def result(reason):
            actual = ik.fk(measured).copy()
            pe, ae = errors(actual, target_pose)
            cq = measured.copy(); cq[ik.indices] = self.previous
            cp, ca = errors(actual, ik.fk(cq))
            ge = abs(float(latest.gripper)-target[6]) if grip is not None else None
            reached = pe <= cfg.position_tolerance and ae <= cfg.angle_tolerance
            complete = grip is None or abs(grip-target[6]) <= 1e-12
            return MoveResult(status="rejected" if reason=="move_too_long" else "reached" if reached and complete else "partial",
                reason="measured_target_reached" if reached and complete and reason!="move_too_long" else reason,
                steps_executed=executed, position_error_m=pe, angle_error_rad=ae, actual_pose=actual,
                gripper_error=ge, nonconverged_steps=nonconverged, resync_steps=resyncs,
                held_steps=holds, rate_limited_steps=rates, progress_fraction=float(executed/steps) if executed else 0.,
                arm_reached=bool(reached), gripper_target_reached=None if ge is None else bool(ge<=cfg.gripper_tolerance),
                execution_mode=mode, command_position_error_m=cp, command_angle_error_rad=ca,
                max_joint_tracking_error=float(np.max(np.abs(self.previous-measured[ik.indices]))),
                gripper_command=grip, gripper_command_complete=None if grip is None else bool(complete))
        steps = max(1, math.ceil(headed)) if math.isfinite(headed) else budget+1
        if steps > budget:
            return result("move_too_long")
        self.goal, self.goal_updates = target.copy(), updates.copy()
        for fraction in np.linspace(1./steps, 1., steps):
            waypoint = start+(target-start)*fraction
            command = self.previous.copy()
            resynced = False
            if self._holds:
                self._holds -= 1
                holds += 1
                if not self._holds:
                    self._reversals.clear(); self._delta=None
            else:
                origin = self.previous.copy()
                if np.any(np.abs(origin-measured[ik.indices]) > cfg.joint_values(cfg.resync_thresholds,ik,"resync_thresholds")):
                    origin = np.clip(measured[ik.indices], ik.low[ik.indices], ik.high[ik.indices])
                    resyncs += 1; resynced = True
                seed = measured.copy(); seed[ik.indices] = origin
                answer = ik.reference_solve(values_pose(waypoint[:6],self.reference),seed)
                candidate = np.asarray(answer.q,dtype=float)
                nonconverged += int(not answer.success)
                if candidate.shape != measured.shape:
                    raise ValueError("invalid IK joint shape")
                if answer.reason in ("qp_no_solution","nonfinite_solution") or not np.all(np.isfinite(candidate)):
                    holds += 1
                else:
                    if len(ik.frozen) and not np.allclose(candidate[ik.frozen],seed[ik.frozen],atol=1e-8,rtol=0):
                        raise ValueError("IK changed a non-commanded joint")
                    bounded = np.clip(candidate[ik.indices],ik.low[ik.indices],ik.high[ik.indices])
                    raw_delta = bounded-origin
                    desired = origin+np.clip(raw_delta,-step_limits,step_limits)
                    # Extra G1 guard: resync cannot bypass actual sent-command slew.
                    delta = np.clip(desired-self.previous,-step_limits,step_limits)
                    rates += int(np.any(np.abs(raw_delta)>step_limits) or np.any(np.abs(desired-self.previous)>step_limits))
                    reversal = np.zeros(len(delta),dtype=bool)
                    dead = cfg.joint_values(cfg.oscillation_deadbands,ik,"oscillation_deadbands")
                    if not resynced and self._delta is not None:
                        reversal = (np.abs(delta)>dead)&(np.abs(self._delta)>dead)&(np.signbit(delta)!=np.signbit(self._delta))
                    self._reversals.append(reversal)
                    self._reversals = self._reversals[-cfg.oscillation_window:]
                    if not resynced: self._delta=delta.copy()
                    if np.any(np.sum(self._reversals,axis=0)>cfg.oscillation_reversals):
                        self._holds=cfg.oscillation_hold_steps
                        self._reversals.clear();self._delta=None;holds+=1
                    else:
                        command = np.clip(self.previous+delta,ik.low[ik.indices],ik.high[ik.indices])
            kwargs = dict(episode_id=latest.episode_id,expected_step=latest.step)
            if grip is not None:
                grip=float(waypoint[6]);kwargs['gripper']=grip
            updated=self.port.command_joints(dict(zip(ik.arm_joints,map(float,command))),**kwargs)
            if updated.episode_id!=latest.episode_id or updated.step!=latest.step+1:
                raise RuntimeError("invalid bridge response: episode/step changed")
            self.previous=command.copy()
            if grip is not None:self.previous_gripper=grip
            latest=updated;self.last_step=updated.step;measured=ik.read_q(updated);executed+=1
        return result("playout_complete_not_reached")


def move_to(port: RobotPort, ik: MinkIK, targets, *, config: MoveConfig, mode="replace"):
    """One-shot convenience. Use persistent MoveToController for an episode."""
    return MoveToController(port, ik, config=config).move_to(targets, mode=mode)

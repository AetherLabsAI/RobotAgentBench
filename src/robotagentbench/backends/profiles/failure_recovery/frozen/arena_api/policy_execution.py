"""Versioned, task-independent joint target controller; no asset/drive edits.

Requested native targets are held unchanged across every physics substep.
Legacy math helpers below are retained only for historical experiment tests;
the runtime registry and worker no longer select or execute them.
Tracking error is recorded, not used as a stopping threshold. This is NOT
collision avoidance or identical robots, tasks, drives, or evaluation behavior.
"""
import math
import os

from . import bimanual as b, robot_profiles as rp
from .policy_contract import ORDER, GRIP_RANGE, validate_chunk
from . import physics_defaults

TASK_ID = "isaac_policy_execution"
PROFILE = "arena.g1.position_controller.v14_raw_target_hold_20hz"
TARGET_INTERPOLATION = os.environ.get("ARENA_TARGET_INTERPOLATION", "none")
if TARGET_INTERPOLATION != "none":
    raise ValueError("Raw trajectory control requires ARENA_TARGET_INTERPOLATION=none")
DEFAULT_GRIPPER_MODE = "direct_target.v1"
CLOSE_DURATION_S = 0.6
CLOSE_THRESHOLD = 0.5
GRIPPER_INTERPOLATION = os.environ.get("ARENA_GRIPPER_INTERPOLATION", "measured")
if GRIPPER_INTERPOLATION not in ("measured", "previous_target"):
    raise ValueError("Invalid ARENA_GRIPPER_INTERPOLATION")
PHYSICS_DT = physics_defaults.PHYSICS_DT
SUBSTEPS = physics_defaults.steps_for_seconds(b.CONTROL_DT)
MAX_STEPS = 5000
GRIPPER_MAX_LEAD_FRACTION = 0.2
# Reset validation: the user-approved waist-only settle tolerance is 0.02 deg.
# This is measurement acceptance, not a physical joint/drive or command limit.
# Other joints retain the strict numerical tolerance. Never project raw state.
SETTLED_LIMIT_TOLERANCE = {"rad": 1e-5, "m": 1e-5}
SETTLED_LIMIT_TOLERANCE_BY_JOINT = {"idx02_body_joint2": math.radians(0.02)}


def spec(robot_profile=rp.JOINT16):
    p = rp.get(robot_profile)
    return {"profile_id": profile_id(robot_profile), "robot_profile": p.id,
            "robot_adapter": rp.specification(p.id),
            "mode": "joint_position", "unit": p.wire_unit, "joint_units": p.units,
            "joint_names": list(p.joints), "all_joints_required": True,
            "control_dt_s": b.CONTROL_DT, "physics_dt_s": PHYSICS_DT,
            "physics_substeps": SUBSTEPS, "max_episode_steps": MAX_STEPS,
            "physics_profile": physics_defaults.PROFILE,
            "solver_position_iterations": physics_defaults.POSITION_ITERATIONS,
            "solver_velocity_iterations": physics_defaults.VELOCITY_ITERATIONS,
            "limits_source": "USD joint limits, no reset-relative envelope",
            "arm_out_of_range": "reject", "gripper_mapping": "native trajectory values unchanged; normalized model IO uses n * pi/4, out-of-range rejected",
            "target_interpolation_mode": TARGET_INTERPOLATION,
            "interpolation": ("none; requested target held across all physics substeps"
                if TARGET_INTERPOLATION == "none" else
                "measured start to target, alpha=1/4,1/2,3/4,1"
                if GRIPPER_INTERPOLATION == "measured" else
                "arms: measured start; grippers: previous issued target; alpha=1/4,1/2,3/4,1"),
            "measured_limit_tolerance_by_unit": None,
            "settled_limit_tolerance_by_unit": dict(SETTLED_LIMIT_TOLERANCE),
            "settled_limit_tolerance_by_joint": {
                n: v for n, v in SETTLED_LIMIT_TOLERANCE_BY_JOINT.items() if n in p.joints},
            "settled_limit_policy": "waist_0.02deg_v1; joint overrides unit default; acceptance only; commands remain within USD limits; raw state unchanged",
            "measured_limit_policy": "record_only; preserve raw state; reject nonfinite feedback",
            "arm_target_slew_limit": None,
            "gripper_target_slew_limit": None,
            "gripper_execution_mode": DEFAULT_GRIPPER_MODE,
            "gripper_close_ramp": None,
            "gripper_close_zero": None,
            "gripper_max_measured_lead_fraction": None,
            "gripper_bound_reference": "native USD limits only",
            "gripper_surface_stop": None,
            "max_tracking_error_rad": None,
            "tracking_error_policy": "record_only; no threshold stop",
            "reference_commit": "RoboDojo ee67a1468510da7624a089164402359f2afc72c8",
            "reference_scope": "Historical reference only; current execution holds original native targets without interpolation or re-control",
            "bound_scope": "native joint target limits; no gripper measured-lead or collision safety",
            "physics_configuration_changed": False, "task_scoring_ready": False}


def profile_id(robot_profile=rp.JOINT16):
    rp.get(robot_profile)
    selected = PROFILE
    if TARGET_INTERPOLATION == "measured":
        selected += "_measured"
        if GRIPPER_INTERPOLATION == "previous_target":
            selected += "_gripper_previous_target_probe1"
    return selected if robot_profile == rp.JOINT16 else selected + ".joint20"


def _values(values, label, robot_profile=rp.JOINT16):
    rp.values(values, robot_profile, label)


def _limits(limits, robot_profile=rp.JOINT16):
    if set(limits) != set(rp.get(robot_profile).joints):
        raise ValueError("Incomplete USD limits")
    for name, pair in limits.items():
        if (len(pair) != 2 or any(isinstance(q, bool) or not isinstance(q, (int, float))
                                 or not math.isfinite(q) for q in pair) or pair[0] >= pair[1]):
            raise ValueError(f"Invalid USD limits: {name}")


def validate(targets, limits, robot_profile=rp.JOINT16):
    _values(targets, "targets", robot_profile)
    _limits(limits, robot_profile)
    for name, value in targets.items():
        if not limits[name][0] <= value <= limits[name][1]:
            raise ValueError(f"Target outside USD limits: {name}")


def build_limits(initial, asset_limits, robot_profile=rp.JOINT16):
    limits = {name: list(asset_limits[name]) for name in rp.get(robot_profile).joints if name in asset_limits}
    _limits(limits, robot_profile)
    _values(initial, "initial state", robot_profile)
    reference = {}
    for name, q in initial.items():
        low, high = limits[name]
        tolerance = SETTLED_LIMIT_TOLERANCE_BY_JOINT.get(
            name, SETTLED_LIMIT_TOLERANCE[rp.get(robot_profile).units[name]])
        if not low - tolerance <= q <= high + tolerance:
            raise ValueError(f"Settled joint outside USD limits: {name}")
        reference[name] = min(high, max(low, q))  # command reference only, never measured/physical state
    for name in b.GRIPPERS:
        if abs(limits[name][0]) > 1e-6 or abs(limits[name][1] - GRIP_RANGE) > 1e-6:
            raise ValueError("Asset gripper limits disagree with joint16 IO profile")
    return reference, limits


def map_action(row, limits, robot_profile=rp.JOINT16):
    validate_chunk([row] * 30, robot_profile)
    targets, changes = {}, []
    for name, value in zip(rp.get(robot_profile).order, row):
        raw = float(value)
        if name in b.GRIPPERS:
            if not 0. <= raw <= 1.:
                raise ValueError('Normalized model gripper outside [0,1]: ' + name)
            targets[name] = raw * GRIP_RANGE  # IO unit conversion only; native replay bypasses this.
        else:
            targets[name] = raw
    validate(targets, limits, robot_profile)  # Never silently clip arm predictions
    return targets, changes


def tracking_errors(measured, previous, robot_profile=rp.JOINT16):
    """Absolute errors for audit only; finite-state validation remains mandatory."""
    _values(measured, "measured state", robot_profile)
    _values(previous, "previous issued target", robot_profile)
    return {name: abs(measured[name] - previous[name]) for name in rp.get(robot_profile).joints}


def plan_interval(targets, measured, previous, limits, robot_profile=rp.JOINT16, *, physics_hz=240):
    """One control interval of raw targets; no hidden warmup or interpolation."""
    validate(targets, limits, robot_profile)
    validate(previous, limits, robot_profile)
    initial_errors = tracking_errors(measured, previous, robot_profile)
    from .physics_rate import validate_hz
    sequence = [dict(targets) for _ in range(validate_hz(physics_hz) // b.CONTROL_HZ)]
    return {"profile_id": profile_id(robot_profile), "robot_profile": robot_profile,
            "target_interpolation_mode": TARGET_INTERPOLATION,
            "joint_units": rp.get(robot_profile).units, "requested_targets_rad": dict(targets),
            "measured_start_rad": dict(measured), "previous_issued_targets_rad": dict(previous),
            "tracking_error_start_rad": initial_errors,
            "interpolated_substep_targets_rad": sequence,
            "substep_targets_rad": [], "measured_before_substep_rad": [],
            "measured_after_substep_rad": [], "tracking_error_after_substep_rad": [],
            "interventions": [],
            "applied_substeps": 0}


def prepare_substep(interpolated, measured, limits, index, robot_profile=rp.JOINT16):
    """Call again after EVERY physical tick; never precompute from stale feedback.

    Arms pass through. Each primary gripper target may lead the fresh measured
    position (projected to USD limits) by at most 20% of its USD travel
    (9 degrees on this G1). Raw feedback is preserved. The robot adapter expands
    paired inner targets after this bound. No reference to prior commands.
    """
    validate(interpolated, limits, robot_profile)
    _values(measured, "measured state", robot_profile)
    issued, changes = dict(interpolated), []
    for name in b.GRIPPERS:
        lead = GRIPPER_MAX_LEAD_FRACTION * (limits[name][1] - limits[name][0])
        desired, raw_actual = interpolated[name], measured[name]
        low, high = limits[name]
        actual = min(high, max(low, raw_actual))
        if actual != raw_actual:
            changes.append({"substep": index, "joint": name, "measured_rad": raw_actual,
                            "bounded_reference_rad": actual, "correction": actual - raw_actual,
                            "lower_limit": low, "upper_limit": high,
                            "reason": "gripper_measured_reference_projection"})
        issued[name] = min(actual + lead, max(actual - lead, desired))
        if abs(issued[name] - desired) > 1e-12:
            changes.append({"substep": index, "joint": name, "interpolated_rad": desired,
                            "measured_rad": raw_actual, "bounded_reference_rad": actual, "max_lead_rad": lead,
                            "issued_rad": issued[name], "reason": "gripper_measured_lead_bound"})
    validate(issued, limits, robot_profile)
    return issued, changes


def direct_gripper_substep(requested, interpolated, measured, limits, robot_profile=rp.JOINT16):
    """Pass through ALL native targets, irrespective of contact or tracking lag."""
    validate(requested, limits, robot_profile)
    validate(interpolated, limits, robot_profile)
    _values(measured, 'measured state', robot_profile)
    if interpolated != requested:
        raise ValueError('Raw-target mode forbids intermediate target rewriting')
    issued = dict(requested)
    validate(issued, limits, robot_profile)
    return issued, []


class CloseZero:
    """Close commands target zero immediately, even when already holding an object.

    Preserve the existing half-travel close/open classification. Only the drive
    target changes; feedback, actual joint positions, drives and contacts do not.
    There is no pre-opening, timer, measured-lead bound or contact-stop override.
    """
    def apply(self, requested, interpolated, measured, limits, dt,
              robot_profile=rp.JOINT16):
        validate(requested, limits, robot_profile)
        validate(interpolated, limits, robot_profile)
        _values(measured, 'measured state', robot_profile)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid close-zero physics dt')
        for name in b.GRIPPERS:
            if abs(limits[name][0]) > 1e-6 or abs(limits[name][1] - GRIP_RANGE) > 1e-6:
                raise ValueError('Close zero requires native G1 gripper limits')
        issued, evidence = dict(interpolated), []
        for name in b.GRIPPERS:
            low, high = limits[name]
            closing = requested[name] <= low + CLOSE_THRESHOLD * (high - low)
            if closing:
                issued[name] = 0.0
            evidence.append(dict(joint=name, phase='hold_zero' if closing else 'open',
                requested_native=requested[name], issued_native=issued[name],
                mode='close_zero.v1', contact_stop_applied=False, measured_lead_applied=False))
        validate(issued, limits, robot_profile)
        return issued, evidence


class ContactCloseZero:
    """Original target until current physical opposing-pad contact is confirmed.

    Reuse the existing SurfaceStop latch criterion, not its force controller.
    No persistent grasp latch: losing contact restores the requested trajectory.
    No actual position, drive, object, scoring or collision changes.
    """
    def __init__(self):
        self.last_sample = None

    def apply(self, requested, interpolated, measured, limits, dt,
              robot_profile=rp.JOINT16, *, contacts, sample):
        from .gripper_surface_stop import Config, norm

        # Reuse existing target/feedback/dt/limit validation and close classifier.
        _, classification = CloseZero().apply(
            requested, interpolated, measured, limits, dt, robot_profile)
        if type(sample) is not int or sample <= 0:
            raise RuntimeError('Missing completed gripper contact sample')
        if self.last_sample is not None and sample <= self.last_sample:
            raise RuntimeError('Stale/reused gripper contact sample')
        if not isinstance(contacts, dict) or set(contacts) != set(b.GRIPPERS):
            raise ValueError('Incomplete gripper contact feedback')
        for groups in contacts.values():
            if not isinstance(groups, dict):
                raise ValueError('Invalid gripper contact groups')
            for body, forces in groups.items():
                if not isinstance(body, str) or not body:
                    raise ValueError('Invalid contact body')
                if len(forces) != 2 or any(len(v) != 3 for v in forces):
                    raise ValueError('Invalid pad forces')
                if any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for v in forces for x in v):
                    raise ValueError('Nonfinite or invalid pad force')
        issued, evidence = dict(interpolated), []
        threshold = Config().latch_force_n
        for item in classification:
            name = item['joint']
            closing = item['phase'] == 'hold_zero'
            eligible = []
            for body, (a, c) in contacts[name].items():
                loads = (norm(a), norm(c))
                if min(loads) >= threshold and sum(x*y for x, y in zip(a, c)) < 0:
                    eligible.append(body)
            confirmed = bool(eligible)
            if closing and confirmed:
                issued[name] = 0.0
            evidence.append(dict(
                joint=name, sample=sample, mode='contact_close_zero.v1',
                phase='contact_zero' if closing and confirmed else 'original_target',
                closing_command=closing, contact_confirmed=confirmed,
                contact_bodies=sorted(eligible), contact_threshold_n=threshold,
                requested_native=requested[name], issued_native=issued[name],
                contact_stop_applied=False, measured_lead_applied=False))
        validate(issued, limits, robot_profile)
        self.last_sample = sample
        return issued, evidence


class CloseRamp:
    """Episode-local, per-hand 1->0 target ramp, followed by persistent zero.

    This is a position-target schedule, not force control or a physical pose write.
    Native targets <= half travel trigger closing. Opening cancels immediately;
    wall time, contacts and stalled measured motion never restart the ramp.
    Targets below are sampled at the END of each physics interval.
    """
    def __init__(self):
        self.elapsed = {name: None for name in b.GRIPPERS}

    def apply(self, requested, interpolated, measured, limits, dt,
              robot_profile=rp.JOINT16):
        validate(requested, limits, robot_profile)
        validate(interpolated, limits, robot_profile)
        _values(measured, 'measured state', robot_profile)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid close-ramp physics dt')
        # Validate both hands before changing either state.
        for name in b.GRIPPERS:
            if abs(limits[name][0]) > 1e-6 or abs(limits[name][1] - GRIP_RANGE) > 1e-6:
                raise ValueError('Close ramp requires native G1 gripper limits')
        issued, evidence = dict(interpolated), []
        for name in b.GRIPPERS:
            low, high = limits[name]
            closing = requested[name] <= low + CLOSE_THRESHOLD * (high - low)
            if not closing:
                self.elapsed[name] = None
                phase = 'open'
            else:
                elapsed = (self.elapsed[name] or 0.0) + dt
                elapsed = CLOSE_DURATION_S if elapsed >= CLOSE_DURATION_S - 1e-12 else elapsed
                self.elapsed[name] = elapsed
                phase = 'hold' if elapsed == CLOSE_DURATION_S else 'closing'
                issued[name] = 0.0 if phase == 'hold' else high * (1.0 - elapsed / CLOSE_DURATION_S)
            evidence.append(dict(joint=name, phase=phase, elapsed_sim_s=self.elapsed[name],
                requested_native=requested[name], issued_native=issued[name],
                mode='close_ramp.v1', contact_stop_applied=False, measured_lead_applied=False))
        validate(issued, limits, robot_profile)
        return issued, evidence


def evidence_keys(robot_profile):
    """The 20-axis profile never labels torso meters as radians."""
    rp.get(robot_profile)
    return {key: (key if robot_profile == rp.JOINT16 else key.replace("_rad", "_native"))
            for key in ("requested_targets_rad", "measured_start_rad", "previous_issued_targets_rad",
                        "tracking_error_start_rad", "interpolated_substep_targets_rad",
                        "substep_targets_rad", "measured_before_substep_rad", "measured_after_substep_rad",
                        "tracking_error_after_substep_rad")}


def wire_evidence(execution, robot_profile):
    names = evidence_keys(robot_profile)
    return {names.get(k, k): v for k, v in execution.items()}


def action_metadata(initial, limits, robot_profile):
    validate(initial, limits, robot_profile)
    suffix = "rad" if robot_profile == rp.JOINT16 else "native"
    return {"action_limits_" + suffix: limits, "initial_action_targets_" + suffix: dict(initial),
            "action_spec": spec(robot_profile)}

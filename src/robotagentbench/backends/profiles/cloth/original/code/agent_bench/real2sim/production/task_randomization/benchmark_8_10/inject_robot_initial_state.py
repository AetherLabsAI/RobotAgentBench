"""Sample and apply Sheet 8.10 robot reset perturbations before physics.

The benchmark owns this entrypoint. A data producer and an evaluation reset call
exactly the same functions. This module does not alter cloth state or task goals.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

SCHEMA = "tablecloth.robot_initial_perturbation.v1"
DEFAULT_BASE_CENTER = np.array([.009729688908293765, -.0016407338778981157, .37994640163620486])
DEFAULT_HALF_WIDTH = np.array([.0015, .0015, .08] + [.025] * 14)


def sample_initial_perturbation(seed: int) -> dict:
    """Independent x/y/yaw and 14 arm-joint offsets in the tested local range.

    These bounds describe the sampling distribution, not a robustness guarantee.
    All samples, including physical failures, retain their original seed.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    value = np.random.default_rng(int(seed)).uniform(-1., 1., 17) * DEFAULT_HALF_WIDTH
    value[:3] += DEFAULT_BASE_CENTER
    return dict(
        schema=SCHEMA, benchmark_task_id="8.10", seed=int(seed),
        base_delta_xyyaw_m_deg=value[:3].tolist(), arm_delta_rad=value[3:].tolist(),
        sampling=dict(distribution="independent_uniform", base_center_xyyaw_m_deg=DEFAULT_BASE_CENTER.tolist(),
                      half_width=DEFAULT_HALF_WIDTH.tolist(), units=["m", "m", "deg"] + ["rad"] * 14),
    )


def validate_initial_perturbation(injection: dict) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(injection, dict) or injection.get("schema") != SCHEMA:
        raise ValueError("Expected a tablecloth robot initial perturbation record")
    if injection.get("benchmark_task_id", "8.10") != "8.10":
        raise ValueError("Initial perturbation belongs to another Sheet task")
    seed = injection.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Missing or invalid perturbation seed")
    base = np.asarray(injection.get("base_delta_xyyaw_m_deg"), dtype=float)
    arm = np.asarray(injection.get("arm_delta_rad"), dtype=float)
    if base.shape != (3,) or arm.shape != (14,) or not np.isfinite(base).all() or not np.isfinite(arm).all():
        raise ValueError("Expected finite base[3] and both-arm[14] offsets")
    return base, arm


def apply_initial_perturbation(sim, injection: dict) -> dict:
    """Reset a constructed simulator once, with no running step or attachment.

    Validate the complete target before changing any robot value. Then update
    previous/current prescribed rigid poses together, so the first step starts
    from the perturbed state rather than sweeping from the nominal robot.
    """
    base_delta, arm_delta = validate_initial_perturbation(injection)
    if str(sim.args.task) != "8.10":
        raise ValueError("This initializer belongs to Sheet 8.10")
    if sim.t != 0 or sim.frames or sim.clamps or getattr(sim, "initial_perturbation_applied", False):
        raise ValueError("Perturbation must occur once before physics with unattached cloth")
    names = sim.robot.arm_names("L") + sim.robot.arm_names("R")
    if len(names) != 14 or len(set(names)) != 14:
        raise ValueError("Expected 14 distinct arm joints")
    nominal = np.asarray([sim.hands.values[n] for n in sim.robot.names], dtype=float)
    nominal_base = sim.robot.base.copy()
    targets = sim.hands.values.copy()
    for name, delta in zip(names, arm_delta):
        target = targets[name] + float(delta)
        low, high = sim.robot.by_name[name]["limits"]
        if not low + .001 <= target <= high - .001:
            raise ValueError(f"Initial perturbation exceeds joint limit: {name}")
        targets[name] = target
    sim.robot.base[:2, 3] += base_delta[:2]
    angle = np.deg2rad(base_delta[2])
    cosine, sine = np.cos(angle), np.sin(angle)
    yaw_rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    sim.robot.base[:3, :3] = yaw_rotation @ nominal_base[:3, :3]
    sim.hands.values = targets
    sim.hands.apply_grips()
    body = sim.body_transforms()
    sim.previous_body = body.copy()
    sim.previous_grip = sim.hands.grip.copy()
    sim.start.assign(body)
    sim.stop.assign(body)
    for state in (sim.a, sim.b):
        state.body_q.assign(body)
    sim.initial_perturbation_applied = True
    actual_joint = np.asarray([sim.hands.values[n] for n in sim.robot.names], dtype=float)
    actual_base = np.r_[sim.robot.base[:2, 3], np.degrees(np.arctan2(sim.robot.base[1, 0], sim.robot.base[0, 0]))]
    evidence = copy.deepcopy(injection)
    evidence.update(
        applied_before_first_physics_step=True, cloth_unattached_at_injection=True,
        actual_initial_joint_position=actual_joint.tolist(), nominal_initial_joint_position=nominal.tolist(),
        joint_names=list(sim.robot.names), actual_base_xyyaw_m_deg=actual_base.tolist(),
        nominal_base_transform=nominal_base.tolist(), arm_joint_names=names,
    )
    return evidence


def check_initial_perturbation(state, evidence: dict) -> dict:
    """Recompute applied offsets from saved initial arrays; never trust a flag alone."""
    base_delta, arm_delta = validate_initial_perturbation(evidence)
    names = list(state["joint_names"])
    arm_names = evidence.get("arm_joint_names") or [f"idx{21 + j}_arm_l_joint{1 + j}" for j in range(7)] + [f"idx{61 + j}_arm_r_joint{1 + j}" for j in range(7)]
    if len(arm_names) != 14 or any(n not in names for n in arm_names):
        raise ValueError("Cannot identify both arm joint arrays")
    nominal = np.asarray(evidence["nominal_initial_joint_position"], dtype=float)
    actual = np.asarray(state["initial_joint_position"], dtype=float)
    expected = np.asarray(evidence["actual_initial_joint_position"], dtype=float)
    base = np.asarray(state["base_xyyaw_m_deg"], dtype=float)
    nominal_base = np.asarray(evidence["nominal_base_transform"], dtype=float)
    if actual.shape != (len(names),) or nominal.shape != actual.shape or expected.shape != actual.shape or nominal_base.shape != (4, 4) or base.ndim != 2 or base.shape[1] != 3 or not len(base):
        raise ValueError("Invalid initial state evidence dimensions")
    indices = [names.index(n) for n in arm_names]
    nominal_yaw = np.degrees(np.arctan2(nominal_base[1, 0], nominal_base[0, 0]))
    actual_delta = base[0] - np.r_[nominal_base[:2, 3], nominal_yaw]
    yaw_error = (actual_delta[2] - base_delta[2] + 180) % 360 - 180
    gates = dict(
        finite_initial_state=bool(np.isfinite(actual).all() and np.isfinite(base).all()),
        actual_joint_record_matches=bool(np.allclose(actual, expected, atol=1e-8, rtol=0.)),
        arm_offsets_applied=bool(np.allclose(actual[indices] - nominal[indices], arm_delta, atol=1e-8, rtol=0.)),
        base_offsets_applied=bool(np.allclose(actual_delta[:2], base_delta[:2], atol=1e-8, rtol=0.) and abs(yaw_error) <= 1e-8),
        fixed_initial_base=bool(np.allclose(base, base[0], atol=1e-8, rtol=0.)),
        applied_before_first_physics_step=evidence.get("applied_before_first_physics_step") is True,
        cloth_initially_unattached=evidence.get("cloth_unattached_at_injection") is True,
    )
    return dict(success=all(gates.values()), gates=gates, reasons=[k for k, v in gates.items() if not v])



def check_episode_perturbation(episode_directory) -> dict:
    """Owner adapter for the central pre-render gate, using measured arrays."""
    root = Path(episode_directory)
    try:
        evidence = json.loads((root / "initial_perturbation.json").read_text())
        with np.load(root / "full_state.npz", allow_pickle=False) as state:
            result = check_initial_perturbation(state, evidence)
            gates = dict(result["gates"])
            with np.load(root / "action.npz", allow_pickle=False) as action:
                gates["action_state_correspondence"] = bool(
                    np.array_equal(action["joint_target"], state["joint_position"])
                    and np.array_equal(action["time_s"], state["time_s"])
                    and np.isfinite(action["joint_target"]).all())
            gates["initial_pose_observed_before_action"] = bool(
                np.allclose(state["joint_position"][0], state["initial_joint_position"], atol=1e-8, rtol=0)
                and np.allclose(state["body_q"][0], state["initial_body_q"], atol=1e-8, rtol=0))
        base, arm = validate_initial_perturbation(evidence)
        gates["nonzero_base_delta"] = bool(np.any(base != 0))
        gates["nonzero_arm_delta"] = bool(np.any(arm != 0))
        return dict(status="passed" if all(gates.values()) else "unverified", gates=gates,
                    metrics=dict(base_delta_xyyaw_m_deg=base.tolist(), arm_max_abs_rad=float(abs(arm).max())),
                    reasons=[key for key, value in gates.items() if not value])
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return dict(status="unverified", gates={"evidence_readable": False}, metrics={},
                    reasons=[f"{type(exc).__name__}: {exc}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = sample_initial_perturbation(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()

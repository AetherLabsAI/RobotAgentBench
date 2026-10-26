"""Execute independent, physically perturbed Sheet 8.10 benchmark episodes.

Setup, injection and success functions are also reusable by evaluation. This
entrypoint produces action/state and evidence only; the central renderer owns
RGB, and explicit visual review is required before a dataset can be selected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
# Existing owner simulator modules use local imports when executed as scripts.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/physics_fluid"))
from real2sim.production.task_randomization.benchmark_8_10.inject_robot_initial_state import (
    sample_initial_perturbation, validate_initial_perturbation, check_episode_perturbation,
)


def require_clean_source() -> dict:
    """Reject mutable source before starting costly physics, and record Git identity."""
    from real2sim.production.task_data_compatibility import current_compatibility_source_revision
    from real2sim.production.git_provenance import source_revision
    paths = set(current_compatibility_source_revision("8.10")["paths"])
    paths.update(str(path.relative_to(REPO)) for path in HERE.glob("*.py"))
    paths.update(str(path.relative_to(REPO)) for path in HERE.glob("*.json"))
    paths.update(str(path.relative_to(REPO)) for path in (REPO / "experiments/physics_fluid").glob("tablecloth_*.py"))
    paths.add("experiments/physics_fluid/tablecloth_canvas_physics.json")
    revision = source_revision(sorted(paths), repo=REPO)
    if revision["worktree_dirty"]:
        raise ValueError("Commit benchmark source before generating release candidates")
    return revision


def configuration(geometry, robot, output, *, initial=None, policy=None) -> SimpleNamespace:
    config = json.loads((HERE / "simulation_config.json").read_text())
    config.update(geometry=str(Path(geometry).resolve()), robot=str(Path(robot).resolve()),
                  output=str(Path(output).resolve()), initial=str(Path(initial).resolve()) if initial else None,
                  policy=str(Path(policy or HERE / "policy.json").resolve()),
                  physics_profile=str(REPO / "experiments/physics_fluid/tablecloth_canvas_physics.json"),
                  diagnostic_branch=False, pickup_only=False, injection=None)
    return SimpleNamespace(**config)


def prepare_initial_cloth(geometry, robot, output) -> Path:
    """Settle a free folded sheet against the current basket and visible linens.

    This is scene preparation, not task data. It has no attachment or recorded
    reference requirement. Each subsequent episode resets the cloth independently.
    """
    from real2sim.production.task_randomization.benchmark_8_10.initial_scene import SupportedCanvasInitial as CanvasInitial

    config = configuration(geometry, robot, output)
    config.policy = None
    config.settle = 3.0
    config.clamp_radius = .018
    sim = CanvasInitial(config)
    failure = None
    try:
        sim.run_policy()
        if sim.clamps or not np.isfinite(sim.a.particle_q.numpy()).all():
            raise ValueError("Initial cloth must be finite and unpinned")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sim.save(failure)
    return Path(output) / "full_state.npz"


def create_episode(geometry, robot, output, injection, *, initial, policy=None):
    """Create the benchmark reset state; the caller can execute its own policy."""
    from tablecloth_capture import BoundedCloseCapture

    validate_initial_perturbation(injection)
    config = configuration(geometry, robot, output, initial=initial, policy=policy)
    config.injection_data = injection
    return BoundedCloseCapture(config)


def write_episode_evidence(sim) -> dict:
    """Normalize metadata while preserving physical/action arrays exactly."""
    from real2sim.production.task_success import evaluate_task_success

    root = sim.output
    with np.load(root / "action.npz", allow_pickle=False) as archive:
        action = {key: archive[key] for key in archive.files}
    names = action.pop("joint_names")
    action["joint_names_json"] = np.asarray(json.dumps(names.tolist()))
    with np.load(root / "full_state.npz", allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    count = len(state["time_s"])
    if not np.array_equal(action["joint_target"], state["joint_position"]) or not np.array_equal(action["time_s"], state["time_s"]):
        raise ValueError("Action/state correspondence failed")
    state.update(frame_index=np.arange(count, dtype=np.int64), fps=np.asarray(sim.args.fps))
    np.savez_compressed(root / "action.npz", **action)
    np.savez_compressed(root / "full_state.npz", **state)
    evidence = dict(schema="tablecloth.spread_evidence.v1", target_identity="clean_tablecloth",
                    destination_prim_path="/World/Assets/CoffeeTable",
                    destination_table_bounds_m=sim.table_bounds.tolist(),
                    geometry_source=sim.args.geometry, grasp_model="closed_jaw_surface_gated_point_attachment")
    (root / "tablecloth_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    revision = require_clean_source()
    metadata = dict(schema="real2sim.tablecloth_review_episode.v1", benchmark_subtask="8.10",
                    task_key="benchmark_8_10", environment_scope="canonical_house_default",
                    execution_and_evaluation_source=revision, original_physics_source=sim.source,
                    owner_review="pending", release_approved=False,
                    physical_quality_status="independent_whole_trajectory_audit_required",
                    visual_review="pending", strict_dataset_accepted=False)
    (root / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
    audit = check_episode_perturbation(root)
    (root / "initial_perturbation_check.json").write_text(json.dumps(audit, indent=2) + "\n")
    result = evaluate_task_success("8.10", root)
    (root / "registered_success.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_episode(geometry, robot, output, injection, *, initial, policy=None) -> dict:
    source_at_start = require_clean_source()
    sim = create_episode(geometry, robot, output, injection, initial=initial, policy=policy)
    failure = None
    try:
        sim.run_policy()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        sim.save(failure)
    source_at_end = require_clean_source()
    if source_at_start != source_at_end:
        raise RuntimeError("Benchmark source changed during physics execution")
    result = write_episode_evidence(sim) if sim.frames else {"status": "unverified"}
    return dict(episode=str(sim.output), semantic_status=result["status"], failure=failure,
                independent_quality_review="required", visual_review="pending", release_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--robot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial", type=Path, help="Optional explicitly prepared free cloth state; otherwise settle from current basket geometry")
    parser.add_argument("--policy", type=Path, default=HERE / "policy.json")
    parser.add_argument("--seed", type=int, default=810202609065000)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--injection", type=Path, help="Explicit perturbation record, only for count=1")
    args = parser.parse_args()
    if args.count < 1 or (args.injection and args.count != 1):
        parser.error("count must be positive; explicit injection requires count=1")
    if args.output.exists():
        raise FileExistsError("Use a new run directory; failed attempts are preserved")
    require_clean_source()
    import warp as wp
    wp.init()
    args.output.mkdir(parents=True)
    initial = args.initial or prepare_initial_cloth(args.geometry, args.robot, args.output / "scene_initialization")
    rows = []
    for i in range(args.count):
        injection = json.loads(args.injection.read_text()) if args.injection else sample_initial_perturbation(args.seed + i)
        rows.append(run_episode(args.geometry, args.robot, args.output / f"episode_{i:03d}", injection,
                                initial=initial, policy=args.policy))
        (args.output / "production.json").write_text(json.dumps(dict(task="8.10", attempts=rows,
            requested_attempts=args.count, strict_accepted=0, actual_visual_review_required=True), indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()

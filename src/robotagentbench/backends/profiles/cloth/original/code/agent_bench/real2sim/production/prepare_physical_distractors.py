#!/usr/bin/env python3
"""Sample collidable clutter once before an owner's action rollout."""
import argparse
import copy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "real2sim/teleop")]


def requires_coffee_table_workspace(task_key):
    """The main-table split does not apply to independent side-table picks."""
    from real2sim.production.benchmark_eval_registry import resolve
    task = resolve(task_key.removeprefix("benchmark_").replace("_", "."))["task"]
    return task["family"] == 7 and task["runtime"]["environment"]["layout_frame"] == "coffee_table"


def ensure_target_physics(stage, target, policy):
    """Materialize a policy-listed visual-only target before physical replay."""
    from pxr import Usd, UsdPhysics
    from physical_distractors import ensure_collision_geometry, validate_bodies
    prim = stage.GetPrimAtPath(target)
    if not prim or not prim.IsActive():
        raise ValueError(f"physical target is missing or inactive: {target}")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
            raise ValueError(f"target physics belongs to a different nested body: {target}")
        candidates = list(policy["surface_objects"])
        for group in policy["shelf"].values():
            if isinstance(group, list):
                candidates.extend(p for p in group if isinstance(p, dict))
        mass = next((p["mass_kg"] for p in candidates
                     if p.get("prim_path") == target and p.get("mass_kg")), None)
        if mass is None:
            raise ValueError(f"visual-only target has no registered physical mass: {target}")
        ensure_collision_geometry(stage, target)
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    ensure_collision_geometry(stage, target)
    return validate_bodies(stage, [target])


def prepare(*, scene, task_key, target, seed, protected=(), runtime_task_id=None):
    from pxr import Usd
    from real2sim.production.benchmark_eval_registry import resolve
    import episode_scene_randomization as clutter
    from init_family7_code_control import ensure_session_assets
    import session_scene
    if (session_scene.DEMO / "scene_room.usda").resolve() != scene.resolve():
        raise ValueError("REAL2SIM_DEMO_ROOT must select the supplied working House scene")
    from session_scene import _atomic_sidecar
    if resolve(task_key.removeprefix("benchmark_").replace("_", "."))["task"]["key"] != task_key:
        raise ValueError("expected the canonical Sheet task key")
    sidecar = scene.with_suffix(scene.suffix + ".session.json")
    session = json.loads(sidecar.read_text())
    if "physical_distractor_seed" in session:
        raise ValueError("physical clutter is fixed for the House session; do not resample between tasks")
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"missing House working scene {scene}")
    ensure_session_assets(stage)
    policy = copy.deepcopy(clutter.load_policy())
    target_physics = ensure_target_physics(stage, target, policy)
    from house_topology_visual import DECOR_SLOT_ROOTS
    from visual_scene_geometry import visible_bounds
    from physical_distractors import ensure_scene_distractor_collisions
    decor_collisions = ensure_scene_distractor_collisions(stage,
        [p for group in DECOR_SLOT_ROOTS.values() for p in group
         if visible_bounds(stage, p) is not None])
    stage.GetRootLayer().Save()
    policy["eligible_production_task_ids"] = [task_key]
    from coffee_table_partition import validate_cleanup_objects
    if requires_coffee_table_workspace(task_key):
        validate_cleanup_objects(stage, [target])
    descriptor = clutter._target_descriptor(stage, target, policy)
    protected_objects = [
        item for path in protected
        if (item := clutter._target_descriptor(stage, path, policy)) is not None
    ]
    plan = clutter.build_layout_plan(task_id=task_key, seed_override=seed,
        target=descriptor, protected_prim_paths=[target, *protected],
        protected_objects=protected_objects, policy=policy)
    # Current inventory contract counts only distractors, not the target.
    objects = list(plan["objects"])
    for surface in ("shelf_snack", "shelf_beverage"):
        candidates = sorted([o for o in objects if o["surface"] == surface], key=lambda o:o["x_m"])
        if len(candidates) < 5:
            raise ValueError(f"insufficient safe physical {surface} distractors: {len(candidates)}/5")
        selected = {candidates[round(i*(len(candidates)-1)/4)]["prim_path"] for i in range(5)}
        objects = [o for o in objects if o["surface"] != surface or o["prim_path"] in selected]
    plan["objects"] = objects
    plan["interactive_prim_paths"] = sorted(o["prim_path"] for o in objects)
    if runtime_task_id is not None:
        from make_compare3 import runtime_target_prim
        if runtime_target_prim(str(runtime_task_id), {"track_prim": target}) != target:
            raise ValueError("physical randomization runtime target mismatch")
        plan["runtime_task_id"] = str(runtime_task_id)
    plan["target_physics"] = target_physics
    plan["decor_collision_geometry"] = decor_collisions
    applied = clutter.apply_plan_to_stage(scene_path=scene, plan=plan,
        target_prim_path=target, protected_prim_paths=protected,
        policy=policy, demo_root=scene.parent)
    session = json.loads(sidecar.read_text())
    session["physical_distractor_seed"] = int(seed)
    _atomic_sidecar(session)
    return applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protected", action="append", default=[])
    args = parser.parse_args()
    result = prepare(scene=args.scene, task_key=args.task_key, target=args.target,
                     seed=args.seed, protected=args.protected)
    print(json.dumps({"ok": True, "paths": result["applied_prim_paths"]}))


if __name__ == "__main__":
    main()

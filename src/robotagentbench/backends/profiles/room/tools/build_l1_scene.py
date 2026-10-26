"""Prepare a standalone L1-aligned scene or one of L1's final 50 whole-house layouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_api import l1_scene, l1_runtime
from arena_randomization import shelf_supports, countertops
from vendor.l1_scene.real2sim.production import house_topology, house_topology_visual


def prepare(base, output, ordinal=None, *, seed=0, appearance=True):
    from pxr import Usd, Sdf, UsdGeom
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    if (output.exists() or not l1_runtime.is_scene_output_directory(output.parent)
            or output.name != "scene_room.usda"):
        raise ValueError("Use a new scene_room.usda in an approved output tree; source scenes are never overwritten")
    from tools.install_l1_base import validate_receipt
    receipt = validate_receipt(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Usd.Stage.Open(str(base))
    source_hashes = {Path(layer.realPath): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                     for layer in source.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    robot_source = UsdGeom.Xformable(source.GetPrimAtPath("/World/Robot")).GetLocalTransformation()
    if source.GetPrimAtPath("/World").GetAttribute("arena:l1ScenePrepared").Get():
        raise ValueError("Use the base scene, not a previously randomized output")
    stage = Usd.Stage.CreateNew(str(output))
    # Stage metadata is root-owned, not inherited from sublayers. Never let a
    # meter-scale source silently become USD's centimetre-scale default.
    for key in ("upAxis", "metersPerUnit", "timeCodesPerSecond", "framesPerSecond"):
        stage.SetMetadata(key, source.GetMetadata(key))
    stage.GetRootLayer().subLayerPaths = [
        os.path.relpath(p, output.parent) for p in (*l1_scene.LAYERS, base)]
    stage.OverridePrim("/World").CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim("/World").CreateAttribute("arena:l1ScenePrepared", Sdf.ValueTypeNames.Bool).Set(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    l1_scene.verify(stage)
    # The source itself supplies the L1 station. Never select a task pose.
    robot_before = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robot")).GetLocalTransformation()
    microwave = dict(source="approved_microwave.usda", independent_snapshot=True,
                     pose_source="L1 scene_room_baseline.usda",
                     xyz_m=[2.3878, -.643, .92], yaw_deg=-90.)
    kitchen_bindings = l1_runtime.restore_kitchen_bindings(stage)
    shelf = shelf_supports.apply(stage)
    stage.GetRootLayer().Save()
    l1_runtime.open_session(output)
    result = dict(schema="arena.l1_scene_port.v2", scene=str(output), baseline=str(base),
                  clean_baseline_receipt=receipt,
                  shelf_supports=shelf, layout_ordinal=ordinal, seed=seed,
                  microwave_alignment=microwave,
                  robot_station_modified=robot_before != robot_source,
                  robot_station_source="L1 scene_room_baseline.usda",
                  kitchen_bindings=kitchen_bindings,
                  physics_replay_validated=False)
    if appearance:
        plan = l1_runtime.appearance_plan(seed)
        result["appearance_plan"] = plan
        result["appearance"] = l1_runtime.apply_appearance(output, plan)
    if ordinal is not None:
        catalog = house_topology.load_catalog()
        if isinstance(ordinal, bool) or not 1 <= ordinal <= len(catalog["candidates"]):
            raise ValueError("L1 final layout ordinal must be in 1..50")
        candidate = catalog["candidates"][ordinal - 1]
        modules, counter_plan = countertops.configure(catalog["interaction_modules"])
        result["generated_countertops"] = counter_plan
        # Path mapping only: recovered fridge is a clean sibling, not an old/new reference stack.
        modules["refrigerator_cluster"]["source_prim_roots"] = [l1_scene.FRIDGE]
        # Preserve the approved stand/lamp together with the plate/box and microwave.
        extra = {p: "microwave" for p in ("/World/Assets/MicrowavePlateStand",
                                         "/World/MicrowaveCavityLamp") if stage.GetPrimAtPath(p)}
        result["topology"] = house_topology_visual.apply_topology_overlay(
            output, candidate, modules, robot_path=None,
            extra_prim_frames=extra, asset_root=l1_scene.VENDOR, physics_overlay=True)
        if appearance:
            visual = house_topology_visual.build_visual_plan(candidate, modules, plan)
            result["decor"] = house_topology_visual.apply_visual_plan(output, candidate, modules, visual)
        result["catalog_design_revision"] = catalog["design_revision"]
        result["candidate_status"] = catalog["status"]
    reopened = Usd.Stage.Open(str(output))
    l1_scene.verify(reopened)
    # Scene physics, not a task initializer: persist previously approved fixes
    # after variant selection so replaced/hidden decor leaves no ghost collider.
    from arena_api.scene_physics_export import persist
    result["retained_physics_repairs"], result["physics_export"] = persist(reopened, output)
    if ordinal is not None:
        result["generated_countertop_measurements"] = countertops.validate(
            Usd.Stage.Open(str(output)), counter_plan)
    robot_after = UsdGeom.Xformable(reopened.GetPrimAtPath("/World/Robot")).GetLocalTransformation()
    if robot_after != robot_before:
        raise ValueError("Scene preparation changed the aligned L1 robot station")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in source_hashes.items()):
        raise ValueError("Source asset layer changed during scene preparation")
    result["source_layers_unchanged"] = len(source_hashes)
    result["robot_transform_unchanged"] = robot_after == robot_source
    result["robot_transform_unchanged_after_alignment"] = True
    result["usd_author_save_reopen"] = True
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=l1_scene.BASE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layout", type=int, choices=range(1, 51))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-layout", action="store_true")
    p.add_argument("--no-appearance", action="store_true")
    args = p.parse_args()
    if args.layout and args.random_layout:
        p.error("Choose --layout or --random-layout, not both")
    ordinal = l1_runtime.layout_ordinal(args.seed) if args.random_layout else args.layout
    print(json.dumps(prepare(args.base, args.output, ordinal, seed=args.seed,
                            appearance=not args.no_appearance), indent=2))


if __name__ == "__main__":
    main()

"""Persist approved scene policies without changing their session-only API use.

The outer scene layer composes physics > content > original assets. Keeping
physics above *all* content opinions matters: simply adding a sublayer below
the original root would lose to physics attributes authored in that root.
No task initializer, simulator, tuning, or source-asset writes occur here.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import (l1_scene, robot_collision_filters, finger_pad_material,
               finger_pad_contact, microwave_physics, scene_obstacles,
               coffee_box_physics)

POLICIES = (robot_collision_filters, finger_pad_material, finger_pad_contact,
            microwave_physics, scene_obstacles, coffee_box_physics)
METADATA = ("upAxis", "metersPerUnit", "timeCodesPerSecond", "framesPerSecond")
VERIFIED_PREFIX = "ARENA_PHYSICS_EXPORT_VERIFIED "


def verify_reports(stage, reports):
    l1_scene.verify(stage)
    for policy in POLICIES:
        if policy is robot_collision_filters:
            policy.verify(stage)
        else:
            policy.verify(stage, reports[policy.__name__]["configuration"])


def export_session_layers(stage, scene):
    """Export an owned generated root and its strongest session as two files."""
    from pxr import Usd, Sdf
    from .l1_runtime import is_scene_output_directory
    scene = Path(scene).resolve(strict=True)
    root, session = stage.GetRootLayer(), stage.GetSessionLayer()
    if (scene != Path(root.realPath).resolve() or session.subLayerPaths
            or scene.name != "scene_room.usda"
            or not is_scene_output_directory(scene.parent)):
        raise ValueError("Only a generated scene_room.usda in an approved output tree with a flat session may be exported")
    content_file = scene.with_name("scene_room.content.usda")
    physics_file = scene.with_name("scene_room.physics.usda")
    if any(p.exists() or p.is_symlink() for p in (content_file, physics_file)):
        raise ValueError("Refusing to overwrite existing exported scene layers")
    wrapper = Usd.Stage.CreateInMemory()
    for key in METADATA:
        wrapper.SetMetadata(key, stage.GetMetadata(key))
    wrapper.GetRootLayer().defaultPrim = root.defaultPrim
    if not root.Export(str(content_file)) or not session.Export(str(physics_file)):
        raise RuntimeError("Scene layer export failed")
    # Never flatten asset references or copy physics over entire prim specs.
    # Relative paths must be authored on the file-backed root, not the
    # anonymous staging layer (which has no directory to resolve them from).
    with Sdf.ChangeBlock():
        root.TransferContent(wrapper.GetRootLayer())
        root.subLayerPaths = [physics_file.name, content_file.name]
        session.Clear()
    if not root.Save():
        raise RuntimeError("Could not save scene wrapper")
    stage.SetEditTarget(root)
    return dict(content_layer=content_file.name, physics_layer=physics_file.name,
                composition_order=[physics_file.name, content_file.name],
                physics_layer_sha256=hashlib.sha256(physics_file.read_bytes()).hexdigest())


def persist(stage, scene):
    """Apply reviewed policies, persist, then verify in a fresh CPU-only process."""
    scene = Path(scene).resolve(strict=True)
    manifest_path = scene.with_name("scene_room.physics.json")
    if manifest_path.exists() or not stage.GetSessionLayer().empty:
        raise ValueError("Fresh generated scene and empty session required")
    if stage.GetPrimAtPath("/World").GetAttribute("arena:l1ScenePrepared").Get() is not True:
        raise ValueError("Refusing to persist physics into a non-generated source scene")
    root_before = stage.GetRootLayer().ExportToString()
    reports = {p.__name__: p.apply(stage) for p in POLICIES}
    if stage.GetRootLayer().ExportToString() != root_before:
        raise RuntimeError("A session-only policy unexpectedly changed the content layer")
    verify_reports(stage, reports)
    export = export_session_layers(stage, scene)
    # This check is explicitly made AFTER clearing the anonymous session.
    verify_reports(stage, reports)
    manifest = dict(schema="arena.persisted_scene_physics.v1", status="pending_verification",
                    scene=scene.name, reports=reports, **export,
                    physics_replay_validated=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = subprocess.run(
        [sys.executable, "-m", "arena_api.scene_physics_export", "--verify", str(scene),
         "--expected", str(manifest_path)], cwd=l1_scene.ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    scene.with_name("scene_room.physics.verify.log").write_text(
        completed.stdout + completed.stderr)
    lines = [line[len(VERIFIED_PREFIX):] for line in completed.stdout.splitlines()
             if line.startswith(VERIFIED_PREFIX)]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError("Fresh-process physics verification failed; see scene_room.physics.verify.log")
    manifest.update(status="verified_fresh_process", verification=json.loads(lines[0]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return reports, {key: value for key, value in manifest.items() if key != "reports"}


def microwave_summary(stage):
    """Optional-asset receipt; missing bodies are not reported as dynamic ones."""
    config = microwave_physics.configuration(stage)
    unavailable = "presence" in config
    return dict(microwave_presence=config.get("presence", "present"),
                microwave_body_kinematic=None if unavailable else config[microwave_physics.BODY]["kinematic"],
                microwave_mount_present=False if unavailable else config[microwave_physics.MOUNT]["exists"])


def verify_saved(scene, expected):
    """Read disk before calling ANY policy.apply; then check API reapplication."""
    from pxr import Usd, UsdGeom
    scene, expected = Path(scene).resolve(strict=True), Path(expected).resolve(strict=True)
    manifest = json.loads(expected.read_text())
    physics = scene.parent / manifest["physics_layer"]
    if hashlib.sha256(physics.read_bytes()).hexdigest() != manifest["physics_layer_sha256"]:
        raise ValueError("Physics file differs from the exported layer")
    stage = Usd.Stage.Open(str(scene))
    if not stage or not stage.GetSessionLayer().empty:
        raise ValueError("Fresh-process check requires no session opinions")
    if list(stage.GetRootLayer().subLayerPaths) != manifest["composition_order"]:
        raise ValueError("Unexpected scene composition order")
    verify_reports(stage, manifest["reports"])
    offsets = finger_pad_contact.configuration(stage)
    materials = finger_pad_material.configuration(stage)
    obstacles = scene_obstacles.configuration(stage)
    robot = stage.GetPrimAtPath("/World/Robot")
    signature = robot_collision_filters.robot_signature(stage)
    result = dict(session_empty_before_api=True, verified_without_api_apply=True,
                  robot_filter_pairs=len(robot_collision_filters.inspect(stage)[0]),
                  robot_transform=str(UsdGeom.Xformable(robot).GetLocalTransformation()),
                  pad_contact_offsets=[v["contact_offset"] for v in offsets.values()],
                  pad_materials=sorted({v["material"] for v in materials["pads"].values()}),
                  **microwave_summary(stage),
                  coffee_box_kinematic={p: v["kinematic"] for p, v in
                                        coffee_box_physics.configuration(stage).items()},
                  enabled_obstacles=dict(Counter(r["category"] for r in
                                               obstacles["colliders"] if r["collision_enabled"])))
    layers_before = {Path(layer.realPath): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                     for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    for policy in POLICIES:
        policy.apply(stage)
    verify_reports(stage, manifest["reports"])
    if robot_collision_filters.robot_signature(stage) != signature:
        raise RuntimeError("API reapplication changed robot properties")
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in layers_before.items()):
        raise RuntimeError("API reapplication modified saved scene/assets")
    result.update(api_reapply_configuration_unchanged=True, saved_layers_unchanged=len(layers_before),
                  physics_replay_validated=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    args = p.parse_args()
    print(VERIFIED_PREFIX + json.dumps(verify_saved(args.verify, args.expected), sort_keys=True))


if __name__ == "__main__":
    main()

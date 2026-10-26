#!/usr/bin/env python3
"""Swap one reviewed family-7 target into the small Task 29 cup-0 pose.

Run this only after ``init_task29_full.py`` in an open scene session.  It edits
the working scene, never ``scene_room_baseline.usda``.  Cup and can therefore
share the exact same X/Y and the same action; their only intentional
difference is the target asset and the Z needed to seat each shape on the
unchanged CoffeeTable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from family7_code_control_contract import (
    PAPER_SUPPORT_PATH,
    PAPER_TARGET_PATH,
    CAN_ASSET_REFERENCE,
    CAN_DIMENSIONS_M,
    CAN_MASS_KG,
    CAN_TARGET_PATH,
    CUP_TARGET_PATH,
    MODES,
    REFERENCE_XY_M,
    SNACK_TARGET_PATH,
    TABLE_PATH,
    ZBAR_SOURCE_PATH,
    mode,
)
from task29_full_reproduce import CUP_ROOT_PATHS
from machine_paths import DEMO_ROOT


DEMO = Path(DEMO_ROOT)
SCENE = DEMO / "scene_room.usda"
SEAT_CLEARANCE_M = 0.003


def ensure_session_assets(stage: Usd.Stage) -> dict[str, object]:
    """Materialize task-owned props omitted from the merged House root.

    The House authoring root deliberately owns only shared structure and fixed
    decor.  Family-7's loose empty can is therefore installed into the claimed
    working session from its reviewed asset registration.  This never edits the
    read-only baseline and is idempotent for older runtimes that still contain
    the prim.
    """
    existing = stage.GetPrimAtPath(CAN_TARGET_PATH)
    if existing and existing.IsValid():
        return {
            "empty_can_materialized": False,
            "empty_can_path": CAN_TARGET_PATH,
            "asset_reference": CAN_ASSET_REFERENCE,
            "baseline_modified": False,
        }

    root = UsdGeom.Xform.Define(stage, CAN_TARGET_PATH)
    root.GetPrim().GetReferences().AddReference(CAN_ASSET_REFERENCE)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    root.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    root.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    body.CreateVelocityAttr(Gf.Vec3f(0.0))
    body.CreateAngularVelocityAttr(Gf.Vec3f(0.0))
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(CAN_MASS_KG)

    radius = CAN_DIMENSIONS_M[0] / 2.0
    height = CAN_DIMENSIONS_M[2]
    collider = UsdGeom.Cylinder.Define(stage, CAN_TARGET_PATH + "/col0")
    collider.CreateAxisAttr(UsdGeom.Tokens.z)
    collider.CreateRadiusAttr(radius)
    collider.CreateHeightAttr(height)
    collider.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height / 2.0))
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(
        True)
    collider.GetPrim().CreateRelationship(
        "material:binding:physics", custom=False,
    ).SetTargets([Sdf.Path("/World/PhysicsMaterials/GripMat")])
    stage.GetRootLayer().Save()
    return {
        "empty_can_materialized": True,
        "empty_can_path": CAN_TARGET_PATH,
        "asset_reference": CAN_ASSET_REFERENCE,
        "dimensions_m": list(CAN_DIMENSIONS_M),
        "mass_kg": CAN_MASS_KG,
        "collision_shape": "cylinder",
        "baseline_modified": False,
    }


def _translate_op(prim: Usd.Prim) -> UsdGeom.XformOp:
    ops = [
        op
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(ops) != 1:
        raise RuntimeError(
            f"{prim.GetPath()} must have exactly one translate op, got "
            f"{[op.GetOpName() for op in ops]}"
        )
    return ops[0]


def _world_bounds(stage: Usd.Stage, path: str) -> tuple[list[float], list[float]]:
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise RuntimeError(f"empty world bound for {path}")
    return (
        [float(value) for value in bounds.GetMin()],
        [float(value) for value in bounds.GetMax()],
    )


def _seat_on_table(stage: Usd.Stage, path: str) -> dict[str, object]:
    table_min, table_max = _world_bounds(stage, TABLE_PATH)
    target = stage.GetPrimAtPath(path)
    translate = _translate_op(target)
    current = translate.Get()
    translate.Set(Gf.Vec3d(
        REFERENCE_XY_M[0],
        REFERENCE_XY_M[1],
        float(current[2]),
    ))
    target_min, target_max = _world_bounds(stage, path)
    dz = table_max[2] + SEAT_CLEARANCE_M - target_min[2]
    current = translate.Get()
    translate.Set(Gf.Vec3d(
        REFERENCE_XY_M[0],
        REFERENCE_XY_M[1],
        float(current[2]) + dz,
    ))
    target_min, target_max = _world_bounds(stage, path)
    if (
        abs(target_min[2] - (table_max[2] + SEAT_CLEARANCE_M)) > 0.001
        or not (
            table_min[0] <= REFERENCE_XY_M[0] <= table_max[0]
            and table_min[1] <= REFERENCE_XY_M[1] <= table_max[1]
        )
    ):
        raise RuntimeError(
            f"{path} did not seat on CoffeeTable at reviewed XY: "
            f"target_min={target_min}, table={table_min}/{table_max}"
        )
    return {
        "root_translate_xyz_m": [float(value) for value in translate.Get()],
        "aabb_min_m": target_min,
        "aabb_max_m": target_max,
        "table_top_m": table_max[2],
    }


def _require_dynamic(stage: Usd.Stage, path: str) -> list[str]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid() or not prim.IsActive():
        raise RuntimeError(f"target is not active: {path}")
    body_paths = []
    collision_paths = []
    for descendant in Usd.PrimRange(prim):
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(descendant)
            if body.GetRigidBodyEnabledAttr().Get() is not False:
                if body.GetKinematicEnabledAttr().Get() is True:
                    raise RuntimeError(
                        f"kinematic target body is forbidden: "
                        f"{descendant.GetPath()}"
                    )
                body_paths.append(str(descendant.GetPath()))
        if (
            descendant.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(
                descendant).GetCollisionEnabledAttr().Get() is not False
        ):
            collision_paths.append(str(descendant.GetPath()))
    if body_paths != [path] or not collision_paths:
        raise RuntimeError(
            f"target must be one dynamic rigid body with collision: "
            f"bodies={body_paths}, collisions={collision_paths}"
        )
    return collision_paths


def initialize(selected: str) -> dict[str, object]:
    task = mode(selected)
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"cannot open working scene {SCENE}")
    session_assets = ensure_session_assets(stage)
    root_layer = stage.GetRootLayer()
    for path in CUP_ROOT_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                "init_task29_full.py must run before family-7 target swap"
            )
        prim.SetActive(selected == "cup" and path == CUP_TARGET_PATH)
    can = stage.GetPrimAtPath(CAN_TARGET_PATH)
    if not can or not can.IsValid():
        raise RuntimeError(f"missing reviewed EmptyCan asset: {CAN_TARGET_PATH}")
    can.SetActive(selected == "can")
    stage.RemovePrim(SNACK_TARGET_PATH)
    if selected == "snack":
        source = stage.GetPrimAtPath(ZBAR_SOURCE_PATH)
        if not source or not source.IsValid():
            raise RuntimeError(f"missing reviewed zbar asset: {ZBAR_SOURCE_PATH}")
        if not Sdf.CopySpec(
            root_layer,
            Sdf.Path(ZBAR_SOURCE_PATH),
            root_layer,
            Sdf.Path(SNACK_TARGET_PATH),
        ):
            raise RuntimeError("failed to clone reviewed zbar asset")
        snack = stage.GetPrimAtPath(SNACK_TARGET_PATH)
        snack.SetActive(True)
        orient_ops = [
            op
            for op in UsdGeom.Xformable(snack).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ]
        if len(orient_ops) != 1:
            raise RuntimeError(
                "reviewed zbar clone must have one orient op")
        # The reviewed box is 135 x 40 x 135 mm.  Rotate only the dynamic
        # target so its 40 mm edge lies on the cup action's jaw axis; the
        # table, bins, robot station and action remain unchanged.
        orient_ops[0].Set(Gf.Quatf(
            0.7071067811865476, 0.0, 0.0, 0.7071067811865475))
    target = stage.GetPrimAtPath(task.target_path)
    target.SetActive(True)
    pose = _seat_on_table(stage, task.target_path)
    from coffee_table_partition import validate_cleanup_objects
    workspace = validate_cleanup_objects(stage, [task.target_path])
    collisions = _require_dynamic(stage, task.target_path)
    for path in CUP_ROOT_PATHS:
        if path != task.target_path and stage.GetPrimAtPath(path).IsActive():
            raise RuntimeError(f"non-target task29 cup stayed active: {path}")
    for path in (CAN_TARGET_PATH, SNACK_TARGET_PATH):
        prim = stage.GetPrimAtPath(path)
        if path != task.target_path and prim and prim.IsValid() and prim.IsActive():
            raise RuntimeError(f"non-target family-7 object stayed active: {path}")
    root_layer.Save()
    output = {
        "ok": True,
        "schema": "real2sim.family7_single_target_scene.v1",
        "mode": selected,
        "label": task.label,
        "data_binding": task.data_binding,
        "target_path": task.target_path,
        "source_asset_path": task.source_asset_path,
        "reference_xy_m": list(REFERENCE_XY_M),
        "pose": pose,
        "collision_paths": collisions,
        "session_assets": session_assets,
        "coffee_table_workspace": workspace,
        "destination": {
            "name": task.destination_name,
            "path": task.destination_path,
        },
        "furniture_moved": False,
    }
    return output


def _activate_paper(stage: Usd.Stage, plate_xy: tuple[float, float]) -> dict[str, object]:
    """Stage the paper target and its plate exactly as the Task 30 initializer does.

    Mirrors ``init_throw``: reviewed Genie paper scale, dynamic body, native
    paper physics material, and a kinematic PaperPlate support with its single
    collider enabled. The plate XY comes from the multi-object plan.
    """
    from task30_throw_reproduce import TASK30_GENIE_PAPER_SCALE_XYZ
    from task29_full_reproduce import (
        TASK29_PAPER_PLATE_BOTTOM_M,
        TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M,
        TASK29_PAPER_PLATE_TOP_M,
    )
    paper = stage.GetPrimAtPath(PAPER_TARGET_PATH)
    if not paper or not paper.IsValid():
        raise RuntimeError(f"missing reviewed paper asset: {PAPER_TARGET_PATH}")
    paper.SetActive(True)
    paper.CreateAttribute(
        "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    for name in ("physics:velocity", "physics:angularVelocity"):
        attribute = paper.GetAttribute(name)
        if attribute and attribute.Get() is not None:
            attribute.Set(Gf.Vec3f(0, 0, 0))
    scale_ops = [
        op for op in UsdGeom.Xformable(paper).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(scale_ops) != 1:
        raise RuntimeError("PaperBall requires exactly one ordered scale op")
    scale_ops[0].Set(Gf.Vec3f(*TASK30_GENIE_PAPER_SCALE_XYZ))
    collider = stage.GetPrimAtPath(PAPER_TARGET_PATH + "/Geom/body/visual")
    if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("PaperBall collision target is invalid")
    collider.RemoveProperty("material:binding:physics")
    binding = collider.GetRelationship("material:binding:physics")
    native = PAPER_TARGET_PATH + "/Geom/PhysicsMaterial"
    if not binding or [str(p) for p in binding.GetTargets()] != [native]:
        raise RuntimeError("PaperBall native physics material binding was not restored")
    plate = stage.GetPrimAtPath(PAPER_SUPPORT_PATH)
    if not plate or not plate.IsValid():
        raise RuntimeError("scene is missing PaperPlate")
    plate.SetActive(True)
    ops = UsdGeom.Xformable(plate).GetOrderedXformOps()
    if [op.GetOpType() for op in ops] != [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeScale,
    ]:
        raise RuntimeError("PaperPlate transform stack changed")
    ops[0].Set(Gf.Vec3d(float(plate_xy[0]), float(plate_xy[1]), 0.448))
    ops[1].Set(Gf.Vec3f(0.0, 0.0, 0.0))
    ops[2].Set(Gf.Vec3f(1.0, 1.0, 1.0))
    rigid = UsdPhysics.RigidBodyAPI(plate)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    colliders = [prim for prim in Usd.PrimRange.AllPrims(plate)
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1:
        raise RuntimeError("PaperPlate must have exactly one authored collision shape")
    UsdPhysics.CollisionAPI(colliders[0]).CreateCollisionEnabledAttr().Set(True)
    low, high = _world_bounds(stage, PAPER_SUPPORT_PATH)
    size = [b - a for a, b in zip(low, high)]
    if (any(abs(a - b) > 0.001 for a, b in zip(size, TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M))
            or abs(low[2] - TASK29_PAPER_PLATE_BOTTOM_M) > 0.001
            or abs(high[2] - TASK29_PAPER_PLATE_TOP_M) > 0.001):
        raise RuntimeError(f"PaperPlate support pose/size changed: {low} {high}")
    return {"paper_scale_xyz": list(TASK30_GENIE_PAPER_SCALE_XYZ),
            "paper_native_material": native,
            "plate_translate_xyz": [float(v) for v in ops[0].Get()],
            "plate_aabb_min_xyz": low, "plate_aabb_max_xyz": high,
            "plate_kinematic": True, "plate_collision_enabled": True}


def initialize_multi(selected: list[str], plate_xy: tuple[float, float] | None) -> dict[str, object]:
    """Activate several family-7 targets in one scene (Task7 multi-object).

    Uses the same per-object activation as ``initialize``; poses are authored
    afterwards by the multi-object scene plan, which verifies footprints.
    """
    if len(set(selected)) != len(selected) or not 2 <= len(selected) <= 3:
        raise ValueError(f"multi-object needs 2-3 distinct modes, got {selected}")
    tasks = [mode(name) for name in selected]
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"cannot open working scene {SCENE}")
    session_assets = ensure_session_assets(stage)
    root_layer = stage.GetRootLayer()
    for path in CUP_ROOT_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                "init_task29_full.py must run before family-7 target swap")
        prim.SetActive("cup" in selected and path == CUP_TARGET_PATH)
    can = stage.GetPrimAtPath(CAN_TARGET_PATH)
    if not can or not can.IsValid():
        raise RuntimeError(f"missing reviewed EmptyCan asset: {CAN_TARGET_PATH}")
    can.SetActive("can" in selected)
    stage.RemovePrim(SNACK_TARGET_PATH)
    if "snack" in selected:
        if not Sdf.CopySpec(root_layer, Sdf.Path(ZBAR_SOURCE_PATH),
                            root_layer, Sdf.Path(SNACK_TARGET_PATH)):
            raise RuntimeError("failed to clone reviewed zbar asset")
        stage.GetPrimAtPath(SNACK_TARGET_PATH).SetActive(True)
    paper = None
    if "paper" in selected:
        if plate_xy is None:
            raise ValueError("paper needs the plan's plate XY")
        paper = _activate_paper(stage, plate_xy)
    collisions = {task.target_path: _require_dynamic(stage, task.target_path)
                  for task in tasks}
    for path in CUP_ROOT_PATHS:
        if path != CUP_TARGET_PATH and stage.GetPrimAtPath(path).IsActive():
            raise RuntimeError(f"non-target task29 cup stayed active: {path}")
    root_layer.Save()
    return {
        "ok": True,
        "schema": "real2sim.family7_multi_target_scene.v1",
        "modes": list(selected),
        "target_paths": [task.target_path for task in tasks],
        "destinations": {task.target_path: task.destination_name for task in tasks},
        "collision_paths": collisions,
        "session_assets": session_assets,
        "paper": paper,
        "poses_authored_by": "family7_experiment_scene.py --multi-plan",
        "furniture_moved": False,
    }


def ensure_session_assets_only() -> dict[str, object]:
    """Materialize Family-7 loose assets without selecting a task target."""
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"cannot open working scene {SCENE}")
    return {
        "ok": True,
        "schema": "real2sim.family7_session_assets.v1",
        "scene": str(SCENE),
        "session_assets": ensure_session_assets(stage),
        "baseline_modified": False,
        "target_selection_changed": False,
        "furniture_moved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("cup", "can", "snack"))
    parser.add_argument("--ensure-session-assets-only", action="store_true")
    parser.add_argument("--multi", help="comma-separated modes of a multi-object episode")
    parser.add_argument("--plate-xy", help="x,y of the PaperPlate for a multi-object paper")
    args = parser.parse_args()
    if sum((args.ensure_session_assets_only, args.mode is not None,
            args.multi is not None)) != 1:
        parser.error(
            "pass exactly one of mode, --multi or --ensure-session-assets-only"
        )
    try:
        output = (
            ensure_session_assets_only()
            if args.ensure_session_assets_only
            else initialize_multi(
                args.multi.split(","),
                tuple(map(float, args.plate_xy.split(","))) if args.plate_xy else None)
            if args.multi is not None
            else initialize(args.mode)
        )
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "mode": args.mode or "session-assets-only",
            "reason": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

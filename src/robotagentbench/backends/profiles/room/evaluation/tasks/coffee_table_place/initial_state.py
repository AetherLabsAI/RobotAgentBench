"""Fixed source-state candidates for coffee-table placement.

Only existing-instance activation and pose opinions are authored. This is not
an accepted reset: physical hold/support validation must happen before timing.
No source controller, material, drive or release helper is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from .source_initial import TASK, COFFEE_TABLE, load_recorded_hold_targets

REPO = TASK.parents[2]
# Existing approved object assets, identical to shelf-pick's current closure.
BASE = REPO / "assets/house_scene_l1/demo_sim/scene_room_shelf_pick.usda"
ROBOT = "/World/Robot"
RESET_OP = "xformOp:transform:coffee_place_initial"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_layer(layer, paths):
    from pxr import Sdf
    errors = []
    def inspect(path):
        obj = layer.GetObjectAtPath(path)
        if isinstance(obj, Sdf.PrimSpec):
            allowed = ("subLayers", "subLayerOffsets", "defaultPrim", "upAxis", "metersPerUnit") if path == Sdf.Path.absoluteRootPath else ("specifier", "active")
            if any(key not in allowed for key in obj.ListInfoKeys()):
                errors.append(str(path))
            if obj.HasInfo("active") and str(path) not in paths:
                errors.append(str(path))
        elif isinstance(obj, Sdf.AttributeSpec):
            if str(path.GetPrimPath()) not in paths or obj.name not in (RESET_OP, "xformOpOrder"):
                errors.append(str(path))
        elif isinstance(obj, Sdf.RelationshipSpec):
            errors.append(str(path))
    layer.Traverse("/", inspect)
    if errors:
        raise ValueError("Non-state opinions in reset layer: " + str(errors))


def physics_facts(stage, roots):
    from pxr import Usd
    return {str(p.GetPath()): dict(
        schemas=list(p.GetAppliedSchemas()),
        attributes={a.GetName(): str(a.Get()) for a in p.GetAttributes()
                    if a.GetName().startswith(("physics:", "physx"))},
        relationships={r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships()
                       if r.GetName().startswith(("physics:", "physx", "material:binding"))})
        for root in roots for p in Usd.PrimRange(stage.GetPrimAtPath(root))}


def prepare(api_id, output, *, base=BASE):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    data, hold_targets = load_recorded_hold_targets(api_id)
    # This entry point restores the recorded fixed frame only. Layout transforms
    # will be added through the existing randomization contract, not guessed here.
    source = Path(base).resolve(strict=True)
    if source != BASE.resolve(strict=True):
        raise ValueError("Fixed candidate builder requires the existing canonical object scene")
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to((TASK / "artifacts").resolve()):
        raise ValueError("Use a new directory in coffee_table_place/artifacts")
    stage = Usd.Stage.Open(str(source))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Expected a metres/Z-up scene")
    table = stage.GetPrimAtPath(COFFEE_TABLE)
    if not table or not table.IsActive():
        raise ValueError("Missing active coffee table")
    # AABB/zone is provenance, not an excuse to overwrite the shared table pose.
    bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(table).ComputeAlignedRange()
    dest = data["source_destination"]
    measured_min, measured_max = dest["furniture_aabb_min_xyz_m"], dest["furniture_aabb_max_xyz_m"]
    if not measured_min:
        if dest["body_path"] != COFFEE_TABLE:
            raise ValueError("No source furniture frame")
        cx, cy, hx, hy, z = dest["zone"]
        expected = (cx-hx, cy-hy, cx+hx, cy+hy, z)
        provenance = "source destination zone (no measured furniture frame)"
    else:
        expected = (*measured_min[:2], *measured_max[:2], measured_max[2])
        provenance = "source measured furniture AABB"
    actual = (*bounds.GetMin()[:2], *bounds.GetMax()[:2], bounds.GetMax()[2])
    if not all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(actual, expected)):
        raise ValueError("Source/current coffee-table frame mismatch; explicit relative adaptation required")
    dependencies = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers() if l.realPath}
    # Use an anonymous layer until every state/asset check has passed.
    layer = stage.GetSessionLayer()
    stage.SetEditTarget(layer)
    roots = [ROBOT, data["target_path"]]
    for path in roots:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError("Missing existing instance: " + path)
        prim.SetActive(True)
    target = stage.GetPrimAtPath(data["target_path"])
    body = UsdPhysics.RigidBodyAPI(target)
    if not body or body.GetKinematicEnabledAttr().Get() or body.GetRigidBodyEnabledAttr().Get() is False:
        raise ValueError("Target is not an existing dynamic body: " + data["target_path"])
    if not any(p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
               for p in Usd.PrimRange(target)):
        raise ValueError("Target has no enabled existing collider")
    before = physics_facts(stage, roots)
    for path, state in ((ROBOT, data["robot"]), (data["target_path"], data["target"])):
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        scale = Gf.Transform(xf.ComputeLocalToWorldTransform(0)).GetScale()
        q = state["quaternion_wxyz"]
        world = Gf.Matrix4d(1).SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        world.SetTranslateOnly(Gf.Vec3d(*state["position_m"]))
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="coffee_place_initial").Set(Gf.Matrix4d(1).SetScale(scale)*world*parent.GetInverse())
    if physics_facts(stage, roots) != before:
        raise ValueError("Physics changed during pose restoration")
    audit_layer(layer, set(roots))
    joint_issues = []
    for name, value in data["joint_positions"].items():
        prim = stage.GetPrimAtPath(ROBOT + "/joints/" + name)
        if not prim:
            joint_issues.append(dict(joint=name, issue="missing joint")); continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            lo, hi = [math.radians(prim.GetAttribute("physics:" + key + "Limit").Get()) for key in ("lower", "upper")]
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            lo, hi = [prim.GetAttribute("physics:" + key + "Limit").Get() for key in ("lower", "upper")]
        else:
            joint_issues.append(dict(joint=name, issue="unsupported joint type")); continue
        if value < lo or value > hi:
            joint_issues.append(dict(joint=name, measured=value, lower=lo, upper=hi, issue="outside native limit; not clamped"))
    if any(digest(p) != h for p, h in dependencies.items()):
        raise ValueError("Source changed during state preparation")
    out.mkdir(parents=True)
    exported = Sdf.Layer.CreateAnonymous("coffee_place_initial.usda")
    exported.TransferContent(layer)
    exported.subLayerPaths = [os.path.relpath(source, out)]
    exported.defaultPrim = "World"
    exported.pseudoRoot.SetInfo("upAxis", "Z")
    exported.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    audit_layer(exported, set(roots))
    exported.Export(str(out / "initial.usda"))
    reopened = Usd.Stage.Open(str(out / "initial.usda"))
    if not reopened or physics_facts(reopened, roots) != before:
        raise ValueError("Exported candidate changed physics or could not reopen")
    record = dict(schema="arena.coffee_table_place.initial_candidate.v1", api_id=api_id,
        scene=str(out / "initial.usda"), source=data, dependency_sha256=dependencies,
        table_frame_check=dict(provenance=provenance, actual=list(actual), expected=list(expected)),
        joint_reset_issues=joint_issues, robot_initial_positions=data["joint_positions"],
        robot_initialization=dict(positions=data['joint_positions'], hold_targets=hold_targets),
        hold_target_provenance='verified full_state command_* frame zero; not measured joint positions',
        authored_state_only=True, physics_unchanged=True,
        physical_acceptance=False, runtime_reset_connected=False,
        note="Candidate only; measured gripper hold and any payload support must pass live settling")
    (out / "initial.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    record = prepare(args.api, args.output)
    print(json.dumps({k: record[k] for k in ("api_id", "scene", "joint_reset_issues", "physics_unchanged", "physical_acceptance")}))


if __name__ == "__main__":
    main()

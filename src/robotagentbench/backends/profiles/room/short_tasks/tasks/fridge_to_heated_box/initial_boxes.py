"""Task-owned restoration of the original three-box scene recipe.

Use the existing sampler unchanged; never resample or move the selected box.
The shared scene composer intentionally keeps only one payload. This hook adds
the other two only to the new episode's anonymous layer, before PhysX starts.
"""
from copy import deepcopy
import hashlib

BOX_IDS = frozenset(("beef_box", "leftover_box", "salad_box"))
SLOTS = frozenset(("low_out", "low_in", "up_out", "up_in"))


def mark_roles(rows, variant):
    rows = deepcopy(rows)
    if variant not in ("beef_box", "leftover_box"):
        raise ValueError("Unsupported heated-box target")
    if len(rows) != 3 or {r["id"] for r in rows} != BOX_IDS:
        raise ValueError("Exactly one beef, leftover and salad box required")
    if len({r["path"] for r in rows}) != 3:
        raise ValueError("Fridge box instances must be distinct")
    if len({r["slot"] for r in rows}) != 3 or any(r["slot"] not in SLOTS for r in rows):
        raise ValueError("Three distinct valid refrigerator slots required")
    for row in rows:
        selected = row["id"] == variant
        row.update(source="refrigerator", task_id="heated_box" if selected else None,
                   task_role="target" if selected else "distractor",
                   scoring_target=selected)
    return rows


def restore(plan, stage, manifest):
    from pxr import Sdf, Usd, UsdGeom
    from layout_task_activation_v1.task_scene_registration_v1.initialize_room_fridge_boxes_v1 import apply, OWNER
    from openend.tasks.meal_tray.prepare_microwave_mechanism import physical_records

    variant = plan["runtime_components"][0]["parameters"]["payload_variant"]
    seed = plan["scene_request"]["seed"]
    existing = [r for r in manifest["objects"] if r.get("id") in BOX_IDS]
    if len(existing) != 1 or existing[0]["id"] != variant:
        raise ValueError("Expected the scene composer's one selected payload")
    selected = existing[0]
    if selected.get("initialization_owner") != OWNER or selected.get("seed") != seed:
        raise ValueError("Selected box must come from the same room seed and recipe")
    layer = stage.GetEditTarget().GetLayer()
    if layer is stage.GetRootLayer() or not layer.anonymous:
        raise ValueError("Three-box preparation requires an anonymous episode layer")
    root_before = stage.GetRootLayer().ExportToString()
    before_physics = physical_records(stage)
    before_pose = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(selected["path"]))
    candidate = Usd.Stage.Open(stage.Flatten())
    rows = mark_roles(apply(candidate, seed=seed), variant)
    sampled = next(r for r in rows if r["id"] == variant)
    after_pose = UsdGeom.XformCache().GetLocalToWorldTransform(candidate.GetPrimAtPath(sampled["path"]))
    if sampled["path"] != selected["path"] or sampled["slot"] != selected["slot"] or max(
            abs(before_pose[i][j] - after_pose[i][j]) for i in range(4) for j in range(4)) > 1e-8:
        raise ValueError("Existing selected box differs from this seed; refusing to move it")
    missing = [r for r in rows if r["id"] != variant]
    for row in missing:
        if stage.GetPrimAtPath(row["path"]):
            raise ValueError("Distractor path already occupied: " + row["path"])
        Sdf.CreatePrimInLayer(layer, Sdf.Path(row["path"]).GetParentPath())
        if not Sdf.CopySpec(candidate.GetRootLayer(), row["path"], layer, row["path"]):
            raise RuntimeError("Cannot restore the original distractor instance")
    after_physics = physical_records(stage)
    changed = {p for p in set(before_physics) | set(after_physics)
               if before_physics.get(p) != after_physics.get(p)}
    if any(not any(p == r["path"] or p.startswith(r["path"] + "/") for r in missing) for p in changed):
        raise RuntimeError("Three-box preparation changed unrelated physical properties")
    if stage.GetRootLayer().ExportToString() != root_before:
        raise RuntimeError("Three-box preparation changed the source root layer")
    result = deepcopy(manifest)
    selected_row = dict(selected, task_role="target", scoring_target=True)
    result["objects"] = [r for r in result["objects"] if r.get("id") not in BOX_IDS] + [selected_row] + missing
    for row in missing:
        result["source_hashes"][row["entry"]] = row["source_sha256"]
    import inspect
    source = inspect.getfile(apply)
    with open(source, "rb") as stream:
        result["source_hashes"][source] = hashlib.sha256(stream.read()).hexdigest()
    result["fridge_box_initialization"] = dict(owner=OWNER, seed=seed, box_count=3,
        scoring_target_count=1, distractor_count=2, selected_variant=variant,
        available_slots=4, selected_pose_unchanged=True,
        sampler="existing_unified_scene_fridge_boxes_v1_unchanged",
        native_settling_validated=False, runtime_docking=False,
        physics_parameter_overrides=False)
    return result

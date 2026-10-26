"""Task 07: fridge to microwave and close door; no knob/lamp or tray stage."""
from copy import deepcopy
import math

RULE_VERSION = "openend.short_task.fridge_to_heated_box.v3"
VARIANTS = ("beef_box", "leftover_box")


def resolve(request):
    for key, allowed_roles in (("source", ("fridge", "refrigerator")), ("destination", ("microwave",))):
        role = request.get(key)
        if role is not None and (not isinstance(role, dict) or set(role) != {"role"}
                or role["role"] not in allowed_roles):
            raise ValueError("Task 07 has a fixed " + key + " role; variant or placement overrides are unsupported")
    options = request.get("options", {})
    if not isinstance(options, dict) or set(options) - {"payload_variant"}:
        raise ValueError("Only payload_variant is supported for task 07")
    targets = request.get("targets")
    variant = options.get("payload_variant", "beef_box")
    if targets:
        if (len(targets) != 1 or targets[0].get("quantity") != 1
                or targets[0].get("category") not in VARIANTS):
            raise ValueError("Task 07 selects exactly one beef_box or leftover_box")
        variant = targets[0]["category"]
        if "payload_variant" in options and variant != options["payload_variant"]:
            raise ValueError("Conflicting food-box selectors")
    if variant not in VARIANTS:
        raise ValueError("payload_variant must be beef_box or leftover_box")
    descriptor = dict(instance_id="heated_box", type="fridge_to_microwave",
        parameters=dict(payload_variant=variant),
        custom_native_port="short_tasks.tasks.fridge_to_heated_box.port.FridgeHeatingPort")
    return dict(task_name="fridge_to_heated_box", rule_version=RULE_VERSION,
        available=True, unavailable=None,
        scene_request=dict(schema="openend.scene_requirements.request.v2",
            seed=request["seed"], layout_id=request.get("layout_id"),
            tasks=[dict(id="heated_box", type="fridge_to_microwave", payload_variant=variant)],
            extras=[]),
        runtime_components=[descriptor], initial_requirements=[
            "Beef, leftover and salad boxes all exist, in three distinct sampled refrigerator slots; only the selected box scores.",
            "Selected box starts inside its declared refrigerator compartment and is not held.",
            "Refrigerator and microwave doors are initially closed.",
            "Refrigerator and microwave bind the same scene-owned box throughout the episode.",
            "The scene supplies the original microwave plate/support."],
        instruction="Take the " + ("beef box" if variant == "beef_box" else "leftover food box")
            + " out of the refrigerator, close the refrigerator door, place the same box inside the microwave, and close the microwave door. Leave the box inside.",
        score_policy=dict(maximum=18, process_maximum=18, final_maximum=0,
            incomplete_hold_settlement_penalty=0),
        source_components=dict(fridge_put_maximum=18,
            no_extraction_awards=True, no_meal_tray_awards=True),
        scene_preparation="short_tasks.tasks.fridge_to_heated_box.scene.prepare",
        required_artifacts=[],
        scene_runtime_available=True,
        native_validation="not_run")


def bind(plan, manifest):
    variant = plan["runtime_components"][0]["parameters"]["payload_variant"]
    matches = [row for row in manifest.get("objects", []) if row.get("id") == variant]
    if len(matches) != 1:
        raise ValueError("One exact scene-owned selected food box is required")
    row = matches[0]
    if row.get("task_id") != "heated_box":
        raise ValueError("Task 07 selected box must be owned by this exact scene recipe")
    path = row.get("path")
    if (not isinstance(path, str) or not path.startswith("/World/") or "//" in path
            or any(part in (".", "..") for part in path.split("/"))):
        raise ValueError("Task 07 selected box needs its canonical native path")
    if row.get("initialization_owner") != "unified_scene_fridge_boxes_v1":
        raise ValueError("Task 07 requires the scene-owned fridge initial recipe")
    if row.get("slot") not in ("low_out", "low_in", "up_out", "up_in"):
        raise ValueError("Selected box needs its refrigerator source compartment")
    plan["target_bindings"] = dict(target=path, payload_variant=variant,
        scene_recipe_owner="heated_box", source_compartment=row["slot"])
    return deepcopy(plan["runtime_components"])


def normalize(plan, component_summaries):
    if not isinstance(component_summaries, dict) or set(component_summaries) != {"heated_box"}:
        raise ValueError("Task 07 has exactly one internal composite")
    summary = component_summaries["heated_box"]
    if summary.get("rule_version") != RULE_VERSION:
        raise ValueError("Task 07 requires its versioned adapter, not a raw legacy summary")
    expected_variant = plan["runtime_components"][0]["parameters"]["payload_variant"]
    if summary.get("payload_variant") != expected_variant:
        raise ValueError("Task 07 summary does not match the selected box category")
    target = plan.get("target_bindings", {}).get("target")
    if target is not None and summary.get("target_prim") != target:
        raise ValueError("Task 07 summary does not retain the exact bound native box")
    scores = deepcopy(summary["score_channels"])
    if scores["maximum"] != 18 or scores["current_final"] != 0:
        raise ValueError("Wrong task 07 score schedule")
    if set(scores) != {"process", "current_final", "deductions", "total", "maximum"} or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Complete finite task 07 score channels required")
    if (not 0 <= scores["process"] <= 18 or not 0 <= scores["total"] <= 18
            or scores["deductions"] < 0
            or abs(scores["total"] - scores["process"] + scores["deductions"]) > 1e-6):
        raise ValueError("Task 07 score arithmetic is inconsistent")
    if any(type(summary[key]) is not bool for key in ("success", "current_inside_microwave", "process_complete")):
        raise ValueError("Measured task 07 success/current-state booleans required")
    return dict(scores=scores, success=summary["success"],
        stages=deepcopy(summary["stages"]),
        targets=[dict(category=summary["payload_variant"],
            path=summary["target_prim"], current_inside_microwave=summary["current_inside_microwave"],
            process_complete=summary["process_complete"], success=summary["success"])])

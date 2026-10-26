"""Document task 08 is selectable but requires an independent scene recipe.

The present upstream runtime requires the real refrigerator predecessor. Its
old initial box/plate composer is not an admitted already-heated Short Task
variant. We reject that missing contract instead of awarding initial heating
or manufacturing a completed parent task.
"""
RULE_VERSION = "openend.short_task.microwave_to_meal_tray.pending.v1"


def resolve(request):
    for key, roles in (("source", ("microwave",)), ("destination", ("meal_tray_large", "large_meal_tray"))):
        value = request.get(key)
        if value is not None and (not isinstance(value, dict) or set(value) != {"role"} or value["role"] not in roles):
            raise ValueError("Task 08 has a fixed " + key + " role and no approved role variant override")
    options = request.get("options", {})
    allowed = {"payload_variant", "carrier_plate", "initial_heating"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unknown task 08 option")
    variant = options.get("payload_variant", "beef_box")
    targets = request.get("targets")
    if targets:
        if (len(targets) != 1 or targets[0].get("quantity") != 1
                or targets[0].get("category") not in ("beef_box", "leftover_box")):
            raise ValueError("Task 08 selects exactly one supported food-box category")
        variant = targets[0]["category"]
        if "payload_variant" in options and variant != options["payload_variant"]:
            raise ValueError("Conflicting food-box selectors")
    if variant not in ("beef_box", "leftover_box"):
        raise ValueError("Task 08 requires beef_box or leftover_box")
    if "carrier_plate" in options and type(options["carrier_plate"]) is not bool:
        raise ValueError("carrier_plate must explicitly be boolean")
    if options.get("initial_heating", "scene_declared_completed") != "scene_declared_completed":
        raise ValueError("Task 08 starts after heating; unheated starts would extend its boundary")
    return dict(task_name="microwave_to_meal_tray", rule_version=RULE_VERSION,
        available=False, unavailable=dict(kind="load",
            reason="independent_heated_microwave_initial_state_not_supported",
            message="当前统一场景及微波炉端口要求冰箱真实交接；独立已加热、灯灭、门关闭初态和可选小盘的准入契约尚未接入。不能伪造已完成冰箱任务或灯周期。"),
        scene_request=None, runtime_components=[],
        initial_requirements=[
            "Scene-owned declaration of a completed prior heating cycle, lamp off, closed microwave door.",
            "One exact selected box fully inside the current microwave cavity and not held.",
            "Explicit carrier_plate present/absent variant; no refrigerator task is registered.",
            "One designated large meal tray; any carrier plate must not remain on it."],
        instruction="Take the " + ("beef box" if variant == "beef_box" else "leftover food box")
            + " out of the microwave, close the door, and place the box on the large meal tray. Do not leave the small carrier plate on the large meal tray.",
        score_policy=dict(maximum=None, process_maximum=None, final_maximum=3),
        requested_variant=dict(payload_variant=variant, carrier_plate=options.get("carrier_plate", True),
            initial_heating="scene_declared_completed"),
        pending=["independent_scene_heated_initial_admission", "optional_plate_initial_recipe",
            "remove_upstream_fridge_dependency_without_faking_handoff", "freeze_extraction_score_without_initial_heat_award"],
        native_validation="not_run")


def bind(plan, manifest):
    raise ValueError(plan["unavailable"]["message"])


def normalize(plan, component_summaries):
    if not isinstance(component_summaries, dict) or component_summaries:
        raise ValueError("Unavailable task 08 must not consume old fridge-dependent summaries")
    return dict(scores=dict(process=None, current_final=None, deductions=None, total=None, maximum=None),
        success=None, stages=[], targets=[])

"""One public standalone microwave task, without predecessor/destination."""
from copy import deepcopy
import math
from short_tasks.tasks.stock_wooden_tray._contract import plan as make_plan, summaries, result, number
from .registration import KIND, INSTANCE, KNOB, bind_manifest

TASK_NAME = "microwave_heat_take"
RULE_VERSION = "openend.short_task.microwave_heat_take.v1"
INSTRUCTION = (
    "Turn the microwave knob to start heating. Wait for the light to turn off, "
    "then open the door, take out the plate with the food box still on it, "
    "and close the door. Hold the plate steadily for at least five seconds "
    "after closing the door."
)


def resolve(request):
    if (not isinstance(request, dict) or request.get("task_name") != TASK_NAME
            or request.get("rule_version") != RULE_VERSION):
        raise ValueError("Wrong task or rule identity")
    options = request.get("options", {})
    if not isinstance(options, dict) or set(options)-{"payload_variant"}:
        raise ValueError("Only the existing payload selector is supported")
    variant = options.get("payload_variant", "beef_box")
    if variant not in ("beef_box", "leftover_box"):
        raise ValueError("Unknown source payload variant")
    if any(k in request for k in ("source", "destination", "targets")):
        raise ValueError("No new target or destination overrides are approved")
    task = dict(id=INSTANCE, type=KIND, payload_variant=variant)
    component = dict(instance_id=INSTANCE, type=KIND, parameters=dict(payload_variant=variant))
    out = make_plan(TASK_NAME, RULE_VERSION, request, [task], [component], 14, 14, 0,
        INSTRUCTION, ["Closed, unheated microwave with the original food-bearing plate inside.",
            "No objects held and no finger contact at native initial admission.",
            "Original contact-triggered rotary/light/return cycle before extraction.",
            "Same plate, food retained, door closed and original five-second post-close hold."])
    out.update(requested_variant=dict(payload_variant=variant),
        training_sources=[dict(task="10.1", name="microwave startup"),
                          dict(task="10.2", name="take food out of microwave")],
        excluded_operations=["refrigerator handoff", "meal-tray placement"],
        evidence_policy=dict(interaction_rewards_owner="task_port", other_object_monitoring=False))
    return out


def bind(plan, manifest):
    declaration, = plan["runtime_components"]
    bind_manifest(declaration["parameters"], manifest)
    plan["target_bindings"] = [dict(instance_id=INSTANCE, target_path=KNOB)]
    return deepcopy(plan["runtime_components"])


def normalize(plan, component_summaries):
    s = summaries(plan, component_summaries)[INSTANCE]
    if (s.get("schema") != "arena.short_microwave_heat_take.rules.v1"
            or s.get("maximum_score") != 14 or s.get("evidence_valid") is not True
            or s.get("public_runtime_id") != TASK_NAME or s.get("instance_id") != INSTANCE
            or s.get("evidence_status") != "native_physx" or type(s.get("success")) is not bool):
        raise ValueError("Complete task-bound native physical summary required")
    stages = s.get("stage_points")
    maxima = dict(approach=1, contact=1, illumination_cycle=2,
                  open=3, grasp=1, take_out=3, payload_retained=2, close=1)
    if not isinstance(stages, dict) or set(stages) != set(maxima):
        raise ValueError("Every approved stage must be reported")
    if any(number(stages[k]) > v+1e-9 for k,v in maxima.items()):
        raise ValueError("Stage maximum exceeded")
    gross = sum(stages.values())
    deductions = number(s["deducted_points"])
    bonuses = sum(stages[k] for k in ("approach", "contact", "illumination_cycle"))
    expected = bonuses+max(0., gross-bonuses-deductions)
    if not math.isclose(number(s["score"]), expected, abs_tol=1e-9):
        raise ValueError("Score does not preserve original component clipping")
    if s["success"] and (not s.get("heating_completed") or not s.get("extraction",{}).get("hold_requirement_status") == "satisfied"):
        raise ValueError("Success lacks a completed cycle and current post-close hold")
    return result(plan, [dict(instance_id=INSTANCE, target_path=KNOB,
        process=gross, current_final=0., deductions=deductions, component_total=s["score"],
        process_complete=s["success"], success=s["success"], stage_points=deepcopy(stages))])


def factory_override(plan, declaration):
    if declaration not in plan["runtime_components"] or declaration["type"] != KIND:
        raise ValueError("Unrecognized standalone native declaration")
    from .port import HeatTakePort
    return HeatTakePort

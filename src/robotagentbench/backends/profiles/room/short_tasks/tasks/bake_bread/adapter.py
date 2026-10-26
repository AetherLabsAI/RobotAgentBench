"""Single public fryer task with a task-owned first-opening score policy."""
from copy import deepcopy
from short_tasks.common.contracts import TaskError
from .evaluator import RULE_VERSION, MAXIMUM_SCORE, score_channels


def factory_override(plan, declaration):
    if plan["task_name"] != "bake_bread" or declaration["type"] != "fryer":
        raise TaskError("Only the bake_bread fryer port is supported", kind="binding")
    from .native_port import BakeBreadPort
    return BakeBreadPort


def resolve(request):
    options = request.get("options", {})
    if set(options) - {"bread_variant", "finish_on_small_plate"}:
        raise TaskError("Unsupported fryer option")
    variant = options.get("bread_variant", "no_sugar")
    if variant not in ("sugar", "no_sugar"):
        raise TaskError("Choose sugar or no_sugar")
    finish = options.get("finish_on_small_plate", False)
    if type(finish) is not bool or finish:
        raise TaskError("Return-to-small-plate is outside the approved mechanical task")
    if any(request.get(k) for k in ("targets", "source", "destination")):
        raise TaskError("Fryer uses its approved one-target/two-cake assembly")
    color = "white" if variant == "sugar" else "yellow"
    return dict(task_name="bake_bread", rule_version=RULE_VERSION, available=True,
        scene_request=dict(schema="openend.scene_requirements.request.v2", seed=request["seed"],
            layout_id=request.get("layout_id"), tasks=[dict(id="fryer", type="fryer")], extras=[]),
        runtime_components=[dict(instance_id="fryer", type="fryer", parameters=dict(bread_variant=variant))],
        initial_requirements=["two_real_cakes_on_plate", "one_shared_closed_guided_fryer_basket"],
        instruction=f"Pull out the air fryer basket, place the {color} pastry from the plate into the basket, and push the basket closed. Then pull out the same basket, set it down securely, remove the target pastry, and hold it steadily. Heating and returning the pastry to the small plate are not required.",
        score_policy=dict(maximum=MAXIMUM_SCORE, process_maximum=MAXIMUM_SCORE, final_maximum=0),
        heating_required=False, scene_runtime_available=True)


def bind(plan, manifest):
    roles = {}
    for role, asset in (("fryer", "air_fryer"), ("plate", "paper_plate_task2"),
                        ("yellow", "cake_plain_yellow"), ("white", "cake_frosted_white")):
        rows = [o for o in manifest["objects"] if o["id"] == asset]
        if len(rows) != 1:
            raise TaskError("Exactly one actual fryer role required: " + role, kind="binding")
        roles[role] = rows[0]["path"]
    if len(set(roles.values())) != 4:
        raise TaskError("Fryer roles must bind different physical instances", kind="binding")
    variant = plan["runtime_components"][0]["parameters"]["bread_variant"]
    roles["target"] = roles["white" if variant == "sugar" else "yellow"]
    roles["other"] = roles["yellow" if variant == "sugar" else "white"]
    plan["target_bindings"] = roles
    return deepcopy(plan["runtime_components"])


def normalize(plan, component_summaries):
    if set(component_summaries) != {"fryer"}:
        raise TaskError("Exactly one mechanical fryer composite required", kind="binding")
    row = component_summaries["fryer"]
    scores = score_channels(row)
    variant = plan["runtime_components"][0]["parameters"]["bread_variant"]
    expected = "2.2" if variant == "sugar" else "2.1"
    if row["phase_results"][0]["key"] != expected:
        raise TaskError("Fryer result uses the wrong cake variant", kind="binding")
    return dict(scores=scores, success=row["success"], stages=deepcopy(row["phase_results"]),
        targets=[dict(category="cake", variant=variant, roles=deepcopy(plan.get("target_bindings", {})))])

"""Single-task room declaration around the pinned coffee preparation evaluator."""
from copy import deepcopy
from short_tasks.common.contracts import TaskError

from .evaluator import RULE_VERSION, MAXIMUM_SCORE


def resolve(request):
    if request.get("options", {}):
        raise TaskError("Coffee has no scoring or physics overrides")
    targets = request.get("targets")
    if targets is not None and (len(targets) != 1 or targets[0]["quantity"] != 1
                               or targets[0]["category"] not in ("coffee_cup", "coffee_cup_blue_compact")):
        raise TaskError("Coffee prepares one scene-owned cup")
    if request.get("source", {}).get("role", "counter") != "counter" or request.get("source", {}).get("variant"):
        raise TaskError("Coffee starts with its approved cup/bag counter assembly")
    if request.get("destination", {}).get("role", "robot_hand") != "robot_hand" or request.get("destination", {}).get("variant"):
        raise TaskError("Coffee finishes held by the robot outside its receiving bay")
    return dict(task_name="prepare_coffee", rule_version=RULE_VERSION, available=True,
                scene_request=dict(schema="openend.scene_requirements.request.v2", seed=request["seed"],
                    layout_id=request.get("layout_id"), tasks=[dict(id="coffee", type="coffee")], extras=[]),
                runtime_components=[dict(instance_id="coffee", type="coffee", parameters={},
                    custom_native_port="short_tasks.tasks.prepare_coffee.port.CoffeeShortPort")],
                initial_requirements=["empty_cup_on_counter", "bag_with_100_beans", "idle_machine", "released_button"],
                instruction="Place the cup under the coffee machine spout, refill the coffee beans, press the brew button, and finally take away the cup of coffee.",
                score_policy=dict(maximum=MAXIMUM_SCORE, process_maximum=MAXIMUM_SCORE, final_maximum=0, arithmetic="gross_minus_deductions"))


def bind(plan, manifest):
    roles = {}
    for role, asset in (("cup", "coffee_cup_blue_compact"), ("bag", "coffee_bag"), ("machine", "coffee_machine"), ("beans", "coffee_beans")):
        rows = [o for o in manifest["objects"] if
                (o["id"].startswith("coffee_cup_") if role == "cup" else o["id"] == asset)]
        if len(rows) != 1:
            raise TaskError("Exactly one coffee role required: " + role, kind="binding")
        if role == "cup" and rows[0]["id"] not in (
                "coffee_cup_blue_compact", "coffee_cup_cream_classic", "coffee_cup_green_tall"):
            raise TaskError("Unsupported scene-owned coffee cup style", kind="binding")
        roles[role] = rows[0]["path"]
    if len(set(roles.values())) != len(roles):
        raise TaskError("Coffee roles cannot overlap", kind="binding")
    plan["target_bindings"] = roles
    return deepcopy(plan["runtime_components"])


def normalize(plan, component_summaries):
    row = component_summaries["coffee"]
    if row.get("task_name") != "prepare_coffee" or row.get("rule_version") != RULE_VERSION:
        raise TaskError("Only the pinned preparation evaluator may score coffee", kind="binding")
    return dict(scores=deepcopy(row["scores"]), success=row["success"],
                stages=deepcopy(row["phase_results"]),
                targets=[dict(category="coffee_cup", roles=deepcopy(plan.get("target_bindings", {})),
                              current=deepcopy(row["current_result"]))])

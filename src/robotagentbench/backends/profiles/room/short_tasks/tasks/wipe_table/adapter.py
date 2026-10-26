"""One public wiping task; reuse the existing scene and native evidence."""
from copy import deepcopy
import math

from openend.tasks.wipe_table.registration import WipeRegistration, TOOL
from short_tasks.tasks.stock_wooden_tray._contract import (
    options, role, plan as make_plan, summaries, number, result,
)

TASK_NAME = "wipe_table"
RULE_VERSION = "openend.short_task.wipe_table.v1"
INSTANCE = "wipe"
INSTRUCTION = ("Pick up the brush, clean the stains on the coffee table, "
               "then return the brush to its original place and release it.")


def resolve(request):
    if options(request, TASK_NAME):
        raise ValueError("Wiping has no approved task parameter overrides")
    if "targets" in request:
        raise ValueError("The scene binds one brush and its stain; no target overrides")
    role(request, "source", "coffee_table")
    role(request, "destination", "original_brush_location")
    out = make_plan(TASK_NAME, RULE_VERSION, request,
        [dict(id=INSTANCE, type="wipe_table")],
        [dict(instance_id=INSTANCE, type="wipe_table", parameters={})],
        8, 8, 0, INSTRUCTION,
        ["One initially unheld brush, supported on the coffee table.",
         "Use the existing furniture-relative brush placement and seeded stain sampler.",
         "Physical grasp starts stain fading; contact alone cannot clear the stain.",
         "Approach/contact/grasp-lift/wiping maxima are 1/1/1/5.",
         "Return/release stability is retained without a return bonus."])
    out["evidence_policy"] = dict(
        source="native_brush_contacts_and_collision_surfaces",
        target_only=True, other_object_monitoring=False,
        interaction_rewards_owner="task_port", native_trajectory_verified=False)
    return out


def bind(plan, manifest):
    row = WipeRegistration.resolve_object({}, manifest)
    if row.get("task_id") != INSTANCE:
        raise ValueError("Brush belongs to a different task instance")
    tasks = manifest.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("id") != INSTANCE or tasks[0].get("type") != "wipe_table":
        raise ValueError("Exactly one wiping operation required")
    plan["target_bindings"] = [dict(instance_id=INSTANCE, target_path=TOOL)]
    return deepcopy(plan["runtime_components"])


def normalize(plan, component_summaries):
    s = summaries(plan, component_summaries)[INSTANCE]
    if (s.get("schema") != "arena.short_wipe.score.v1"
            or s.get("maximum_score") != 8
            or s.get("public_runtime_id") != TASK_NAME
            or s.get("instance_id") != INSTANCE
            or s.get("evidence_status") != "native_physx"):
        raise ValueError("Wrong task, scorer, instance or evidence source")
    maxima = dict(approach=1., contact=1., grasp_lift=1., wipe_progress=5.)
    stages = s.get("stage_points")
    if not isinstance(stages, dict) or set(stages) != set(maxima):
        raise ValueError("Complete eight-point stage evidence required")
    if any(number(stages[k]) > v+1e-9 for k, v in maxima.items()):
        raise ValueError("Stage exceeds its approved maximum")
    earned = sum(stages.values())
    deductions = number(s["deducted_points"])
    if (not math.isclose(earned, number(s["earned_points"]), abs_tol=1e-9)
            or not math.isclose(number(s["score"]), max(0., earned-deductions), abs_tol=1e-9)
            or type(s.get("success")) is not bool):
        raise ValueError("Unreconciled native score")
    row = dict(instance_id=INSTANCE, target_path=TOOL, process=earned,
        current_final=0., deductions=deductions, component_total=s["score"],
        process_complete=s["success"], success=s["success"],
        stage_points=deepcopy(stages), coverage=s["coverage"],
        failure=s["failure"])
    return result(plan, [row])


def factory_override(plan, declaration):
    if declaration != dict(instance_id=INSTANCE, type="wipe_table", parameters={}):
        raise ValueError("Unrecognized wiping native declaration")
    from .port import ShortWipePort
    return ShortWipePort

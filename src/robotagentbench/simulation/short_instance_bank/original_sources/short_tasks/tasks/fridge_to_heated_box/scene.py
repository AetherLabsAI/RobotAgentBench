"""Task-owned door boxes and L1 cabinet backgrounds; no rotary/lamp assets."""
from copy import deepcopy


def prepare(plan, stage, manifest):
    from .sources import verify_sources
    from .initial_boxes import restore
    from .interior_contents import add
    verify_sources()
    if plan["task_name"] != "fridge_to_heated_box":
        raise ValueError("Task 07 scene hook only")
    result = deepcopy(manifest)
    result.setdefault("source_hashes", {})
    result["short_task_scene_preparation"] = dict(task_name=plan["task_name"],
        rule_version=plan["rule_version"], recipe="three_fridge_boxes_and_l1_interior_backgrounds",
        no_extra_runtime_tasks=True, rotary_lamp_installed=False,
        preparation_owner="short_task_scene_builder", native_validation="not_run")
    return add(plan, stage, restore(plan, stage, result))

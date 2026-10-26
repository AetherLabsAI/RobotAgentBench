"""Task 10 selection is connected; whole-cloth collection rules are absent."""
RULE_VERSION = "openend.short_task.collect_tablecloth.pending.v1"
GPU_PREVIEW_RULE_VERSION = "openend.short_task.collect_tablecloth.gpu_preview.v1"


def resolve(request):
    if request.get("rule_version") == GPU_PREVIEW_RULE_VERSION:
        from short_tasks.gpu_cloth.interface import resolve_preview
        return resolve_preview(request, "collect_tablecloth")
    if request.get("options"):
        raise ValueError("Task 10 has no approved folding/containment/physics override options")
    targets = request.get("targets")
    if targets and (len(targets) != 1 or targets[0].get("quantity") != 1
                    or targets[0].get("category") not in ("tablecloth", "tablecloth_dirty")):
        raise ValueError("Task 10 selects one cloth, not tabletop rigid objects")
    return dict(task_name="collect_tablecloth", rule_version=RULE_VERSION,
        available=False, unavailable=dict(kind="rules", reason="live_cloth_collection_rules_not_bound",
            message="现有软布移除/交互来源未接成统一房间收布任务；整张布当前入篮、脱离桌面、放手及折叠容差的版本未冻结。桌布篮不能按名字替换成物品篮或垃圾桶。"),
        scene_request=None, runtime_components=[],
        initial_requirements=["One live cloth already spread on the designated coffee-table region.",
            "No objects on the cloth; do not append a hidden cloth-clear task.",
            "One exact cloth-basket destination, separate from storage-basket and trash-bin roles."],
        instruction="Collect the tablecloth from the coffee table, place it inside the designated cloth basket, and release it.",
        score_policy=dict(maximum=None, process_maximum=None, final_maximum=None),
        pending=["shared_room_live_deformable_runtime", "measured_cloth_grasp_and_collection",
            "versioned_entire_cloth_in_basket_predicate", "folding_policy_and_points", "actual_cloth_basket_role"],
        candidate_sources=["evaluation/tasks/tabletop_pick/soft_cloth/remove_trajectory.py",
            "evaluation/tasks/tabletop_pick/soft_cloth/newton_free_advance.py",
            "evaluation/tasks/tabletop_pick/soft_cloth/live_surface.py"], native_validation="not_run")


def bind(plan, manifest):
    if plan.get("execution_mode") == "gpu_cloth_preview":
        from short_tasks.gpu_cloth.interface import bind_preview
        return bind_preview(plan, manifest)
    raise ValueError(plan["unavailable"]["message"])


def normalize(plan, component_summaries):
    if plan.get("execution_mode") == "gpu_cloth_preview":
        from short_tasks.gpu_cloth.interface import normalize_preview
        return normalize_preview(plan, component_summaries)
    if not isinstance(component_summaries, dict) or component_summaries:
        raise ValueError("Static cloth or object-clear summaries cannot score cloth collection")
    return dict(scores=dict(process=None, current_final=None, deductions=None, total=None, maximum=None),
        success=None, stages=[], targets=[])


def native_factory(component_type):
    from short_tasks.gpu_cloth.interface import reject_cpu_factory
    return reject_cpu_factory(component_type)

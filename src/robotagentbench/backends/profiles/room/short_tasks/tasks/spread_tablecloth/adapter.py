"""Task 09 selection is connected; shared-room deformable rules are absent."""
RULE_VERSION = "openend.short_task.spread_tablecloth.pending.v1"
GPU_PREVIEW_RULE_VERSION = "openend.short_task.spread_tablecloth.gpu_preview.v1"


def resolve(request):
    if request.get("rule_version") == GPU_PREVIEW_RULE_VERSION:
        from short_tasks.gpu_cloth.interface import resolve_preview
        return resolve_preview(request, "spread_tablecloth")
    if request.get("options"):
        raise ValueError("Task 09 has no approved coverage/folding/physics override options")
    targets = request.get("targets")
    if targets and (len(targets) != 1 or targets[0].get("quantity") != 1
                    or targets[0].get("category") not in ("tablecloth", "tablecloth_clean")):
        raise ValueError("Task 09 selects one cloth, not tabletop rigid objects")
    return dict(task_name="spread_tablecloth", rule_version=RULE_VERSION,
        available=False, unavailable=dict(kind="rules", reason="live_cloth_spread_rules_not_bound",
            message="仓库有动态桌布候选，但朋友的统一房间尚未绑定其模拟、抓布及铺开/覆盖规则版本。静态 tablecloth_dirty 贴图不能作为铺布完成证据。"),
        scene_request=None, runtime_components=[],
        initial_requirements=["One dynamic cloth in its scene-owned storage source.",
            "Designated coffee-table region physically clear; collisions remain enabled.",
            "Same live cloth topology/state and exact source/target bindings for all stages."],
        instruction="Take the tablecloth from its designated storage location, spread it over the designated area of the coffee table, and release it.",
        score_policy=dict(maximum=None, process_maximum=None, final_maximum=None),
        pending=["shared_room_live_deformable_runtime", "measured_cloth_grasp",
            "versioned_unfold_and_surface_coverage_predicates", "coverage_fold_tolerance_points"],
        candidate_sources=["evaluation/tasks/tabletop_pick/soft_cloth/native.py",
            "evaluation/tasks/tabletop_pick/soft_cloth/live_surface.py",
            "evaluation/tasks/tabletop_pick/soft_cloth/newton_free_advance.py"], native_validation="not_run")


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
        raise ValueError("Static cloth placement summaries cannot score a dynamic spread task")
    return dict(scores=dict(process=None, current_final=None, deductions=None, total=None, maximum=None),
        success=None, stages=[], targets=[])


def native_factory(component_type):
    from short_tasks.gpu_cloth.interface import reject_cpu_factory
    return reject_cpu_factory(component_type)

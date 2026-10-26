"""One public Short Task; native ports remain private implementation details."""
from copy import deepcopy
from importlib import import_module
import hashlib
import json
from pathlib import Path

from .common.contracts import TaskError
from .common.request import validate_public_request
from .common.episode import NO_PROGRESS_TIMEOUT_S


TASK_NAMES = (
    "prepare_coffee", "bake_bread", "stock_wooden_tray", "retrieve_from_drawer",
    "hang_cap", "clear_table_trash", "fridge_to_heated_box", "microwave_to_meal_tray",
    "spread_tablecloth", "collect_tablecloth", "clear_tablecloth", "pack_object",
    "prepare_cutlery", "carry_meal_tray", "sofa_to_coffee_table", "table_to_storage_basket",
    "stock_wooden_tray_interference", "sofa_to_drawer", "stock_wooden_tray_lighting",
    "select_trash", "select_sofa", "select_drink", "select_snack",
    "coffee_bag_interference", "coffee_machine_push_interference",
    "coffee_bag_then_machine_interference", "coffee_button_lighting_interference",
    "failure_recovery_shelf_drink", "failure_recovery_coffee_bag",
    'failure_recovery_sofa', 'failure_recovery_table_waste_empty_grasp', 'failure_recovery_table_waste_tipped_cup',
)

FAILURE_RECOVERY_TASKS = frozenset(('failure_recovery_shelf_drink', 'failure_recovery_coffee_bag', 'failure_recovery_sofa', 'failure_recovery_table_waste_empty_grasp', 'failure_recovery_table_waste_tipped_cup'))


def failure_recovery_adapter():
    return import_module('short_task_robustness.tasks.grasp_failure_recovery.public_adapter')


ATOMIC_COFFEE_TASKS = frozenset((
    'coffee_bag_interference', 'coffee_machine_push_interference',
    'coffee_bag_then_machine_interference', 'coffee_button_lighting_interference',
))


def atomic_coffee_adapter():
    return import_module('short_task_robustness.tasks.coffee_machine_push_interference.public_integration_v1.adapter')

CONTEXT_TASKS = {
    'select_trash': 'select_trash', 'select_sofa': 'select_sofa',
    'select_drink': 'select_shelf', 'select_snack': 'select_shelf',
}


def task_directory(name):
    if name not in TASK_NAMES:
        raise TaskError("Unknown Short Task", reason="unknown_task")
    root = Path(__file__).resolve().parents[1]
    if name in FAILURE_RECOVERY_TASKS:
        return failure_recovery_adapter().HERE
    if name in ATOMIC_COFFEE_TASKS:
        return atomic_coffee_adapter().NAMES[name][2]
    if name in CONTEXT_TASKS:
        return root / 'short_task_ContextReasoning/tasks' / CONTEXT_TASKS[name]
    return root / "short_tasks/tasks" / name


def is_gpu_cloth_preview(plan):
    """Only the two explicit, trusted cloth versions may bypass scored admission."""
    name = plan.get("task_name")
    return (name in ("spread_tablecloth", "collect_tablecloth")
            and plan.get("rule_version") == "openend.short_task." + name + ".gpu_preview.v1"
            and plan.get("execution_mode") == "gpu_cloth_preview"
            and plan.get("scoring_available") is False)


def adapter(name):
    if name not in TASK_NAMES:
        raise TaskError("Unknown Short Task", reason="unknown_task")
    if name in FAILURE_RECOVERY_TASKS:
        return failure_recovery_adapter()
    if name in ATOMIC_COFFEE_TASKS:
        return atomic_coffee_adapter()
    if name in CONTEXT_TASKS:
        return import_module('short_task_ContextReasoning.tasks.' + CONTEXT_TASKS[name] + '.adapter')
    return import_module("short_tasks.tasks." + name + ".adapter")


def resolve(request):
    request = validate_public_request(request)
    module = adapter(request["task_name"])
    try:
        plan = deepcopy(module.resolve(deepcopy(request)))
    except TaskError:
        raise
    except ValueError as exc:
        raise TaskError(str(exc)) from exc
    instruction = plan.get("instruction")
    # All consumers receive the environment's English instruction. Reject a
    # regressed adapter rather than translating differently for each policy.
    if (not isinstance(instruction, str) or not instruction.strip()
            or any("\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff"
                   or "\U00020000" <= c <= "\U000323af" for c in instruction)):
        raise TaskError("Task instruction must be nonempty English text", kind="rules",
                        reason="non_english_task_instruction")
    if (plan.get("task_name") != request["task_name"]
            or not isinstance(plan.get("available"), bool)
            or plan.get("rule_version") != request["rule_version"]):
        raise TaskError("Requested rule version does not match this adapter", kind="rules", reason="rule_version_mismatch")
    plan["request"] = request
    plan["schema"] = "openend.short_task.plan.v1"
    plan["public_task_count"] = 1
    plan["top_level_registrations"] = [request["task_name"]]
    plan["control_hz"] = 20
    preview = is_gpu_cloth_preview(plan)
    plan["no_progress_timeout_s"] = None if preview else NO_PROGRESS_TIMEOUT_S
    plan["physics_policy"] = ("physx_gpu_fixed_240hz" if preview
                              else "friend_cpu_single_workspace_60_240_hz")
    plan["action_policy_included"] = False
    # The current shared scene builder rejects extras. Unsupported input must
    # never disappear in a successfully admitted plan.
    for key in ("background", "position_constraints"):
        if request.get(key):
            plan.update(available=False, unavailable=dict(kind="load", reason=key+"_adapter_unavailable",
                message="The current scene provider cannot honor explicit " + key + "; no input was ignored"))
    if plan["available"] and not preview:
        policy = plan.get("score_policy", {})
        maximum = policy.get("maximum")
        if type(maximum) not in (int, float) or maximum <= 0:
            raise TaskError("Available task needs a resolved scoring maximum", kind="rules", reason="unresolved_score_policy")
    if preview and any(v is not None for v in plan.get("score_policy", {}).values()):
        raise TaskError("Unscored cloth preview cannot award benchmark points", kind="rules", reason="preview_score_forbidden")
    if request['task_name'] in FAILURE_RECOVERY_TASKS:
        # Preserve original atomic/shelf scoring and deadlines for these failed starts.
        if plan['failure_recovery_variant'] in ('shelf_drink', 'coffee_bag'):
            plan['no_progress_timeout_s'] = None
            plan['physics_policy'] = 'original_atomic_cpu_fixed_240hz'
        else:
            plan.update(plan['recovery_original_execution'])
        return plan
    if request['task_name'] in ATOMIC_COFFEE_TASKS:
        # Retain the atomic scoring. Generic interaction,
        # navigation and no-deduction policies would change its benchmark.
        plan['no_progress_timeout_s'] = None
        plan['physics_policy'] = 'original_atomic_cpu_fixed_240hz'
        from short_tasks.agent_probe_v1.cap_v1.native_tools.codex_adapter_v1.robustness6_v1.timing120_v1.integration import attach as attach_robustness_timing
        attach_robustness_timing(plan)
        return plan
    from .tasks.interaction_progress_v1.policy import attach
    attach(plan)
    from .tasks.no_deductions_v1.policy import attach as attach_deduction_policy
    attach_deduction_policy(plan)
    from .tasks.navigation_progress_v1.rules import attach as attach_navigation
    attach_navigation(plan)
    from short_tasks.agent_probe_v1.cap_v1.native_tools.codex_adapter_v1.robustness6_v1.timing120_v1.integration import attach as attach_robustness_timing
    attach_robustness_timing(plan)
    return plan


def verify_sources(plan):
    """Reject upstream drift before constructing any native readers."""
    if plan['task_name'] in FAILURE_RECOVERY_TASKS:
        return failure_recovery_adapter().verify_sources_for_plan(plan)
    if plan['task_name'] in ATOMIC_COFFEE_TASKS:
        return atomic_coffee_adapter().verify_sources(plan)
    folder = task_directory(plan["task_name"])
    path = folder / "component_sources.json"
    if not path.exists():
        raise TaskError("No pinned component source record", kind="rules", reason="missing_source_record")
    record = json.loads(path.read_text())
    sources = record.get("sources", record.get("files_sha256", record.get("source_files", {})))
    if not sources:
        raise TaskError("Empty component source record", kind="rules")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in sources.items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise TaskError("Pinned source missing: " + relative, kind="load", reason="missing_component_source")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise TaskError("Pinned source changed: " + relative, kind="rules", reason="component_source_changed")
    return dict(verified=True, file_count=len(sources), source_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bind(plan, manifest):
    if not plan["available"]:
        error = plan["unavailable"]
        raise TaskError(error["message"], kind=error["kind"], reason=error["reason"])
    plan, manifest = deepcopy(plan), deepcopy(manifest)
    if plan['task_name'] in FAILURE_RECOVERY_TASKS:
        plan['runtime_components'] = failure_recovery_adapter().bind(plan, manifest)
        plan['actual_layout_id'] = None  # Original atomic layout identity, not an invented room ID.
        return plan
    if plan['task_name'] in ATOMIC_COFFEE_TASKS:
        return atomic_coffee_adapter().bind(plan, manifest)
    request = plan["request"]
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise TaskError("Scene-owned object manifest required", kind="binding")
    paths = [o.get("path") for o in objects if isinstance(o, dict)]
    if len(paths) != len(objects) or any(not isinstance(p, str) or not p.startswith("/World/")
            or "//" in p or any(part in (".", "..") for part in p.split("/")) for p in paths):
        raise TaskError("Canonical absolute object paths required", kind="binding")
    if len(set(paths)) != len(paths):
        raise TaskError("Duplicate scene object paths", kind="binding", reason="duplicate_scene_identity")
    owners = {t["id"] for t in plan["scene_request"]["tasks"]}
    owners.update(c["instance_id"] for c in plan["runtime_components"])
    if any(o.get("task_id") not in owners for o in objects if o.get("task_id")):
        raise TaskError("Scene contains objects owned by an unrelated task", kind="binding", reason="unrelated_task_objects")
    if any(t.get("id", t.get("instance_id")) not in owners for t in manifest.get("tasks", [])):
        raise TaskError("Only this task's internal components may be registered", kind="binding", reason="multiple_public_tasks")
    if manifest.get("seed") != request["seed"]:
        raise TaskError("Scene seed differs from request", kind="binding", reason="seed_mismatch")
    actual = manifest.get("layout_id")
    isolated = (is_gpu_cloth_preview(plan)
                and manifest.get("layout_scope") == "isolated_component_preview")
    if not (isolated and actual is None) and (type(actual) is not int or not 1 <= actual <= 50):
        raise TaskError("Actual layout identity is missing", kind="binding", reason="missing_layout")
    if not isolated and request.get("layout_id") is not None and request["layout_id"] != actual:
        raise TaskError("An explicit layout cannot be replaced", kind="layout", reason="layout_mismatch")
    module = adapter(plan["task_name"])
    try:
        components = module.bind(plan, manifest) if hasattr(module, "bind") else deepcopy(plan["runtime_components"])
    except TaskError:
        raise
    except (ValueError, KeyError) as exc:
        raise TaskError(str(exc), kind="binding", reason="instance_binding_failed") from exc
    if isinstance(components, dict):
        components = components.get("components", components.get("runtime_components"))
    if not isinstance(components, list) or not components:
        raise TaskError("Available task has no native components", kind="binding", reason="missing_components")
    identifiers = [c.get("instance_id") for c in components]
    if any(not isinstance(i, str) or not i for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise TaskError("Internal component identities must be unique", kind="binding", reason="duplicate_components")
    for c in components:
        if not isinstance(c.get("type"), str) or not isinstance(c.get("parameters"), dict):
            raise TaskError("Explicit typed native component required", kind="binding", reason="bad_component")
    plan["runtime_components"] = deepcopy(components)
    plan["actual_layout_id"] = None if isolated else actual
    return plan


def native_factory(plan, declaration):
    """Import simulator-dependent classes only when an owner builds a scene."""
    module = adapter(plan["task_name"])
    custom = declaration.get("custom_native_port")
    if custom:
        module_name, class_name = custom.rsplit(".", 1)
        if not module_name.startswith("short_tasks.tasks."):
            raise TaskError("Native override must belong to Short Tasks", kind="binding")
        return getattr(import_module(module_name), class_name)
    if hasattr(module, "factory_override"):
        value = module.factory_override(plan, deepcopy(declaration))
        if value is not None:
            return value
    if hasattr(module, "native_factory"):
        value = module.native_factory(declaration["type"])
        if value is not None:
            return value
    from openend.room import FACTORIES
    return FACTORIES[declaration["type"]]


def normalize(plan, summaries):
    expected = {c["instance_id"] for c in plan["runtime_components"]}
    if not isinstance(summaries, dict) or set(summaries) != expected:
        raise TaskError("Every bound component must supply one current summary", kind="binding", reason="incomplete_component_snapshot")
    try:
        from .tasks.interaction_progress_v1.policy import component_plan
        return adapter(plan["task_name"]).normalize(component_plan(plan), deepcopy(summaries))
    except TaskError:
        raise
    except (ValueError, KeyError) as exc:
        raise TaskError(str(exc), kind="binding", reason="invalid_component_snapshot") from exc

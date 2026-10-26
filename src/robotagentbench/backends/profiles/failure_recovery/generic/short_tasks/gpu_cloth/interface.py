"""Pure contracts for scene-owned GPU cloth previews, without simulator imports.

This preview admits a native diagnostic component. It does not freeze grasp,
coverage, containment, or folding predicates and cannot issue benchmark points.
The manifest is a declaration; the native owner still verifies the real stage.
"""
from copy import deepcopy
import json
import math

from short_tasks.common.contracts import TaskError


EXECUTION_MODE = "gpu_cloth_preview"
PHYSICS_POLICY = "physx_gpu_fixed_240hz"
SCENE_SCHEMA = "openend.short_task.gpu_cloth_scene.v1"
SAMPLE_SCHEMA = "openend.short_task.gpu_cloth_sample.v1"
BODY_ROOT = "/World/Assets/TableclothRuleState"
MESH = BODY_ROOT + "/Surface"
SCENE_OWNER = "gpu_cloth_scene_owner.v1"
PHYSICS_PROFILE = dict(profile="openend.short_task.gpu_cloth_physx.v1",
    physics_backend="physx", physics_dt_s=1 / 240,
    physics_hz=240, control_hz=20, deformable_beta_enabled=True,
    enable_gpu_dynamics=True, broadphase="GPU", solver="TGS",
    gravity_m_s2=[0, 0, -9.81], tensor_pipeline="gpu", tensor_frontend="torch")
SCORE_KEYS = ("process", "current_final", "deductions", "total", "maximum")
TASK_ROLES = {
    "spread_tablecloth": ("cloth_storage", "coffee_table"),
    "collect_tablecloth": ("coffee_table", "cloth_basket"),
}


def null_scores():
    return {key: None for key in SCORE_KEYS}


def preview_rule_version(task_name):
    if task_name not in TASK_ROLES:
        raise ValueError("Only the two cloth tasks support this diagnostic profile")
    return "openend.short_task." + task_name + ".gpu_preview.v1"


def _finite(value, label, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(label + " must not be negative")
    return value


def _vector(value, label, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + " must have three coordinates")
    for number in value:
        _finite(number, label)
        if positive and number <= 0:
            raise ValueError(label + " dimensions must be positive")
    return list(value)


def _roles(request, task_name):
    source, destination = TASK_ROLES[task_name]
    for field, required_role in (("source", source), ("destination", destination)):
        supplied = request.get(field)
        if supplied is not None and (not isinstance(supplied, dict)
                or set(supplied) != {"role"} or supplied["role"] != required_role):
            raise ValueError("GPU cloth preview requires exact " + field + " role " + required_role)
    return source, destination


def resolve_preview(request, task_name):
    if request.get("rule_version") != preview_rule_version(task_name):
        raise ValueError("Explicit GPU cloth preview rule version required")
    options = request.get("options", {})
    if not isinstance(options, dict) or options:
        raise ValueError("Physics, initial positions and scoring overrides are scene-owned")
    categories = ("tablecloth", "tablecloth_clean") if task_name == "spread_tablecloth" else ("tablecloth", "tablecloth_dirty")
    targets = request.get("targets")
    if targets is not None and (not isinstance(targets, list) or len(targets) != 1
            or not isinstance(targets[0], dict) or type(targets[0].get("quantity")) is not int
            or targets[0]["quantity"] != 1 or targets[0].get("category") not in categories):
        raise ValueError("GPU cloth preview selects one live cloth, not a texture or rigid object")
    source, destination = _roles(request, task_name)
    return dict(task_name=task_name, rule_version=preview_rule_version(task_name),
        available=True, unavailable=None, execution_mode=EXECUTION_MODE,
        benchmark_available=False, scoring_available=False,
        task_initial_state_verified=False, occupancy_verification="requires_native_pre_attach_geometry_check",
        physics_policy=PHYSICS_POLICY, physics_profile=deepcopy(PHYSICS_PROFILE),
        physics_hz=240, control_hz=20, no_progress_timeout_s=None,
        scene_runtime_available=True, layout_scope="isolated_component_preview",
        scene_request=dict(schema="openend.scene_requirements.request.v2",
            seed=request["seed"], layout_id=request.get("layout_id"),
            tasks=[dict(id="gpu_cloth", type="gpu_tablecloth")], extras=[]),
        runtime_components=[dict(instance_id="gpu_cloth", type="gpu_tablecloth", parameters={})],
        source_role=source, destination_role=destination,
        target_category=targets[0]["category"] if targets else categories[1],
        initial_requirements=[
            "One actual PhysX deformable cloth with the scene-owned native body and surface mesh.",
            "A fixed GPU physics scene at 240 Hz, with control observations at 20 Hz.",
            "The scene owns the initial cloth pose, designated table/storage/basket fixtures and clear occupancy.",
            "Native admission verifies stage units, topology and cloth native vertices; pre-attach fixture bounds and obstruction checks remain required.",
            "Geometric clearance is diagnostic evidence; native support contact, grasp and benchmark initial-state acceptance are not verified."],
        instruction=("Preview the dynamic tablecloth at its storage location and the coffee table area. Observe the actual cloth state only; this preview is unscored." if task_name == "spread_tablecloth"
            else "Preview the dynamic tablecloth on the coffee table and the cloth basket. Observe the actual cloth state only; this preview is unscored."),
        score_policy=dict(maximum=None, process_maximum=None, final_maximum=None),
        unsupported_operations=["robot_cloth_grasp", "cloth_release_detection", "benchmark_success"],
        pending=["versioned_cloth_grasp", "versioned_surface_coverage" if task_name == "spread_tablecloth"
            else "versioned_entire_cloth_in_basket", "benchmark_score_policy"],
        native_validation="not_run",
        limitations=["This explicit version is a GPU cloth component diagnostic, not the scored document task.",
            "A requested room layout is provenance only; the diagnostic fixture is not that room.",
            "Table/storage/basket and clear occupancy declarations do not establish benchmark initial-state acceptance.",
            "No source/destination placement, native grasp or full task success is inferred from mesh motion."])


def bind_preview(plan, manifest):
    if plan.get("execution_mode") != EXECUTION_MODE or plan.get("scoring_available") is not False:
        raise ValueError("Only the explicit unscored GPU cloth preview can bind this component")
    task_name = plan["task_name"]
    if plan.get("rule_version") != preview_rule_version(task_name):
        raise ValueError("GPU preview rule version mismatch")
    if plan.get("request") is not None:
        _roles(plan["request"], task_name)
    data = manifest.get("gpu_cloth")
    if not isinstance(data, dict) or data.get("schema") != SCENE_SCHEMA:
        raise ValueError("Scene owner must provide a versioned GPU cloth manifest")
    if data.get("body_root") != BODY_ROOT or data.get("mesh") != MESH:
        raise ValueError("GPU cloth must bind the approved native body and mesh")
    source, destination = TASK_ROLES[task_name]
    if data.get("source_role") != source or data.get("destination_role") != destination:
        raise ValueError("Scene cloth source/destination roles differ from the selected task")
    if type(data.get("meters_per_unit")) not in (int, float) or data["meters_per_unit"] != 1 or data.get("up_axis") != "Z":
        raise ValueError("The scene declaration must use meters and Z up; native admission verifies the real stage")
    _vector(data.get("initial_center_world_m"), "initial cloth center")
    _finite(data.get("initial_yaw_rad"), "initial cloth yaw")
    region_paths = []
    for name in ("table_region", "storage_region", "basket_region"):
        region = data.get(name)
        if not isinstance(region, dict):
            raise ValueError("Scene owner must bind " + name)
        _vector(region.get("center_world_m"), name + " center")
        _vector(region.get("size_m"), name + " size", positive=True)
        path = region.get("prim_path")
        if (not isinstance(path, str) or not path.startswith("/World/") or "//" in path
                or any(part in (".", "..", "") for part in path.split("/")[1:])):
            raise ValueError("Each cloth fixture region needs its exact canonical native prim_path")
        region_paths.append(path)
    if len(set(region_paths)) != 3 or any(path == BODY_ROOT or path.startswith(BODY_ROOT + "/") for path in region_paths):
        raise ValueError("Cloth table/storage/basket fixture identities must be distinct from each other and the cloth")
    occupancy = data.get("occupancy")
    if not isinstance(occupancy, dict) or occupancy.get("table_clear") is not True or occupancy.get("cloth_clear") is not True:
        raise ValueError("Clear table/cloth occupancy must be declared by the scene owner")
    rows = manifest.get("objects")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Scene-owned cloth object rows required")
    cloth = [row for row in rows if row.get("role") == "tablecloth" or row.get("id") == "tablecloth"
             or row.get("path") == BODY_ROOT]
    if len(cloth) != 1:
        raise ValueError("Exactly one scene-owned live cloth is required")
    row = cloth[0]
    if (row.get("path") != BODY_ROOT or row.get("role") != "tablecloth"
            or row.get("task_id") != "gpu_cloth" or row.get("initialization_owner") != SCENE_OWNER):
        raise ValueError("Cloth identity and initial-state ownership must match this exact scene recipe")
    tasks = manifest.get("tasks")
    if (not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict)
            or tasks[0].get("id", tasks[0].get("instance_id")) != "gpu_cloth"
            or tasks[0].get("type") != "gpu_tablecloth"):
        raise ValueError("GPU cloth preview has one internal component and no other task registration")
    scope = manifest.get("layout_scope")
    if scope not in ("isolated_component_preview", "shared_world"):
        raise ValueError("GPU cloth layout scope must distinguish a diagnostic fixture from a shared room")
    actual = manifest.get("layout_id")
    if scope == "shared_world" and (type(actual) is not int or not 1 <= actual <= 50):
        raise ValueError("A shared-world component must retain its actual room layout identity")
    plan["layout_scope"] = scope
    plan["actual_layout_id"] = actual if scope == "shared_world" else None
    plan["target_bindings"] = dict(target=BODY_ROOT, mesh=MESH,
        scene_recipe_owner="gpu_cloth", initialization_owner=SCENE_OWNER,
        source_role=source, destination_role=destination)
    plan["gpu_cloth"] = deepcopy(data)
    return deepcopy(plan["runtime_components"])


def _measured_geometry(measured):
    if not isinstance(measured, dict) or measured.get("finite") is not True:
        raise ValueError("A finite, actual native cloth sample is required")
    for name in ("vertex_count", "triangle_count"):
        if type(measured.get(name)) is not int or measured[name] <= 0:
            raise ValueError("Actual positive cloth " + name + " required")
    _vector(measured.get("centroid_world_m"), "cloth centroid")
    bounds = measured.get("aabb_world_m")
    if not isinstance(bounds, dict):
        raise ValueError("Measured cloth world bounds required")
    low = _vector(bounds.get("low"), "cloth bounds low")
    high = _vector(bounds.get("high"), "cloth bounds high")
    if any(a > b for a, b in zip(low, high)):
        raise ValueError("Measured cloth bounds are inverted")
    for name in ("displacement_rms_m", "displacement_max_m", "edge_length_change_rms_m"):
        _finite(measured.get(name), name, nonnegative=True)
    if measured["displacement_rms_m"] > measured["displacement_max_m"] + 1e-9:
        raise ValueError("Measured RMS displacement cannot exceed maximum displacement")


def _native_vertices(sample):
    state = sample.get("native_state")
    if not isinstance(state, dict):
        raise ValueError("GPU cloth evidence must contain actual native vertices and topology")
    if (state.get("physics_step") != sample["evidence_physics_step"]
            or type(state.get("physics_step")) is not int
            or state.get("time_s") != sample["evidence_time_s"]):
        raise ValueError("Cloth vertices and measurements must come from the same physics boundary")
    _finite(state.get("time_s"), "native vertex evidence time", nonnegative=True)
    measured = sample["measured"]
    points, triangles = state.get("points_world_m"), state.get("triangles")
    if not isinstance(points, list) or len(points) != measured["vertex_count"]:
        raise ValueError("Measured vertex count differs from the native tensor state")
    points = [_vector(point, "native cloth vertex") for point in points]
    if not isinstance(triangles, list) or len(triangles) != measured["triangle_count"]:
        raise ValueError("Measured triangle count differs from the native cloth topology")
    for triangle in triangles:
        if (not isinstance(triangle, (list, tuple)) or len(triangle) != 3
                or any(type(index) is not int or not 0 <= index < len(points) for index in triangle)
                or len(set(triangle)) != 3):
            raise ValueError("Native cloth triangles require three distinct valid vertex indices")
    for axis in range(3):
        values = [point[axis] for point in points]
        expected = (sum(values) / len(points), min(values), max(values))
        reported = (measured["centroid_world_m"][axis], measured["aabb_world_m"]["low"][axis],
                    measured["aabb_world_m"]["high"][axis])
        if any(abs(a - b) > 1e-6 for a, b in zip(expected, reported)):
            raise ValueError("Cloth centroid/bounds must describe the actual sampled native vertices")


def normalize_preview(plan, summaries):
    if not isinstance(summaries, dict) or set(summaries) != {"gpu_cloth"}:
        raise ValueError("Exactly one actual GPU cloth component summary is required")
    sample = summaries["gpu_cloth"]
    if (not isinstance(sample, dict) or sample.get("schema") != SAMPLE_SCHEMA
            or sample.get("instance_id") != "gpu_cloth" or sample.get("task_name") != plan["task_name"]):
        raise ValueError("GPU cloth sample must identify this exact task component")
    binding = plan.get("target_bindings")
    if not isinstance(binding, dict):
        raise ValueError("Native cloth evidence must bind a scene-owned body before normalization")
    if sample.get("body_root") != binding["target"] or sample.get("mesh") != binding["mesh"]:
        raise ValueError("Native cloth sample belongs to a different body or mesh")
    if (sample.get("state_source") != "physx_deformable" or sample.get("diagnostics_only") is not True
            or sample.get("scoring_available") is not False or sample.get("success", "missing") is not None
            or sample.get("scores") != null_scores()):
        raise ValueError("Cloth diagnostics cannot supply scores, success or texture-based state")
    if sample.get("live_physics_verified") is not True:
        raise ValueError("GPU preview evidence requires an actual native tensor sample")
    step = sample.get("evidence_physics_step")
    if type(step) is not int or step < 0:
        raise ValueError("Actual nonnegative native physics step required")
    _finite(sample.get("evidence_time_s"), "native evidence time", nonnegative=True)
    _measured_geometry(sample.get("measured"))
    _native_vertices(sample)
    try:
        json.dumps(sample, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Native GPU cloth evidence must be finite JSON data") from exc
    return dict(scores=null_scores(), success=None,
        execution_mode=EXECUTION_MODE, scoring_available=False,
        stages=[dict(name="live_cloth_observation", status="measured",
            state_source="physx_deformable", evidence_time_s=sample["evidence_time_s"],
            evidence_physics_step=step),
            dict(name="robot_cloth_grasp", status="not_observable", scoring_available=False),
            dict(name="benchmark_completion", status="unavailable", scoring_available=False)],
        targets=[dict(category=plan["target_category"], path=binding["target"], mesh=binding["mesh"],
            source_role=binding["source_role"], destination_role=binding["destination_role"],
            measurements=deepcopy(sample["measured"]), success=None)],
        native_state=deepcopy(sample))


def preview_result(plan, summary, *, episode_id, status="running", termination_reason=None):
    """The same public result envelope, explicitly without benchmark semantics."""
    if status not in ("running", "finished"):
        raise ValueError("Preview failures use the public typed failure receipt")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("A real preview episode identity is required")
    if (status == "running" and termination_reason is not None) or (status == "finished"
            and (not isinstance(termination_reason, str) or not termination_reason or termination_reason == "success")):
        raise ValueError("GPU diagnostics cannot finish with benchmark success")
    result = normalize_preview(plan, {"gpu_cloth": summary})
    result.update(schema="openend.short_task.result.draft.v1", task_name=plan["task_name"],
        episode_id=episode_id, actual_layout_id=plan.get("actual_layout_id"),
        requested_layout_id=plan.get("request", {}).get("layout_id"),
        layout_scope=plan.get("layout_scope", "isolated_component_preview"), rule_version=plan["rule_version"],
        status=status, termination_reason=termination_reason,
        evidence_time_s=summary["evidence_time_s"], evidence_step=summary["evidence_physics_step"])
    return result


def reject_cpu_factory(_component_type):
    raise TaskError("Explicit GPU cloth preview must use its trusted GPU scene/session builder",
        kind="load", reason="gpu_preview_requires_gpu_builder")

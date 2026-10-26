"""One independent microwave recipe using the existing room planner/assembly."""
from copy import deepcopy
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = dict(task_name="microwave_heat_take", version="short_tasks.microwave_heat_take.scene.v1")
INSTANCE = "heat_take"


def selected(request):
    return (isinstance(request, dict) and isinstance(request.get("tasks"), list)
            and any(isinstance(t, dict) and t.get("type") == "microwave_heat_take"
                    for t in request["tasks"]))


def validate_request(request, *, resolved=False):
    allowed = {"schema", "seed", "layout_id", "tasks", "extras"}
    if (set(request)-allowed or request.get("schema") != "openend.scene_requirements.request.v2"
            or type(request.get("seed")) is not int or not 1 <= request["seed"] <= 4294967295
            or request.get("extras", []) != []):
        raise ValueError("Exact independent microwave room request required")
    tasks = request.get("tasks")
    if (not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict)
            or set(tasks[0]) != {"id", "type", "payload_variant"}
            or tasks[0]["id"] != INSTANCE or tasks[0]["type"] != "microwave_heat_take"
            or tasks[0]["payload_variant"] not in ("beef_box", "leftover_box")):
        raise ValueError("One exact heat/take operation and original payload required")
    layout = request.get("layout_id")
    if layout is not None and (type(layout) is not int or not 1 <= layout <= 50):
        raise ValueError("Unknown layout")
    if resolved and layout is None:
        raise ValueError("Room planner requires an actual layout")
    return tasks[0]


def compile_request(request):
    from planner import compile_plan
    task = validate_request(request, resolved=True)
    base = compile_plan(dict(schema="openend.scene_requirements.request.v1",
                             seed=request["seed"], tasks=[], extras=[]))
    station = dict(id=INSTANCE, type="microwave", placement={})
    return dict(schema="openend.scene_requirements.plan.v2", executable=False,
        fixed_task_requirements=[], shelf_task_requirements=[], meal_task_requirements=[],
        destination_task_requirements=[], portable_inventory=base,
        microwave_heat_take_requirements=[deepcopy(task)],
        station_request=dict(layout_id=request["layout_id"], seed=request["seed"], stations=[station]),
        assembly_requirements=[dict(task_id=INSTANCE, type="microwave",
            initial_profile="existing_unheated_closed_loaded_microwave_plate_assembly",
            transform_policy="one rigid station transform; preserve component relative transforms",
            asset_instance_expansion_complete=False)],
        pending=["actual assembly composition and native initial acceptance"])


def resolve_request(request):
    validate_request(request)
    request = deepcopy(request)
    layouts = json.loads((ROOT/"openend/tasks/scene_requirements/placement_sources/random-layouts-v1.json").read_text())["layouts"]
    order = sorted(l["id"] for l in layouts)
    random.Random(request["seed"]).shuffle(order)
    specified = request.get("layout_id")
    if specified is not None:
        if specified not in order:
            raise ValueError("Layout not in current source library")
        order = [specified]
    request["layout_id"] = order[0]
    layout = next(l for l in layouts if l["id"] == order[0])
    return dict(request=request, requirements=compile_request(request), layout=layout,
        layout_candidates=order, layout_selection=dict(
            mode="specified" if specified is not None else "seeded_random",
            requested=specified, selected=layout["id"], seed=request["seed"], rerolls=0))


def validate_output(source, root, data):
    source, root = Path(source).resolve(), Path(root).resolve()
    if HERE != root/"short_tasks/tasks/microwave_heat_take":
        raise ValueError("Generator is outside the registered task directory")
    if (not source.is_relative_to(HERE/"execution") or source.name != "placement.json"
            or source.is_symlink() or (source.parent/"room.usda").exists()
            or data.get("task_scene_policy") != POLICY
            or data.get("openend_episode_policy") is not None):
        raise ValueError("Fresh task-owned microwave scene required")
    task = validate_request(data["request"], resolved=True)
    if data["requirements"].get("microwave_heat_take_requirements") != [task]:
        raise ValueError("Scene requirement does not match the exact request")


def compose(stage, data, rows, tasks, project, source_hashes, sha):
    from openend.tasks.meal_tray.microwave_initialization import compose as original
    task = validate_request(data["request"], resolved=True)
    if tasks or any(r.get("task_id") for r in rows):
        raise ValueError("No predecessor or unrelated scored assembly permitted")
    # This is ONLY the existing asset initializer's input, not a registered
    # meal task, fabricated fridge handoff, or completed process declaration.
    assembly_request = dict(id=INSTANCE, type="microwave_box_to_meal_tray",
                            payload_variant=task["payload_variant"])
    added = original(stage, [assembly_request], rows, project, source_hashes, sha,
                     payload_from_fridge=False)
    for row in added:
        row["task_id"] = INSTANCE
    tasks.append(dict(id=INSTANCE, type="microwave_heat_take", source="microwave",
        target_path="/World/Assets/DiningTableMicrowave/Asset/StartKnob",
        object_paths=["/World/Assets/DiningTableMicrowave", *[r["path"] for r in added]],
        parameters=dict(payload_variant=task["payload_variant"]),
        status="scene_bound", score=None, runtime_started=False))
    source_hashes[str(Path(__file__))] = sha(Path(__file__))
    return added


def registration(request, manifest):
    rows = request["tasks"]
    if len(rows) != 1 or rows[0]["type"] != "microwave_heat_take":
        raise ValueError("Only one standalone Short Task may register this recipe")
    return dict(owner="short_tasks", task_name="microwave_heat_take",
        all_requested_tasks_bound=True, pending=[], runtime_started=False,
        native_physics_validated=False, identity_bound_not_physics_verified=True)

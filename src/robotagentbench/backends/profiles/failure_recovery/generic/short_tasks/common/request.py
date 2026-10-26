"""Admission for the two fixed-layout diagnostic implementations.

The larger JSON schema also describes future semantic inputs. Those inputs
remain unavailable here: accepting their shape must not silently ignore them.
"""
from copy import deepcopy


RULE_VERSIONS = {
    "prepare_coffee": "openend.short_task.prepare_coffee.v1",
    "bake_bread": "short_tasks.bake_bread.cake_preview.v1",
}


def validate_public_request(request):
    """Check structure without pretending to prove variant/scene availability."""
    from .contracts import TaskError
    from short_tasks.registry import TASK_NAMES
    if not isinstance(request, dict):
        raise TaskError("One Short Task request object required")
    required = {"schema", "task_name", "seed", "rule_version"}
    allowed = required | {"layout_id", "targets", "source", "destination", "background", "position_constraints", "options"}
    if not required <= set(request) or set(request) - allowed:
        raise TaskError("Only the public single-task fields are accepted; hidden IDs and additional tasks are forbidden")
    if request["schema"] not in ("openend.short_task.request.draft.v1", "openend.short_task.request.v1"):
        raise TaskError("Unknown Short Task request schema")
    if not isinstance(request["task_name"], str) or request["task_name"] not in TASK_NAMES:
        raise TaskError("Unknown Short Task", reason="unknown_task")
    if type(request["seed"]) is not int or not 1 <= request["seed"] <= 4294967295:
        raise TaskError("Valid integer seed required")
    layout = request.get("layout_id")
    if layout is not None and (type(layout) is not int or not 1 <= layout <= 50):
        raise TaskError("Unknown explicit layout")
    if not isinstance(request["rule_version"], str) or not request["rule_version"]:
        raise TaskError("Explicit frozen rule version required")
    for name in ("options", "position_constraints"):
        if name in request and not isinstance(request[name], dict):
            raise TaskError(name + " must be an object")
    for name in ("targets", "background"):
        if name not in request:
            continue
        rows = request[name]
        if not isinstance(rows, list) or (name == "targets" and not rows):
            raise TaskError(name + " must be a semantic selection list")
        for row in rows:
            if (not isinstance(row, dict) or not {"category", "quantity"} <= set(row)
                    or set(row) - {"category", "quantity", "description"}
                    or not isinstance(row["category"], str) or not row["category"]
                    or type(row["quantity"]) is not int or row["quantity"] <= 0
                    or ("description" in row and (not isinstance(row["description"], str) or not row["description"]))):
                raise TaskError("Targets use category/quantity; scene IDs are internal")
    for name in ("source", "destination"):
        if name in request:
            row = request[name]
            if (not isinstance(row, dict) or "role" not in row or set(row) - {"role", "variant"}
                    or not isinstance(row["role"], str) or not row["role"]
                    or ("variant" in row and (not isinstance(row["variant"], str) or not row["variant"]))):
                raise TaskError("Source/destination need explicit semantic roles")
    return deepcopy(request)


def validate_request(request, *, manifest):
    if not isinstance(request, dict):
        raise ValueError("One Short Task request object required")
    required = {"schema", "task_name", "seed", "layout_id", "rule_version", "options"}
    if set(request) != required:
        raise ValueError("Fixed-layout preview accepts only explicit single-task inputs; semantic selectors are pending")
    if request["schema"] != "openend.short_task.request.draft.v1":
        raise ValueError("Unsupported Short Task request schema")
    name = request["task_name"]
    if not isinstance(name, str) or name not in RULE_VERSIONS:
        raise ValueError("Only tasks 01 and 02 have diagnostic implementations")
    if request["rule_version"] != RULE_VERSIONS[name]:
        raise ValueError("Request does not identify this adapter's pinned rules")
    seed, layout = request["seed"], request["layout_id"]
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("Explicit valid source seed required")
    if type(layout) is not int or not 1 <= layout <= 50:
        raise ValueError("Explicit fixed layout required")
    if seed != manifest.get("seed") or layout != manifest.get("layout_id"):
        raise ValueError("Request seed/layout differs from the prepared actual scene")
    if manifest.get("short_task", {}).get("name") != name:
        raise ValueError("Prepared scene belongs to another task")
    options = request["options"]
    if not isinstance(options, dict):
        raise ValueError("Explicit task options object required")
    if name == "prepare_coffee":
        if options:
            raise ValueError("Coffee has no approved option or scoring overrides")
    else:
        if (set(options) != {"bread_variant", "finish_on_small_plate"}
                or options["bread_variant"] not in ("no_sugar", "sugar")
                or options["finish_on_small_plate"] is not False):
            raise ValueError("Only an explicit real cake variant without an added plate finish is supported")
        if options["bread_variant"] != manifest["short_task"].get("bread_variant"):
            raise ValueError("Requested target cake differs from the scene-owned variant")
    return deepcopy(request)

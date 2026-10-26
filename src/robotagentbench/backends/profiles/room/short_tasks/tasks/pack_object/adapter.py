"""One scene-owned personal object; approved per-asset process/final rules.

The additional current-release gate is local to Short Tasks. Upstream room
scorers and their process milestones are never modified.
"""
from copy import deepcopy
import math

RULE_VERSION = "openend.short_task.pack_object.v1"
VARIANTS = {
    "wallet": ("wallet_transfer", 7, ("small_table", "sofa", "coffee_table.objects")),
    "glasses_case": ("glasses_transfer", 7, ("small_table", "sofa", "coffee_table.objects")),
    "power_bank": ("powerbank_transfer", 9, ("sofa",)),
    "water_bottle": ("water_bottle_transfer", 9, ("sofa",)),
}


def request_parts(request, task_name, rule_version, allowed):
    if not isinstance(request, dict) or request.get("task_name") != task_name:
        raise ValueError("Explicit matching Short Task required")
    permitted = {"schema", "task_name", "seed", "layout_id", "options", "rule_version",
        "targets", "source", "destination", "background", "position_constraints"}
    if set(request) - permitted:
        raise ValueError("Unknown Short Task request fields")
    if request.get("schema", "openend.short_task.request.draft.v1") not in (
        "openend.short_task.request.draft.v1", "openend.short_task.request.v1"
    ):
        raise ValueError("Unsupported Short Task request schema")
    seed = request.get("seed")
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("seed must be an integer in 1..4294967295")
    layout = request.get("layout_id")
    if layout is not None and (type(layout) is not int or layout < 1):
        raise ValueError("layout_id must be a positive integer or null")
    options = deepcopy(request.get("options", {}))
    if not isinstance(options, dict) or set(options) - set(allowed):
        raise ValueError("Unknown task options")
    if request.get("rule_version", rule_version) != rule_version:
        raise ValueError("Unsupported frozen rule version")
    # These singleton tasks accept the public semantic selector directly.
    # Scene IDs remain private; a selector cannot claim an additional target.
    aliases = {"glasses": "glasses_case", "powerbank": "power_bank", "cap": "baseball_cap",
        "remote": "remote_tv", "book": "book_red"}
    fixed = {"retrieve_from_drawer": {"source": "coffee_table.drawer"},
        "hang_cap": {"asset": "baseball_cap", "destination": "coat_rack_hooks"},
        "sofa_to_coffee_table": {"source": "sofa", "destination": "coffee_table.objects"}}

    def select(key, value):
        if key in fixed.get(task_name, {}):
            if value != fixed[task_name][key]:
                raise ValueError("Semantic selector contradicts this task's explicit " + key)
            return
        if key not in allowed:
            raise ValueError("Semantic selector is not supported for this task: " + key)
        if key in options and options[key] != value:
            raise ValueError("Conflicting semantic and option selectors: " + key)
        options[key] = value

    if "targets" in request:
        targets = request["targets"]
        if not isinstance(targets, list) or len(targets) != 1:
            raise ValueError("Exactly one semantic target is required for this task")
        target = targets[0]
        if (not isinstance(target, dict) or not {"category", "quantity"} <= set(target)
                or set(target)-{"category", "quantity", "description"}
                or not isinstance(target["category"], str) or not target["category"]
                or type(target["quantity"]) is not int or target["quantity"] != 1
                or ("description" in target and (not isinstance(target["description"], str) or not target["description"]))):
            raise ValueError("One category with quantity=1 required; no hidden target IDs")
        select("asset", aliases.get(target["category"], target["category"]))

    for name in ("source", "destination"):
        if name not in request:
            continue
        row = request[name]
        if (not isinstance(row, dict) or "role" not in row or set(row)-{"role", "variant"}
                or not isinstance(row["role"], str) or not row["role"]
                or ("variant" in row and (not isinstance(row["variant"], str) or not row["variant"]))):
            raise ValueError("Semantic source/destination role and optional variant required")
        role, variant = row["role"], row.get("variant")
        if role == "coffee_table":
            variant = variant or ("drawer" if task_name == "retrieve_from_drawer" else "objects")
            if variant not in ("objects", "drawer"):
                raise ValueError("Unsupported coffee-table source/destination variant")
            role += "." + variant
        elif role in ("coat_rack", "coat_rack_hooks"):
            if variant not in (None, "hooks"):
                raise ValueError("Unsupported coat-rack destination variant")
            role = "coat_rack_hooks"
        elif role in ("fixed_cpu_bag", "fixed_open_bag"):
            if variant not in (None, "fixed_open"):
                raise ValueError("Only the fixed open bag variant is supported")
            role = "fixed_cpu_bag"
        elif variant is not None:
            raise ValueError("Unknown semantic role variant")
        select(name, role)
    # Root admission reports nonempty background/placement constraints as
    # loading-unavailable before building. Do not turn them into score targets.
    if "background" in request and not isinstance(request["background"], list):
        raise ValueError("Background must be a semantic selection list")
    if "position_constraints" in request and not isinstance(request["position_constraints"], dict):
        raise ValueError("Position constraints must be an object")
    return deepcopy(options), seed, layout


def plan_base(request, name, version, component, asset, source, destination, maximum,
              instruction, initial_requirements):
    scene_task = dict(id=name, type=component, source=source)
    if component in ("wallet_transfer", "glasses_transfer"):
        scene_task["destination"] = destination
    parameters = {"destination": destination} if component in ("wallet_transfer", "glasses_transfer") else {}
    return dict(task_name=name, rule_version=version, available=True, unavailable=None,
        asset=asset, source=source, destination=destination,
        scene_request=dict(schema="openend.scene_requirements.request.v2", seed=request["seed"],
            layout_id=request.get("layout_id"), tasks=[scene_task], extras=[]),
        runtime_components=[dict(instance_id=name, type=component, parameters=parameters)],
        initial_requirements=list(initial_requirements),
        instruction=instruction,
        score_policy=dict(maximum=maximum, process_maximum=maximum-2, final_maximum=2),
        factory_descriptor="short_tasks.tasks.%s.adapter:factory_override" % name,
        evidence_policy=dict(current_release_required=True,
            legacy_geometry_retained=True, whole_object_containment_verified=False,
            grasp="position_follow_2s_3cm_10cm", timeout_owner="single_short_episode"))


def unavailable(plan, reason, message, kind="rules"):
    plan["available"] = False
    plan["unavailable"] = dict(kind=kind, reason=reason, message=message)
    plan["score_policy"] = dict(maximum=None, process_maximum=None, final_maximum=None)
    return plan


def resolve(request):
    options, _, _ = request_parts(request, "pack_object", RULE_VERSION,
        ("asset", "source", "destination"))
    asset = options.get("asset", "wallet")
    source = options.get("source", "sofa")
    destination = options.get("destination", "fixed_cpu_bag")
    if not isinstance(asset, str) or not isinstance(source, str) or not isinstance(destination, str):
        raise ValueError("Asset, source and destination must be strings")
    typ, maximum, sources = VARIANTS.get(asset, ("wallet_transfer", 7, ()))
    label = {"wallet": "wallet", "glasses_case": "sunglasses case", "power_bank": "power bank", "water_bottle": "water bottle"}.get(asset, asset)
    plan = plan_base(request, "pack_object", RULE_VERSION, typ, asset, source, destination,
        maximum, "Pick up the %s, carry it to the storage tote bag, and place it inside." % label,
        ["One unique scene-owned target of the exact asset type, initially unheld.",
         "Target initially supported at the requested approved source.",
         "One fixed open bag; original triangle-mesh cavity and real support contacts.",
         "No unrelated targets; no source reset, assisted attachment or destination teleport."])
    if asset not in VARIANTS:
        return unavailable(plan, "unsupported_target_variant", "该物体没有已批准的放包规则。")
    if source not in sources:
        return unavailable(plan, "unsupported_source_variant", "此物体当前仅支持来源：" + ", ".join(sources), "load")
    if destination != "fixed_cpu_bag":
        return unavailable(plan, "pack_requires_fixed_open_bag", "收包任务目的地必须为固定开口包。", "load")
    return plan


def bind(plan, manifest):
    if not plan.get("available"):
        raise ValueError("Unavailable task cannot be bound")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), list):
        raise ValueError("Unified scene manifest required")
    rows = [r for r in manifest["objects"] if r.get("id") == plan["asset"]]
    if len(rows) != 1:
        raise ValueError("Exactly one selected scene object required")
    row = rows[0]
    if (row.get("task_id") != plan["task_name"]
            or row.get("role") in ("extra", "background", "infrastructure")
            or row.get("task_role") == "cloth_clear_target"):
        raise ValueError("Target is not explicitly owned by this Short Task")
    if row.get("source") != plan["source"]:
        raise ValueError("Scene target source differs from requested source")
    path = row.get("path")
    if not isinstance(path, str) or not path.startswith("/World/"):
        raise ValueError("Exact scene target path required")
    if sum(r.get("path") == path for r in manifest["objects"]) != 1:
        raise ValueError("Two manifest objects alias the same target path")
    if plan["source"] == "coffee_table.drawer" and not row.get("drawer_path"):
        raise ValueError("Actual drawer identity required")
    destination_path = {"fixed_cpu_bag": "/World/Assets/Task5PackStation/ToteSpawn/Tote",
        "storage_basket": "/World/Assets/TableclothBasket",
        "coat_rack_hooks": "/World/Assets/CoatRack"}[plan["destination"]]
    plan["target_bindings"] = [dict(instance_id=plan["runtime_components"][0]["instance_id"],
        asset=plan["asset"], target_path=path, source=plan["source"],
        destination_role=plan["destination"], destination_path=destination_path, task_role="target",
        drawer_path=row.get("drawer_path"))]
    return deepcopy(plan["runtime_components"])


def _number(value, name, nonnegative=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Finite numeric score channel required: " + name)
    return float(value)


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError("Measured boolean required: " + name)
    return value


def adapt_summary(summary, frame, *, require_drawer=False, drawer_opened=None):
    """Enrich only from the component's reader-owned last physical frame."""
    if frame is None:
        raise ValueError("Fresh component physical frame required")
    held = _boolean(frame.held, "held")
    robot = _boolean(frame.robot_contact, "robot_contact")
    # Preserve a component's measured release predicate if it exposes one.
    # Otherwise the document's current-final gate is no longer held. Original
    # placement-process contact gates remain entirely inside the component.
    released = _boolean(frame.released, "released") if hasattr(frame, "released") else not held
    result = deepcopy(summary)
    target = getattr(frame, "target_path", getattr(frame, "target", None))
    destination = getattr(frame, "destination_path", getattr(frame, "destination", None))
    evidence = dict(held=held, robot_contact=robot, released=released,
        target_path=target, destination_path=destination, time_s=float(frame.time_s),
        source="component_reader_owned_physical_frame",
        release_predicate="component_frame.released" if hasattr(frame, "released") else "current_not_held")
    if require_drawer:
        evidence["drawer_opened"] = _boolean(drawer_opened, "drawer_opened")
        evidence["source_clear"] = _boolean(frame.source_clear, "source_clear")
    result["short_task_evidence"] = evidence
    return result


def normalize(plan, component_summaries):
    if not plan.get("available"):
        return dict(task_name=plan["task_name"], rule_version=plan["rule_version"],
            scores=dict(process=None, current_final=None, deductions=None, total=None, maximum=None),
            success=False, stages=[], targets=[], unavailable=deepcopy(plan["unavailable"]))
    expected = {r["instance_id"] for r in plan["runtime_components"]}
    if not isinstance(component_summaries, dict) or set(component_summaries) != expected or len(expected) != 1:
        raise ValueError("Exactly the resolved component summaries required")
    instance = next(iter(expected)); s = component_summaries[instance]
    if not isinstance(s, dict) or not isinstance(s.get("pickup"), dict) or not isinstance(s.get("placement"), dict):
        raise ValueError("Actual approved pickup/placement summary required")
    p, q = s["pickup"], s["placement"]
    maximum = _number(s.get("maximum_score"), "maximum_score")
    if maximum != plan["score_policy"]["maximum"]:
        raise ValueError("Component maximum differs from frozen asset variant")
    pickup_earned = _number(p.get("earned_points"), "pickup.earned_points")
    pickup_raw = _number(p.get("raw_score"), "pickup.raw_score", False)
    if "operation_score" in q:
        placement_process = _number(q["operation_score"], "placement.operation_score")
        final = _number(q.get("final_state_points"), "placement.final_state_points")
        q_raw = _number(q.get("raw_score"), "placement.raw_score", False)
        q_earned = placement_process + final
    else:
        stages = q.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("Approved placement stages required")
        key = "hang" if plan["task_name"] == "hang_cap" else "place"
        if set(stages) - {key, "stable"}:
            raise ValueError("Unknown placement score stage")
        placement_process = _number(stages.get(key, 0), "placement." + key)
        final = _number(stages.get("stable", 0), "placement.stable")
        q_earned = _number(q.get("earned"), "placement.earned")
        q_raw = _number(q.get("raw_score"), "placement.raw_score", False)
        if abs(q_earned-placement_process-final) > 1e-8:
            raise ValueError("Placement stage arithmetic mismatch")
    if placement_process not in (0, 2) or final not in (0, 2):
        raise ValueError("Unknown destination point schedule")
    if pickup_earned not in ((0, 1, 3) if maximum == 7 else (0, 3, 5)):
        raise ValueError("Pickup exceeds approved maximum")
    deductions = pickup_earned-pickup_raw + q_earned-q_raw
    if deductions < 0 or deductions != int(deductions) or abs(_number(s.get("raw_score"), "raw_score", False)-pickup_raw-q_raw) > 1e-8:
        raise ValueError("Component score arithmetic mismatch")
    if abs(_number(s.get("score"), "score")-max(0, pickup_raw+q_raw)) > 1e-8:
        raise ValueError("Component clamped total mismatch")
    operation = _boolean(s.get("operation_complete"), "operation_complete")
    current = _boolean(q.get("current_final"), "placement.current_final")
    _boolean(s.get("success"), "success")
    e = s.get("short_task_evidence")
    if not isinstance(e, dict) or e.get("source") != "component_reader_owned_physical_frame":
        raise ValueError("Current reader-owned release evidence required")
    held = _boolean(e.get("held"), "held"); robot = _boolean(e.get("robot_contact"), "robot_contact")
    released = _boolean(e.get("released"), "released")
    if (held and released or e.get("release_predicate") not in ("current_not_held", "component_frame.released")
            or (e.get("release_predicate") == "current_not_held" and released != (not held))):
        raise ValueError("Contradictory release evidence")
    time_s = _number(e.get("time_s"), "time_s")
    for name in ("target_path", "destination_path"):
        if not isinstance(e.get(name), str) or not e[name].startswith("/"):
            raise ValueError("Exact physical identity required: " + name)
    if "target_path" in p and p["target_path"] != e["target_path"]:
        raise ValueError("Pickup/release evidence target identity mismatch")
    if "instance_id" in s and s["instance_id"] != instance:
        raise ValueError("Component instance differs from resolved single target")
    bindings = plan.get("target_bindings")
    if bindings and (len(bindings) != 1 or bindings[0]["target_path"] != e["target_path"]):
        raise ValueError("Current target differs from uniquely bound scene instance")
    if bindings and bindings[0].get("destination_path") != e["destination_path"]:
        raise ValueError("Current destination differs from explicitly bound role")
    if final != (2 if current else 0):
        raise ValueError("Current final flag and numeric channel disagree")
    if plan["task_name"] == "retrieve_from_drawer":
        opened = _boolean(e.get("drawer_opened"), "drawer_opened")
        operation = operation and opened
    final = final if operation and current and released else 0
    process = pickup_earned + placement_process
    success = bool(operation and current and released and final == 2)
    stages = [dict(name="pickup", score=pickup_earned, maximum=maximum-4),
              dict(name="destination_process", score=placement_process, maximum=2),
              dict(name="current_destination", score=final, maximum=2, current=True)]
    if plan["task_name"] == "retrieve_from_drawer":
        stages.insert(0, dict(name="open_drawer", complete=opened, score=0, maximum=0))
    return dict(task_name=plan["task_name"], rule_version=plan["rule_version"],
        scores=dict(process=process, current_final=final, deductions=deductions,
                    total=max(0, process+final-deductions), maximum=maximum),
        success=success, stages=stages,
        targets=[dict(instance_id=instance, asset=plan["asset"], source=plan["source"],
            destination=plan["destination"], target_path=e["target_path"],
            destination_path=e["destination_path"], operation_complete=operation,
            current_final=bool(final), released=released, success=success,
            evidence_time_s=time_s)],
        geometry_limitations=deepcopy(plan.get("evidence_policy", {})))


class ReleaseEvidencePort:
    """Local subclass mixin: read state, preserve inherited readers and scores."""
    def _short_summary(self):
        self.score = adapt_summary(self.composite.summary(), self.composite.last)
        self.score["evidence_status"] = "native_physx"
        return self.score

    def accept_initial(self):
        receipt = super().accept_initial()
        self._short_summary()
        return receipt

    def consume_control(self, active):
        super().consume_control(active)
        return self._short_summary()

    def finalize(self, reason):
        if reason == "success":
            current = self._short_summary()
            evidence = current["short_task_evidence"]
            if (current.get("success") is not True or current.get("operation_complete") is not True
                    or not evidence["released"]
                    or ("drawer_opened" in evidence and not evidence["drawer_opened"])):
                raise ValueError("Only the current measured Short Task success may settle")
            # Upstream reversible cores intentionally do not end on observe.
            # Their legacy finalize requires its reason to be latched first.
            # This is episode bookkeeping, never a physical state mutation.
            if self.composite.finalized and self.composite.reason != "success":
                raise ValueError("Cannot replace an existing settlement")
            self.composite.reason = "success"
        legacy_reason = {"no_new_net_progress_timeout": "no_score_increase_timeout"}.get(reason, reason)
        super().finalize(legacy_reason)
        return self._short_summary()


def factory_override(plan, declaration):
    if declaration not in plan["runtime_components"] or not plan.get("available"):
        raise ValueError("Resolved available declaration required")
    from openend.tasks.scene_requirements.runtime_ports import WalletPort, GlassesPort
    from openend.tasks.all_objects_integration.ports import PowerBankPort, WaterBottlePort
    classes = dict(wallet_transfer=WalletPort, glasses_transfer=GlassesPort,
        powerbank_transfer=PowerBankPort, water_bottle_transfer=WaterBottlePort)
    return type("Short" + classes[declaration["type"]].__name__,
        (ReleaseEvidencePort, classes[declaration["type"]]), {})

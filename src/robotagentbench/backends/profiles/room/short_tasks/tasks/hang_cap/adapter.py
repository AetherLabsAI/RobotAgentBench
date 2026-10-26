"""Approved sofa-cap pickup and rack support, retaining rocking admission."""
from short_tasks.tasks.pack_object.adapter import (
    request_parts, plan_base, unavailable, bind, normalize, ReleaseEvidencePort)

RULE_VERSION = "openend.short_task.hang_cap.v1"


def resolve(request):
    options, _, _ = request_parts(request, "hang_cap", RULE_VERSION, ("source",))
    source = options.get("source", "sofa")
    if not isinstance(source, str):
        raise ValueError("Explicit source string required")
    plan = plan_base(request, "hang_cap", RULE_VERSION, "cap_transfer", "baseball_cap",
        source, "coat_rack_hooks", 9, "Pick up the hat from the sofa, carry it to the coat rack, and hang it up.",
        ["Exactly one supported, unheld baseball cap and one existing coat rack.",
         "Source placement follows the approved sofa policy.",
         "Use cap_supported_rocking_v1 initial motion admission; do not tighten drift tolerance.",
         "A real load-bearing allowed rack hook is required; floor/proximity is insufficient."])
    plan["evidence_policy"]["initial_motion"] = "cap_supported_rocking_v1"
    plan["evidence_policy"]["source_height_reference"] = "world_aabb_center"
    if source != "sofa":
        return unavailable(plan, "unsupported_cap_source", "当前帽子生成器仅接入沙发来源。", "load")
    return plan


def factory_override(plan, declaration):
    if declaration not in plan["runtime_components"] or not plan["available"]:
        raise ValueError("Available resolved cap declaration required")
    from openend.tasks.all_objects_integration.ports import CapPort
    return type("ShortCapPort", (ReleaseEvidencePort, CapPort), {})

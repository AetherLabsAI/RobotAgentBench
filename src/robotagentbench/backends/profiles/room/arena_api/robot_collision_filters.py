"""Versioned, fail-closed G1 filter override; source USDs are never saved.

Restore six base/arm contacts. Shoulder geometry and its link1/link2 filters,
fixed assemblies, gripper linkages, and body/head filters stay.
This changes physics filtering, NOT the policy action or collision geometry.
"""
import hashlib
import json

ROBOT = "/World/Robot"
POLICY = "g1.restore_base_arm_contacts.v1"


def expected_pairs():
    arms = [f"arm_{side}_{part}" for side in ("l", "r")
            for part in ("base_link", "link1", "link2")]
    pairs = {(ROBOT + "/base_link", ROBOT + "/" + name)
             for name in ["body_link2", "head_link2", *arms]}
    pairs.update((ROBOT + "/body_link2", ROBOT + "/" + name)
                 for name in [*arms, "head_link2"])
    for side in ("l", "r"):
        for finger in ("inner", "outer"):
            stem = ROBOT + f"/gripper_{side}_{finger}_link"
            pairs.update(((stem + "1", stem + "2"), (stem + "2", stem + "3")))
    return pairs


def restored_pairs():
    pairs = {(ROBOT + "/base_link", ROBOT + f"/arm_{side}_{part}")
             for side in ("l", "r") for part in ("base_link", "link1", "link2")}
    return pairs


def welded(a, b, fixed_edges):
    reached = {a}
    while True:
        expanded = reached | {y for x, y in fixed_edges if x in reached} | {
            x for x, y in fixed_edges if y in reached}
        if expanded == reached:
            return b in reached
        reached = expanded


def plan(current_pairs, fixed_edges):
    current_pairs = set(current_pairs)
    retained = expected_pairs() - restored_pairs()
    if current_pairs not in (expected_pairs(), retained):
        raise ValueError("Robot filters differ from the audited 23/17-pair policy; review before editing")
    for side in ("l", "r"):
        if not welded(ROBOT + "/body_link2", ROBOT + f"/arm_{side}_base_link", fixed_edges):
            raise ValueError("Expected shoulder fixed assembly is not connected by fixed joints")
    rows = []
    for a, b in sorted(retained):
        if a.startswith(ROBOT + "/gripper_"):
            reason = "gripper_four_bar_assembly_preserved"
        elif b.endswith(("arm_l_base_link", "arm_r_base_link")):
            reason = "same_fixed_assembly"
        elif b.endswith(("arm_l_link1", "arm_r_link1", "arm_l_link2", "arm_r_link2")):
            reason = "shoulder_shell_and_filter_deferred_by_user"
        else:
            reason = "unrelated_body_head_filter_preserved"
        rows.append(dict(source=a, target=b, reason=reason))
    return dict(policy_id=POLICY, before_count=len(current_pairs), after_count=len(retained),
                restored_count=len(restored_pairs()), removed_now_count=len(current_pairs - retained),
                restored=[dict(source=a, target=b) for a, b in sorted(restored_pairs())],
                retained=rows, application_layer="session", action_filtering=False,
                automatic_adjacent_link_filtering="unchanged",
                shoulder_geometry_changed=False, source_assets_modified=False)


def inspect(stage):
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(ROBOT)
    if not root or root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
        raise ValueError("Robot must declare self-collisions enabled")
    pairs, fixed_edges = set(), set()
    for p in Usd.PrimRange(root):
        relation = p.GetRelationship("physics:filteredPairs")
        if relation and relation.GetTargets():
            if not p.HasAPI(UsdPhysics.FilteredPairsAPI):
                raise ValueError("Authored filter relation has no active FilteredPairsAPI")
            pairs.update((str(p.GetPath()), str(t)) for t in relation.GetTargets())
        if p.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(p)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) == len(b) == 1:
                fixed_edges.add((str(a[0]), str(b[0])))
    return pairs, fixed_edges


def value_signature(value):
    # Numeric USD arrays expose lossless buffers; avoid formatting millions of
    # mesh coordinates. Unsupported scalars/strings use their stable value repr.
    try:
        view = memoryview(value)
    except TypeError:
        return repr(value)
    return (type(value).__name__, view.format, view.shape,
            hashlib.sha256(view.tobytes()).hexdigest())


def attribute_signature(attribute):
    # Sdf.ValueTypeName.__repr__ contains a Python memory address, not its USD
    # type token. Convert it explicitly; different wrapper instances must match.
    return (str(attribute.GetTypeName()), value_signature(attribute.Get()),
            [(t, value_signature(attribute.Get(t))) for t in attribute.GetTimeSamples()])


def robot_signature(stage):
    """Protect every composed robot property except the edited relation."""
    from pxr import Usd
    excluded = {owner + ".physics:filteredPairs" for owner, _ in restored_pairs()}
    values = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT), Usd.TraverseInstanceProxies()):
        values[str(p.GetPath())] = str((p.GetTypeName(), p.IsActive(), p.GetAppliedSchemas()))
        for a in p.GetAttributes():
            values[str(a.GetPath())] = str(attribute_signature(a))
        for rel in p.GetRelationships():
            if str(rel.GetPath()) not in excluded:
                values[str(rel.GetPath())] = str(rel.GetTargets())
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def verify(stage):
    pairs, _ = inspect(stage)
    if pairs != expected_pairs() - restored_pairs():
        raise RuntimeError("Robot collision filter policy changed")


def apply(stage):
    """Apply before physics initialization; never mutate the caller's source layer."""
    from pxr import Usd
    pairs, edges = inspect(stage)
    report = plan(pairs, edges)  # Validate entire baseline before any write.
    before = robot_signature(stage)
    if report["removed_now_count"]:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for owner in sorted({a for a, _ in restored_pairs()}):
                rel = stage.GetPrimAtPath(owner).GetRelationship("physics:filteredPairs")
                targets = [r["target"] for r in report["retained"] if r["source"] == owner]
                if not rel.SetTargets(targets):
                    raise RuntimeError("Could not apply robot filter override")
    verify(stage)
    after = robot_signature(stage)
    if before != after:
        raise RuntimeError("Unexpected non-filter robot modification")
    return {**report, "non_filter_robot_properties_unchanged": True,
            "non_filter_robot_signature": before}

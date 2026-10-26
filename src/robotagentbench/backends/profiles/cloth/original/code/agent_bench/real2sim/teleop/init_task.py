# -*- coding: utf-8 -*-
"""THE single uniform init entry point for every real2sim task. One paradigm, every family:

    TASK=<canonical_id> conda run -n base python real2sim/teleop/init_task.py

Looks the task up in task_registry.REGISTRY, dispatches to that family's TUNED init
(init_pick / init_place / init_throw / init_snack / init_tier2 — unchanged internals), and
emits ONE uniform JSON contract so every caller (video generator, reproduce runners) is
family-agnostic:

    {ok, task, family, target, track_prim, distract_prims, hand, kin:[x,y,z], base:[x,y], reason?}

Family-specific placement knobs pass straight through the environment (PICK_ANCHOR, PICK_BITE,
BITE_Z, GRAB_FRAC, EDGE_OVERHANG, SNACK_BITE, T2_BITE, RGBD_REC, RGBD_SAM3, GRASP_MODE, ...) —
the dispatcher forwards the whole env, so nothing family-specific is lost.
"""
import json, math, os, re, subprocess, sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO, "real2sim", "teleop"))
from task_registry import (  # noqa: E402
    REGISTRY,
    contact_prims,
    distract_prims,
    track_prim,
)
from managed_subprocess import run_argv  # noqa: E402
from episode_scene_randomization import (  # noqa: E402
    DEFAULT_RECORD_ROOT,
    audit_episode_hand,
    context_object_descriptors,
    randomization_enabled,
    reconcile_hand,
    run_randomization_subprocess,
)
from shelf_asset_policy import (  # noqa: E402
    DEFAULT_DEMO_ROOT,
    require_ready_scene_assets,
    task_asset_decision,
)

# family -> (script, task environment variable or None, positional key?)
DISPATCH = {
    "pick": ("real2sim/teleop/init_pick.py", "PICK_TASK", False),
    "place": ("real2sim/teleop/init_place.py", "PLACE_TASK", False),
    "throw": ("real2sim/teleop/init_throw.py", "THROW_TASK", False),
    "snack": ("real2sim/teleop/init_snack.py", None, True),
    "tier2": ("real2sim/teleop/init_tier2.py", None, True),
}
SCALAR_ENV = {
    "BITE_Z",
    "EDGE_OVERHANG",
    "FIXED_YAW",
    "GRAB_FRAC",
    "PICK_BITE",
    "PLACE_BITE",
    "SAM3_THR",
    "SNACK_BITE",
    "T2_BITE",
    "THROW_BITE",
    "THROW_MAX_RISER",
    "COFFEE_TABLE_WORLD_X_M",
    "COFFEE_TABLE_WORLD_Y_M",
    "COFFEE_TABLE_WORLD_Z_M",
    "COFFEE_TABLE_BOTTOM_Z_M",
}
# Non-numeric initializer knobs a job may forward (validated below like the
# scalars; the queue's split_env unions this with SCALAR_ENV so a job-level
# assignment reaches the initializer instead of being rejected as unknown).
STRING_ENV = {
    "ALIGN_HAND",
    "ALIGN_RUN_ORDINAL",
}

# Direct supports are part of the task's physical initial condition, not
# optional clutter.  The randomizer must never deactivate or reposition them.
REQUIRED_SUPPORT_PRIMS_BY_TASK = {
    "30": ("/World/Assets/PaperPlate",),
}


def required_support_prims(task_id):
    return tuple(REQUIRED_SUPPORT_PRIMS_BY_TASK.get(str(task_id), ()))


def validate_forwarded_environment(environment):
    """Reject values that downstream legacy init scripts interpolate into shells/code."""
    instance = environment.get("INSTANCE", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]*", instance):
        return "INSTANCE contains unsafe characters"
    if environment.get("GRASP_MODE", "") not in ("", "rise"):
        return "GRASP_MODE must be empty or 'rise'"
    if "PICK_HAND" in environment and not re.fullmatch(
            r"[LlRr?]", environment["PICK_HAND"]):
        return "PICK_HAND must be L, R, or ?"
    for key in SCALAR_ENV:
        if key not in environment:
            continue
        try:
            value = float(environment[key])
        except ValueError:
            return f"{key} must be a finite number"
        if not math.isfinite(value):
            return f"{key} must be a finite number"
    for key, count in (
        ("FIXED_BASE", 2),
        ("PICK_ANCHOR", 3),
        ("SNACK_KIN_OVERRIDE", 3),
        ("T2_TARGET_XY_OVERRIDE", 2),
        ("T2_KIN_OVERRIDE", 3),
        ("THROW_KIN_OVERRIDE", 3),
        ("THROW_PLATE_XY", 2),
    ):
        if key not in environment:
            continue
        try:
            values = [float(value) for value in environment[key].split(",")]
        except ValueError:
            return f"{key} must contain {count} finite comma-separated numbers"
        if len(values) != count or not all(math.isfinite(value) for value in values):
            return f"{key} must contain {count} finite comma-separated numbers"
    if "RGBD_REC" in environment:
        value = environment["RGBD_REC"]
        if (not value or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
                or ".." in value.split("/")):
            return "RGBD_REC contains unsafe path characters"
    if "RGBD_SAM3" in environment and environment["RGBD_SAM3"].lower() not in (
            "", "0", "1", "false", "true", "no", "yes"):
        return "RGBD_SAM3 must be boolean"
    if "ALIGN_HAND" in environment and environment[
            "ALIGN_HAND"].strip().lower() not in ("", "l", "r"):
        return "ALIGN_HAND must be exactly l or r when provided"
    if "ALIGN_RUN_ORDINAL" in environment and not re.fullmatch(
            r"[0-9]*", environment["ALIGN_RUN_ORDINAL"].strip()):
        return "ALIGN_RUN_ORDINAL must be a non-negative integer"
    return None


def dispatch_command(family, key, task_id=None):
    """Build an argv and child environment without shell interpolation."""
    script, task_environment, positional_key = DISPATCH[family]
    # init_task already runs in the selected base interpreter. ``conda run``
    # closes the inherited managed-place lock fd before lock-aware init_place
    # starts, leaving only a stale fd number in the environment.
    command = [sys.executable, script]
    if positional_key:
        command.append(str(key))
    child_env = os.environ.copy()
    if task_environment:
        child_env[task_environment] = str(key)
    if task_id is not None:
        child_env["REAL2SIM_TASK_ID"] = str(task_id)
    return command, child_env


def main():
    cid = os.environ.get("TASK") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if cid not in REGISTRY:
        print(json.dumps({"ok": False, "task": cid, "reason": f"unknown task id {cid!r} (not in registry)"}))
        return 2
    e = REGISTRY[cid]
    fam = e["family"]
    if fam == "cloth":
        print(json.dumps({"ok": False, "task": cid, "family": "cloth",
                          "reason": "cloth uses the Newton pipeline, not init dispatch"}))
        return 3
    if fam not in DISPATCH:
        print(json.dumps({"ok": False, "task": cid, "reason": f"no dispatch for family {fam}"}))
        return 3
    environment_error = validate_forwarded_environment(os.environ)
    if environment_error:
        print(json.dumps({
            "ok": False,
            "task": cid,
            "family": fam,
            "reason": environment_error,
        }))
        return 2
    init_randomization = os.environ.get(
        "REAL2SIM_INIT_SCENE_RANDOMIZATION", "1")
    if init_randomization not in ("0", "1"):
        print(json.dumps({
            "ok": False,
            "task": cid,
            "family": fam,
            "reason": (
                "REAL2SIM_INIT_SCENE_RANDOMIZATION must be exactly 0 or 1"),
        }))
        return 2
    scene_path = (
        DEFAULT_DEMO_ROOT
        / f"scene_room{os.environ.get('INSTANCE', '')}.usda"
    )
    asset = task_asset_decision(cid)
    if asset.applicable and not asset.usable:
        print(json.dumps({
            "ok": False,
            "task": cid,
            "family": fam,
            "asset_id": asset.asset_id,
            "asset_status": asset.status,
            "artifact_url": asset.artifact_url,
            "reason": f"shelf asset blocked: {asset.reason}",
        }, ensure_ascii=False))
        return 10
    try:
        if asset.applicable:
            asset = task_asset_decision(
                cid,
                verify_runtime=True,
                scene_path=scene_path,
                demo_root=DEFAULT_DEMO_ROOT,
            )
            if not asset.usable:
                raise RuntimeError(asset.reason)
        require_ready_scene_assets(
            scene_path, demo_root=DEFAULT_DEMO_ROOT
        )
    except RuntimeError as exc:
        print(json.dumps({
            "ok": False,
            "task": cid,
            "family": fam,
            "asset_id": asset.asset_id,
            "asset_status": (
                asset.status
                if asset.applicable
                else "scene_whitelist_failed"
            ),
            "artifact_url": asset.artifact_url,
            "reason": f"shelf asset verification failed: {exc}",
        }, ensure_ascii=False))
        return 10
    command, child_env = dispatch_command(fam, e["key"], cid)
    try:
        r = run_argv(
            command,
            cwd=REPO,
            env=child_env,
            timeout=1300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({
            "ok": False,
            "task": cid,
            "family": fam,
            "reason": f"sub-init execution failed: {type(exc).__name__}: {exc}",
        }))
        return 6
    try:
        sub = json.loads(r.stdout.strip().splitlines()[-1])
        if not isinstance(sub, dict):
            raise ValueError("sub-init result must be a JSON object")
    except Exception:
        sub = {"ok": False, "reason": "sub-init produced no JSON: " + (r.stdout + r.stderr)[-160:]}
    # uniform normalization — every family now yields the same shape
    resolved = sub.get("target") or e.get("prim")
    _zone = sub.get("zone")
    _destination = sub.get("destination")
    _destination_kind = (
        _destination.get("kind")
        if isinstance(_destination, dict)
        else None
    )
    _placement = sub.get("placement")
    _table_plate_context = sub.get("table_plate_context")
    if (
        _table_plate_context is None
        and isinstance(_placement, dict)
    ):
        _table_plate_context = _placement.get("table_plate_context")
    if _table_plate_context is None and e.get("table_plate_context"):
        # Older throw initializers did not echo this already-validated scene
        # state.  Preserve it in the normalized audit instead of reporting a
        # misleading null, including this episode's supported plate override.
        _table_plate_context = dict(e["table_plate_context"])
        _plate_xy = os.environ.get("THROW_PLATE_XY", "").strip()
        if _plate_xy:
            _px, _py = (float(value) for value in _plate_xy.split(","))
            _old = _table_plate_context["paper_plate_translate_xyz"]
            _table_plate_context["paper_plate_translate_xyz"] = (
                _px, _py, _old[2])
            _half = [
                float(value) / 2.0
                for value in _table_plate_context[
                    "paper_plate_expected_size_xyz_m"]
            ]
            _table_plate_context["paper_plate_aabb_min_xyz"] = [
                _px - _half[0], _py - _half[1],
                _table_plate_context["paper_plate_bottom_m"],
            ]
            _table_plate_context["paper_plate_aabb_max_xyz"] = [
                _px + _half[0], _py + _half[1],
                _table_plate_context["paper_plate_top_m"],
            ]
    out = {
        "ok": r.returncode == 0 and sub.get("ok") is True,
        "task": cid,
        "family": fam,
        "target": resolved,
        "track_prim": track_prim(cid, resolved),
        "distract_prims": distract_prims(cid, resolved),
        "hand": sub.get("hand", "?"),
        "kin": sub.get("kin"),
        "base": sub.get("base"),
        "yaw": sub.get("yaw"),
        "up_axis": sub.get("up_axis"),
        "seat_mid": sub.get("seat_mid"),
        "seat_long": sub.get("seat_long"),
        "grasp_long_plane_rotation_deg": sub.get(
            "grasp_long_plane_rotation_deg"
        ),
        "approach_offset_xyz": sub.get("approach_offset_xyz"),
        "grip_width": sub.get("grip_width"),
        "release_retreat": sub.get("release_retreat"),
        "zone": _zone,
        "destination": _destination,
        # The fifth zone component is a support surface for ordinary place
        # tasks but a rim height for the task-28 bin.  Never relabel the latter
        # as a surface merely to satisfy the old playback diagnostic.
        "surface_z": (
            _zone[4]
            if (_destination_kind in (None, "surface")
                and isinstance(_zone, (list, tuple))
                and len(_zone) >= 5)
            else None
        ),
        "new": e.get("new"),
        "required_count": e.get("required_count", 1),
        "probe_scope": e.get("probe_scope", "full_task"),
        "preferred_probes": list(e.get("preferred_probes", ())),
        "required_evidence_contract": e.get("required_evidence_contract"),
        "asset_id": asset.asset_id,
        "asset_status": asset.status,
        "asset_revision": asset.asset_revision,
        "task53_support_mode": sub.get("task53_support_mode"),
        "current_secondary_context": sub.get(
            "current_secondary_context"
        ),
        "table_plate_context": _table_plate_context,
        "required_support_prim_paths": list(required_support_prims(cid)),
    }
    out["init_scene_randomization_enabled"] = (
        init_randomization == "1")
    if (
        out["ok"]
        and init_randomization == "1"
        and randomization_enabled(cid)
    ):
        episode_uid = os.environ.get("EPISODE_UID", "").strip()
        try:
            hand_audit = audit_episode_hand(
                episode_uid,
                record_root=os.environ.get(
                    "REAL2SIM_RECORD_ROOT",
                    str(DEFAULT_RECORD_ROOT),
                ),
            )
            resolved_hand, hand_resolution = reconcile_hand(
                out["hand"], hand_audit
            )
            out["hand"] = resolved_hand
            protected = set(
                contact_prims(
                    cid, episode_uid if episode_uid else None
                )
            )
            protected.update(required_support_prims(cid))
            if sub.get("task53_support_mode"):
                protected.add("/World/Assets/CushionOnBook")
            randomized = run_randomization_subprocess(
                scene_path=scene_path,
                task_id=cid,
                episode_uid=episode_uid,
                seed_override=os.environ.get(
                    "SCENE_RANDOMIZATION_SEED"
                ),
                target_prim_path=out["track_prim"],
                hand=resolved_hand,
                place_zone=_zone,
                protected_prim_paths=protected,
                protected_objects=context_object_descriptors(
                    sub.get("current_secondary_context")
                ),
            )
            out["scene_randomization"] = randomized
            out["distract_prims"] = randomized[
                "interactive_prim_paths"
            ]
            out["hand_audit"] = hand_audit
            out["hand_resolution"] = hand_resolution
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as exc:
            out["ok"] = False
            out["reason"] = (
                "episode scene randomization failed: "
                f"{type(exc).__name__}: {exc}"
            )
    # A randomization failure above already recorded its specific reason;
    # never clobber it with the sub-init fallback text.
    if not out["ok"] and "reason" not in out:
        if r.returncode != 0 and sub.get("ok") is True:
            out["reason"] = (
                f"sub-init exited {r.returncode} despite an ok JSON result")
        else:
            out["reason"] = sub.get("reason", "init failed")
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())

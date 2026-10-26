#!/usr/bin/env python3
"""Open and close an isolated scene session.

A session is one run whose state is *meant* to carry: a single calibration
episode, or a whole multi-task end-to-end sequence. Either way it starts from a
restored copy of the read-only baseline, so no session inherits the previous
one's leftovers.

  session start -> restore scene_room.usda from the baseline and activate exactly
                   the objects this session needs; place nothing
  during        -> tasks run in order and mutate that scene on purpose
  session end   -> optionally snapshot the scene as it ended, for forensics

Isolation is by *restore*, not by a separate file: make_compare3 deliberately
refuses INSTANCE because its own intermediates (playback_traj, the compare mp4)
are not suffixed, so a suffixed scene would desynchronise init from playback.
Restoring the canonical path before each session gives the same guarantee for
serial runs — which is what we have, one GPU and one shared container.

The distinction that matters: isolation is *between* sessions, accumulation is
*within* one. A calibration job is a one-task session so its results are
comparable; an end-to-end run is a many-task session so the second task sees
what the first one left, which is the whole point of running it.

    conda run -n mesh311 python real2sim/teleop/session_scene.py open  NAME --activate PRIM [PRIM ...]
    conda run -n mesh311 python real2sim/teleop/session_scene.py close NAME [--keep]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pxr import Usd

from machine_paths import DEMO_ROOT as DEMO  # noqa: E402

# Sessions restore from the deployed runtime mirror.  The deployer is the only
# bridge from the Git-owned source baseline into this writable runtime; keeping
# that boundary explicit prevents a server job from accidentally reaching back
# into (or requiring) a developer checkout.
BASELINE = DEMO / "scene_room_baseline.usda"
MANIFEST = Path(__file__).resolve().parent / "scene_baseline_manifest_v1.json"

SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


def instance_suffix(name: str) -> str:
    if not name or any(c not in SAFE for c in name):
        raise SystemExit(f"session name must be [A-Za-z0-9_-]+, got {name!r}")
    return f"_{name}"


CANONICAL = DEMO / "scene_room.usda"
SNAPSHOTS = DEMO / "_session_snapshots"
SIDECAR = DEMO / "scene_room.usda.session.json"


def scene_path(name: str) -> Path:
    """Sessions run on the canonical scene; the name only labels snapshots."""
    instance_suffix(name)          # still validate the name
    return CANONICAL


def conflicts_with(manifest: dict, target_prim: str, staged: list[str]) -> list[str]:
    """Staged stock that stands in the episode target's authored volume.

    The real shelf is packed and its items lean on each other, so an episode
    should stage the whole surface rather than the target alone — t37's box lost
    its neighbours and its grasp went from 6.7 mm to 0.04 mm. But two objects
    cannot share a volume: oreo_gf and pring_g overlap by 75 x 34 x 132 mm
    because each stands where its own recording put it. The target owns the
    spot, so the other item is what gives way.
    """
    if not target_prim:
        return []
    staged_set = set(staged)
    dropped = []
    for pair in manifest.get("interpenetrating_pairs", ()):
        other = None
        if pair["a"] == target_prim:
            other = pair["b"]
        elif pair["b"] == target_prim:
            other = pair["a"]
        if other and other in staged_set:
            dropped.append(other)
    return sorted(set(dropped))


def active_pair_conflicts(manifest: dict, active: list[str]) -> list[dict]:
    """Return every authored overlap whose two bodies would both be active."""
    active_set = set(map(str, active))
    return [
        dict(pair)
        for pair in manifest.get("interpenetrating_pairs", ())
        if pair.get("a") in active_set and pair.get("b") in active_set
    ]


def _atomic_sidecar(payload: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=SIDECAR.parent,
            prefix=SIDECAR.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o666)
        os.replace(temporary, SIDECAR)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _claim_session(name: str, *, recover_stale: bool) -> None:
    instance_suffix(name)
    if recover_stale and SIDECAR.exists():
        SIDECAR.unlink()
    payload = json.dumps({
        "schema": "real2sim.scene_session.v1",
        "status": "opening",
        "session": name,
        "scene": str(CANONICAL),
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=1) + "\n"
    try:
        descriptor = os.open(
            # The claim is created by the host user, but Isaac updates it from
            # a container running as a different uid. The demo directory is
            # already shared for that host/container boundary.
            SIDECAR, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as exc:
        try:
            owner = json.loads(SIDECAR.read_text(encoding="utf-8"))
            detail = f"session={owner.get('session')!r}, status={owner.get('status', 'open')!r}"
        except (OSError, json.JSONDecodeError):
            detail = "unreadable owner sidecar"
        raise RuntimeError(
            "canonical scene is already claimed (" + detail + "); close it "
            "or explicitly use --recover-stale after verifying no producer is running"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # A normal host umask masks O_CREAT down to 0644, so restore the intended
    # cross-uid write permission explicitly.
    os.chmod(SIDECAR, 0o666)


def open_session(
    name: str, activate: list[str], target_prim: str = "", *,
    recover_stale: bool = False,
    task_ids: list[str] | None = None,
    layout: str = "canonical",
    choice_groups: dict[str, str] | None = None,
) -> dict:
    if not BASELINE.is_file():
        raise SystemExit("baseline missing; run build_scene_baseline.py first")
    target = scene_path(name)
    _claim_session(name, recover_stale=recover_stale)
    try:
        shutil.copy2(BASELINE, target)  # restore only after exclusive claim

        manifest = json.loads(MANIFEST.read_text())
        known = set(manifest["session_objects_inactive"])
        dropped = conflicts_with(manifest, target_prim, activate)
        activate = [p for p in activate if p not in dropped]
        pair_conflicts = active_pair_conflicts(manifest, activate)
        if pair_conflicts:
            pairs = [f"{row['a']} <> {row['b']}" for row in pair_conflicts]
            raise RuntimeError(
                "session active set contains mutually exclusive authored "
                "poses: " + "; ".join(pairs)
            )
        stage = Usd.Stage.Open(str(target))
        if stage is None:
            raise RuntimeError("could not open restored canonical scene")
        if stage.GetPrimAtPath("/World/Assets/ShelvingUnit"):
            from shelf_support_geometry import validate_shelf_supports
            validate_shelf_supports(stage)
        activated, unknown = [], []
        for path in activate:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                unknown.append(path)
                continue
            prim.SetActive(True)
            activated.append(path)
        if unknown:
            raise RuntimeError(f"session {name}: no such prim: {unknown}")
        # The same environment repair applies to every single- and multi-task
        # session. Subsequent tasks inherit it; extend never reapplies it.
        production = Path(__file__).resolve().parents[1] / "production"
        if str(production) not in sys.path:
            sys.path.insert(0, str(production))
        import house_environment_appearance
        shared_appearance = house_environment_appearance.apply(
            stage, house_environment_appearance.default_plan())
        stage.GetRootLayer().Save()

    # Declare what this session deliberately left out. Playback's asset gate
    # force-activates every approved retail prim, which silently re-stocked the
    # whole shelf and made this activation list cosmetic — and put two approved
    # items at authored positions that overlap by 75 mm. The gate keeps its real
    # guarantee (an unapproved prim is never active) and now also honours a
    # session's declared exclusions, so an episode can be composed of exactly
    # the objects it recorded.
        held = sorted(set(known) - set(activated))
        _atomic_sidecar({
            "schema": "real2sim.scene_session.v1",
            "status": "open",
            "session": name,
            "scene": str(target),
            "baseline": str(BASELINE),
            "restored_from_baseline": True,
            "shared_default_appearance": shared_appearance,
            "task_ids": list(task_ids or ()),
            "layout": str(layout),
            "choice_groups": dict(choice_groups or {}),
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "activated": activated,
            "held_inactive": held,
        })

    # Anything a session activates that the baseline does not list is a prim the
    # baseline considers structure — activating it is a scene edit in disguise.
        stray = [p for p in activated if p not in known]
        return {
            "session": name,
            "instance": "",          # canonical paths; see module docstring
            "scene": str(target),
            "cam_preview": str(DEMO / "cam_preview"),
            "activated": activated,
            "dropped_conflicts": dropped,
            "held_inactive": len(held),
            "sidecar": str(SIDECAR),
            "task_ids": list(task_ids or ()),
            "layout": str(layout),
            "activated_outside_baseline_object_set": stray,
        }
    except BaseException:
        # A failed open must not leave a false owner claim.  The restored scene
        # is still treated as dirty because it has no sidecar afterwards.
        try:
            owner = json.loads(SIDECAR.read_text(encoding="utf-8"))
            if owner.get("session") == name:
                SIDECAR.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        raise


def extend_session_activation(
    activate: list[str], *, authored_poses_replaced: bool = False,
) -> dict:
    """Authorize episode props added after the clean session was opened.

    Scene randomization happens after target initialization, so its selected
    stock cannot be known by ``open_session``.  Updating only USD ``active``
    is insufficient: playback intentionally reapplies ``held_inactive`` and
    would otherwise remove those physical distractors before frame zero.
    """
    if not SIDECAR.is_file():
        raise RuntimeError("cannot extend activation without an open session")
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    if (
        sidecar.get("schema") != "real2sim.scene_session.v1"
        or Path(str(sidecar.get("scene", ""))).resolve()
        != CANONICAL.resolve()
    ):
        raise RuntimeError("open-session sidecar is invalid")
    stage = Usd.Stage.Open(str(CANONICAL))
    if stage is None:
        raise RuntimeError("could not open canonical working scene")
    requested = sorted(set(map(str, activate)))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    combined = sorted(set(map(str, sidecar.get("activated", ()))) | set(requested))
    pair_conflicts = (
        [] if authored_poses_replaced
        else active_pair_conflicts(manifest, combined)
    )
    if pair_conflicts:
        pairs = [f"{row['a']} <> {row['b']}" for row in pair_conflicts]
        raise RuntimeError(
            "session extension would activate mutually exclusive authored "
            "poses: " + "; ".join(pairs)
        )
    missing = []
    inactive = []
    for path in requested:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            missing.append(path)
        elif not prim.IsActive():
            inactive.append(path)
    if missing or inactive:
        raise RuntimeError(
            f"session extension requires active scene prims; "
            f"missing={missing}, inactive={inactive}")
    prior_activated = set(map(str, sidecar.get("activated", ())))
    prior_held = set(map(str, sidecar.get("held_inactive", ())))
    sidecar["activated"] = sorted(prior_activated | set(requested))
    sidecar["held_inactive"] = sorted(prior_held - set(requested))
    sidecar["activation_extended"] = True
    temporary = SIDECAR.with_name(SIDECAR.name + ".tmp")
    temporary.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, SIDECAR)
    return {
        "session": sidecar["session"],
        "extended": requested,
        "extended_count": len(requested),
        "held_inactive": len(sidecar["held_inactive"]),
        "authored_poses_replaced": bool(authored_poses_replaced),
        "sidecar": str(SIDECAR),
    }


def record_episode_materialization(name: str, receipt: dict) -> dict:
    """Attach an actual frame-zero write receipt to the open session."""
    if not SIDECAR.is_file():
        raise RuntimeError("cannot record materialization without an open session")
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    if sidecar.get("session") != name or sidecar.get("status") != "open":
        raise RuntimeError(f"session {name!r} does not own the working scene")
    sidecar["episode_materialization"] = dict(receipt)
    _atomic_sidecar(sidecar)
    return dict(sidecar["episode_materialization"])


def close_session(name: str, keep: bool = False) -> dict:
    """Leave the canonical scene alone; snapshot it when the run failed.

    Deleting the canonical scene would break every other tool, so closing a
    session is a no-op unless we want the wreck: then it is copied aside so the
    room can be opened exactly as it died.
    """
    try:
        sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot close a scene without a readable owner sidecar") from exc
    owner = str(sidecar.get("session", ""))
    if owner != name:
        raise RuntimeError(
            f"refusing to close session {name!r}; canonical scene belongs to {owner!r}"
        )
    snapshot = None
    if keep and CANONICAL.is_file():
        SNAPSHOTS.mkdir(exist_ok=True)
        snapshot = SNAPSHOTS / f"scene_room_{name}.usda"
        shutil.copy2(CANONICAL, snapshot)
    SIDECAR.unlink()
    return {"session": name, "kept": keep, "released": True,
            "snapshot": str(snapshot) if snapshot else None}


def session_status() -> dict:
    """Who the canonical scene currently belongs to.

    CLAUDE.md tells every agent to run this before measuring anything, because
    reading someone else's session is how small Task 37 reported 51.7 mm on a
    dirty scene and 6.7 mm from the baseline. The command it names —
    `make session-status` — did not exist: session_scene had only open/close,
    so the documented pre-flight check exited 2 on an argparse error.

    The answer was already on disk in the session sidecar; nothing read it.
    """
    if not CANONICAL.is_file():
        return {"session": None, "reason": "canonical scene is missing"}
    try:
        sidecar = json.loads(SIDECAR.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        clean = bool(
            BASELINE.is_file()
            and filecmp.cmp(BASELINE, CANONICAL, shallow=False)
        )
        return {
            "session": None,
            "scene": str(CANONICAL),
            "scene_mtime": _mtime(CANONICAL),
            "state": "clean_baseline" if clean else "unclaimed_modified",
            "reason": (
                "unclaimed working scene directly matches the runtime baseline"
                if clean else
                "unclaimed working scene differs from the runtime baseline"
            ),
        }
    return {
        "session": sidecar.get("session"),
        "status": sidecar.get("status", "open"),
        "scene": str(CANONICAL),
        "scene_mtime": _mtime(CANONICAL),
        "activated": len(sidecar.get("activated", ())),
        "held_inactive": len(sidecar.get("held_inactive", ())),
        # The sidecar is written at open. A scene newer than its sidecar has
        # been mutated since — by that session's own render, or by anything
        # that skipped the session protocol.
        "mutated_since_open": _mtime(CANONICAL) > _mtime(SIDECAR),
    }


def _mtime(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    import datetime
    return datetime.datetime.fromtimestamp(stamp).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    opener = sub.add_parser("open")
    opener.add_argument("name")
    opener.add_argument("--activate", nargs="*", default=[])
    opener.add_argument("--target", default="",
                        help="the episode's manipulation target; staged stock "
                             "that overlaps its authored volume is dropped")
    opener.add_argument(
        "--recover-stale", action="store_true",
        help="replace an existing claim only after externally verifying it is stale",
    )
    opener.add_argument("--task", action="append", default=[])
    opener.add_argument("--layout", default="canonical")
    closer = sub.add_parser("close")
    # `make session-close` passes no name, so this defaulted to an argparse
    # error too. Closing the session that is actually open is the only sensible
    # reading of a bare close.
    closer.add_argument("name", nargs="?", default="")
    closer.add_argument("--keep", action="store_true")
    extender = sub.add_parser("extend")
    extender.add_argument("--activate", nargs="+", required=True)
    sub.add_parser("status")
    args = parser.parse_args()

    if args.action == "open":
        print(json.dumps(open_session(
            args.name, args.activate, args.target,
            recover_stale=args.recover_stale,
            task_ids=args.task, layout=args.layout),
                         ensure_ascii=False))
    elif args.action == "status":
        print(json.dumps(session_status(), ensure_ascii=False))
    elif args.action == "extend":
        print(json.dumps(
            extend_session_activation(args.activate), ensure_ascii=False))
    else:
        name = args.name or (session_status().get("session") or "")
        if not name:
            raise SystemExit("no open session to close; pass a name explicitly")
        print(json.dumps(close_session(name, args.keep), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

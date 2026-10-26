"""Loading and structural validation for the living-room production contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
TELEOP = REPO / "real2sim" / "teleop"
CONTRACT_PATH = Path(__file__).with_name(
    "livingroom_data_production_v1.json")
SCHEMA = "real2sim.livingroom_data_production.v1"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
GEN_SEED_MIN = 1_000_000


def _repo_path(value: str) -> Path:
    path = REPO / value
    if not path.is_file():
        raise ValueError(f"contract source is missing: {value}")
    return path


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(contract)
    if value.get("schema") != SCHEMA or value.get("status") != "current":
        raise ValueError("living-room production contract is not current v1")

    scene = value.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("contract.scene must be an object")
    for key in (
        "baseline_manifest",
        "room_layout_contract",
        "reproduce_contract",
        "session_contract",
    ):
        _repo_path(str(scene.get(key, "")))
    camera = scene.get("camera_profile", {})
    if camera != {
        "head_adjustment_deg": 0.0,
        "mount_adjustment_deg": 0.0,
        "head_focal_scale": 1.0,
        "wrist_focal_scale": 1.0,
        "randomization": "forbidden",
    }:
        raise ValueError("camera profile drifted from the current contract")

    regions = value.get("region_profiles")
    groups = value.get("task_groups")
    if not isinstance(regions, dict) or not isinstance(groups, dict):
        raise ValueError("region_profiles and task_groups must be objects")
    task_owner: dict[str, str] = {}
    for group_name, group in groups.items():
        region = group.get("region")
        if region not in regions:
            raise ValueError(
                f"task group {group_name} references unknown region {region!r}"
            )
        ids = list(group.get("small_task_ids", ())) + list(
            group.get("synthetic_task_ids", ())
        )
        if not ids:
            raise ValueError(f"task group {group_name} has no task identities")
        for task_id in ids:
            task_id = str(task_id)
            if task_id in task_owner:
                raise ValueError(
                    f"task {task_id} belongs to both {task_owner[task_id]} "
                    f"and {group_name}"
                )
            task_owner[task_id] = group_name

    bindings = json.loads(
        (TELEOP / "fixed_task_bindings_v1.json").read_text(encoding="utf-8")
    )
    fixed = {
        str(task_id)
        for task_ids in bindings["bindings"].values()
        for task_id in task_ids
    }
    missing = sorted(fixed - set(task_owner), key=int)
    if missing:
        raise ValueError(
            "fixed current small tasks are absent from production groups: "
            + ", ".join(missing)
        )

    appearance = value.get("appearance", {})
    looks = appearance.get("material_looks", {})
    styles = appearance.get("coordinated_styles", {})
    if "canonical" not in styles or "canonical" not in looks:
        raise ValueError("appearance must define canonical style and look")
    for style_name, assignments in styles.items():
        for furniture_id, look in assignments.items():
            if look not in looks:
                raise ValueError(
                    f"style {style_name} assigns unknown look {look!r} "
                    f"to {furniture_id}"
                )
    if not appearance.get("lighting_schemes"):
        raise ValueError("appearance must define lighting schemes")

    statuses = value.get("acceptance", {}).get("statuses", [])
    if statuses != [
        "diagnostic",
        "mother_candidate",
        "accepted_mother",
        "accepted_episode",
        "review_sample",
        "publishable",
    ]:
        raise ValueError("acceptance status vocabulary drifted")
    if value["acceptance"].get("training_statuses") != ["publishable"]:
        raise ValueError("only human-reviewed publishable episodes may train")
    artifacts = value.get("episode_artifacts", {}).get(
        "required_for_training", [])
    expected_artifacts = {
        "episode.json", "action.npz", "full_state.npz",
        "pick_contact_trace.npz", "videos/head.mp4",
        "videos/hand_left.mp4", "videos/hand_right.mp4", "score.json",
        "review.json", "validation.json", "simulator.log",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("required training artifact set drifted")
    return value


def load_contract(path: Path | str = CONTRACT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_contract(payload)


def task_group(task_id: str, contract: Mapping[str, Any]) -> tuple[str, dict]:
    task_id = str(task_id)
    for name, group in contract["task_groups"].items():
        identities = {
            str(value)
            for key in ("small_task_ids", "synthetic_task_ids")
            for value in group.get(key, ())
        }
        if task_id in identities:
            return name, dict(group)
    raise ValueError(f"task {task_id!r} is not in the production contract")


def validate_identity(task_id: str, source_episode_uid: str, gen_seed: int) -> None:
    if not isinstance(gen_seed, int) or isinstance(gen_seed, bool):
        raise ValueError("gen_seed must be an integer")
    if gen_seed < GEN_SEED_MIN:
        raise ValueError(
            f"gen_seed must be >= {GEN_SEED_MIN}; lower seeds are eval-reserved"
        )
    if task_id.startswith("family7:"):
        if source_episode_uid:
            raise ValueError(
                "family7 synthetic identities take their source binding from "
                "family7_code_control_contract.py, not a caller UUID"
            )
        return
    if not UUID_RE.fullmatch(source_episode_uid.lower()):
        raise ValueError(
            "real-data-backed small tasks require one exact source episode UUID"
        )


def _teleop_import(name: str):
    entry = str(TELEOP)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return __import__(name)


def target_candidates(task_id: str) -> list[str]:
    """Resolve stable target prim candidates without opening a USD stage."""
    if task_id.startswith("family7:"):
        family = _teleop_import("family7_code_control_contract")
        return [family.mode(task_id.split(":", 1)[1]).target_path]
    registry = _teleop_import("task_registry")
    entry = registry.REGISTRY.get(str(task_id))
    if entry is None:
        if task_id in {"68", "69"}:
            return ["/World/Assets/Tablecloth"]
        raise ValueError(f"small Task {task_id} is absent from task_registry.py")
    prim = entry.get("prim")
    if prim:
        return [f"{entry['root']}/{prim}"]
    if entry.get("family") == "snack":
        snack = _teleop_import("init_snack")
        resolved = snack.TARGETS[entry["key"]]
        names = tuple(resolved) if isinstance(resolved, tuple) else (resolved,)
        return [f"{entry['root']}/{name}" for name in names]
    raise ValueError(f"small Task {task_id} target cannot be resolved")


def station(task_id: str, group: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve canonical station when it has a shared robot_positions key."""
    source = str(group["station_source"])
    prefix = "robot_positions.py:"
    if not source.startswith(prefix):
        return {"source": source, "resolved": False}
    key = source[len(prefix) :]
    positions = _teleop_import("robot_positions")
    pose = positions.robot_position(key)
    return {
        "source": source,
        "resolved": True,
        "key": key,
        "xy_m": [float(value) for value in pose.xy_m],
        "yaw_deg": float(pose.yaw_deg),
        "quat_wxyz": [float(value) for value in pose.quat_wxyz],
    }

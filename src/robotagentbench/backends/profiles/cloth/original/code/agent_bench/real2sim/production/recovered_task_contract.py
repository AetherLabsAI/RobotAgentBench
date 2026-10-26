#!/usr/bin/env python3
"""Assets, identities, and initializer policy for recovered personal items."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

try:
    from .git_provenance import source_revision
except ImportError:
    from git_provenance import source_revision
try:
    from .house_paths import SOURCE_ROOT, WORKING_SCENE
except ImportError:
    from house_paths import SOURCE_ROOT, WORKING_SCENE

try:
    from .dining_task_contract import ASSETS as DINING_ASSETS
except ImportError:
    from dining_task_contract import ASSETS as DINING_ASSETS

try:
    from .recovered_scene_materialization import (
        AUDIT_SCHEMA as RANDOMIZATION_SCHEMA,
        CANONICAL_BASELINE,
        CANONICAL_SCENE,
        PLAN_SCHEMA,
        SOURCE_DEMO,
        binding as runtime_binding,
    )
except ImportError:
    from recovered_scene_materialization import (
        AUDIT_SCHEMA as RANDOMIZATION_SCHEMA,
        CANONICAL_BASELINE,
        CANONICAL_SCENE,
        PLAN_SCHEMA,
        SOURCE_DEMO,
        binding as runtime_binding,
    )


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "real2sim" / "output"
HOUSE_DEMO = SOURCE_ROOT
BOTTLE_ASSET = (
    HOUSE_DEMO / "house_task_assets"
    / "task73" / "bottle_02" / "bottle_02.usdc"
)
ASSETS = {
    "alcoholic_drink": HOUSE_DEMO / "newcans" / "cans"
    / "cayman_jacked_margarita" / "asset.usda",
    "caffeinated_drink": HOUSE_DEMO / "newcans" / "cans"
    / "starbucks_espresso_light" / "asset.usda",
    "sparkling_water": HOUSE_DEMO / "newcans" / "cans"
    / "grapefruit_sparkling" / "asset.usda",
    "oreo_pack": HOUSE_DEMO / "newcans" / "bags"
    / "oreo_gf_pack" / "asset.usda",
    "glasses_case": HOUSE_DEMO / "usdz" / "glasses_case.usdz",
    "wallet": HOUSE_DEMO / "usdz" / "wallet.usda",
    "bottle": BOTTLE_ASSET,
    # Task5 uses the user-selected white foodbox. The historical salad_box /
    # lunchbox keys are compatibility names, not a second asset choice.
    "salad_box": DINING_ASSETS["meal_box"],
    "power_bank": HOUSE_DEMO / "house_task_assets" / "recovered"
    / "power_bank_v1.usda",
    "remote": HOUSE_DEMO / "usdz" / "remote_tv.usdz",
    "old_tablecloth": HOUSE_DEMO / "house_task_assets" / "recovered"
    / "old_tablecloth.usda",
    "clean_tablecloth": HOUSE_DEMO / "house_task_assets" / "recovered"
    / "clean_tablecloth.usda",
}
# Compatibility for owner scripts that still call the shared salad container
# ``lunchbox``.  Both names point to the same file; new task metadata uses
# ``salad_box``.
ASSETS["lunchbox"] = ASSETS["salad_box"]
TOTE_SCENE = WORKING_SCENE
SOURCE_AUDIT_PATH = HERE / "recovered_task_source_audit_20260808.json"

TASKS = {
    "3.1": {
        "real_data_reference_ids": ("85",), "target": "alcoholic_drink",
        "action": "pick", "support": "snack_shelf",
        "source_initializer": "real2sim/teleop/init_task.py",
        "robot_station": "shelf_pick",
        "runtime_ready": True,
        "owner_decision": "small Task 85 / Cayman margarita confirmed",
    },
    "3.7": {
        "real_data_reference_ids": ("48",), "target": "caffeinated_drink",
        "action": "pick", "support": "snack_shelf",
        "source_initializer": "real2sim/teleop/init_task.py",
        "robot_station": "shelf_pick",
        "runtime_ready": True,
        "owner_decision": "Starbucks variant only; Red Bull is noncanonical",
    },
    "3.11": {
        "real_data_reference_ids": ("50",), "target": "sparkling_water",
        "action": "pick", "support": "snack_shelf",
        "source_initializer": "real2sim/teleop/init_task.py",
        "robot_station": "shelf_pick",
        "runtime_ready": True,
        "owner_decision": (
            "small Task 50 / grapefruit sparkling confirmed"
        ),
    },
    "3.24": {
        "real_data_reference_ids": ("45",), "target": "oreo_pack",
        "action": "surface_place", "support": "coffee_table_target_zone",
        "source_initializer": "real2sim/teleop/init_task.py",
        "robot_station": "shelf_place_table",
        "destination_contract": "shared_shelf_to_coffee_table_live_footprint",
        "runtime_ready": True,
        "owner_decision": (
            "use the shared shelf-to-CoffeeTable place station: robot faces "
            "the table with its back to the shelf"
        ),
    },
    "4.4": {
        "real_data_reference_ids": (), "target": "glasses_case",
        "donor_real_data_reference_ids": ("98",),
        "action": "drawer_pick",
        "support": "coffee_table_black_box_1",
        "robot_station": "coffee_table_drawer_wallet",
        "drawer_contract": {
            "prim_path": "/World/Assets/CoffeeTable/black_box_1",
            "joint_path": (
                "/World/Assets/CoffeeTable/black_box_1_slide"
            ),
            "initial_state": "fully_closed",
            "required_sequence": "physically_pull_open_then_pick_glasses_case",
            "final_state": "open_allowed",
        },
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "shared_code_control_drawer_open_pick",
        "canary_gate": "physical_open_pick_acceptance_required",
        "owner_decision": (
            "black_box_1 starts closed; both Family-4 drawer picks use the "
            "same proven front station, physically open it, then pick the "
            "target; the Sheet does not require closing the drawer"
        ),
    },
    "4.10": {
        "real_data_reference_ids": ("92",), "target": "wallet",
        "action": "drawer_pick",
        "support": "coffee_table_black_box_1",
        "robot_station": "coffee_table_drawer_wallet",
        "drawer_contract": {
            "prim_path": "/World/Assets/CoffeeTable/black_box_1",
            "joint_path": "/World/Assets/CoffeeTable/black_box_1_slide",
            "initial_state": "fully_closed",
            "required_sequence": "physically_pull_open_then_pick_wallet",
            "final_state": "open_allowed",
        },
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "shared_code_control_drawer_open_pick",
        "canary_gate": "physical_open_pick_acceptance_required",
        "owner_decision": (
            "Task 92 supplies optional wallet-pick motion; both drawer tasks "
            "use the physical canonical black_box_1 drawer"
        ),
    },
    "4.6": {
        "real_data_reference_ids": (), "target": "glasses_case",
        "action": "surface_place",
        "support": "episode_selected_destination",
        "destination_contract": "episode_selected_user_or_live_surface",
        "episode_destination_required": True,
        "allowed_destination_modes": (
            "specified_surface", "user_handoff",
        ),
        "implemented_destination_modes": ("specified_surface",),
        "controller_profile": "code_control_synthetic_surface_place",
        "runtime_ready": True,
        "production_ready": False,
        "canary_gate": "physical_surface_place_acceptance_required",
        "owner_decision": (
            "the episode selects user handoff or a specified live surface; "
            "there is no implicit CoffeeTable destination"
        ),
    },
    "4.16": {
        "real_data_reference_ids": (), "target": "remote",
        "action": "pick", "support": "snack_shelf_beverage_layer",
        "robot_station": "shelf_pick",
        "spawn_policy": (
            "sample any collision-free support-local pose on the beverage "
            "layer; keep the full remote footprint inside the shelf"
        ),
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "code_control_synthetic_shelf_pick",
        "canary_gate": "remote_specific_shelf_pick_acceptance_required",
    },
    "4.17": {
        "real_data_reference_ids": ("102",), "target": "remote",
        "action": "surface_place", "support": "tv_cabinet",
        "source_trajectory": "small Task 102",
        "destination_contract": "live_tv_cabinet_top_footprint",
        "robot_station": "tv_cabinet_pick",
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "real_replay_then_code_control_surface_place",
        "canary_gate": "task102_station_and_place_acceptance_required",
        "owner_decision": "the only destination is the live TV-cabinet top",
    },
    "4.12": {
        "real_data_reference_ids": (), "target": "wallet",
        "action": "surface_place",
        "support": "episode_selected_destination",
        "destination_contract": "episode_selected_user_or_live_surface",
        "episode_destination_required": True,
        "allowed_destination_modes": (
            "specified_surface", "user_handoff",
        ),
        "implemented_destination_modes": ("specified_surface",),
        "controller_profile": "code_control_synthetic_surface_place",
        "runtime_ready": True,
        "production_ready": False,
        "canary_gate": "physical_surface_place_acceptance_required",
        "owner_decision": (
            "the episode selects user handoff or a specified live surface; "
            "there is no implicit CoffeeTable destination"
        ),
    },
    "5.3": {
        "real_data_reference_ids": ("107",), "target": "salad_box",
        "target_compatibility_aliases": ("lunchbox",),
        "action": "pick", "support": "dining_table",
        "robot_station": "kitchen_table_place",
        "source_trajectory": "small Task 107",
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "real_replay_then_code_control_pick",
        "canary_gate": "task107_pick_acceptance_required",
        "owner_decision": "use the same station as the other bag tasks",
    },
    "5.4": {
        "real_data_reference_ids": ("99",), "target": "salad_box",
        "target_compatibility_aliases": ("lunchbox",),
        "action": "tote_place", "support": "l2_soft_tote",
        "robot_station": "kitchen_table_place",
        "runtime_ready": True,
    },
    "5.5": {
        "real_data_reference_ids": ("111",), "target": "power_bank",
        "action": "pick", "support": "sofa",
        "robot_station": "sofa_pick",
        "source_trajectory": "small Task 111",
        "runtime_ready": True,
        "production_ready": False,
        "controller_profile": "real_replay_then_code_control_pick",
        "canary_gate": "power_bank_sofa_pick_acceptance_required",
        "owner_decision": "2026-09-17 user: pick power bank on the sofa at the 5.11 station",
    },
    "5.6": {
        "real_data_reference_ids": ("103",), "target": "power_bank",
        "action": "tote_place", "support": "l2_soft_tote",
        "robot_station": "kitchen_table_place",
        "runtime_ready": True,
    },
    "5.11": {
        "real_data_reference_ids": ("73",), "target": "bottle",
        "action": "pick", "support": "sofa",
        "robot_station": "sofa_pick",
        "source_trajectory": "small Task 73 pick segment",
        "segment_contract": "pick_segment_of_same_long_real_episode",
        "controller_profile": "real_long_episode_pick_segment_replay_or_repair",
        "runtime_ready": True,
        "production_ready": False,
        "canary_gate": "task73_sofa_pick_segment_acceptance_required",
        "owner_decision": (
            "5.11 is the sofa water-bottle pick segment; 5.12 is the bag-place "
            "segment of the same long REAL episode"
        ),
    },
    "8.8": {
        "real_data_reference_ids": (), "target": "old_tablecloth",
        "action": "cloth_place", "support": "laundry_zone",
        "runtime_ready": False,
        "blocker": (
            "strict deformable destination checker exists; laundry-zone "
            "geometry and an accepted cloth initializer remain absent"
        ),
    },
    "8.9": {
        "real_data_reference_ids": (), "target": "clean_tablecloth",
        "action": "cloth_pick", "support": "clean_cloth_storage",
        "runtime_ready": False,
        "blocker": (
            "strict deformable pick checker exists; clean-cloth initial state "
            "and robot-relative initializer remain absent"
        ),
    },
}

ROLLOUT_OUTPUT = OUTPUT / "recovered_personal_item_rollouts"


def _source_audit(benchmark_subtask: str) -> dict:
    payload = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "real2sim.recovered_task_source_audit.v1":
        raise ValueError("unsupported recovered source-audit schema")
    row = payload.get("tasks", {}).get(str(benchmark_subtask))
    if not isinstance(row, dict):
        raise ValueError(
            f"source audit has no benchmark {benchmark_subtask} row"
        )
    return dict(row)


def _station_pose(station: str) -> dict:
    import sys

    teleop = REPO / "real2sim" / "teleop"
    if str(teleop) not in sys.path:
        sys.path.insert(0, str(teleop))
    from robot_positions import robot_position

    pose = robot_position(str(station))
    return {
        "key": pose.key,
        "xy_m": [float(value) for value in pose.xy_m],
        "yaw_deg": float(pose.yaw_deg),
    }


def preflight(
    benchmark_subtask: str,
    *,
    episode_destination: dict | None = None,
) -> dict:
    task = str(benchmark_subtask)
    if task not in TASKS:
        raise KeyError(f"unknown recovered benchmark task {task}")
    contract = dict(TASKS[task])
    scene_binding = runtime_binding(task, episode_destination)
    source = _source_audit(task)
    required = [
        ASSETS[contract["target"]], CANONICAL_SCENE, CANONICAL_BASELINE,
        SOURCE_AUDIT_PATH,
    ]
    if contract["action"] == "tote_place":
        required.append(TOTE_SCENE)
    missing = [str(path) for path in required if not path.is_file()]
    blockers = [contract["blocker"]] if contract.get("blocker") else []
    if missing:
        blockers.append("missing runtime dependency: " + ", ".join(missing))
    if scene_binding["canonical_target_available"] is not True:
        blockers.append(
            "canonical House baseline is missing isolated target prim "
            + scene_binding["target_prim"]
        )
    if scene_binding.get("episode_destination_required"):
        if scene_binding.get("destination_selected") is not True:
            blockers.append(
                "episode must select user_handoff or a specified_surface "
                "destination; CoffeeTable is not a default"
            )
        elif scene_binding.get("runtime_supported") is not True:
            blockers.append(str(scene_binding["unsupported_reason"]))
    source_status = str(source.get("status", "unknown"))
    accepted_strict_trajectory = source.get(
        "accepted_strict_trajectory"
    ) is True
    pipeline_ready = bool(
        contract["runtime_ready"]
        and not missing
        and scene_binding["canonical_target_available"] is True
        and scene_binding.get("runtime_supported") is True
    )
    drawer_pick_producer = None
    if contract["action"] == "drawer_pick":
        try:
            from .family4_drawer_pick_task_owned_producer import (
                preflight as drawer_pick_preflight,
            )
        except ImportError:
            from family4_drawer_pick_task_owned_producer import (
                preflight as drawer_pick_preflight,
            )
        drawer_pick_producer = drawer_pick_preflight(task)
        blockers.extend(map(str, drawer_pick_producer.get("blockers", ())))
    return {
        "benchmark_subtask": task,
        "defined": True,
        "strict_checker_ready": True,
        "runtime_ready": pipeline_ready,
        "pipeline_ready_for_materialization": pipeline_ready,
        "runtime_materialized": False,
        "production_ready": bool(
            contract.get("production_ready", contract["runtime_ready"])
            and pipeline_ready
        ),
        "canary_required_before_unattended_production": bool(
            contract.get("canary_gate")
        ),
        "canary_gate": str(contract.get("canary_gate", "")),
        "task_owned_producer": drawer_pick_producer,
        "source_status": source_status,
        "accepted_strict_trajectory_available": accepted_strict_trajectory,
        "source_audit": source,
        "source_audit_path": str(SOURCE_AUDIT_PATH),
        "contract": contract,
        "runtime_binding": scene_binding,
        "episode_destination_required": bool(
            scene_binding.get("episode_destination_required")
        ),
        "episode_destination": scene_binding.get("episode_destination"),
        "asset": str(ASSETS[contract["target"]]),
        "base_scene": str(TOTE_SCENE) if contract["action"] == "tote_place"
        else "",
        "missing_paths": missing,
        "blockers": blockers,
        "randomization": {
            "fresh_scene_per_rollout": True,
            "target_xy_jitter_m": (
                0.0 if contract["real_data_reference_ids"] else 0.015
            ),
            "target_yaw_jitter_deg": (
                0.0 if contract["real_data_reference_ids"] else 5.0
            ),
            "full_footprint_collision_rejection": True,
            "furniture_transforms_changed": False,
            "camera_randomized": False,
            "object_pose_writes_during_rollout": False,
        },
    }


def prepare_rollout(
    benchmark_subtask: str,
    *,
    rollout_index: int,
    output_root: Path = ROLLOUT_OUTPUT,
    initializer_receipt: Path | None = None,
    episode_destination: dict | None = None,
) -> dict:
    """Plan a rollout, or materialize it inside an initialized open session.

    Normal schedulers call this once to create the deterministic plan.  The
    runtime calls it again with ``initializer_receipt`` after initialization;
    that path measures the canonical scene and returns ``run_strict_rollout``
    only on a fresh, collision-free, identity-bound session.
    """
    task = str(benchmark_subtask)
    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    decision = preflight(
        task, episode_destination=episode_destination,
    )
    if not decision["runtime_ready"]:
        return {
            **decision,
            "schema": RANDOMIZATION_SCHEMA,
            "rollout_index": int(rollout_index),
            "schedulable": False,
            "decision": (
                "episode_destination_required"
                if decision.get("episode_destination_required")
                and decision.get("episode_destination") is None
                else "blocked"
            ),
        }
    selected_destination = decision["runtime_binding"].get(
        "episode_destination"
    )
    destination_identity = (
        ":destination:"
        + json.dumps(
            selected_destination,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if selected_destination is not None else ""
    )
    plan_id = (
        f"recovered:{task}{destination_identity}:"
        f"rollout:{int(rollout_index):06d}"
    )
    seed = (
        int(rollout_index) * 1_000_003
        + sum((index + 1) * ord(char) for index, char in enumerate(task))
    )
    rng = random.Random(seed)
    real_derived = bool(decision["contract"]["real_data_reference_ids"])
    target_offset = (
        [0.0, 0.0, 0.0]
        if real_derived else [
            rng.uniform(-0.015, 0.015),
            rng.uniform(-0.015, 0.015),
            0.0,
        ]
    )
    yaw_delta = 0.0 if real_derived else rng.uniform(-5.0, 5.0)
    rollout_dir = Path(output_root) / f"benchmark_{task.replace('.', '_')}"
    if selected_destination is not None:
        readable_destination = "_".join((
            str(selected_destination["mode"]),
            str(selected_destination.get("prim_path", "")).strip("/").replace(
                "/", "_"
            ),
            str(selected_destination.get("layout_frame", "")),
            str(selected_destination.get("robot_station", "")),
        )).rstrip("_")
        rollout_dir /= f"destination_{readable_destination}"
    rollout_dir /= f"rollout_{int(rollout_index):06d}"
    plan_path = rollout_dir / "initializer_plan.json"
    audit_path = rollout_dir / "randomization_audit.json"
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": plan_id,
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "seed": seed,
        "restore_source": str(CANONICAL_BASELINE),
        "working_scene": str(CANONICAL_SCENE),
        "source_baseline": str(SOURCE_DEMO / "scene_room_baseline.usda"),
        "source_revision": source_revision((
            HERE / "recovered_task_contract.py",
            HERE / "recovered_scene_materialization.py",
            SOURCE_AUDIT_PATH,
            REPO / "real2sim" / "teleop" / "scene_baseline_manifest_v1.json",
        )),
        "runtime_scene_basename": "scene_room.usda",
        "target_asset": decision["asset"],
        "action": decision["contract"]["action"],
        "episode_destination": selected_destination,
        "target_prim": decision["runtime_binding"]["target_prim"],
        "support_prim": decision["runtime_binding"]["support_prim"],
        "intentional_collision_roots": decision["runtime_binding"][
            "intentional_collision_roots"
        ],
        "initial_relation": decision["runtime_binding"]["initial_relation"],
        "support_surface_z_m": decision["runtime_binding"].get(
            "support_surface_z_m"
        ),
        "activation_prims": decision["runtime_binding"][
            "activation_prims"
        ],
        "canonical_real_data_reference_ids": list(
            decision["contract"]["real_data_reference_ids"]
        ),
        "source_initializer": decision["contract"].get(
            "source_initializer", ""
        ),
        "source_trajectory": decision["contract"].get(
            "source_trajectory", ""
        ),
        "source_segment_contract": decision["contract"].get(
            "segment_contract", ""
        ),
        "controller_profile": decision["contract"].get(
            "controller_profile", ""
        ),
        "robot_station": decision["runtime_binding"].get(
            "robot_station", decision["contract"].get("robot_station", "")
        ),
        "robot_station_pose": _station_pose(
            decision["runtime_binding"].get(
                "robot_station",
                decision["contract"].get("robot_station", ""),
            )
        ) if decision["runtime_binding"].get(
            "robot_station", decision["contract"].get("robot_station", "")
        ) else {},
        "support": decision["contract"]["support"],
        "spawn_policy": decision["contract"].get("spawn_policy", ""),
        "destination_contract": decision["contract"].get(
            "destination_contract", ""
        ),
        "drawer_contract": decision["contract"].get("drawer_contract", {}),
        "target_initial_pose_policy": (
            "extract_from_real_episode_unchanged"
            if real_derived else "synthetic_support_local_randomized"
        ),
        "target_frame_zero_offset_m": target_offset,
        "target_frame_zero_yaw_delta_deg": yaw_delta,
        "robot_adaptation": (
            "move robot station and robot-relative trajectory together; "
            "never move furniture"
        ),
        "furniture_transforms_changed": False,
        "camera_randomized": False,
        "object_pose_writes_during_rollout": False,
        "materialization_hook": (
            "conda run -n mesh311 python real2sim/production/"
            "recovered_scene_materialization.py --plan <plan> "
            "--receipt <initializer-receipt> --audit <audit>"
        ),
        "runtime_evidence_required": [
            "exclusive canonical session restored from baseline",
            "exact task object activation",
            "initializer-complete target world bounds",
            "target versus every active CollisionAPI body audit",
            "unchanged inherited furniture transforms",
        ],
        "execution_phase": (
            "production" if decision["production_ready"] else "canary"
        ),
        "canary_gate": decision["canary_gate"],
    }
    # This function currently materializes a deterministic initializer plan,
    # not a USD scene.  Freshness and collision freedom are runtime facts and
    # must never be inferred from a plan with an empty conflict list.
    audit = {
        "schema": RANDOMIZATION_SCHEMA,
        "benchmark_subtask": task,
        "rollout_index": int(rollout_index),
        "episode_destination": selected_destination,
        "scene_materialized": False,
        "fresh_scene_per_rollout": False,
        "collision_free": None,
        "collision_audit": {
            "status": "not_run",
            "required_method": (
                "full target footprint versus all active staged scene bodies"
            ),
            "intentional_contacts": (
                ["target", "gripper pads"]
                if TASKS[task]["action"] in {"tote_place", "surface_place"}
                else ["target", decision["runtime_binding"]["support_prim"]]
            ),
            "conflicts": None,
        },
        "furniture_transforms_changed": False,
        "camera_randomized": False,
        "object_pose_writes_during_rollout": None,
        "initializer_plan": str(plan_path),
    }
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((plan_path, plan), (audit_path, audit)):
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    pending = {
        **audit,
        "defined": True,
        "pipeline_ready_for_materialization": True,
        "runtime_materialized": False,
        "schedulable": False,
        "production_ready": False,
        "execution_phase": plan["execution_phase"],
        "canary_gate": decision["canary_gate"],
        "initializer_plan": str(plan_path),
        "decision": "requires_runtime_materialization",
        "randomization_audit": str(audit_path),
    }
    if initializer_receipt is None:
        return pending
    try:
        from .recovered_scene_materialization import materialize_and_audit
    except ImportError:
        from recovered_scene_materialization import materialize_and_audit
    materialize_and_audit(
        plan_path, Path(initializer_receipt), audit_path,
    )
    return require_materialized_rollout(
        task,
        initializer_plan=plan_path,
        randomization_audit=audit_path,
    )


def require_materialized_rollout(
    benchmark_subtask: str,
    *,
    initializer_plan: Path,
    randomization_audit: Path,
) -> dict:
    """Authorize physics only from this rollout's measured audit."""
    try:
        from .recovered_scene_materialization import validate_plan
    except ImportError:
        from recovered_scene_materialization import validate_plan

    task = str(benchmark_subtask)
    plan_path = Path(initializer_plan).resolve()
    audit_path = Path(randomization_audit).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if str(plan.get("benchmark_subtask")) != task:
        raise ValueError("initializer plan benchmark identity mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    collision = audit.get("collision_audit")
    observation = audit.get("observation")
    failures = []
    if audit.get("schema") != RANDOMIZATION_SCHEMA:
        failures.append("randomization schema mismatch")
    if str(audit.get("benchmark_subtask")) != task:
        failures.append("randomization benchmark identity mismatch")
    if int(audit.get("rollout_index", -1)) != int(plan["rollout_index"]):
        failures.append("rollout index mismatch")
    if audit.get("scene_materialized") is not True:
        failures.append("scene was not materialized")
    if audit.get("fresh_scene_per_rollout") is not True:
        failures.append("scene was not restored from baseline")
    if audit.get("collision_free") is not True:
        failures.append("initial scene is not collision-free")
    if not isinstance(collision, dict) or collision.get("status") != "measured_pass":
        failures.append("collision audit is not a measured pass")
    elif (
        collision.get("conflicts") != []
        or collision.get("errors") != []
        or isinstance(collision.get("measured_non_target_body_count"), bool)
        or not isinstance(collision.get("measured_non_target_body_count"), int)
        or collision["measured_non_target_body_count"] < 1
    ):
        failures.append("measured collision audit is incomplete")
    if not isinstance(observation, dict):
        failures.append("measured scene observation is missing")
    else:
        receipt = observation.get("initializer_receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema")
            != "real2sim.recovered_initializer_receipt.v1"
            or str(receipt.get("initializer_plan_id", ""))
            != str(plan.get("plan_id", ""))
            or str(observation.get("initializer_plan_id", ""))
            != str(plan.get("plan_id", ""))
        ):
            failures.append("initializer receipt is missing or identity-unbound")
    if audit.get("furniture_transforms_changed") is not False:
        failures.append("furniture transforms changed")
    if audit.get("object_pose_writes_during_rollout") is not False:
        failures.append("rollout object-pose writes are not forbidden")
    if failures:
        return {
            "schema": RANDOMIZATION_SCHEMA,
            "benchmark_subtask": task,
            "defined": True,
            "runtime_materialized": False,
            "schedulable": False,
            "decision": "blocked",
            "blockers": failures,
            "initializer_plan": str(plan_path),
            "randomization_audit": str(audit_path),
        }
    return {
        **audit,
        "defined": True,
        "runtime_materialized": True,
        "schedulable": True,
        "decision": "run_strict_rollout",
        "initializer_plan": str(plan_path),
        "randomization_audit": str(audit_path),
    }

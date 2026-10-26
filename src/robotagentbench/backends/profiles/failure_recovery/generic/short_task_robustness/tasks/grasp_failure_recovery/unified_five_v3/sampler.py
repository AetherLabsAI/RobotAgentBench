"""Fresh original sofa room plus the approved eight frozen arm configurations."""
from copy import deepcopy
from pathlib import Path
from ..common import HERE, owned, digest, finish, write_json


def sofa(seed, output):
    from ..closeout_v1.sampler import room_base
    from ..frozen_initials_v3.fast import sofa_context
    from ..frozen_initials_v3 import adapted_bank
    from ..frozen_initials_v1.cup import base_matrix
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from .sofa_geometry import Gate
    out = owned(output, fresh=True)
    metadata = out.parent / "base_generation"
    metadata.mkdir(exist_ok=False)
    key = room_base("sofa", seed, metadata)
    path = HERE / "frozen_initials_v3/compile/sofa_offline_adapt_v3/pool.json"
    pool = adapted_bank.load(path)
    stage, record, before = sofa_context(out, key)
    gate = Gate(stage, record)
    checks, stations = {}, {}
    for profile in pool["profiles"]:
        index = profile["fit"]["station"]["candidate_index"]
        station = next((s for s in gate.bases if s["candidate_index"] == index), None)
        if station is None:
            checks[profile["id"]] = dict(accepted_cpu=False, reason="station_not_in_original_domain")
            continue
        stations[profile["id"]] = station
        checks[profile["id"]] = gate.check(profile["robot_initialization"], profile["active_hand"],
            base=base_matrix(station["initial_base_xyyaw"]))
    write_json(out / "frozen_pose_eligibility.json", checks)
    profile = adapted_bank.select(pool, seed, {k for k, v in checks.items() if v["accepted_cpu"]})
    station = stations[profile["id"]]
    record.update(robot_initialization=deepcopy(profile["robot_initialization"]),
        active_hand=profile["active_hand"], gripper_closure=profile["gripper_closure"],
        initial_base_xyyaw=station["initial_base_xyyaw"],
        robot_base_initialization=initialize_base_pose(stage, station["initial_base_xyyaw"]),
        frozen_profile_id=profile["id"], training_pose_provenance=profile["training_provenance"],
        provenance_kind=profile["provenance_kind"], initialization_policy="offline_adapted_sofa_frozen_bank.v1",
        room_seed=record["seed"], seed=seed, preparation_accepted=True,
        new_episode_native_acceptance_inherited=False, source_motion_read=False, source_checkpoint=None,
        online_ik_calls=0, generation_reads_recorded_trajectory=False,
        runtime_training_dataset_required=False, object_relocated_for_ik=False, seed_retried=False,
        fresh_full_room_generated=True, procedural_grasp=checks[profile["id"]],
        code_sources={str(p): digest(p) for p in (Path(__file__), path,
            HERE / "unified_five_v3/sofa_geometry.py", Path(adapted_bank.__file__),
            HERE / "frozen_initials_v3/geometry.py", HERE / "closeout_v1/sampler.py")})
    return finish(stage, out, record, before)

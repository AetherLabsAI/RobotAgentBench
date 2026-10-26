"""CPU contracts and launch admission only; never call a model or advance physics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from short_tasks import registry
from short_tasks.agent_probe_v1.probe_random_layout import request_for as ordinary_request
from ..common import HERE, write_json
from . import entry


def ordinary():
    rows = {}
    for name in registry.TASK_NAMES:
        if name.startswith("failure_recovery_"):
            continue
        try:
            rows[name] = dict(plan=registry.resolve(ordinary_request(name)))
        except Exception as exc:
            rows[name] = dict(error=type(exc).__name__ + ": " + str(exc))
    return rows


def contracts():
    import io
    import unittest
    import numpy as np
    from .sofa_geometry import stable_side
    from ..expansion_v2.catalog import base_request
    from ..shelf_continuous_v2 import public_entry as shelf
    checks = []
    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    check(set(registry.FAILURE_RECOVERY_TASKS) == set(entry.NAMES), "exact_five_names")
    check(len(registry.TASK_NAMES) == len(set(registry.TASK_NAMES)), "unique_registrations")
    for name, variant in entry.NAMES.items():
        request = registry.failure_recovery_adapter().request_for(name)
        plan = registry.resolve(request)
        check(registry.verify_sources(plan)["verified"], name + ":source_pins")
        check(plan["control_hz"] == 20, name + ":20hz")
        check(plan["failure_recovery_variant"] == variant, name + ":variant")
        changed = dict(request, options={"override": True})
        try:
            registry.resolve(changed)
        except Exception:
            checks.append(name + ":reject_semantic_override")
        else:
            raise AssertionError("Semantic override admitted")
        if variant in ("shelf_drink", "coffee_bag"):
            check(plan["score_policy"]["maximum"] == (4 if variant == "shelf_drink" else 6), name + ":score")
            check(plan["no_progress_timeout_s"] is None, name + ":atomic_deadline")
            check(plan["physics_policy"] == "original_atomic_cpu_fixed_240hz", name + ":atomic_physics")
        else:
            old = registry.resolve(base_request(variant, 1,
                sofa_asset="remote_tv" if variant == "sofa" else None,
                waste_categories=None if variant == "sofa" else ["cup"]))
            for key in ("score_policy", "instruction", "no_progress_timeout_s", "physics_policy", "no_progress_timeout_by_phase_s"):
                if variant == "sofa" and key == "instruction":
                    from ..closeout_v1.sofa_whole_table import INSTRUCTION, RULE_VERSION as sofa_version
                    check(plan[key] == INSTRUCTION and plan['rule_version'] == sofa_version,
                          name + ":approved_whole_table_instruction_and_version")
                    continue
                check(plan.get(key) == old.get(key), name + ":original_" + key)
    apis = set()
    for seed in range(1, 101):
        request = entry.request_for(shelf.NAME)
        request["seed"] = seed
        plan = registry.resolve(request)
        check(plan["initialization_policy"] == shelf.POLICY, "shelf_continuous_seed_" + str(seed))
        apis.add(plan["source_api"])
    check(apis == set(shelf.APIS), "all_eight_drink_apis_preserved")
    for hand in ("l", "r"):
        side, witness = stable_side(np.eye(4), np.array([1., 0., .5]), hand)
        check(side == hand and witness["kind"] == "floating_point_centerline_tie", "centerline_" + hand)
    check(stable_side(np.eye(4), np.array([1., -1e-8, .5]), "l")[0] == "r", "real_right_not_tie")
    check(stable_side(np.eye(4), np.array([1., 1e-8, .5]), "r")[0] == "l", "real_left_not_tie")
    from ..public_contract_tests import Robot22
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Robot22))
    check(result.wasSuccessful(), "original_robot22_transport_tests")
    return dict(passed=True, assertions=len(checks), checks=checks,
        robot22_tests=result.testsRun, robot22_log=stream.getvalue(), physics_run=False, models_called=False)


def admit(path):
    import os
    import sys
    from unittest.mock import patch
    from short_tasks.agent_probe_v1.native_owner import admit_launch
    launch = json.loads(Path(path).read_text())
    request, manifest, plan = admit_launch(launch, HERE / "unified_five_v3/unused_native_output")
    if request["task_name"] != launch["task_name"] or plan["task_name"] != launch["task_name"]:
        raise AssertionError("Three-value parent admission mismatch")
    adapter = registry.failure_recovery_adapter()
    for key in ("seed", "variant", "scene_sha256"):
        changed = deepcopy(manifest)
        changed[key] = "tampered"
        try:
            registry.bind(plan, changed)
        except Exception:
            pass
        else:
            raise AssertionError("Tampered manifest accepted: " + key)
    altered = dict(launch, source_record_sha256="0" * 64)
    try:
        adapter.admit_launch(altered, HERE / "unified_five_v3/unused_native_output")
    except Exception:
        pass
    else:
        raise AssertionError("Source-record mismatch accepted")
    initial = json.loads(Path(manifest["initial"]).read_text())
    if initial["online_ik_calls"] != 0 or initial["runtime_training_dataset_required"] is not False:
        raise AssertionError("Unexpected runtime dependency")
    # Exercise the real trusted dispatch without allocating a GPU or executing
    # the owner. Actual launch admission above is not mocked.
    class ExecReplaced(Exception):
        pass
    with patch.object(sys, "argv", ["native_owner", "--launch", str(path), "--output",
            str(HERE / "unified_five_v3/unused_native_output"), "--gpu", "0",
            "--gpu-uuid", "dispatch-contract-only", "--smoke"]), \
            patch.object(os, "execv", side_effect=ExecReplaced) as execv, \
            patch.object(os, "execve", side_effect=ExecReplaced) as execve:
        try:
            adapter.exec_native()
        except ExecReplaced:
            pass
        if launch["task_name"] == "failure_recovery_coffee_bag":
            if execve.call_count != 1 or execv.call_count:
                raise AssertionError("Coffee child environment was not selected")
            native_module = execve.call_args.args[1][4]
            if not native_module.endswith(".unified_five_v3.coffee_worker"):
                raise AssertionError("Wrong coffee dependency bridge")
        else:
            if execv.call_count != 1 or execve.call_count:
                raise AssertionError("Expected exactly one trusted native dispatch")
            native_module = execv.call_args.args[1][4]
            suffix = (".shelf_continuous_v2.public_owner" if initial["variant"] == "shelf_drink"
                else ".closeout_v1.room_owner_v3" if initial["variant"] == "sofa"
                else ".closeout_v1.room_owner_v2")
            if not native_module.endswith(suffix):
                raise AssertionError("Wrong trusted native owner: " + native_module)
    return dict(passed=True, task_name=launch["task_name"], instruction=plan["instruction"],
        initial=manifest["initial"], profile=initial["frozen_profile_id"],
        score_policy=plan["score_policy"], source_api=initial.get("source_api"),
        variant=initial["variant"], initial_three_value_admission=True,
        online_ik_calls=0, training_dataset_required=False, models_called=False,
        trusted_native_dispatch_verified=True, native_module=native_module, gpu_used=False,
        physical_actions_executed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "contracts", "admit"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch")
    args = parser.parse_args()
    value = ordinary() if args.mode == "baseline" else contracts() if args.mode == "contracts" else admit(args.launch)
    write_json(Path(args.output), value)

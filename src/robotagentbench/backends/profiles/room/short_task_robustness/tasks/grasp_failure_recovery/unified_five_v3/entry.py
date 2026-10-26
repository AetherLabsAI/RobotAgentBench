"""Trusted five-way router; preserve the continuous shelf, original scorers."""
import json
import os
import subprocess
import sys
from ..common import HERE, CURRENT, COFFEE_SOURCE, owned, digest, write_json

NAMES = {"failure_recovery_" + v: v for v in (
    "shelf_drink", "coffee_bag", "sofa", "table_waste_empty_grasp", "table_waste_tipped_cup")}
RULE_VERSION = "short_task_robustness.grasp_failure_recovery.frozen5.v2"


def coffee_environment():
    # The atomic coffee scorer is an existing pinned dependency, not the slim
    # Arena distribution. Match its original isolated interpreter recipe.
    env = dict(os.environ)
    paths = [str(COFFEE_SOURCE), str(CURRENT), *env.get("PYTHONPATH", "").split(":")]
    env["PYTHONPATH"] = ":".join(dict.fromkeys(p for p in paths if p))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def module(name):
    if name not in NAMES:
        raise ValueError("Only the five approved failure recoveries")
    if NAMES[name] == "shelf_drink":
        from ..shelf_continuous_v2 import public_entry
        return public_entry
    from ..closeout_v1 import public_adapter
    return public_adapter


def request_for(name):
    return module(name).request_for(name)


def resolve(request):
    return module(request["task_name"]).resolve(request)


def verify_sources_for_plan(plan):
    return module(plan["task_name"]).verify_sources_for_plan(plan)


def bind(plan, manifest):
    return module(plan["task_name"]).bind(plan, manifest)


def prepare_registered(request, ready_pose, sensor_file, output):
    if request["task_name"] == "failure_recovery_coffee_bag":
        if ready_pose is not None or sensor_file is not None:
            raise ValueError("Preserve original coffee posture/cameras")
        out = owned(output, fresh=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        log = owned(out.with_name(out.name + ".coffee_worker.log"))
        with log.open("x") as stream:
            result = subprocess.run([sys.executable, "-B", "-u", "-m",
                __package__ + ".coffee_worker", "prepare", "--request-json", json.dumps(request),
                "--output", str(out)], env=coffee_environment(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Pinned coffee preparation failed; see " + str(log))
        return json.loads((out / "launch.json").read_text())
    if request["task_name"] != "failure_recovery_sofa":
        return module(request["task_name"]).prepare_registered(request, ready_pose, sensor_file, output)
    from short_tasks import registry
    from .sampler import sofa
    if ready_pose is not None or sensor_file is not None:
        raise ValueError("Preserve original posture/cameras")
    plan = registry.resolve(request)
    verify_sources_for_plan(plan)
    out = owned(output, fresh=True)
    out.mkdir(parents=True)
    write_json(out / "request.json", request)
    value = sofa(request["seed"], out / "initial")
    from ..closeout_v1.sofa_whole_table import RULE_VERSION as sofa_version
    value['completion_policy'] = 'same_target_released_on_whole_coffee_table'
    value['placement_policy_version'] = sofa_version
    write_json(out / 'initial/initial.json', value)
    manifest = dict(schema="arena.failure_recovery.frozen5_manifest.v2", variant="sofa",
        seed=value["seed"], initial=str(out / "initial/initial.json"),
        initial_sha256=digest(out / "initial/initial.json"), scene=value["scene"],
        scene_sha256=value["scene_sha256"], generated=True, native_physics_validated=False,
        source_progress_inherited=False)
    registry.bind(plan, manifest)
    write_json(out / "manifest.json", manifest)
    launch = dict(schema="openend.short_task.launch.v1", task_name=plan["task_name"],
        request=str(out / "request.json"), request_sha256=digest(out / "request.json"),
        manifest=str(out / "manifest.json"), manifest_sha256=digest(out / "manifest.json"),
        scene=value["scene"], scene_sha256=value["scene_sha256"],
        source_record_sha256=digest(HERE / "closeout_v1/component_sources.json"),
        native_physics_validated=False, action_acceptance_verified=False)
    write_json(out / "launch.json", launch)
    return launch


def admit_launch(launch, output):
    selected = module(launch["task_name"])
    if NAMES[launch["task_name"]] == "shelf_drink":
        return selected.admit_launch(launch, output)
    # Shared native owner requires the same three-value contract for all five.
    from short_tasks import registry
    owned(output, fresh=True)
    manifest, plan = selected.admit_launch(launch)
    request = json.loads(owned(launch["request"]).read_text())
    return request, manifest, registry.bind(plan, manifest)


def exec_native():
    if "--launch" not in sys.argv:
        raise ValueError("A prepared launch is required")
    launch = json.loads(owned(sys.argv[sys.argv.index("--launch") + 1]).read_text())
    if launch["task_name"] == "failure_recovery_coffee_bag":
        os.execve(sys.executable, [sys.executable, "-B", "-u", "-m",
            __package__ + ".coffee_worker", "native", *sys.argv[1:]], coffee_environment())
    return module(launch["task_name"]).exec_native()

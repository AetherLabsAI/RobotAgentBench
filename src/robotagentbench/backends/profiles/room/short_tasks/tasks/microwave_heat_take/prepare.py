"""Task-owned preparation through the existing CPU composer and 22D owner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .adapter import TASK_NAME, RULE_VERSION
from .generation import HERE, POLICY, validate_output

ROOT = HERE.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE):
        raise ValueError("Task output escaped its directory")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def prepare_registered(request, ready_pose, sensor_file, output):
    from pxr import Usd, UsdUtils
    from short_tasks import registry
    from short_tasks.scene import plan_scene, verify_manifest, verify_scene_sources
    from short_tasks.agent_probe_v1.prepare_task import localize
    from short_tasks.agent_probe_v1 import target_spawn
    from openend.tasks.layout_repair_v1.ready_envelope import REFERENCE
    from short_tasks.tools.prepare_scene import checked_dependencies
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    from short_tasks.agent_probe_v1.native_owner import admit_launch

    out = Path(output).resolve()
    if not out.is_relative_to(HERE/"execution") or out.exists():
        raise ValueError("Use a fresh output under the microwave task's execution directory")
    plan = registry.resolve(request)
    if plan["task_name"] != TASK_NAME or plan["rule_version"] != RULE_VERSION or not plan["available"]:
        raise ValueError("Unavailable or mismatched microwave request")
    registry.verify_sources(plan)
    verify_scene_sources()
    ready_pose = Path(ready_pose).resolve(strict=True)
    sensor_file = Path(sensor_file).resolve(strict=True)
    sensor = read(sensor_file)["sensor"]
    out.mkdir(parents=True, exist_ok=False)
    (out/"tmp").mkdir()
    previous_tmp = tempfile.tempdir
    tempfile.tempdir = str(out/"tmp")
    report = dict(status="planning", task_name=TASK_NAME, rule_version=RULE_VERSION,
                  maximum=14, native_physics_validated=False)
    try:
        write(out/"request.json", request)
        planned = plan_scene(request)
        write(out/"workspace_plan.json", planned)
        recipe = deepcopy(planned["recipe"])
        recipe["task_scene_policy"] = dict(POLICY)
        composed = out/"composed"
        composed.mkdir()
        placement = composed/"placement.json"
        write(placement, recipe)
        validate_output(placement, ROOT, recipe)
        command = [sys.executable, "-B", str(ROOT/"openend/tasks/scene_requirements/build_selected_room.py"), str(placement)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 timeout=180, check=False)
        (composed/"composition.log").write_text(process.stdout+process.stderr)
        if process.returncode:
            raise RuntimeError("Original CPU room composition failed; see composition.log")
        manifest = read(composed/"composition.json")
        verify_manifest(manifest)
        bound = registry.bind(plan, manifest)
        native = out/"native"
        native.mkdir()
        manifest = localize(manifest, native)
        stage = Usd.Stage.Open(manifest["scene"])
        if not stage:
            raise ValueError("Prepared scene cannot be opened")
        physics = stage.GetPrimAtPath("/World/PhysicsScene")
        if (physics.GetAttribute("physxScene:enableGPUDynamics").Get() is not False
                or physics.GetAttribute("physxScene:timeStepsPerSecond").Get() != 60):
            raise ValueError("Existing CPU/60 Hz startup contract required")
        layers, assets, special = checked_dependencies(Path(manifest["scene"]))
        files = [l.realPath for l in layers if l.realPath]+list(assets)
        manifest["source_hashes"].update({str(p): sha(p) for p in files})
        selector = from_scene(stage, manifest, sensor)
        # Reuse the pre-existing measured folded-ready envelope and generic
        # target-facing search. No invented dataset medoid for a new task.
        navigation = read(REFERENCE/'navigation_result.json')['navigation_posture']
        if read(ready_pose) != navigation:
            raise ValueError("Ready posture differs from its measured envelope reference")
        from .registration import KNOB
        from pxr import UsdGeom
        bounds = UsdGeom.BBoxCache(0, ['default','render','proxy'], False, True).ComputeWorldBound(
            stage.GetPrimAtPath(KNOB)).ComputeAlignedRange()
        center = (bounds.GetMin()+bounds.GetMax())/2
        support = manifest['supports']['microwave']
        target = dict(paths=[KNOB], source='microwave', target_xy=list(center[:2]),
            front=list(target_spawn.unit(support['front'])), front_origin=list(center[:2]),
            direction_source='existing_scene_declared_microwave_front', reference_files=[])
        proposals = (dict(initial_base_xyyaw=[*map(float, xy), float(yaw)],
                          targets_rad=navigation['targets_rad'])
                     for _, xy, yaw in target_spawn.candidates(
                         target['target_xy'], target['front'], target['front_origin'],
                         manifest['overview']['floors'],
                         float(sensor['range']), float(sensor['halfAngle'])))
        spawn = target_spawn.choose(stage, manifest, TASK_NAME, sensor,
                                    candidate_rows=proposals, target=target)
        initial = [*spawn['position'][:2], spawn['yaw_rad']]
        spawn['reference_hashes'] = {str(path): sha(path) for path in (
            REFERENCE/'navigation_result.json', REFERENCE/'native_poses.npz',
            Path(target_spawn.__file__), *map(Path, target['reference_files']))}
        write(native/"spawn.json", spawn)
        write(native/"ready_pose.json", read(ready_pose))
        write(native/"sensor.json", read(sensor_file))
        manifest["source_hashes"].update(spawn["reference_hashes"])
        manifest["source_hashes"][str(native/"ready_pose.json")] = sha(native/"ready_pose.json")
        manifest["source_hashes"][str(native/"sensor.json")] = sha(native/"sensor.json")
        write(native/"manifest.json", manifest)
        write(native/"bindings.json", bound)
        write(native/"scene_validation.json", dict(
            workspace_count=len(selector.workspaces),
            dependencies=dict(layers=len(layers), assets=len(assets), resolver_modules=special),
            initial_pose="existing_measured_generic_folded_ready_and_target_front_search",
            native_physics_validated=False))
        launch = native/"launch.json"
        write(launch, dict(schema="openend.short_task.launch.v1", task_name=TASK_NAME,
            manifest=str(native/"manifest.json"), request=str(out/"request.json"),
            ready_pose=str(native/"ready_pose.json"), sensor=sensor,
            initial_base_xyyaw=initial, source_robot_pose_preserved_in_scene=True,
            startup_spawn_scope="task_front_folded_envelope_search_requires_native_acceptance",
            startup_physics_hz=60, control_hz=20, physics_backend="cpu",
            physics_frequency_owner="selected_workspace",
            preparation_receipt=str(out/"preparation.json")))
        _, checked, _ = admit_launch(read(launch), out/"native_smoke")
        verify_manifest(checked)
        report.update(status="prepared_not_native_accepted", launch=str(launch),
                      native_admission_passed=True, actual_layout_id=manifest["layout_id"])
        return report
    except BaseException as exc:
        report.update(status="failed", error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write(out/"preparation.json", report)
        tempfile.tempdir = previous_tmp

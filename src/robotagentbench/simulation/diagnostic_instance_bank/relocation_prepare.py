"""Task-local launch wiring around the original reserve-first CPU generator."""
import hashlib
import json
from pathlib import Path


def prepare(request,sensor_file,output):
    from pxr import Usd
    from short_tasks import registry
    from short_tasks.tasks.stock_wooden_tray_interference.generate_reserved_room import generate
    from short_tasks.tasks.stock_wooden_tray_interference.startup_request import request_for_generated_room
    from short_tasks.tasks.stock_wooden_tray import prepare_initial
    from short_tasks.agent_probe_v1.native_owner import admit_launch
    from short_tasks.tasks.stock_wooden_tray_interference.preflight_startup import check
    output=Path(output)
    # The original generate() enforces its task output allowlist and owns all
    # target/background/layout/reservation sampling and dependency checks.
    generated=generate(request,output)
    manifest_path=output/'composition.json'
    manifest=json.loads(manifest_path.read_text())
    plan=json.loads((output/'task_plan.json').read_text())
    actual=request_for_generated_room(plan,manifest)
    request_path=output/'request.json';request_path.write_text(json.dumps(actual,indent=2)+'\n')
    stage=Usd.Stage.Open(manifest['scene'])
    station=prepare_initial.choose(stage,manifest,seed=manifest['seed'])
    native=output/'native';native.mkdir()
    (native/'station.json').write_text(json.dumps(station,indent=2)+'\n')
    ready=native/'ready_pose.json'
    ready.write_text(json.dumps({'targets_rad':station['targets_rad']},indent=2)+'\n')
    sensor=json.loads(Path(sensor_file).read_text())['sensor']
    launch=dict(schema='openend.short_task.launch.v1',task_name=plan['task_name'],
        manifest=str(manifest_path),request=str(request_path),ready_pose=str(ready),
        initial_base_xyyaw=[*station['position'][:2],station['yaw_rad']],
        scene_sha256=hashlib.sha256(Path(manifest['scene']).read_bytes()).hexdigest(),
        sensor=sensor,startup_physics_hz=60,control_hz=20,physics_backend='cpu',
        physics_frequency_owner='selected_workspace',native_physics_validated=False,
        initialization_video_only=False,
        startup_wiring_receipt=dict(original_generator='generate_reserved_room.generate',
            original_station='stock_wooden_tray.prepare_initial.choose',
            robot_or_scene_pose_written=False,relocation_distance_unchanged=True))
    launch_path=native/'launch.json';launch_path.write_text(json.dumps(launch,indent=2)+'\n')
    # Original binding/admission and route-with-original-station gates. No
    # rejected route is edited, reoriented, widened or called a native pass.
    registry.verify_sources(registry.resolve(actual))
    admitted=admit_launch(launch,output/'original_cpu_admit')
    preflight=check(launch_path)
    (output/'original_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    if preflight['passed']is not True:
        raise ValueError('Original relocation startup preflight rejected this unchanged instance')
    return dict(launch=str(launch_path),preparation='original_reserve_first',
        generated=generated,original_admission_returned_items=len(admitted),
        original_preflight_passed=True,native_started=False)

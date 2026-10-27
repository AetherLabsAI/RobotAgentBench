"""Prepare a private lighting variant from an original ordinary-shelf launch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(original_launch, output):
    from pxr import Usd, UsdGeom
    from short_tasks.scene import verify_manifest
    from short_tasks.tasks.stock_wooden_tray import adapter as base
    from .adapter import TASK_NAME, RULE_VERSION, resolve, bind, prepare_scene
    source_launch = json.loads(Path(original_launch).read_text())
    request = json.loads(Path(source_launch['request']).read_text())
    if request['task_name'] != base.TASK_NAME:
        raise ValueError('Original ordinary shelf task required; do not use human-interference stock')
    manifest = json.loads(Path(source_launch['manifest']).read_text())
    verify_manifest(manifest)
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(HERE):
        raise ValueError('Fresh output inside lighting task required')
    request.update(task_name=TASK_NAME, rule_version=RULE_VERSION)
    plan = resolve(request)
    bind(plan, manifest)
    source_scene = Path(manifest['scene']).resolve()
    source_stage = Usd.Stage.Open(str(source_scene))
    if not source_stage:
        raise ValueError('Ordinary shelf scene did not open')
    output.mkdir(parents=True)
    scene = output / 'room.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    stage.GetRootLayer().subLayerPaths = [str(source_scene)]
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    stage.GetRootLayer().Save()
    prepared = prepare_scene(plan, stage, manifest)
    prepared['scene'] = str(scene)
    prepared['scene_sha256'] = hashlib.sha256(scene.read_bytes()).hexdigest()
    prepared['source_hashes'][str(source_scene)] = hashlib.sha256(source_scene.read_bytes()).hexdigest()
    request_path, manifest_path = output / 'request.json', output / 'composition.json'
    request_path.write_text(json.dumps(request, indent=2) + '\n')
    manifest_path.write_text(json.dumps(prepared, indent=2) + '\n')
    (output / 'task_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    launch = deepcopy(source_launch)
    launch.update(schema='openend.short_task.launch.v1', task_name=TASK_NAME,
                  request=str(request_path), manifest=str(manifest_path),
                  scene_sha256=prepared['scene_sha256'],
                  native_physics_validated=False,
                  lighting_preparation=dict(geometry_or_initial_pose_changes=False,
                                            source_launch=str(Path(original_launch).resolve())))
    (output / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    return launch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.original_launch, args.output), indent=2))

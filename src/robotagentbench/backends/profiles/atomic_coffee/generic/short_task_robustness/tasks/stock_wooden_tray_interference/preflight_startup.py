"""CPU startup checks only; no native physics, score or trigger claim."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def check(launch_path):
    from pxr import Usd, UsdGeom, Gf
    from short_tasks import registry
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from short_tasks.tasks.stock_wooden_tray_interference.prepare_scene import prepare_reserved
    launch = json.loads(launch_path.read_text())
    manifest_path = Path(launch['manifest'])
    manifest = json.loads(manifest_path.read_text())
    from short_tasks.tasks.stock_wooden_tray_interference.startup_request import request_for_generated_room
    generated_plan = json.loads((manifest_path.parent/'task_plan.json').read_text())
    plan = registry.resolve(request_for_generated_room(generated_plan, manifest))
    result = dict(layout_id=manifest['layout_id'], seed=manifest['seed'],
                  launch=str(launch_path), native_physics_run=False, checks={})
    checks = result['checks']
    for label, action in (
        ('shared_sources', verify_scene_sources),
        ('scene_sources', lambda: verify_manifest(manifest)),
        ('component_sources', lambda: registry.verify_sources(plan)),
        ('task_binding', lambda: registry.bind(plan, manifest))):
        try:
            action()
            checks[label] = dict(passed=True)
        except Exception as exc:
            checks[label] = dict(passed=False, error=str(exc))
    try:
        stage = Usd.Stage.Open(manifest['scene'])
        stage.SetEditTarget(stage.GetSessionLayer())
        x, y, yaw = launch['initial_base_xyyaw']
        matrix = Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), math.degrees(yaw)))
        matrix.SetTranslateOnly(Gf.Vec3d(x,y,0))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(matrix)
        # Same pre-attach transform as the native harness; no root-layer save.
        prepare_reserved(plan, stage, manifest)
        checks['route_with_prepared_base'] = dict(passed=True,
            limitation='USD geometry only; native ready-joint/settled route check still required')
    except Exception as exc:
        checks['route_with_prepared_base'] = dict(passed=False,error=str(exc),traceback=traceback.format_exc())
    result['passed'] = all(c['passed'] for c in checks.values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('launch', type=Path, nargs='+')
    print(json.dumps([check(p) for p in parser.parse_args().launch], indent=2))

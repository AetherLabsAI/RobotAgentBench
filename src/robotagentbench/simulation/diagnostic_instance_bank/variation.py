"""Read composed initial bindings/rigid transforms; never advance physics."""
import json
from pathlib import Path


def _semantic(value):
    if isinstance(value, dict):
        return {k: _semantic(v) for k, v in value.items()
                if k not in ('seed', 'scene', 'initial', 'source_hashes', 'files_sha256')
                and not any(token in k for token in ('sha256', 'output', 'log_path', 'source_file'))
                and not (isinstance(v, str) and (v.startswith('/') or v.endswith(('.json', '.usda', '.usdc', '.npz'))))}
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


def physical_signature(launch_path, plan):
    from pxr import Usd, UsdGeom, UsdPhysics
    launch = json.loads(Path(launch_path).read_text())
    manifest = json.loads(Path(launch['manifest']).read_text())
    stage = Usd.Stage.Open(manifest['scene'])
    if not stage:
        raise ValueError('Prepared original scene is unreadable')
    cache = UsdGeom.XformCache()
    transforms = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.IsA(UsdGeom.Xformable):
            transforms[str(prim.GetPath())] = [list(row) for row in cache.GetLocalToWorldTransform(prim)]
    if not transforms:
        raise ValueError('Prepared scene has no actual rigid-body transform inventory')
    result = dict(task_name=plan['task_name'], original_rule_version=plan['rule_version'],
                  public_question=_semantic(plan.get('public_question', {})),
                  selection=_semantic(plan.get('selection', {})),
                  objects=_semantic(manifest.get('objects', [])),
                  initial_base_xyyaw=launch.get('initial_base_xyyaw'),
                  rigid_body_world_transforms=transforms)
    initial = manifest.get('initial_file', manifest.get('initial'))
    if initial:
        result['original_initial_state'] = _semantic(json.loads(Path(initial).read_text()))
    if launch.get('ready_pose'):
        result['ready_pose'] = _semantic(json.loads(Path(launch['ready_pose']).read_text()))
    return result

"""Task-only background exclusion, before physics; source layers stay intact.

This is a composition filter, not a change to the shared random sampler.
It removes same-product background instances from this episode's scene and
manifest together. Other registered targets are an error, never auto-removed.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json

from .unique_target import require_unique_target

TASK = 'stock_wooden_tray_interference'
HERE = Path(__file__).resolve().parent


def filtered_manifest(plan, manifest):
    if plan.get('task_name') != TASK:
        raise ValueError('Background exclusion belongs only to '+TASK)
    components = plan['runtime_components']
    if len(components) != 1:
        raise ValueError('Exactly one target declaration is required')
    declaration = components[0]
    product = declaration['parameters']['product']
    identity = declaration['instance_id']
    objects = manifest['objects']
    paths = [o.get('path') for o in objects]
    if any(not isinstance(p, str) or not p.startswith('/World/') or
           any(x in ('', '.', '..') for x in p.split('/')[1:]) for p in paths):
        raise ValueError('Every object needs a canonical /World/ instance path')
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate manifest instance paths')
    matches = [o for o in objects if o.get('id') == product]
    owned = [o for o in matches if o.get('task_id') == identity]
    if len(owned) != 1:
        raise ValueError('Exactly one explicitly owned target is required')
    target = owned[0]
    excluded = []
    for obj in matches:
        if obj is target:
            continue
        if obj.get('task_id') or obj.get('task_role') not in ('background', 'background_stock'):
            raise ValueError('Same product is not unowned background: '+obj['path'])
        path = obj['path']
        if any(p != path and (p.startswith(path+'/') or path.startswith(p+'/')) for p in paths):
            raise ValueError('Background instance overlaps another manifest root: '+path)
        # Reject references from task records even if the object role is stale.
        for task in manifest.get('tasks', []):
            refs = list(task.get('object_paths', [])) + [task.get('target_path')]
            if path in refs:
                raise ValueError('Background is referenced by a registered task: '+path)
        excluded.append(obj)
    removed = {o['path'] for o in excluded}
    result = deepcopy(manifest)
    result['objects'] = [o for o in result['objects'] if o['path'] not in removed]
    require_unique_target(result, product, target['path'])
    previous = result.get('interference_background_exclusion', {})
    result['interference_background_exclusion'] = dict(
        task_name=TASK, product=product, target_path=target['path'],
        excluded_paths=list(dict.fromkeys(previous.get('excluded_paths', []) + sorted(removed))),
        source_scene_unchanged=True, method='episode_layer_inactive_background',
        physical_validation=False)
    return result, excluded


def prepare(plan, stage, manifest):
    """Existing scene_preparation hook; only author to an episode/session layer."""
    result, excluded = filtered_manifest(plan, manifest)
    target = result['interference_background_exclusion']['target_path']
    prim = stage.GetPrimAtPath(target)
    if not prim or not prim.IsActive():
        raise ValueError('Owned target is absent or inactive in the scene')
    for obj in excluded:
        if not stage.GetPrimAtPath(obj['path']):
            raise ValueError('Background manifest path is absent: '+obj['path'])
    from pxr import Usd
    layer = stage.GetSessionLayer()
    before = layer.ExportToString()
    try:
        with Usd.EditContext(stage, layer):
            for obj in excluded:
                if not stage.GetPrimAtPath(obj['path']).SetActive(False):
                    raise RuntimeError('Cannot exclude background: '+obj['path'])
    except Exception:
        layer.ImportFromString(before)
        raise
    return result


def prepare_reserved(plan, stage, manifest):
    """Task hook: unique target AND a frozen, same-level geometry reservation."""
    from .initialization import prepare_reservation
    layer = stage.GetSessionLayer()
    before = layer.ExportToString()
    try:
        return prepare_reservation(plan, stage, prepare(plan, stage, manifest))
    except Exception:
        layer.ImportFromString(before)
        raise


def generate(plan, manifest, output, *, reserve=True):
    """Generate a private room layer from a composed room, with no GPU needed."""
    output = Path(output).resolve()
    if output == HERE or not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Output must be a fresh directory inside this task')
    filtered_manifest(plan, manifest)  # Reject conflicts before writing anything.
    from pxr import Usd
    source = Path(manifest['scene']).resolve(strict=True)
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise ValueError('Cannot open source room')
    sources = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
               for l in stage.GetUsedLayers() if l.realPath}
    # reserve=False exists for the older filter-only composition regression;
    # the public CLI and production hook always require a reservation.
    result = (prepare_reserved if reserve else prepare)(plan, stage, manifest)
    output.mkdir(parents=True)
    scene = output/'room.usda'
    layer = stage.GetSessionLayer()
    # The private root contains only inactive opinions and a source sublayer.
    for key in ('metersPerUnit', 'upAxis', 'defaultPrim', 'timeCodesPerSecond',
                'framesPerSecond', 'startTimeCode', 'endTimeCode'):
        if stage.HasAuthoredMetadata(key):
            layer.pseudoRoot.SetInfo(key, stage.GetMetadata(key))
    layer.subLayerPaths = [str(source)]
    layer.Export(str(scene))
    result.update(scene=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest())
    result.setdefault('source_hashes', {}).update(sources)
    result['source_hashes'][str(scene)] = result['scene_sha256']
    reopened = Usd.Stage.Open(str(scene))
    receipt = result['interference_background_exclusion']
    if not reopened.GetPrimAtPath(receipt['target_path']).IsActive():
        raise RuntimeError('Exported room lost its target')
    if any(reopened.GetPrimAtPath(p).IsActive() for p in receipt['excluded_paths']):
        raise RuntimeError('Exported room retained excluded background')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in sources.items()):
        raise RuntimeError('Source layer changed during preparation')
    (output/'composition.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from .adapter import resolve
    plan = resolve(json.loads(args.request.read_text()))
    result = generate(plan, json.loads(args.manifest.read_text()), args.output)
    print(json.dumps(result['interference_background_exclusion'], indent=2))


if __name__ == '__main__':
    main()

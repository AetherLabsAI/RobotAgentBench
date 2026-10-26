"""Select coffee placement with the unchanged Openend workspace composer.

Run only in the task-owned remote source snapshot: the existing composer writes
its own generated_rooms cache, which must remain inside this task's deployment.
No changes to Openend packing, collision policies or native acceptance rules.
"""
import argparse
import json
import os
from pathlib import Path
from copy import deepcopy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def bind_supports(recipe):
    """Preserve registered support identity; bind physical contacts separately.

    V2 inventory already declares collider_paths and a hash of its own path.
    Never copy a wrapper hash onto a child-path alias. Legacy inventories without
    that contract retain their existing unique, identical-bounds resolution.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    value = deepcopy(recipe)
    stage = Usd.Stage.Open(value['layout']['scene'])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    receipts = []
    def enabled(prim):
        return prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
    surfaces = value['layout']['surfaces']
    from openend.tasks.scene_requirements.surface_registry import SCHEMA, verify_surfaces
    registered = bool(surfaces) and all(s.get('support_schema') == SCHEMA for s in surfaces)
    if registered:
        verify_surfaces(stage, surfaces)
    for placement in value['plan']['placements']:
        root_path = placement['support']
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsActive():
            raise ValueError('Missing selected support '+root_path)
        if registered:
            matches = [s for s in surfaces if s['path'] == root_path]
            if len(matches) != 1:
                raise ValueError('Ambiguous registered support inventory: '+root_path)
            paths = matches[0].get('collider_paths', [])
            if not paths or len(paths) != len(set(paths)):
                raise ValueError('Missing or duplicate registered support colliders: '+root_path)
            for path in paths:
                child = stage.GetPrimAtPath(path)
                if not (path == root_path or path.startswith(root_path + '/')) or not child or not enabled(child):
                    raise ValueError('Invalid registered support collider: '+path)
            if not enabled(root):
                receipts.append(dict(surface_wrapper=root_path, collider_paths=list(paths),
                    support_inventory_unchanged=True, geometry_signature_preserved=True,
                    placement_unchanged=True, physics_unchanged=True))
            continue
        if enabled(root):
            continue
        children = [p for p in Usd.PrimRange(root) if enabled(p)]
        if len(children) != 1:
            raise ValueError('Selected support needs exactly one enabled descendant collider: '+root_path)
        child = children[0]
        before = cache.ComputeWorldBound(root).ComputeAlignedRange()
        after = cache.ComputeWorldBound(child).ComputeAlignedRange()
        error = max(abs(a-b) for x,y in ((before.GetMin(),after.GetMin()),(before.GetMax(),after.GetMax())) for a,b in zip(x,y))
        if error > 1e-5:  # Same existing surface inventory verification tolerance.
            raise ValueError('Support wrapper and collider geometry differ: '+root_path)
        path = str(child.GetPath())
        matches = [s for s in value['layout']['surfaces'] if s['path'] == root_path]
        if len(matches) != 1:
            raise ValueError('Ambiguous support inventory')
        value['layout']['surfaces'].append(dict(matches[0], path=path, source_wrapper=root_path))
        placement['support'] = path
        receipts.append(dict(surface_wrapper=root_path, collider=path, max_bounds_error_m=error,
                             placement_unchanged=True, physics_unchanged=True))
    return value, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ready-pose', required=True)
    parser.add_argument('--sensor-file', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    # Refuse accidental use of a production checkout/shared generated_rooms.
    if (ROOT.name != 'source' or not ROOT.parent.name.startswith('deployment_')
            or ROOT.parent.parent.name != 'agent_probe_v1'):
        raise ValueError('Use the task-owned isolated source snapshot, not the shared repository')
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Fresh output inside agent_probe_v1 required')
    out.mkdir(parents=True)
    (out/'tmp').mkdir()
    os.environ['TMPDIR'] = str(out/'tmp')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # The unchanged room composer expects the historical root/api-protocol
    # nesting. This snapshot already contains those exact read-only inputs at
    # root. Supply aliases for asset reads only; no source or parameters edited.
    aliases = ROOT/'api-protocol'
    aliases.mkdir(exist_ok=True)
    for name in ('resources', 'all_assets'):
        target = ROOT/name
        if not target.is_dir():
            raise ValueError('Missing deployed source dependency: '+str(target))
        link = aliases/name
        if not link.exists():
            link.symlink_to(Path('..')/name, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError('Unexpected source alias: '+str(link))
    from short_tasks import registry
    from short_tasks.scene import plan_scene, LocalFriendProvider, verify_manifest
    from .prepare_native import prepare
    request = json.loads((HERE.parent/'tasks/prepare_coffee/request_v2.json').read_text())
    request.update(layout_id=args.layout, seed=args.seed)
    write(out/'request.json', request)
    report = dict(status='planning', layout_id=args.layout, seed=args.seed,
                  placement_owner='unchanged_openend_workspace_planner', native_physics_validated=False)
    try:
        planned = plan_scene(request)
        write(out/'workspace_plan.json', planned)
        if planned['scene_status'] != 'planar_candidate':
            raise ValueError('No workspace placement candidate')
        recipe, binding_receipts = bind_supports(planned['recipe'])
        write(out/'support_bindings.json', binding_receipts)
        report['support_binding_receipts'] = binding_receipts
        provider = LocalFriendProvider()
        manifest = provider.compose(recipe, out/'composed')
        verify_manifest(manifest)
        if manifest['layout_id'] != args.layout or manifest.get('scene_generated') is not True:
            raise ValueError('Unexpected generated layout')
        registry.bind(planned['task_plan'], manifest)
        prepare(out/'composed/composition.json', args.ready_pose, args.sensor_file,
                out/'native', request=request, local_random=True)
        report.update(status='prepared_not_native_accepted',
            placement=planned['recipe']['plan']['placements'],
            station_clearance=manifest['station_clearance'],
            launch=str(out/'native/prepared/launch_v2.json'))
        report['initial_randomization'] = json.loads((out/'native/local_random/randomization.json').read_text())
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(out/'workspace_receipt.json', report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

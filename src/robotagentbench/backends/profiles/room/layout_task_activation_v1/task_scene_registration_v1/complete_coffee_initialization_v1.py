"""Scene-owned additive coffee initialization; no scoring or simulation startup.

Use existing all_assets rigid beans and the standard 1-2 bag-local initial poses.
Do not alter existing object poses, physics parameters, assets, or old previews.
Coffee liquid remains visual_cpu_v1; this does not construct the live device.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
BEANS = '/World/RoomEpisode/CoffeeBeans'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(flat):
    require(len(flat) == 16, 'Expected a source 4x4 row-major matrix')
    return Gf.Matrix4d(*flat)


def world(stage, path):
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))


def fingerprint(stage, root):
    """All physics values and local geometry, independent of reference root path."""
    rows = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        attrs = [(a.GetName(), str(a.Get())) for a in p.GetAttributes()
                 if a.GetName().startswith(('physics:', 'physx', 'xformOp'))
                 or a.GetName() in ('points', 'faceVertexCounts', 'faceVertexIndices')]
        # Root pose is the only property intentionally supplied by initialization.
        if str(p.GetPath()) == root:
            attrs = [(k, v) for k, v in attrs if not k.startswith('xformOp')]
        rows.append((str(p.GetPath())[len(root):], p.GetTypeName(),
                     list(p.GetAppliedSchemas()), sorted(attrs)))
    return rows


def load_source(repo):
    from evaluation.tasks.coffee_machine import specs
    source_path = specs.EPISODES / '1-2' / specs.episodes('1-2')[0] / 'public_binding.json'
    source = json.loads(source_path.read_text())
    require(len(source['beans']) == 100, 'Standard source must contain 100 beans')
    cluster_path = repo / 'resources/shared_assets/coffee_v1/cluster.usda'
    cluster = Usd.Stage.Open(str(cluster_path))
    assets, variants = {}, []
    for i in range(100):
        prim = cluster.GetPrimAtPath(f'/CoffeeAssets/CoffeeBeans/Bean_{i:03d}')
        refs = prim.GetMetadata('references').GetAddedOrExplicitItems()
        require(len(refs) == 1, 'Expected one source bean asset')
        variant = Path(refs[0].assetPath).stem
        require(variant in ('bean_a', 'bean_b', 'bean_c'), 'Unknown bean variant')
        variants.append('coffee_' + variant)
        entry = repo / 'all_assets' / ('coffee_' + variant) / 'asset.usda'
        if variant not in assets:
            asset = Usd.Stage.Open(str(entry))
            native_path = cluster_path.parent / (variant + '.usda')
            native = Usd.Stage.Open(str(native_path))
            require(fingerprint(asset, '/Asset') == fingerprint(native, '/Asset'),
                    'all_assets bean differs from source geometry/physics: ' + variant)
            assets[variant] = asset
    return source_path, source, cluster_path, variants


def complete(manifest_path, folder, repo, source_data):
    from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import ReadOnlySceneBinding, resolve
    source_path, source, cluster_path, variants = source_data
    manifest_path = Path(manifest_path).resolve(strict=True)
    original = json.loads(manifest_path.read_text())
    scene = Path(original['scene']).resolve(strict=True)
    old = Usd.Stage.Open(str(scene))
    require(old is not None, 'Cannot open input room')
    resolved = resolve(old, original)
    require(not old.GetPrimAtPath(BEANS) and 'beans' not in resolved['roles'],
            'Beans already exist: refusing duplicate initialization')
    require(not any(p.GetName().startswith('Bean_') for p in old.Traverse()),
            'Unbound existing beans require explicit ownership')
    # This fixes missing outer metadata only, not an actual authored centimetre/Y-up scene.
    for key, expected in (('upAxis', 'Z'), ('metersPerUnit', 1.0)):
        require(not old.HasAuthoredMetadata(key) or old.GetMetadata(key) == expected,
                'Conflicting authored stage metadata: ' + key)
        require(any(l.pseudoRoot.HasInfo(key) and l.pseudoRoot.GetInfo(key) == expected
                    for l in old.GetLayerStack() if l.realPath),
                'No underlying layout evidence for ' + key)
    input_hashes = {str(p): digest(p) for p in [manifest_path, source_path, cluster_path]
                    + [Path(l.realPath) for l in old.GetUsedLayers() if l.realPath]}
    old_poses = {str(p.GetPath()): world(old, str(p.GetPath())) for p in old.Traverse()
                 if p.IsA(UsdGeom.Xformable)}
    source_bag = matrix(source['world']['bag'])
    bag = world(old, resolved['roles']['bag'])
    require(abs(abs(bag.GetDeterminant()) - 1) < 1e-8, 'Scaled bag frame unsupported')
    folder.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.CreateNew(str(folder / 'room.usda'))
    stage.GetRootLayer().subLayerPaths = [str(scene)]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.Xform.Define(stage, BEANS)
    parent_inverse = world(stage, BEANS).GetInverse()
    max_relative_error = 0.0
    asset_stages = {}
    for i, (flat, variant) in enumerate(zip(source['beans'], variants)):
        path = f'{BEANS}/Bean_{i:03d}'
        entry = repo / 'all_assets' / variant / 'asset.usda'
        prim = stage.DefinePrim(path, 'Xform')
        prim.GetReferences().AddReference(str(entry))
        relative = matrix(flat) * source_bag.GetInverse()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(relative * bag * parent_inverse)
        require(prim.HasAPI(UsdPhysics.RigidBodyAPI), 'Missing bean rigid body')
        require(prim.GetAttribute('physics:rigidBodyEnabled').Get() is True
                and prim.GetAttribute('physics:kinematicEnabled').Get() is not True,
                'Bean is not dynamic')
        asset_stages.setdefault(variant, Usd.Stage.Open(str(entry)))
        require(fingerprint(stage, path) == fingerprint(asset_stages[variant], '/Asset'),
                'Bean geometry/physics changed during composition')
        got = world(stage, path) * bag.GetInverse()
        max_relative_error = max(max_relative_error,
                                 max(abs(got[r][c] - relative[r][c]) for r in range(4) for c in range(4)))
    require(max_relative_error < 1e-10, 'Bean bag-local initial pose not preserved')
    require(all(Gf.IsClose(world(stage, p), pose, 1e-12) for p, pose in old_poses.items()),
            'An existing room object moved')
    # Fail if this additive layer authored physics values or anything outside its bean group.
    for p in stage.GetRootLayer().rootPrims:
        def inspect(spec):
            for prop in spec.properties:
                require(not prop.name.startswith(('physics:', 'physx')), 'Physics override prohibited')
            for child in spec.nameChildren:
                inspect(child)
        inspect(p)
    stage.GetRootLayer().Save()
    report = deepcopy(original)
    report['scene'] = str(folder / 'room.usda')
    report['objects'].append(dict(id='coffee_beans', path=BEANS, count=100,
                                 entries=sorted(str(repo / 'all_assets' / v / 'asset.usda') for v in set(variants))))
    report['status'] = 'coffee_initial_geometry_complete_native_settling_and_runtime_pending'
    report['coffee_initialization'] = dict(
        owner='unified_scene', source_manifest=str(manifest_path), source_binding=str(source_path),
        source_binding_sha256=digest(source_path), source_api='1-2',
        transform='source_bean_world * inverse(source_bag_world) * existing_room_bag_world',
        bean_count=100, bean_variant_counts={v: variants.count(v) for v in sorted(set(variants))},
        liquid_mode='visual_cpu_v1', gpu_fluid_solver=False, initial_coffee_ml=0,
        persistent_device_constructed=False, scoring_initialized=False,
        existing_object_poses_unchanged=True, physics_parameters_unchanged=True,
        physical_settling_test=False, manipulation_test=False)
    binding = ReadOnlySceneBinding(stage, report)
    geometry = binding.geometry_snapshot()
    require(geometry['beans_bound'] == geometry['beans_geometrically_in_bag'] == 100,
            'Bean count or static bag containment failed')
    binding.task_binding('initialization-probe', 'coffee')
    require(not any('Particle' in str(p.GetTypeName()) or any('Particle' in a for a in p.GetAppliedSchemas())
                    for p in Usd.PrimRange(stage.GetPrimAtPath(BEANS))), 'Unexpected fluid/particle schema')
    require(all(digest(p) == h for p, h in input_hashes.items()), 'Input source changed')
    report['limitations'] = list(original.get('limitations', [])) + [
        'Coffee initialized statically only; no native settling or continuous actions tested',
        'visual_cpu_v1 declared, persistent animation device and live scoring bridge not connected']
    (folder / 'composition.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    audit = dict(layout_id=report['layout_id'], scene=report['scene'], static_geometry=geometry,
                 existing_xforms_checked=len(old_poses), max_bean_bag_local_error=max_relative_error,
                 metadata=dict(up_axis='Z', meters_per_unit=1), input_sha256=input_hashes,
                 all_source_files_unchanged=True, bean_source_geometry_physics_match=True,
                 no_physics_parameter_overrides=True, static_checks_passed=True,
                 native_physics_test=False, runtime_registered=False, scoring_ready=False,
                 pending=binding.requirements())
    (folder / 'coffee_initialization_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('input_sha256', 'static_geometry')}), flush=True)
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo, output = args.repo.resolve(strict=True), args.output.resolve()
    require(output.is_relative_to(HERE) and output != HERE and not output.exists(),
            'Use a new output directory inside the unified generator directory')
    sys.path.insert(0, str(repo))
    source = load_source(repo)
    ids = [json.loads(Path(m).read_text())['layout_id'] for m in args.manifest]
    require(len(set(ids)) == len(ids), 'Duplicate layout IDs')
    rows = [complete(m, output / f'layout{n:02d}', repo, source) for m, n in zip(args.manifest, ids)]
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Prepare the first two Short Tasks from a pinned, relocated room on CPU.

The room shell, furniture and robot are retained from an explicit readonly
basis. Task assemblies are referenced from this checkout's actual all_assets,
at the basis's original initial transforms. This is fixed-layout preparation,
not a random scene generator or a native physics acceptance certificate.
Run with evaluation/tools/run_cpu_usd.sh or an existing CPU USD interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TASK_ASSETS = {
    'prepare_coffee': ('coffee_machine', 'coffee_cup_blue_compact', 'coffee_bag'),
    'bake_bread': ('air_fryer', 'paper_plate_task2', 'cake_plain_yellow', 'cake_frosted_white'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked_dependencies(path):
    """Reject missing file dependencies; retain explicit engine metadata scope.

    Living50 includes Unreal material/source asset metadata that is not a USD
    composition file. NVIDIA MDL modules resolve through Kit's material search
    paths, not the plain CPU USD resolver. Only known installed modules pass.
    """
    from pxr import UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    special = []
    missing = []
    for name in unresolved:
        if name.startswith('/Game/') and not name.lower().endswith(('.usd', '.usda', '.usdc', '.usdz')):
            special.append(dict(asset=name, reason='non_runtime_unreal_source_metadata'))
        elif name in ('OmniPBR.mdl', 'OmniPBR_Opacity.mdl', 'OmniGlass.mdl'):
            spec = importlib.util.find_spec('isaacsim')
            module = Path(spec.origin).parent / 'kit/mdl/core/Base' / name if spec else None
            if module and module.is_file():
                special.append(dict(asset=name, reason='installed_kit_mdl_search_path', resolved=str(module), sha256=sha(module)))
            else:
                missing.append(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError('Unresolved file dependencies: ' + repr(missing[:20]))
    return layers, assets, special


def room_overview(source, basis_manifest, pinned):
    """Restore omitted surface inventory only from this exact layout receipt.

    The old prepared bundle predates overview metadata. Its kitchen consists
    of adjacent support colliders; declaring just one collider loses measured
    bag contacts on its neighbour. Preserve the friend's connected-counter
    classifier by providing its actual scene inventory, not a scoring waiver.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    from openend.tasks.scene_requirements.surface_registry import verify_surfaces
    existing = basis_manifest.get('overview', {})
    if existing.get('surfaces'):
        return existing, dict(status='existing_declared_inventory')
    profile_path = REPO / 'openend/tasks/scene_requirements/placement_sources/random-layouts-v1.json'
    extract_path = REPO / 'layout_task_activation_v1/extract_random_layouts_v1.py'
    profiles = json.loads(profile_path.read_text())['layouts']
    rows = [r for r in profiles if r['id'] == basis_manifest.get('layout_id')]
    if len(rows) != 1:
        raise ValueError('Missing surface inventory and no unique layout profile')
    profile = rows[0]
    if basis_manifest.get('source_sha256') != profile['scene_sha256']:
        raise ValueError('Omitted surface inventory cannot be restored from a different layout revision')
    pinned[str(profile_path)] = sha(profile_path)
    pinned[str(extract_path)] = sha(extract_path)
    verify_surfaces(source, profile['surfaces'])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    verified = []
    for row in profile['surfaces']:
        prim = source.GetPrimAtPath(row['path'])
        if not prim or not prim.IsActive():
            raise ValueError('Declared source surface is missing: ' + row['path'])
        box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        actual_bounds = np.asarray([box.GetMin(), box.GetMax()])
        error = float(np.max(np.abs(actual_bounds - np.asarray([row['min'], row['max']]))))
        if error > 1e-5:
            raise ValueError('Surface geometry differs from matching layout profile: ' + row['path'])
        # Exact mesh/collider signature above verifies the cached face-derived
        # height; do not reintroduce the old vertex-hull height selector here.
        measured_top = row['top']
        verified.append(dict(path=row['path'], max_bound_error_m=error,
                             broad_support_top_m=float(measured_top),
                             whole_prim_max_z_m=float(actual_bounds[1, 2])))
    overview = dict(bounds=profile['bounds'], floors=profile['floors'],
                    surfaces=profile['surfaces'], blockers=profile['blockers'],
                    furniture=[dict(path=r['path'], bounds=[r['min'], r['max']])
                               for r in profile['blockers'] if source.GetPrimAtPath(r['path'])])
    return overview, dict(status='restored_exact_layout_surface_inventory',
                          source_profile=str(profile_path), source_profile_sha256=sha(profile_path),
                          layout_scene_sha256=profile['scene_sha256'], verified_surfaces=verified,
                          object_poses_changed=False, contact_classifier_changed=False)


def prepare(task_name, source_launch, output, *, target_variant='no_sugar', stand_off_m=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    if task_name not in TASK_ASSETS:
        raise ValueError('Only prepare_coffee and bake_bread are supported')
    if target_variant not in ('no_sugar', 'sugar'):
        raise ValueError('Target variant must be no_sugar or sugar')
    if task_name != 'prepare_coffee':
        stand_off_m = 1.10 if stand_off_m is None else stand_off_m
        if not math.isfinite(stand_off_m) or not 0.8 <= stand_off_m <= 1.5:
            raise ValueError('Startup standing distance must be within 0.8 to 1.5 metres')
    source_launch = Path(source_launch).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Preparation requires a fresh output directory')
    basis_launch = json.loads(source_launch.read_text())
    basis_manifest_path = Path(basis_launch['manifest']).resolve(strict=True)
    basis_ready_path = Path(basis_launch['ready_pose']).resolve(strict=True)
    basis_manifest = json.loads(basis_manifest_path.read_text())
    basis_scene_path = Path(basis_manifest['scene']).resolve(strict=True)
    source = Usd.Stage.Open(str(basis_scene_path))
    if not source:
        raise ValueError('Cannot compose the explicit basis scene')
    if UsdGeom.GetStageMetersPerUnit(source) != 1 or UsdGeom.GetStageUpAxis(source) != 'Z':
        raise ValueError('Basis must use metres and Z up')
    if not source.GetPrimAtPath('/World/Robot') or not source.GetPrimAtPath('/World/PhysicsScene'):
        raise ValueError('Basis must contain the real robot and physics scene')
    physics = source.GetPrimAtPath('/World/PhysicsScene')
    if physics.GetAttribute('physxScene:enableGPUDynamics').Get() is not False:
        raise ValueError('This prototype preserves the friend CPU scene policy')
    if physics.GetAttribute('physxScene:timeStepsPerSecond').Get() != 60:
        raise ValueError('Expected the original 60 Hz startup scene')
    dependencies, dependency_assets, special_dependencies = checked_dependencies(basis_scene_path)
    pinned = {str(source_launch): sha(source_launch), str(basis_manifest_path): sha(basis_manifest_path),
              str(basis_ready_path): sha(basis_ready_path)}
    pinned.update({layer.realPath: sha(layer.realPath) for layer in dependencies if layer.realPath})
    pinned.update({str(p): sha(p) for p in dependency_assets if Path(p).is_file()})
    selected = {}
    for asset_id in TASK_ASSETS[task_name]:
        rows = [row for row in basis_manifest['objects'] if row['id'] == asset_id]
        if len(rows) != 1:
            raise ValueError('Missing or ambiguous initial object: ' + asset_id)
        selected[asset_id] = rows[0]
        entry = REPO / 'all_assets' / asset_id / 'asset.usda'
        layers, assets, special = checked_dependencies(entry)
        special_dependencies.extend(special)
        pinned.update({layer.realPath: sha(layer.realPath) for layer in layers if layer.realPath})
        pinned.update({str(p): sha(p) for p in assets if Path(p).is_file()})
    kind = 'coffee' if task_name == 'prepare_coffee' else 'fryer'
    overview, overview_receipt = room_overview(source, basis_manifest, pinned)
    support = dict(basis_manifest['supports'][kind])
    measured_support = [s for s in overview['surfaces'] if s['path'] == support['path']]
    if len(measured_support) != 1:
        raise ValueError('Task support lacks a unique measured surface inventory entry')
    support['actual_top_m'] = measured_support[0]['top']
    support_prim = source.GetPrimAtPath(support['path'])
    if not support_prim or not support_prim.IsActive():
        raise ValueError('Task source support does not exist')
    # Preserve the initial assembly transforms before hiding its source subtree.
    cache = UsdGeom.XformCache()
    original_matrices = {asset_id: cache.GetLocalToWorldTransform(source.GetPrimAtPath(row['path']))
                         for asset_id, row in selected.items()}
    output.mkdir(parents=True, exist_ok=False)
    out = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(out))
    stage.GetRootLayer().subLayerPaths = [str(basis_scene_path)]
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    removed = {'/World/RoomEpisode', '/World/Assets/DiningTableMicrowave'}
    # These are task-owned targets/devices. Ordinary shell/furniture remain.
    removed.update(row['path'] for row in basis_manifest['objects']
                   if not row['path'].startswith('/World/RoomEpisode/'))
    for path in sorted(removed):
        if path == support['path'] or support['path'].startswith(path + '/'):
            raise ValueError('Task support is also marked as an unrelated task object')
        if stage.GetPrimAtPath(path):
            stage.OverridePrim(path).SetActive(False)
    root = '/World/ShortTask'
    UsdGeom.Xform.Define(stage, root)
    rows = []
    max_matrix_error = 0.0
    for asset_id, original in selected.items():
        path = root + '/' + Path(original['path']).name
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(original_matrices[asset_id])
        asset = stage.DefinePrim(path + '/Asset')
        entry = REPO / 'all_assets' / asset_id / 'asset.usda'
        asset.GetReferences().AddReference(str(entry))
        actual = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        error = max(abs(actual[i][j] - original_matrices[asset_id][i][j]) for i in range(4) for j in range(4))
        max_matrix_error = max(max_matrix_error, error)
        rows.append(dict(original, path=path, entry=str(entry), task_id=task_name,
                         source_initial_wrapper=original['path'], initial_transform_preserved=True))
    if task_name == 'prepare_coffee':
        beans = [row for row in basis_manifest['objects'] if row['id'] == 'coffee_beans']
        if len(beans) != 1:
            raise ValueError('Explicit source coffee bean inventory required')
        beans_source = beans[0]['path']
        beans_target = root + '/CoffeeBeans'
        flat = source.Flatten()
        if not Sdf.CopySpec(flat, beans_source, stage.GetRootLayer(), beans_target):
            raise ValueError('Cannot copy the exact initial bean subtree')
        # The source bean wrapper may have a parent transform. Preserve its
        # world matrix under the new identity parent, without moving a bean.
        UsdGeom.Xformable(stage.GetPrimAtPath(beans_target)).MakeMatrixXform().Set(
            cache.GetLocalToWorldTransform(source.GetPrimAtPath(beans_source)))
        bean_paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(beans_target))
                      if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bean_paths) != 100:
            raise ValueError('Expected the original 100 dynamic coffee beans')
        for target_path in bean_paths:
            source_path = beans_source + target_path[len(beans_target):]
            before = cache.GetLocalToWorldTransform(source.GetPrimAtPath(source_path))
            after = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(target_path))
            error = max(abs(after[i][j] - before[i][j]) for i in range(4) for j in range(4))
            max_matrix_error = max(max_matrix_error, error)
        rows.append(dict(beans[0], path=beans_target, task_id=task_name, count=len(bean_paths),
                         source_initial_wrapper=beans_source, initial_transform_preserved=True))
    if max_matrix_error > 1e-9:
        raise ValueError('An initial task transform changed')
    manifest = dict(schema='openend.short_task.scene.v1', episode_id=output.parent.name,
                    scene=str(out), source_layout=basis_manifest.get('source_layout'),
                    layout_id=basis_manifest.get('layout_id'), seed=basis_manifest.get('seed'),
                    status='prepared_static_not_native_accepted',
                    preparation_mode='derived_fixed_layout_initial_state',
                    objects=rows, supports={kind: support}, overview=overview,
                    tasks=[dict(id=task_name, type=kind, target_path=rows[0]['path'])],
                    short_task=dict(name=task_name, index=1 if kind == 'coffee' else 2,
                                    component_mode='visual_cpu_v1' if kind == 'coffee' else 'release145_v2',
                                    bread_variant=target_variant if kind == 'fryer' else None),
                    source_initial_states_preserved=True, source_sha256=pinned)
    from evaluation.tasks.air_fryer.continuous_openended_v1.native import physical_signature
    signatures = {}
    for row in rows:
        if row['id'] == 'coffee_beans':
            continue
        asset_stage = Usd.Stage.Open(row['entry'])
        actual = physical_signature(stage, row['path'] + '/Asset')
        approved = physical_signature(asset_stage, str(asset_stage.GetDefaultPrim().GetPath()))
        if actual != approved:
            raise ValueError('Current asset physical signature differs: ' + row['id'])
        signatures[row['id']] = 'matches_current_all_assets'
    if task_name == 'prepare_coffee':
        from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import ReadOnlySceneBinding
        binding = ReadOnlySceneBinding(stage, manifest)
        binding.task_binding(manifest['episode_id'], task_name)
        static = binding.geometry_snapshot()
        if static['beans_bound'] != 100 or static['beans_geometrically_in_bag'] != 100:
            raise ValueError('Source bean inventory is not wholly inside the source bag')
        if not static['placement']['outside_bay']:
            raise ValueError('Initial cup is already inside the receiving bay')
        bindings = binding.roles
    else:
        from evaluation.tasks.air_fryer.continuous_openended_v1.native import bind_room
        bindings = bind_room(stage, manifest, target_variant)
        static = dict(source='authored_usd_static_geometry_not_live_physx',
                      target_variant=target_variant, native_scoring_evidence_available=False)
    scene_layers, scene_assets, special = checked_dependencies(out)
    special_dependencies.extend(special)
    ready = json.loads(basis_ready_path.read_text())
    write(output / 'ready_pose.json', ready)
    request = dict(schema='openend.short_task.request.draft.v1', task_name=task_name,
                   seed=manifest['seed'], layout_id=manifest['layout_id'],
                   rule_version=('openend.short_task.prepare_coffee.v1' if kind == 'coffee'
                                 else 'short_tasks.bake_bread.cake_preview.v1'), options=(
                       {'bread_variant': target_variant, 'finish_on_small_plate': False}
                       if kind == 'fryer' else {}))
    write(output / 'request.json', request)
    fx, fy = map(float, support['front'])
    norm = math.hypot(fx, fy)
    if norm < 1e-8:
        raise ValueError('Task support has no usable front direction')
    fx, fy = fx / norm, fy / norm
    ox, oy = map(float, support['origin'])
    coffee_station = None
    if task_name == 'prepare_coffee':
        from short_tasks.tasks.prepare_coffee import initial_station
        # Measure the actual asset origin, never the visible bounding-box centre.
        machine_prim = stage.GetPrimAtPath(bindings['machine'])
        machine_xyz = UsdGeom.XformCache().GetLocalToWorldTransform(machine_prim).ExtractTranslation()
        if math.hypot(float(machine_xyz[0])-ox, float(machine_xyz[1])-oy) > 1e-5:
            raise ValueError('Coffee support origin differs from actual machine origin')
        coffee_station = initial_station.sample(manifest['seed'], [ox,oy], [fx,fy], stand_off_m=stand_off_m)
        pinned[str(Path(initial_station.__file__).resolve())] = sha(initial_station.__file__)
        initial = coffee_station['initial_base_xyyaw']
    else:
        initial = [ox + fx * stand_off_m, oy + fy * stand_off_m, math.atan2(-fy, -fx)]
    launch = dict(schema='openend.short_task.launch.v1', task_name=task_name,
                  manifest=str(output / 'manifest.json'), request=str(output / 'request.json'),
                  ready_pose=str(output / 'ready_pose.json'), initial_base_xyyaw=initial,
                  sensor=basis_launch.get('sensor', {'halfAngle': 45, 'range': 1.32}),
                  startup_spawn_scope='support_relative_standing_candidate_requires_native_collision_check',
                  source_robot_pose_preserved_in_scene=True,
                  startup_physics_hz=60, control_hz=20, active_physics_hz=240,
                  physics_backend='cpu', liquid_mode='visual_cpu_v1' if kind == 'coffee' else None)
    if coffee_station is not None:
        launch['initial_station'] = coffee_station
    report = dict(schema='openend.short_task.preparation_receipt.v1', status='prepared_static',
                  task_name=task_name, source_launch=str(source_launch), source_manifest=str(basis_manifest_path),
                  source_scene=str(basis_scene_path), preparation_mode=manifest['preparation_mode'],
                  removed_source_task_roots=sorted(removed), bindings=bindings,
                  surface_inventory=overview_receipt,
                  asset_physical_signatures=signatures, max_initial_matrix_error=max_matrix_error,
                  static_geometry=static, initial_base_xyyaw=initial,
                  task_object_pose_writes_during_rollout=0, native_physics_run=False,
                  native_initial_acceptance=False, task_success=False,
                  dependency_layers=len(scene_layers), dependency_assets=len(scene_assets),
                  unresolved_file_dependencies=[], external_engine_metadata_and_installed_modules=special_dependencies,
                  source_sha256=pinned,
                  source_files_unchanged=all(sha(path) == digest for path, digest in pinned.items()))
    if coffee_station is not None:
        report['initial_station'] = coffee_station
    if not report['source_files_unchanged']:
        raise ValueError('Source inputs changed during preparation')
    stage.GetRootLayer().Save()
    manifest['scene_sha256'] = sha(out)
    write(output / 'manifest.json', manifest)
    write(output / 'launch.json', launch)
    write(output / 'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-name', required=True, choices=tuple(TASK_ASSETS))
    parser.add_argument('--source-launch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-variant', choices=('no_sugar', 'sugar'), default='no_sugar')
    args = parser.parse_args()
    report = prepare(args.task_name, args.source_launch, args.output, target_variant=args.target_variant)
    print(json.dumps({k: report[k] for k in ('status', 'task_name', 'bindings', 'native_physics_run')}, indent=2))


if __name__ == '__main__':
    main()

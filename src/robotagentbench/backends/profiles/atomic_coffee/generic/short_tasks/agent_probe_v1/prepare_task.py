"""Trusted registry-driven preparation, separate from the robot action protocol.

Run in an isolated deployment_* source tree owned by agent_probe_v1. A launch
is a static preparation result, never evidence of native or action acceptance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def range_initializer(task_name):
    """Trusted task-local reset samplers; no caller-supplied module names."""
    if task_name == 'bake_bread':
        from short_tasks.tasks.bake_bread import prepare_initial
        return prepare_initial
    if task_name == 'retrieve_from_drawer':
        from short_tasks.tasks.retrieve_from_drawer import prepare_initial
        return prepare_initial
    if task_name == 'clear_table_trash':
        from short_tasks.tasks.clear_table_trash import prepare_initial
        return prepare_initial
    if task_name == 'hang_cap':
        from short_tasks.tasks.hang_cap import prepare_initial
        return prepare_initial
    if task_name == 'stock_wooden_tray':
        from short_tasks.tasks.stock_wooden_tray import prepare_initial
        return prepare_initial
    if task_name == 'fridge_to_heated_box':
        from short_tasks.tasks.fridge_to_heated_box import prepare_initial
        return prepare_initial
    return None


def admit_request(request):
    from short_tasks import registry
    from short_tasks.common.contracts import TaskError
    plan = registry.resolve(request)
    if not plan['available']:
        error = plan['unavailable']
        raise TaskError(error['message'], kind=error['kind'], reason=error['reason'])
    if registry.is_gpu_cloth_preview(plan):
        raise TaskError('Use the existing GPU cloth scene owner for this preview',
                        kind='load', reason='gpu_cloth_scene_owner_required')
    registry.verify_sources(plan)
    return plan


def check_output(output):
    if (ROOT.name != 'source' or not ROOT.parent.name.startswith('deployment_')
            or ROOT.parent.parent.name != 'agent_probe_v1'):
        raise ValueError('Use an isolated task-owned deployment source tree')
    out = Path(output).resolve()
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    from short_tasks.registry import task_directory, CONTEXT_TASKS
    context_roots = {task_directory(name).resolve() for name in CONTEXT_TASKS}
    if out in (HERE, *context_roots) or not (out.is_relative_to(HERE) or any(out.is_relative_to(p) for p in context_roots) or permits(out)) or out.exists():
        raise ValueError('Fresh output inside agent_probe_v1 required')
    return out


def aliases():
    base = ROOT / 'api-protocol'
    base.mkdir(exist_ok=True)
    for name in ('resources', 'all_assets'):
        target = ROOT / name
        if not target.is_dir():
            raise ValueError('Missing deployed dependency: ' + str(target))
        link = base / name
        if not link.exists():
            link.symlink_to(Path('..') / name, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError('Unexpected source alias: ' + str(link))


def task_roles(plan, manifest):
    """Separate the composer's placement station ID from scorer ownership.

    Only the exact tray station emitted by the existing meal composer is
    translated. Unknown objects/owners still go through registry rejection.
    """
    value = deepcopy(manifest)
    owners = {t['id'] for t in plan['scene_request']['tasks']}
    changes = []
    if plan['task_name'] not in ('prepare_cutlery', 'carry_meal_tray'):
        return value, changes
    paths = {'meal_tray_large': ('/World/RoomEpisode/MealTray',),
             'cutlery_holder': ('/World/RoomEpisode/CutleryHolder',),
             'meal_fork': ('/World/RoomEpisode/Cutlery_fork_1','/World/RoomEpisode/Cutlery_fork_2'),
             'meal_knife': ('/World/RoomEpisode/Cutlery_knife_0',)}
    for row in value['objects']:
        if row.get('task_id') != 'meal_tray_station':
            continue
        if row.get('path') not in paths.get(row['id'],()):
            raise ValueError('Unknown object in meal-tray placement station')
        declared = [t for t in value['tasks'] if t.get('target_path') == row['path'] and t['id'] in owners]
        before = row.pop('task_id')
        row['placement_station_id'] = before
        if declared:
            if len(declared) != 1:
                raise ValueError('Ambiguous scene-owned target')
            owner = declared[0]['id']
            if row.get('score_owner_task_id', owner) != owner or row.get('meal_task_id', owner) != owner:
                raise ValueError('Target ownership conflict')
            row['task_id'] = owner
        elif row['id'] in ('meal_tray_large', 'cutlery_holder'):
            row['role'] = 'infrastructure'
        else:
            raise ValueError('Unclaimed cutlery cannot be silently added')
        changes.append(dict(path=row['path'], placement_station=before, task_owner=row.get('task_id'),
                            geometry_unchanged=True))
    return value, changes


def compose_room(request, plan, out, *, validate_cloth_station=False, validate_pack_station=False,
                 validate_cutlery_station=False, validate_tray_station=False,
                 validate_basket_station=False, sensor=None):
    from short_tasks.scene import plan_scene, LocalFriendProvider, _scene_imports
    from .prepare_workspace import bind_supports
    from importlib import import_module
    with _scene_imports():
        layouts = import_module('room_request').resolve(plan['scene_request'])['layout_candidates']
    attempts = []
    for index, layout in enumerate(layouts):
        candidate = dict(request, layout_id=layout)
        try:
            planned = plan_scene(candidate)
        except Exception as exc:
            if getattr(exc, 'reason', None) != 'layout_candidates_exhausted' or request.get('layout_id') is not None:
                raise
            attempts.append(dict(layout_id=layout, status='planar_rejected'))
            continue
        write(out / ('workspace_plan_%02d.json' % index), planned)
        recipe, support_receipts = bind_supports(planned['recipe'])
        if plan['task_name'] == 'bake_bread':
            recipe['short_task_fryer_initial'] = dict(
                schema='short_tasks.fryer_initial.v1',
                variant=plan['runtime_components'][0]['parameters']['bread_variant'])
        write(out / ('support_bindings_%02d.json' % index), support_receipts)
        destination = out / ('composed_%02d' % index)
        try:
            manifest = LocalFriendProvider().compose(recipe, destination)
        except Exception as exc:
            # Retry only a known placement infeasibility, never dependencies,
            # source drift, score bindings or arbitrary exceptions.
            prefix = 'Local CPU USD composer rejected the scene; evidence: '
            message = str(exc)
            log = Path(message[len(prefix):]) if message.startswith(prefix) else None
            from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import generated
            allowed = generated(ROOT / 'openend/tasks/scene_requirements/generated_rooms')
            detail = log.read_text() if log and log.resolve().is_relative_to(allowed) and log.is_file() else ''
            placement_failure = detail.rstrip().endswith('ValueError: Books do not fit on cloth without overlap')
            if (validate_cutlery_station or validate_tray_station or plan['task_name']=='bake_bread') and detail.rstrip().rsplit('\n',1)[-1].startswith('ValueError: Station scene overlap:'):
                clearance = log.parent / 'station_clearance.json'
                checks = read(clearance) if clearance.is_file() else {}
                placement_failure = checks.get('passed') is False and bool(checks.get('failures'))
            if plan['task_name']=='bake_bread' and detail.rstrip().rsplit('\n',1)[-1].startswith(
                    'short_tasks.tasks.bake_bread.prepare_initial.NoFeasibleInitial: Approved fryer object proposals exhausted:'):
                placement_failure = True
            if not placement_failure or request.get('layout_id') is not None:
                raise
            attempts.append(dict(layout_id=layout, status='composition_placement_rejected', evidence=str(log)))
            write(out / 'layout_attempts.json', attempts)
            continue
        if plan['task_name'] in ('sofa_to_drawer','sofa_to_coffee_table'):
            from pxr import Usd
            from short_tasks.scene import prepare_runtime_scene
            station_module=import_module('short_tasks.tasks.'+plan['task_name']+'.prepare_initial')
            stage=Usd.Stage.Open(manifest['scene'])
            effective=prepare_runtime_scene(plan,stage,manifest)
            try:
                kwargs={'sensor':sensor} if plan['task_name']=='sofa_to_coffee_table' else {}
                station=station_module.choose(stage,effective,seed=request['seed'],**kwargs)
            except station_module.NoFeasibleInitial as exc:
                evidence=out/('station_rejected_%02d.json'%index)
                write(evidence,dict(layout_id=layout,reason=str(exc),ranges_widened=False))
                attempts.append(dict(layout_id=layout,status='robot_station_rejected',evidence=str(evidence)))
                write(out/'layout_attempts.json',attempts)
                if request.get('layout_id') is not None:raise
                continue
            write(out/('station_candidate_%02d.json'%index),station)
        initializer = range_initializer(plan['task_name'])
        if initializer is not None:
            from pxr import Usd
            from short_tasks.scene import prepare_runtime_scene
            stage = Usd.Stage.Open(manifest['scene'])
            effective = prepare_runtime_scene(plan, stage, manifest)
            try:
                station = initializer.choose(stage, effective, seed=request['seed'])
            except initializer.NoFeasibleInitial as exc:
                evidence = out / ('station_rejected_%02d.json' % index)
                write(evidence, dict(layout_id=layout, reason=str(exc), ranges_widened=False))
                attempts.append(dict(layout_id=layout, status='robot_station_rejected', evidence=str(evidence)))
                write(out / 'layout_attempts.json', attempts)
                if request.get('layout_id') is not None:
                    raise
                continue
            write(out / ('station_candidate_%02d.json' % index), station)
        station_module = None
        if validate_basket_station and plan['task_name'] == 'table_to_storage_basket':
            from short_tasks.tasks.table_to_storage_basket import initial_station as basket_station
            if basket_station.enabled(manifest):
                from short_tasks.tasks.table_to_storage_basket import prepare_initial as station_module
        elif validate_tray_station and plan['task_name'] == 'carry_meal_tray':
            from short_tasks.tasks.carry_meal_tray import prepare_initial as station_module
        elif validate_cutlery_station and plan['task_name'] == 'prepare_cutlery':
            from short_tasks.tasks.prepare_cutlery import prepare_initial as station_module
        elif validate_cloth_station and plan['task_name'] == 'clear_tablecloth':
            from short_tasks.tasks.clear_tablecloth import prepare_initial as station_module
        elif validate_pack_station and plan['task_name'] == 'pack_object':
            from short_tasks.tasks.pack_object import initial_station as pack_station
            if pack_station.enabled(manifest):
                from short_tasks.tasks.pack_object import prepare_initial as station_module
        if station_module is not None:
            try:
                station = station_module.check_room_candidate(manifest, seed=request['seed'])
            except station_module.NoFeasibleInitial as exc:
                evidence = out / ('station_rejected_%02d.json' % index)
                write(evidence, dict(layout_id=layout, reason=str(exc),
                    source_scene=manifest['scene'], ranges_widened=False))
                attempts.append(dict(layout_id=layout, status='robot_station_rejected',
                                     evidence=str(evidence)))
                write(out / 'layout_attempts.json', attempts)
                if request.get('layout_id') is not None:
                    raise
                continue
            write(out / ('station_candidate_%02d.json' % index), station)
        attempts.append(dict(layout_id=layout, status='composed'))
        write(out / 'layout_attempts.json', attempts)
        return manifest, destination, attempts
    raise ValueError('No feasible real composition in the seeded layout list; see layout_attempts.json')


def localize(manifest, output):
    """Same three legacy texture relocations as the admitted coffee preparer."""
    from pxr import Usd, UsdUtils
    original = Usd.Stage.Open(manifest['scene'])
    if not original:
        raise ValueError('Cannot open composed scene')
    source_hashes = {l.realPath: sha(l.realPath) for l in original.GetUsedLayers() if l.realPath}
    prefix = 'D:/00_Sale/TableBasse/TableBasse/PBR_Textures/'
    root = Path('/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/assets_living_room/coffee_table/PBR_Textures')
    names = ('T_Coffee_Table_01_Wood_01_Metallic_4K.1001.png',
             'T_Coffee_Table_01_Wood_01_Normal_4k.1001.png',
             'T_Coffee_Table_01_Wood_01_Roughness_4K.1001.png')
    used = {}
    def relocate(asset):
        if asset in {prefix + n for n in names}:
            used[asset] = str((root / asset.removeprefix(prefix)).resolve(strict=True))
            return used[asset]
        return asset
    layer = original.Flatten()
    UsdUtils.ModifyAssetPaths(layer, relocate)
    scene = output / 'room.usdc'
    layer.Export(str(scene))
    result = deepcopy(manifest)
    result['scene'] = str(scene)
    result['scene_sha256'] = sha(scene)
    result.setdefault('source_hashes', {}).update(source_hashes)
    result['source_hashes'].update({p: sha(p) for p in used.values()})
    result['source_hashes'][str(scene)] = sha(scene)
    write(output / 'localization.json', dict(original_layers_sha256=source_hashes,
        texture_path_map=used, physics_parameters_changed=False, material_parameters_changed=False))
    return result


def initial_pose(stage, manifest, ready_pose, task_name, sensor):
    """Start facing the bound source target, using the measured folded body."""
    initializer = range_initializer(task_name)
    if initializer is not None:
        receipt = initializer.choose(stage, manifest, seed=manifest['seed'])
        return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    if task_name in ('sofa_to_drawer','sofa_to_coffee_table'):
        from importlib import import_module
        module=import_module('short_tasks.tasks.'+task_name+'.prepare_initial')
        kwargs={'sensor':sensor} if task_name=='sofa_to_coffee_table' else {}
        receipt=module.choose(stage,manifest,seed=manifest['seed'],**kwargs)
        return [*receipt['position'][:2],receipt['yaw_rad']],receipt
    if task_name == 'table_to_storage_basket':
        from short_tasks.tasks.table_to_storage_basket.initial_station import enabled
        if enabled(manifest):
            from short_tasks.tasks.table_to_storage_basket.prepare_initial import choose
            receipt = choose(stage, manifest, seed=manifest['seed'])
            return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    if task_name == 'carry_meal_tray':
        from short_tasks.tasks.carry_meal_tray.prepare_initial import choose
        receipt = choose(stage, manifest, seed=manifest['seed'])
        return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    if task_name == 'prepare_cutlery':
        from short_tasks.tasks.prepare_cutlery.prepare_initial import choose
        receipt = choose(stage, manifest, seed=manifest['seed'])
        return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    if task_name == 'clear_tablecloth':
        from short_tasks.tasks.clear_tablecloth.prepare_initial import choose
        receipt = choose(stage, manifest, seed=manifest['seed'])
        return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    if task_name == 'pack_object':
        from short_tasks.tasks.pack_object.initial_station import enabled
        if enabled(manifest):
            from short_tasks.tasks.pack_object.prepare_initial import choose
            receipt = choose(stage, manifest, seed=manifest['seed'])
            return [*receipt['position'][:2], receipt['yaw_rad']], receipt
    from openend.tasks.layout_repair_v1 import ready_envelope
    from . import target_spawn
    reference = read(ready_envelope.REFERENCE / 'navigation_result.json')
    if read(ready_pose) != reference['navigation_posture']:
        raise ValueError('Ready pose differs from the measured spawn envelope reference')
    receipt = target_spawn.choose(stage, manifest, task_name, sensor)
    receipt['posture_source'] = 'approved_generic_folded_ready_not_dataset_fit'
    receipt['posture_scope'] = ('approved_missing_specialized_pack_variant_fallback'
        if task_name == 'pack_object' else 'existing_measured_generic_ready')
    receipt['reference_hashes'] = {str(p): sha(p) for p in (
        ready_envelope.REFERENCE / 'navigation_result.json', ready_envelope.REFERENCE / 'native_poses.npz',
        Path(ready_envelope.__file__), Path(target_spawn.__file__), Path(ready_pose),
        *map(Path, receipt['target']['reference_files']))}
    return [*receipt['position'][:2], receipt['yaw_rad']], receipt


def prepare(request, ready_pose, sensor_file, output):
    from short_tasks import registry
    from short_tasks.scene import verify_manifest, prepare_runtime_scene
    plan = admit_request(request)
    if plan['task_name'] in registry.FAILURE_RECOVERY_TASKS:
        return registry.failure_recovery_adapter().prepare_registered(request, ready_pose, sensor_file, output)
    if plan['task_name'] in registry.ATOMIC_COFFEE_TASKS:
        return registry.atomic_coffee_adapter().prepare_registered(request, ready_pose, sensor_file, output)
    if plan['task_name'] == 'select_trash':
        from short_task_ContextReasoning.tasks.select_trash.registered_prepare import prepare_registered
        return prepare_registered(request, ready_pose, sensor_file, output)
    if plan['task_name'] in registry.CONTEXT_TASKS:
        from short_task_ContextReasoning.tasks.select_shelf.registered_prepare import prepare_registered
        return prepare_registered(request, ready_pose, sensor_file, output)
    out = check_output(output)
    ready_pose, sensor_file = Path(ready_pose).resolve(strict=True), Path(sensor_file).resolve(strict=True)
    sensor = read(sensor_file)['sensor']  # Explicit existing calibration, no guessed range.
    out.mkdir(parents=True)
    (out / 'tmp').mkdir()
    previous_tmp = tempfile.tempdir
    tempfile.tempdir = str(out / 'tmp')
    previous_env_tmp = os.environ.get('TMPDIR')
    os.environ['TMPDIR'] = str(out / 'tmp')
    report = dict(status='planning', task_name=plan['task_name'], native_physics_validated=False,
                  rule_version=plan['rule_version'], requested_layout=request.get('layout_id'),
                  seed=request['seed'], maximum=plan['score_policy']['maximum'])
    write(out / 'request.json', request)
    try:
        aliases()
        manifest, composed, attempts = compose_room(request, plan, out,
            validate_cloth_station=plan['task_name'] == 'clear_tablecloth',
            validate_pack_station=plan['task_name'] == 'pack_object',
            validate_cutlery_station=plan['task_name'] == 'prepare_cutlery',
            validate_tray_station=plan['task_name'] == 'carry_meal_tray',
            validate_basket_station=plan['task_name'] == 'table_to_storage_basket', sensor=sensor)
        verify_manifest(manifest)
        if manifest.get('scene_generated') is not True or manifest['layout_id'] != attempts[-1]['layout_id']:
            raise ValueError('Composition differs from selected layout')
        manifest, role_receipts = task_roles(plan, manifest)
        write(out / 'task_roles.json', role_receipts)
        bound = registry.bind(plan, manifest)
        report.update(actual_layout=manifest['layout_id'], layout_attempts=attempts)
        if plan['task_name'] == 'prepare_coffee':
            from .prepare_native import prepare as coffee_prepare
            # The old coffee owner requires the resolved layout in its input.
            # Preserve the user's random request separately above.
            resolved_request = dict(request, layout_id=manifest['layout_id'])
            coffee_prepare(composed / 'composition.json', ready_pose, sensor_file,
                           out / 'native', request=resolved_request, local_random=True)
            launch_path = out / 'native/prepared/launch_v2.json'
        else:
            from pxr import Usd
            from short_tasks.tools.prepare_scene import checked_dependencies
            from openend.tasks.all_objects_integration.workspace_physics import from_scene
            native = out / 'native'
            native.mkdir()
            manifest = localize(manifest, native)
            if plan['task_name'] == 'prepare_cutlery':
                from short_tasks.tasks.prepare_cutlery.utensil_initial import prepare_manifest, HERE as cutlery_home
                from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import cutlery
                manifest = prepare_manifest(manifest, cutlery(cutlery_home / 'initial_layers' / out.name / 'random.usda', out.name))
            stage = Usd.Stage.Open(manifest['scene'])
            physics = stage.GetPrimAtPath('/World/PhysicsScene')
            if (physics.GetAttribute('physxScene:enableGPUDynamics').Get() is not False
                    or physics.GetAttribute('physxScene:timeStepsPerSecond').Get() != 60):
                raise ValueError('Existing CPU/60 Hz scene startup required')
            layers, assets, special = checked_dependencies(Path(manifest['scene']))
            files = [l.realPath for l in layers if l.realPath] + list(assets)
            manifest['source_hashes'].update({str(p): sha(p) for p in files})
            # Validate on an anonymous session only. Do not bake the mechanism;
            # build_session must install it exactly once before PhysX attaches.
            effective = prepare_runtime_scene(bound, stage, manifest)
            bound = registry.bind(plan, effective)
            selector = from_scene(stage, effective, sensor)
            initial, spawn_receipt = initial_pose(stage, effective, ready_pose, plan['task_name'], sensor)
            initializer = range_initializer(plan['task_name'])
            range_reset = initializer is not None
            if range_reset:
                # This is the fresh localized episode layer, never an asset or
                # original layout. Mechanism hooks remain session-only.
                with Usd.EditContext(stage, stage.GetRootLayer()):
                    initializer.apply_target(stage, spawn_receipt)
                stage.GetRootLayer().Save()
                digest = sha(manifest['scene'])
                for record in (manifest, effective):
                    record['scene_sha256'] = digest
                    record['source_hashes'][manifest['scene']] = digest
                    initializer.refresh_manifest(stage, record, spawn_receipt)
                bound = registry.bind(plan, effective)
            # The existing initializer still owns base stations and target poses.
            # Recheck the actual room using a whole task/variant source posture.
            from short_tasks.tasks.robot_pose_alignment_v1.geometry import align_receipt
            if not spawn_receipt.get('targets_rad'):
                spawn_receipt['targets_rad'] = read(ready_pose)['targets_rad']
            spawn_receipt = align_receipt(stage, effective, plan, spawn_receipt,
                                          seed=manifest['seed'])
            sofa_reset = plan['task_name'] in ('sofa_to_drawer','sofa_to_coffee_table')
            if sofa_reset:
                from importlib import import_module
                import_module('short_tasks.tasks.'+plan['task_name']+'.prepare_initial').refresh_manifest(stage,manifest,spawn_receipt)
            basket_reset = (plan['task_name'] == 'table_to_storage_basket'
                and spawn_receipt.get('method') == 'source_table_to_storage_basket_station_actual_room_fk')
            if basket_reset:
                from short_tasks.tasks.table_to_storage_basket.prepare_initial import refresh_manifest
                refresh_manifest(stage, manifest, spawn_receipt)
            pack_reset = (plan['task_name'] == 'pack_object'
                and spawn_receipt.get('method') == 'source_pack_sofa_station_actual_room_fk')
            cutlery_reset = (plan['task_name'] == 'prepare_cutlery'
                and spawn_receipt.get('method') == 'source_cutlery_station_actual_room_fk')
            tray_reset = (plan['task_name'] == 'carry_meal_tray'
                and spawn_receipt.get('method') == 'source_carry_tray_station_actual_room_fk')
            if tray_reset:
                from short_tasks.tasks.carry_meal_tray.prepare_initial import refresh_manifest
                refresh_manifest(stage, manifest, spawn_receipt)
            if cutlery_reset:
                from short_tasks.tasks.prepare_cutlery.prepare_initial import refresh_manifest
                refresh_manifest(stage, manifest, spawn_receipt)
            if pack_reset:
                from short_tasks.tasks.pack_object.prepare_initial import refresh_manifest
                refresh_manifest(stage, manifest, spawn_receipt)
            if plan['task_name'] == 'clear_tablecloth':
                from short_tasks.tasks.clear_tablecloth.prepare_initial import refresh_manifest
                refresh_manifest(stage, manifest, spawn_receipt)
            from short_tasks.tasks.sofa_background_v1.generation import add as add_sofa_background
            with Usd.EditContext(stage, stage.GetRootLayer()):
                backgrounds = add_sofa_background(plan, stage, effective)
            if backgrounds is not None:
                bound = registry.bind(plan, effective)
                # Keep the selected station and complete joint posture unchanged.
                from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
                check = checker(stage, effective, [*spawn_receipt['position'][:2], spawn_receipt['yaw_rad']])
                reason = check({'targets_rad': spawn_receipt['targets_rad']})
                if reason:
                    raise ValueError('Sofa background intersects the approved robot initial pose: '+reason)
                manifest['objects'].extend(o for o in effective['objects'] if o.get('role') == 'background'
                    and o.get('instance_id', '').startswith('sofa_background_'))
                manifest['sofa_background'] = backgrounds
                stage.GetRootLayer().Save()
                digest = sha(manifest['scene'])
                for record in (manifest, effective):
                    record['scene_sha256'] = digest
                    record['source_hashes'][manifest['scene']] = digest
            write(native / 'spawn.json', spawn_receipt)
            manifest['source_hashes'].update(spawn_receipt['reference_hashes'])
            manifest['source_hashes'].update(effective.get('source_hashes', {}))
            manifest['source_hashes'][str(sensor_file)] = sha(sensor_file)
            if range_reset or plan['task_name'] == 'clear_tablecloth' or pack_reset or cutlery_reset or tray_reset or basket_reset or sofa_reset:
                write(native / 'ready_pose.json', dict(
                    status=('source_task_range_reset_requires_native_acceptance' if range_reset else
                        'sofa_task_reset_requires_native_acceptance' if sofa_reset else
                        'source_basket_reset_requires_native_acceptance' if basket_reset else
                        'approved_carry_tray_reset_requires_native_acceptance' if tray_reset else
                        'source_cutlery_reset_requires_native_acceptance' if cutlery_reset else
                        'source_pack_sofa_reset_requires_native_acceptance' if pack_reset
                        else 'approved_clear_tablecloth_reset_requires_native_acceptance'),
                    targets_rad=spawn_receipt['targets_rad'],
                    sampling=spawn_receipt['candidate']['sampling'], seed=manifest['seed']))
            else:
                shutil.copyfile(ready_pose, native / 'ready_pose.json')
            write(native / 'ready_pose_before_source_alignment.json', read(native / 'ready_pose.json'))
            write(native / 'ready_pose.json', dict(
                status='paired_source_robot_posture_requires_native_acceptance',
                targets_rad=spawn_receipt['targets_rad'],
                sampling='whole_recorded_source_posture_no_joint_jitter',
                seed=manifest['seed'],
                source_alignment=spawn_receipt['robot_posture_alignment']))
            manifest['robot_posture_alignment'] = spawn_receipt['robot_posture_alignment']
            manifest['source_hashes'][str(native / 'ready_pose.json')] = sha(native / 'ready_pose.json')
            write(native / 'manifest.json', manifest)
            write(native / 'bindings.json', bound)
            write(native / 'scene_validation.json', dict(
                scene_hook=effective.get('short_task_scene_preparation'),
                workspace_count=len(selector.workspaces) if hasattr(selector, 'workspaces') else None,
                dependencies=dict(layers=len(layers), assets=len(assets), resolver_modules=special),
                native_physics_validated=False))
            launch_path = native / 'launch.json'
            write(launch_path, dict(schema='openend.short_task.launch.v1', task_name=plan['task_name'],
                manifest=str(native / 'manifest.json'), request=str(out / 'request.json'),
                ready_pose=str(native / 'ready_pose.json'), initial_base_xyyaw=initial, sensor=sensor,
                startup_spawn_scope=('source_task_range_reset_requires_native_acceptance' if range_reset else
                    'sofa_task_station_requires_native_acceptance' if sofa_reset else
                    'source_basket_station_requires_native_acceptance' if basket_reset else
                    'approved_cloth_edge_annulus_requires_native_acceptance'
                    if plan['task_name'] == 'clear_tablecloth' else
                    'source_pack_sofa_station_requires_native_acceptance' if pack_reset else
                    'source_cutlery_station_requires_native_acceptance' if cutlery_reset else
                    'approved_carry_tray_station_requires_native_acceptance' if tray_reset else
                    'task_front_folded_envelope_search_requires_native_acceptance'),
                source_robot_pose_preserved_in_scene=True, startup_physics_hz=60, control_hz=20,
                physics_backend='cpu', physics_frequency_owner='selected_workspace',
                preparation_receipt=str(out / 'preparation.json')))
        from .native_owner import admit_launch
        _, final_manifest, _ = admit_launch(read(launch_path), out / 'native_smoke')
        verify_manifest(final_manifest)
        report.update(status='prepared_not_native_accepted', launch=str(launch_path),
                      native_admission_passed=True)
        return report
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(out / 'preparation.json', report)
        tempfile.tempdir = previous_tmp
        if previous_env_tmp is None:
            os.environ.pop('TMPDIR', None)
        else:
            os.environ['TMPDIR'] = previous_env_tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--request', type=Path)
    source.add_argument('--task', help='Use the registered example request')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--layout', type=int, help='Omit to use seeded random layout selection')
    ap.add_argument('--ready-pose', help='Required for ordinary room tasks; atomic variants use source reset')
    ap.add_argument('--sensor-file', help='Required for ordinary room tasks; atomic variants use source cameras')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.request:
        request = read(args.request)
        if args.layout is not None:
            request['layout_id'] = args.layout
    else:
        from short_tasks import registry
        if args.task in registry.FAILURE_RECOVERY_TASKS:
            request_for = registry.failure_recovery_adapter().request_for
        elif args.task in registry.ATOMIC_COFFEE_TASKS:
            request_for = registry.atomic_coffee_adapter().request_for
        else:
            from .probe_random_layout import request_for
        request = request_for(args.task)
        request['layout_id'] = args.layout
    from short_tasks import registry
    if request.get('task_name') not in (registry.ATOMIC_COFFEE_TASKS | registry.FAILURE_RECOVERY_TASKS) and (not args.ready_pose or not args.sensor_file):
        ap.error('Ordinary room tasks require --ready-pose and --sensor-file')
    if args.seed is not None:
        request['seed'] = args.seed
    print(json.dumps(prepare(request, args.ready_pose, args.sensor_file, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()

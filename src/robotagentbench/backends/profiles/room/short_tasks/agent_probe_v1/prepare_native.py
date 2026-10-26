"""Use the existing fixed-layout scene owner, then bind its receipt to coffee v2.

No physics, geometry, placement, material or scoring parameters are overridden.
All generated artifacts must be under this bridge's task directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(basis_manifest, ready_pose, sensor_file, output, *, request=None, local_random=False):
    from short_tasks.tools.prepare_scene import prepare as source_prepare
    from short_tasks import registry
    from short_tasks.scene import verify_scene_sources, verify_manifest
    output = Path(output).resolve()
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    if not (output.is_relative_to(HERE) or permits(output)) or output.exists():
        raise ValueError('Fresh output must be inside the bridge task directory')
    verify_scene_sources()
    request = request or read(HERE.parent/'tasks/prepare_coffee/request_v2.json')
    plan = registry.resolve(request)
    registry.verify_sources(plan)
    basis = read(basis_manifest)
    if (basis['layout_id'], basis['seed']) != (request['layout_id'], request['seed']):
        raise ValueError('Explicit coffee request and basis layout/seed differ')
    output.mkdir(parents=True)
    # Relocate only the three original Windows texture references. The actual
    # files are the pre-existing L1 source textures, not replacement materials.
    from pxr import Usd, UsdUtils
    original = Usd.Stage.Open(basis['scene'])
    original_hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                       for l in original.GetUsedLayers() if l.realPath}
    texture_root = Path('/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/assets_living_room/coffee_table/PBR_Textures')
    source_root = 'D:/00_Sale/TableBasse/TableBasse/PBR_Textures/'
    names = ('T_Coffee_Table_01_Wood_01_Metallic_4K.1001.png',
             'T_Coffee_Table_01_Wood_01_Normal_4k.1001.png',
             'T_Coffee_Table_01_Wood_01_Roughness_4K.1001.png')
    replacements = {source_root+n: str((texture_root/n).resolve(strict=True)) for n in names}
    flat = original.Flatten()
    used = {}
    def relocate(asset):
        if asset in replacements:
            used[asset] = replacements[asset]
            return replacements[asset]
        return asset
    UsdUtils.ModifyAssetPaths(flat, relocate)
    flat.Export(str(output/'basis_localized.usdc'))
    write(output/'basis_manifest.json', dict(basis, scene=str(output/'basis_localized.usdc')))
    write(output/'localization.json', dict(original_layers_sha256=original_hashes,
        original_scene=basis['scene'], texture_path_map=used,
        texture_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in used.values()},
        physics_parameters_changed=False, material_parameters_changed=False))
    basis_launch = output/'basis_launch.json'
    write(basis_launch, dict(manifest=str(output/'basis_manifest.json'),
        ready_pose=str(Path(ready_pose).resolve()), sensor=read(sensor_file)['sensor']))
    report = source_prepare('prepare_coffee', basis_launch, output/'prepared')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in original_hashes.items()), 'Original scene changed during preparation'
    manifest = read(output/'prepared/manifest.json')
    # The old preparation receipt used this name for its authenticated sources.
    # Preserve it and expose the same bytes through the current session contract.
    manifest['source_hashes'] = dict(manifest['source_sha256'])
    scene = Path(manifest['scene'])
    manifest['source_hashes'][str(scene)] = hashlib.sha256(scene.read_bytes()).hexdigest()
    for row in manifest['objects']:
        if row.get('task_id') == 'prepare_coffee':
            row['task_id'] = 'coffee'
    for row in manifest['tasks']:
        if row['id'] == 'prepare_coffee':
            row['id'] = 'coffee'
    launch = read(output/'prepared/launch.json')
    if local_random:
        from .coffee_randomization import apply
        # Preserve the baseline receipt, then make a stronger task-owned layer.
        write(output/'prepared/manifest_before_local_random.json', manifest)
        manifest, random_receipt = apply(manifest, launch['initial_base_xyyaw'],
                                         output/'local_random', request['seed'])
        launch['initial_randomization_receipt'] = str(output/'local_random/randomization.json')
    registry.bind(plan, manifest)
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import align_receipt
    stage = Usd.Stage.Open(manifest['scene'])
    original_ready = read(launch['ready_pose'])
    aligned = align_receipt(stage, manifest, plan, dict(
        position=[*launch['initial_base_xyyaw'][:2], 0.],
        yaw_rad=launch['initial_base_xyyaw'][2],
        targets_rad=original_ready['targets_rad'], method='existing_coffee_station',
        reference_hashes={}), seed=request['seed'])
    ready_path = output/'prepared/ready_pose.json'
    write(output/'prepared/ready_pose_before_source_alignment.json', original_ready)
    write(ready_path, dict(status='paired_source_robot_posture_requires_native_acceptance',
        targets_rad=aligned['targets_rad'], sampling='whole_recorded_source_posture_no_joint_jitter',
        seed=request['seed'], source_alignment=aligned['robot_posture_alignment']))
    launch['ready_pose'] = str(ready_path)
    launch['robot_posture_alignment'] = aligned['robot_posture_alignment']
    manifest['robot_posture_alignment'] = aligned['robot_posture_alignment']
    manifest['source_hashes'].update(aligned['reference_hashes'])
    manifest['source_hashes'][str(ready_path)] = hashlib.sha256(ready_path.read_bytes()).hexdigest()
    verify_manifest(manifest)
    write(output/'prepared/manifest.json', manifest)
    # Keep the scene owner's historical request instead of erasing its evidence.
    write(output/'prepared/request_v2.json', request)
    launch['request'] = str(output/'prepared/request_v2.json')
    launch['preparation_receipt'] = str(output/'prepared/preparation.json')
    write(output/'prepared/launch_v2.json', launch)
    print(json.dumps(dict(status='prepared_not_native_accepted',
        launch=str(output/'prepared/launch_v2.json'), task_name=plan['task_name'],
        rule_version=plan['rule_version'], maximum=plan['score_policy']['maximum'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basis-manifest', required=True)
    ap.add_argument('--ready-pose', required=True)
    ap.add_argument('--sensor-file', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--request', type=Path, help='Existing approved coffee-v2 request for this layout/seed')
    ap.add_argument('--local-random', action='store_true', help='Apply approved reset-only local coffee randomization')
    args = ap.parse_args()
    prepare(args.basis_manifest, args.ready_pose, args.sensor_file, args.output,
            request=read(args.request) if args.request else None, local_random=args.local_random)

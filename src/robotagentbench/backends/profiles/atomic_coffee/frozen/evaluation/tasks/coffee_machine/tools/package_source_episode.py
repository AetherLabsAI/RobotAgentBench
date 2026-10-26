"""Package the task objects of one source episode exactly as the data producer composed them. CPU USD only
(run through evaluation/tools/run_cpu_usd.sh); read-only on the producer side; no simulation.

For one API + source episode it
  1. opens the producer's own layer stack through path-rewritten copies (tools/producer_layers.py),
  2. for 1-1 / 1-4 (whose per-episode scene files stayed on the producer's host) repeats the producer's pre-physics
     authoring steps from the recorded plan: cup style, machine pose, bag pose, cup pose (the steps and their order are
     the ones in the producer's runner, quoted in AUTHORING below); for 1-4 it also restores the pre-filled liquid from
     the episode's own scene_final.usda, moved rigidly with the cup from its final to its initial pose,
  3. checks the composed object poses against the poses the producer recorded at its first sample,
  4. flattens ONLY the task objects (/World/CoffeeTaskPreview, /World/PourBeans, /World/CoffeeLiquid and the materials
     they bind) into resources/source_episodes/<api>/<episode>/coffee_objects.usdc; referenced textures are copied once into
     resources/source_episodes/assets/ (named by content hash),
  5. writes episode.json: provenance with hashes, the recorded initial robot joint targets (joint20), the recorded
     randomization values, the pose check.

Nothing of the producer's robot, physics scene, lights or house is packaged: the evaluation scene takes those from the
public baseline with the public unified parameters.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from producer_layers import Mirror, WORKSPACE   # noqa: E402

WB = WORKSPACE / 'coffee_task_workbench_20260922'
CONTAINER_WB = '/geniesim/main/demo_sim/coffee_task_workbench_20260922'
ROOT = '/World/CoffeeTaskPreview'
MACHINE, CUP, BAG = ROOT + '/Machine', ROOT + '/Cup', ROOT + '/BeanBag'
BEANS, LIQUID = '/World/PourBeans', '/World/CoffeeLiquid'
MACHINE_ORIGIN = (2.569096825, .166750471, .92)      # producer cup_place/scene_pose.py:30
JOINT20 = ('idx21_arm_l_joint1', 'idx22_arm_l_joint2', 'idx23_arm_l_joint3', 'idx24_arm_l_joint4', 'idx25_arm_l_joint5',
           'idx26_arm_l_joint6', 'idx27_arm_l_joint7', 'idx61_arm_r_joint1', 'idx62_arm_r_joint2', 'idx63_arm_r_joint3',
           'idx64_arm_r_joint4', 'idx65_arm_r_joint5', 'idx66_arm_r_joint6', 'idx67_arm_r_joint7', 'idx01_body_joint1',
           'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2', 'idx41_gripper_l_outer_joint1',
           'idx81_gripper_r_outer_joint1')
AUTHORING = ('producer runner (copy in every 1-4 episode directory, runner.py lines 61-79): apply_asset(cup style) -> '
             'apply_machine_pose(offset, yaw) -> apply_bag_pose(offset, yaw) -> cup world pose; '
             'cup_place/cup_variants.py:65-74, cup_place/scene_pose.py:9-47')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


SELECTION = TASK / 'references/source_sim_videos/selection.json'


def selection(api, label, path=None):
    sel = json.loads((path or SELECTION).read_text())[api]
    return next(e for e in sel['episodes'] if e['label'] == label)


def rows_to_matrix(rows):
    """Producer JSON stores column-vector 4x4 (row i = [r_i0, r_i1, r_i2, t_i]); USD wants row vectors."""
    from pxr import Gf
    return Gf.Matrix4d(*[rows[j][i] for i in range(4) for j in range(4)])


def rotation_about_z(origin, offset, yaw_deg):
    """World-Z rigid delta about `origin` followed by `offset` (producer scene_pose.rigid_delta / machine_delta)."""
    from pxr import Gf
    m = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_deg)))
    o = Gf.Vec3d(*origin)
    m.SetTranslateOnly(o + Gf.Vec3d(*offset) - m.TransformDir(o))
    return m


def set_world(stage, path, world):
    from pxr import UsdGeom, UsdPhysics, Gf
    prim = stage.GetPrimAtPath(path)
    parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * parent.GetInverse())
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        body = UsdPhysics.RigidBodyAPI(prim)
        body.CreateVelocityAttr().Set(Gf.Vec3f(0)); body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0))


def world_of(stage, path):
    from pxr import UsdGeom
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))


def frame_from_axes(position, up, handle):
    """Cup world matrix from the producer's recorded axes: up = cup local +Y, handle_axis = cup local +X."""
    from pxr import Gf
    y = Gf.Vec3d(*up).GetNormalized()
    x = Gf.Vec3d(*handle); x = (x - y * (x * y)).GetNormalized()
    z = Gf.Cross(x, y)
    m = Gf.Matrix4d(1)
    m.SetRow3(0, x); m.SetRow3(1, y); m.SetRow3(2, z)
    m.SetTranslateOnly(Gf.Vec3d(*position))
    return m


def author_cup_stage(stage, plan, mirror, cup_world=None):
    """1-1 / 1-4: the producer's pre-physics authoring, in the producer's order."""
    from pxr import Sdf, Gf
    layer = stage.GetRootLayer()
    if plan.get('cup_style'):
        world = world_of(stage, CUP)
        source = Sdf.Layer.FindOrOpen(mirror.layer(CONTAINER_WB + '/assets/cups/' + plan['cup_style'] + '.usda'))
        Sdf.CreatePrimInLayer(layer, CUP)
        if not Sdf.CopySpec(source, '/CoffeeCup', layer, CUP):
            raise RuntimeError('Could not compose cup asset')
        stage.GetPrimAtPath(CUP).GetReferences().SetReferences([Sdf.Reference(mirror.layer(CONTAINER_WB + '/coffee_cup_dynamic.usda'))])
        set_world(stage, CUP, world)
    if plan.get('machine_offset_m') is not None:
        world = world_of(stage, MACHINE)
        if (world.ExtractTranslation() - Gf.Vec3d(*MACHINE_ORIGIN)).GetLength() > 1e-7:
            raise ValueError('Machine baseline differs from the producer origin')
        set_world(stage, MACHINE, world * rotation_about_z(MACHINE_ORIGIN, plan['machine_offset_m'], plan.get('machine_yaw_deg', 0.)))
    if plan.get('bag_offset_m') is not None:
        world = world_of(stage, BAG)
        set_world(stage, BAG, world * rotation_about_z(world.ExtractTranslation(), plan['bag_offset_m'], plan.get('bag_yaw_deg', 0.)))
    if cup_world is not None:
        set_world(stage, CUP, cup_world)
    elif plan.get('cup_start_xyz') is not None:
        world = world_of(stage, CUP) * Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(plan.get('cup_start_yaw_deg', 0.))))
        world.SetTranslateOnly(Gf.Vec3d(*plan['cup_start_xyz']))
        set_world(stage, CUP, world)


def restore_liquid(stage, scene_final, initial_cup, final_cup):
    """1-4: particles saved in the episode's scene_final.usda, carried from the final to the initial cup frame."""
    from pxr import Sdf, Gf, Vt, UsdGeom
    source = Sdf.Layer.FindOrOpen(str(scene_final))
    if not source.GetPrimAtPath(LIQUID):
        raise ValueError('Episode scene has no liquid record')
    layer = stage.GetRootLayer()
    Sdf.CreatePrimInLayer(layer, LIQUID)
    if not Sdf.CopySpec(source, LIQUID, layer, LIQUID):
        raise RuntimeError('Could not copy the liquid record')
    points = UsdGeom.Points(stage.GetPrimAtPath(LIQUID + '/Particles'))
    move = final_cup.GetInverse() * initial_cup
    values = [Gf.Vec3f(*move.Transform(Gf.Vec3d(*p))) for p in points.GetPointsAttr().Get()]
    points.GetPointsAttr().Set(Vt.Vec3fArray(values))
    points.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0)] * len(values)))
    # local cup coordinates of the restored liquid, for the record
    local = [initial_cup.GetInverse().Transform(Gf.Vec3d(*p)) for p in values]
    return dict(particles=len(values), source=str(scene_final), source_sha256=sha(scene_final),
                method='final particle positions moved rigidly with the cup from its final to its initial recorded pose; velocities zero',
                cup_local_min=[min(p[i] for p in local) for i in range(3)], cup_local_max=[max(p[i] for p in local) for i in range(3)])


def yaw_of(matrix):
    """World-Z heading of the local +X axis, degrees."""
    from pxr import Gf
    x = matrix.TransformDir(Gf.Vec3d(1, 0, 0))
    return math.degrees(math.atan2(x[1], x[0]))


def compare(stage, recorded):
    """Composed vs recorded object world poses. recorded: path -> dict(position=[...], matrix=Gf.Matrix4d | None)."""
    from pxr import Gf
    out = {}
    for path, rec in recorded.items():
        world = world_of(stage, path)
        entry = dict(position_error_m=(world.ExtractTranslation() - Gf.Vec3d(*rec['position'])).GetLength(),
                     composed_position=list(world.ExtractTranslation()), recorded_position=list(rec['position']))
        if rec.get('axes'):
            entry['axis_error_deg'] = max(
                math.degrees(math.acos(max(-1., min(1., world.TransformDir(Gf.Vec3d(*local)).GetNormalized() * Gf.Vec3d(*axis).GetNormalized()))))
                for local, axis in rec['axes'])
        out[path] = entry
    return out


def copy_assets(layer, out):
    """Copy every file the flattened layer references (textures) into the shared, content-addressed asset directory
    resources/source_episodes/assets/ and make the paths relative to the episode package."""
    from pxr import UsdUtils
    assets = out.parents[1] / 'assets'; copied = {}

    def fix(path):
        if not path or path.startswith(('http:', 'https:')):
            return path
        src = Path(path)
        if not src.is_absolute():
            return path
        if not src.is_file():
            copied[path] = None
            return path
        digest = sha(src)
        name = digest[:12] + '_' + src.name
        assets.mkdir(exist_ok=True)
        if not (assets / name).exists():
            shutil.copy2(src, assets / name)
        copied[path] = dict(file='../../assets/' + name, sha256=digest, bytes=src.stat().st_size)
        return '../../assets/' + name

    UsdUtils.ModifyAssetPaths(layer, fix)
    return copied


def package(api, label, force=False, selection_path=None):
    from pxr import Usd, Sdf, UsdGeom, UsdPhysics
    ep = selection(api, label, selection_path)
    ep_dir = Path(ep['episode_dir'])
    out = TASK / 'resources/source_episodes' / api / label
    if out.exists():
        if not force:
            raise ValueError('Package exists: ' + str(out))
        shutil.rmtree(out)
    out.mkdir(parents=True)
    mirror = Mirror(WB, TASK / 'artifacts/producer-mirror' / api / label)
    inputs = {}
    note = {}

    def read(name, base=ep_dir):
        p = Path(base) / name
        inputs[str(p)] = sha(p)
        return json.loads(p.read_text())

    # ---------- open the producer's composition of this episode ----------
    if api in ('1-1', '1-4'):
        plan = read('plan.json'); result = read('result.json')
        first, last = result['trace'][0], result['trace'][-1]
        root = mirror.layer(CONTAINER_WB + '/scene_bag_v1.usda')
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [root]
        cup_world = None
        if api == '1-4':
            cup_world = frame_from_axes(first['cup'], first['up'], first['handle_axis'])
        author_cup_stage(stage, plan, mirror, cup_world)
        if api == '1-4':
            note['liquid'] = restore_liquid(stage, ep_dir / 'scene_final.usda', cup_world,
                                            frame_from_axes(last['cup'], last['up'], last['handle_axis']))
        recorded = {MACHINE: dict(position=first['machine']),
                    CUP: dict(position=first['cup'], axes=[((0, 1, 0), first['up']), ((1, 0, 0), first['handle_axis'])]),
                    BAG: dict(position=first['bag'])}
        rec_machine = first['machine_rotation']
        # the producer stored np.array(Gf.Matrix4d)[:3,:3]: row i is the world direction of local axis i
        recorded[MACHINE]['axes'] = [((1, 0, 0), rec_machine[0]), ((0, 1, 0), rec_machine[1])]
        rec_bag = first['bag_rotation']
        recorded[BAG]['axes'] = [((1, 0, 0), rec_bag[0]), ((0, 1, 0), rec_bag[1])]
        targets = plan['initial_targets']
        robot = {n: float(first['all_joint_positions'][n]) for n in JOINT20}
        robot_source = 'result.json trace[0].all_joint_positions (measured at the producer\'s first sample, t = 0.05 s)'
        hold = {n: float(targets.get(n, robot[n])) for n in JOINT20}
        # the active arm starts at plan.initial (7 joints); the producer commands it from step 0
        arm = ep['categories'].get('arm') or ('left' if plan.get('active_arm') == 'left' else 'right')
        prefix = 'idx2' if arm == 'left' else 'idx6'
        for i, q in enumerate(plan['initial']):
            hold[next(n for n in JOINT20 if n.startswith(f'{prefix}{i + 1}_'))] = float(q)
        randomization = plan.get('randomization') or {}
        seed = randomization.get('seed')
        facts = dict(cup_style=plan.get('cup_style'), arm=arm, machine_offset_m=plan.get('machine_offset_m'), machine_yaw_deg=plan.get('machine_yaw_deg'),
                     bag_offset_m=plan.get('bag_offset_m'), bag_yaw_deg=plan.get('bag_yaw_deg'), initial_cup=result['initial_cup'],
                     retrieval_pose_randomization=plan.get('retrieval_pose_randomization'), continued_from=plan.get('continued_from'),
                     initial_state_status=None)
        trajectory = dict(file=str(ep_dir / 'executed_plan.json'), rows=len(plan['steps']), rate_hz=20)
        inputs[str(ep_dir / 'executed_plan.json')] = sha(ep_dir / 'executed_plan.json')
        base_layer = WB / 'scene_bag_v1.usda'
        inputs[str(base_layer)] = sha(base_layer)
        certificate = read('preflight_certificate.json')
        recorded_files = certificate.get('files') or {}
        note['producer_scene_hash_check'] = {k: dict(certificate=v, on_202=sha(WB / k) if (WB / k).is_file() else None)
                                             for k, v in recorded_files.items() if k.endswith(('.usda', 'cup_variants.json')) }
    else:
        if api == '1-2':
            run = Path(ep['data_list']['local_artifacts'][0]).parent
            wb = Path(str(run).split('/runs/')[0])
            init_dir = run / 'initial'
            plan_file = ep_dir / 'plan.json'
        else:
            run = Path(ep['data_list']['source_path'].split('：', 1)[-1])
            wb = Path(str(run).split('/runs/')[0])
            init_dir = run
            plan_file = ep_dir / 'button_plan.json'
        mirror = Mirror(wb, TASK / 'artifacts/producer-mirror' / api / label)
        init = read('initialization.json', init_dir)
        plan = read(plan_file.name, plan_file.parent)
        root = mirror.layer(CONTAINER_WB + '/' + str((init_dir / 'initial.usda').relative_to(wb)))
        inputs[str(init_dir / 'initial.usda')] = sha(init_dir / 'initial.usda')
        stage = Usd.Stage.Open(root)
        recorded = {MACHINE: dict(position=[r[3] for r in init['machine_world'][:3]]),
                    BAG: dict(position=[r[3] for r in init['bag_world'][:3]]),
                    CUP: dict(position=[r[3] for r in init['cup_world'][:3]])}
        for path, key in ((MACHINE, 'machine_world'), (BAG, 'bag_world'), (CUP, 'cup_world')):
            m = init[key]
            recorded[path]['axes'] = [((1, 0, 0), [m[i][0] for i in range(3)]), ((0, 1, 0), [m[i][1] for i in range(3)])]
        robot = {n: float(init['robot_targets'][n]) for n in JOINT20}
        robot_source = 'initialization.json robot_targets'
        hold = dict(robot)
        seed = init.get('seed')
        randomization = init.get('randomization') or {}
        facts = dict(cup_style=randomization.get('cup_style'), arm='right', randomization=randomization,
                     beans=len(init.get('bean_paths') or []), beans_initially_in_hopper=init.get('beans_initially_in_hopper'),
                     bag_geometry_revision=init.get('bag_geometry_revision'), machine_hopper_revision=init.get('machine_hopper_revision'))
        trajectory = dict(file=str(plan_file), rows=len(plan['steps']), rate_hz=plan.get('control_hz', 20))

    check = compare(stage, recorded)

    # ---------- flatten only the task objects ----------
    paths = [p for p in (ROOT, BEANS, LIQUID) if stage.GetPrimAtPath(p)]
    mask = Usd.StagePopulationMask(paths)
    masked = Usd.Stage.OpenMasked(stage.GetRootLayer(), mask)
    masked.ExpandPopulationMask()
    included = [str(p) for p in masked.GetPopulationMask().GetPaths()]
    outside = [p for p in included if not p.startswith(tuple(paths))]
    if any(p.startswith('/World/Robot') for p in outside):
        raise ValueError('Task objects reference robot prims; refusing to package robot opinions: ' + str(outside))
    flat = masked.Flatten()
    target = out / 'coffee_objects.usdc'   # binary crate: the flattened geometry is large
    copied = copy_assets(flat, out)
    flat.Export(str(target))

    # ---------- inventory of what was packaged (composed physics facts, for the asset alignment check) ----------
    packaged = Usd.Stage.Open(str(target))
    bodies = {}
    for prim in packaged.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            colliders = [c for c in Usd.PrimRange(prim) if c.HasAPI(UsdPhysics.CollisionAPI)]
            bodies[str(prim.GetPath())] = dict(
                kinematic=bool(prim.GetAttribute('physics:kinematicEnabled').Get()),
                mass=mass.GetMassAttr().Get() if mass else None, colliders=len(colliders),
                world=list(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()))
    if len(bodies) > 12:    # keep the 100 beans as a count plus one example
        beans = {k: v for k, v in bodies.items() if k.startswith(BEANS + '/')}
        bodies = {k: v for k, v in bodies.items() if not k.startswith(BEANS + '/')}
        if beans:
            bodies[BEANS + '/*'] = dict(count=len(beans), example=next(iter(beans.values())))

    mirror.save_record()
    shutil.copy2(mirror.out / 'mirror-record.json', out / 'producer-layers.json')
    shutil.rmtree(mirror.out)     # the rewritten layer copies were only needed to compose; the record stays
    manifest = dict(
        schema='arena.coffee_machine.source_episode.v1', api=api, episode=label,
        data_list=ep['data_list'], producer_status_label=ep['data_list']['source_status'],
        seed=seed, facts=facts, robot_base=dict(root_xyz=plan['root_xyz'], yaw=plan['yaw']),
        robot_initialization=dict(positions=robot, hold_targets=hold, source=robot_source),
        trajectory=trajectory, pose_check=check, packaged_prims=paths, mask_paths=included, packaged_bodies=bodies,
        copied_assets=copied, inputs_sha256=inputs, authoring=AUTHORING if api in ('1-1', '1-4') else 'none: the producer\'s own initial.usda of this episode',
        notes=note, coffee_objects_sha256=sha(target),
        scope='Task objects only. Robot, physics scene, lights and house come from the public baseline with unified parameters.')
    (out / 'episode.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + '\n')
    worst = max(v['position_error_m'] for v in check.values())
    worst_axis = max(v.get('axis_error_deg', 0.) for v in check.values())
    print(api, label, 'prims', paths, 'bodies', len(bodies), 'pose error max %.2e m / %.3f deg' % (worst, worst_axis),
          'assets', len(copied), 'missing', [k for k, v in copied.items() if v is None][:3], 'size', target.stat().st_size)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--api', required=True, choices=('1-1', '1-2', '1-3', '1-4'))
    p.add_argument('--episode', nargs='*', help='labels; default: the five displayed source episodes')
    p.add_argument('--force', action='store_true')
    p.add_argument('--selection', type=Path, default=SELECTION, help='episode list (default: the five displayed source episodes)')
    a = p.parse_args(argv)
    sel = json.loads(a.selection.read_text())[a.api]
    for label in a.episode or [e['label'] for e in sel['episodes']]:
        package(a.api, label, a.force, a.selection)


if __name__ == '__main__':
    main()

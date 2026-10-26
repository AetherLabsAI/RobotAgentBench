"""10.2 reset-time randomization; task-local, never a runtime pose controller.

The L1 dining sampler supplies bounds/seeding, NOT a validated initializer.
Materialize the accepted current grasp in the shared layout's dining station.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = Path(__file__).with_name('resources') / 'accepted_initial_v1.json'
POLICY = 'arena.dining_place.randomization.v1'
L1_POLICY = 'real2sim.dining_rollout_randomization.v1'
L1_SOURCE = {
    'path': 'real2sim/production/dining_randomization.py',
    'host': 'L1', 'commit': 'ac2a15021d28c3db634dafd32ec8fb497a84852a',
    'sha256': '72f57ce6acf79c12e75b42e42833eb14aafc143a248a9f1366817f017ef5bcf8',
    'pipeline': 'task108_serve',
    'source_status': 'plan only; L1 task 10.2 runtime_ready=False',
}


def sample_offsets(seed):
    """Exactly L1 rollout_seed('10.2', seed) and its three uniform draws."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('seed must be an integer in [0, 2**31)')
    derived = random.Random(f'{L1_POLICY}\0{10.2}\0{seed}').getrandbits(64)
    rng = random.Random(derived)
    return dict(derived_seed=derived, dx_m=rng.uniform(-.020, .020),
                dy_m=rng.uniform(-.020, .020), yaw_deg=rng.uniform(-8., 8.))


def station_pose(module, offsets):
    """XY jitter in world axes; yaw around robot base, facing table center.

This explicit group-frame adapter is ours; L1 only described robot adaptation.
"""
    x, y = module['interaction_station_xy_m']
    cx, cy = module['center_xy_m']
    if math.hypot(cx-x, cy-y) < 1e-6:
        raise ValueError('Dining station cannot coincide with table center')
    yaw = math.degrees(math.atan2(cy-y, cx-x)) + offsets['yaw_deg']
    return dict(position_m=[x+offsets['dx_m'], y+offsets['dy_m'], 0.],
                quaternion_wxyz=[math.cos(math.radians(yaw)/2), 0., 0.,
                                 math.sin(math.radians(yaw)/2)], yaw_deg=yaw)


def build_plan(seed):
    from arena_api import l1_runtime
    from vendor.l1_scene.real2sim.production import house_topology
    offsets = sample_offsets(seed)
    ordinal = l1_runtime.layout_ordinal(seed)
    candidate = house_topology.load_catalog()['candidates'][ordinal-1]
    module = candidate['module_poses']['dining_table_cluster']
    return dict(schema=POLICY, task_id='10.2', seed=seed, source=L1_SOURCE,
                layout_ordinal=ordinal, layout_id=candidate['id'],
                environment_policy='shared L1 layout + appearance; arena_scene namespace',
                dining_module=copy.deepcopy(module), offsets=offsets,
                robot_station=station_pose(module, offsets),
                bounds=dict(xy_each_m=.020, yaw_deg=8., z_m=0.),
                group=['robot', 'held_plate', 'payload'],
                unchanged=['joint coordinates and PD targets', 'plate-to-hand pose',
                           'payload-to-plate pose', 'asset scale and geometry',
                           'mass, friction, compliance and collision configuration'],
                status='plan_ready', physics_validated=False, api_registered=False)


def prepare(output, seed, *, randomized=True, base=None, reset_variant='accepted',payload_variant='beef_box'):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    from arena_api import l1_scene
    from tools.build_l1_scene import prepare as environment
    from evaluation.tasks.dining_place import initial_state as task
    from evaluation.tasks.dining_place.kinematics import body_transform
    from evaluation.tasks.microwave_take.initial_state import verify_plate_asset, PLATE_LAYER
    from arena_api.beef_asset import DIRECTORY as BEEF_ASSET_DIR
    from evaluation.tools.prepare_dining_place_reference import inspect_destination
    from evaluation.tools.prepare_dining_place_initial import pose, PLATE, PLATE_ROOT, BOX, HAND
    from evaluation.tools.prepare_dining_place_held_reset import matrix, parent_pose_for_body
    from .variants import payload_for,PLACEMENT_VARIANTS
    BOX=payload_for(payload_variant)
    if payload_variant in PLACEMENT_VARIANTS and (randomized or reset_variant!='accepted'):
        raise ValueError('29-2/29-3 currently uses the accepted fixed station only')

    source = json.loads(SNAPSHOT.read_text())
    if reset_variant=='l2_reference':
        from .l2_reference_reset import source_snapshot
        source=source_snapshot(source)
    elif reset_variant!='accepted':raise ValueError('Unknown reset variant')
    record = build_plan(seed) if randomized else dict(
        schema=POLICY,task_id='10.2',seed=seed,source=L1_SOURCE,
        layout_ordinal=None,layout_id='fixed_baseline',
        robot_station=source['robot_station'],offsets=dict(dx_m=0.,dy_m=0.,yaw_deg=0.),
        environment_policy='shared fixed L1 scene; appearance disabled',
        status='plan_ready',physics_validated=False)
    out = Path(output).resolve()
    boundary=(Path(__file__).resolve().parent if payload_variant in PLACEMENT_VARIANTS else ROOT/'runs')
    if out.exists() or not out.is_relative_to(boundary) or out == boundary:
        raise ValueError('Use a new directory inside '+str(boundary))
    verify_plate_asset()
    out.mkdir(parents=True, exist_ok=False)
    manifest = out/'randomization.json'
    def save_record():
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    record['status'] = 'preparing'
    record['reset_variant']=reset_variant
    record['payload_variant']=payload_variant
    save_record()
    try:
        env = out/'environment/scene_room.usda'
        if payload_variant in PLACEMENT_VARIANTS:
            # Existing fixed public layer composition, memory-only. The shared
            # layout writer does not yet allow this task's output directory.
            src=Path(base or l1_scene.BASE).resolve(strict=True)
            baseline=Usd.Stage.Open(str(src))
            l1_scene.apply(baseline)
            layer=Sdf.Layer.CreateAnonymous('dining_fixed.usda')
            layer.TransferContent(baseline.GetSessionLayer())
            layer.subLayerPaths=[*layer.subLayerPaths,str(src)]
            fixed=Usd.Stage.Open(layer)
            for key in ('metersPerUnit','upAxis','timeCodesPerSecond','framesPerSecond'):
                fixed.SetMetadata(key,baseline.GetMetadata(key))
            env.parent.mkdir(parents=True)
            if not layer.Export(str(env)):raise RuntimeError('Fixed scene export failed')
        else:
            environment(base or l1_scene.BASE, env, ordinal=record['layout_ordinal'], seed=seed, appearance=randomized)
        resource = ROOT/'evaluation/tasks/microwave_take/resources'
        stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
        stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, out) for p in
            (PLATE_LAYER, env)]
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        hashes = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers()
                  if l.realPath and Path(l.realPath) != out/'initial.usda'}

        def author(path, transform):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError('Missing task body '+path)
            prim.SetActive(True)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(transform)
            xf.SetResetXformStack(True)

        robot = matrix(record['robot_station'])
        author(task.ROBOT, robot)
        surface = inspect_destination(stage)
        if abs(surface['support_top_m']-.92) > .001:
            raise ValueError('Dining tabletop height changed; do not compensate via arm pose')
        reset = source['joint_reset_coordinates']
        commands = source['joint_targets']
        for name, value in reset.items():
            prim = stage.GetPrimAtPath(task.ROBOT+'/joints/'+name)
            if not prim:
                raise ValueError('Missing native joint '+name)
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            native = value if linear else math.degrees(value)
            low, high = [prim.GetAttribute('physics:'+s+'Limit').Get() for s in ('lower','upper')]
            if not math.isfinite(native) or not low-.001 <= native <= high+.001:
                raise ValueError('Snapshot outside native joint limits: '+name)
            axis = 'linear' if linear else 'angular'
            state = PhysxSchema.JointStateAPI.Apply(prim, axis)
            state.CreatePositionAttr(native)
            state.CreateVelocityAttr(0.)
            # Mirrored/passive finger coordinates are in the plan too. Do not
            # introduce drives where the accepted native robot has none.
            if name in commands and prim.HasAPI(UsdPhysics.DriveAPI, axis):
                drive = UsdPhysics.DriveAPI(prim, axis)
                target = commands[name]['value']
                drive.CreateTargetPositionAttr(target if linear else math.degrees(target))
                drive.CreateTargetVelocityAttr(0.)
        hand = body_transform(stage, HAND, reset, robot)
        relative = Gf.Matrix4d(*sum(source['plate_to_hand'], []))
        payload_relative = Gf.Matrix4d(*sum(source['payload_to_plate'], []))
        plate = relative*hand
        payload = payload_relative*plate
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(PLATE))
        old_scale = Gf.Transform(xf.ComputeLocalToWorldTransform(0)).GetScale()
        author(PLATE_ROOT, parent_pose_for_body(xf.GetLocalTransformation(), plate))
        author(BOX, payload)
        UsdGeom.Imageable(stage.GetPrimAtPath(BOX)).CreateVisibilityAttr().Set('inherited')
        if payload_variant in PLACEMENT_VARIANTS:
            from arena_api.food_box_assets import verify_instance,DIRECTORY
            verify_instance(stage,BOX,payload_variant)
        for name in ('MicrowaveBeefbox','MicrowaveFoodbox','MicrowaveSaladbox','MicrowavePayload'):
            prim = stage.GetPrimAtPath('/World/Assets/'+name)
            if prim and str(prim.GetPath())!=BOX:
                prim.SetActive(False)
        for path in (PLATE, BOX):
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
            if not body or body.GetKinematicEnabledAttr().Get() is True or body.GetRigidBodyEnabledAttr().Get() is False:
                raise ValueError('Expected active dynamic target '+path)
            body.CreateVelocityAttr(Gf.Vec3f(0))
            body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        for prim in Usd.PrimRange(stage.GetPrimAtPath(task.ROBOT)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(prim)
                body.CreateVelocityAttr(Gf.Vec3f(0))
                body.CreateAngularVelocityAttr(Gf.Vec3f(0))

        actual = xf.ComputeLocalToWorldTransform(0)
        scale_error = (Gf.Transform(actual).GetScale()-old_scale).GetLength()
        relative_error = max(abs(a-b) for ra, rb in zip(actual.RemoveScaleShear()*hand.GetInverse(), relative) for a,b in zip(ra,rb))
        box_actual = UsdGeom.Xformable(stage.GetPrimAtPath(BOX)).ComputeLocalToWorldTransform(0)
        payload_error = max(abs(a-b) for ra,rb in zip(box_actual.RemoveScaleShear()*actual.RemoveScaleShear().GetInverse(), payload_relative) for a,b in zip(ra,rb))
        if max(scale_error, relative_error, payload_error) > 1e-6:
            raise ValueError('Randomization changed scale or held relative pose')
        clearance = plate.ExtractTranslation()[2]-surface['support_top_m']
        if clearance < .10:
            raise ValueError('Held plate origin too low; no automatic height correction')
        plan = {**source, 'bindings':dict(robot=task.ROBOT, hand=HAND, plate=PLATE, payload=BOX, destination=task.DESTINATION),
                'payload_variant':payload_variant,
                'source_contact_checkpoint':dict(snapshot=str(SNAPSHOT), provenance=source['provenance'],
                                                  acceptance=source['acceptance']),
                'robot_station':pose(robot), 'surface':surface, 'hand_fk':pose(hand),
                'plate':pose(plate), 'payload':pose(payload), 'physics_validated':False,
                'plate_origin_above_table_m':clearance, 'preserved_plate_world_scale':list(old_scale),
                'assets':dict(plate=str(resource/'black_plate_325mm_filled_recess_v12.usda'),
                              payload_contact=(str(DIRECTORY/('saladbox.usda' if payload_variant=='salad_box' else 'foodbox.usda'))
                                  if payload_variant in PLACEMENT_VARIANTS else str(BEEF_ASSET_DIR/'beefbox_contact_standard243_v1.usda'))),
                'source_layer_sha256':hashes, 'randomization':record,
                'boundary':'Reset only; fixed PD after start; no attachment or runtime object pose writes'}
        if payload_variant in PLACEMENT_VARIANTS:
            label=payload_variant.removesuffix('_box')
            plan['acceptance']=f'candidate_reuses_accepted_plate_pose; {label}_static_only'
            plan['variant_adaptation']=f'Same robot/plate and payload-to-plate pose; canonical fridge {label} asset; no physics tuning'
        record.update(status='static_validated_physics_pending', physics_validated=False,
                      snapshot_sha256=digest(SNAPSHOT), surface=surface,
                      static_checks=dict(scale_error=scale_error, plate_to_hand_error=relative_error,
                                         payload_to_plate_error=payload_error, source_layers_unchanged=True),
                      pending=['physical bilateral hold', 'robot/environment clearance',
                               'placement reachability and full replay'])
        stage.GetRootLayer().customLayerData = {'arenaTask':'10.2', 'initialPlanJson':json.dumps(plan,sort_keys=True), 'physicsValidated':False}
        stage.GetRootLayer().Save()
        if any(digest(p) != h for p,h in hashes.items()):
            raise RuntimeError('Source layer changed during randomization')
        plan['scene_sha256'] = digest(out/'initial.usda')
        (out/'initial.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
        save_record()
        return record
    except Exception as exc:
        record.update(status='failed', error=str(exc), physics_validated=False)
        save_record()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base',type=Path)
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--reset-variant',choices=('accepted','l2_reference'),default='accepted')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output,args.seed,randomized=not args.fixed,base=args.base,reset_variant=args.reset_variant) if args.output else build_plan(args.seed), indent=2))

"""Continuous Task8 cloth clearing with native robot drives and live PhysX.

The initial cloth is a static visible collision mesh. This producer does not
simulate cloth removal. No target transforms are authored after initialization.
All attempts, including failures, retain commands, state and contact evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def base_initial_readback(actual, expected, nominal, *, sampled):
    """Compare XY in metres and wrapped yaw in degrees after physical settling."""
    actual, expected, nominal = [np.asarray(v, dtype=float) for v in (actual, expected, nominal)]
    valid = all(v.shape == (3,) and np.isfinite(v).all() for v in (actual, expected, nominal))
    if not valid:
        return dict(verified=False, reason="invalid finite XY/yaw readback")
    xy_error = float(np.max(np.abs(actual[:2] - expected[:2])))
    yaw_error = float(abs((actual[2] - expected[2] + 180.) % 360. - 180.))
    effective = bool(np.all(np.abs(actual[:2] - nominal[:2]) > 1e-6)
                     and abs((actual[2] - nominal[2] + 180.) % 360. - 180.) > 1e-6)
    return dict(verified=bool(sampled and effective and xy_error < .001 and yaw_error < .01),
                requested=expected.tolist(), nominal=nominal.tolist(), readback=actual.tolist(),
                xy_error_m=xy_error, yaw_error_deg=yaw_error,
                xy_tolerance_m=.001, yaw_tolerance_deg=.01,
                injection_effective=effective,
                measurement_note="Separate physical readback tolerances; sampler and injection ranges unchanged")


def main(task_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=task_id, choices=['8.2', '8.4', '8.6'])
    parser.add_argument('--count', type=int, choices=[1, 2, 3], default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--cloth-scene', type=Path, required=True)
    parser.add_argument('--render-every', type=int, default=4)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--episode-plan', type=Path, help='Pre-resolved production plan; omission is a partial-randomization diagnostic')
    args = parser.parse_args()
    if not args.task:
        parser.error('--task is required')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        run(args, app)
    except Exception:
        import traceback
        (args.output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            if getattr(args,'session_name',None):
                import session_scene
                session_scene.close_session(args.session_name)
        finally:
            app.close()


def run(args, app):
    import omni.usd
    import carb.settings
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.optimize import least_squares
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation, RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import omni.replicator.core as rep
    from PIL import Image
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'real2sim/teleop'))
    from real2sim.production.benchmark_eval_registry import resolve
    from real2sim.production.family4_surface_control import ArmKinematics, command_names
    from real2sim.teleop.task6_drawer_sequence_contract import FAMILY4_STANDARD_ARM_RAD
    from physical_distractors import RobotContactRecorder
    from real2sim.production.robot_collision_planner import CollisionModel, capture_current_gripper_convexes

    is_book = args.task == '8.4'
    out = args.output
    task = resolve(args.task)['task']
    policy_path = REPO / task['implementation']['policy']
    policy = json.loads(policy_path.read_text())
    episode_plan=None
    robot_draw=None
    if args.episode_plan:
        candidates=json.loads(args.episode_plan.read_text())['rows']
        matches=[row for row in candidates if row['task_id']==args.task and row['seed']==args.seed]
        if len(matches)!=1:raise ValueError('Expected one episode plan for this task and seed')
        episode_plan=matches[0]
        from importlib import import_module
        sampler=import_module(policy['robot_initial_state_sampler'].removesuffix('.py').replace('/','.'))
        sample_initial_perturbation=sampler.sample_initial_perturbation
        validate_initial_perturbation=sampler.validate_initial_perturbation
        robot_draw=episode_plan['reference_robot_draw']
        validate_initial_perturbation(robot_draw)
        if robot_draw!=sample_initial_perturbation(args.seed):
            raise ValueError('Shared Task8 robot draw differs from the delivered owner sampler')
        (out/'episode_plan.json').write_text(json.dumps(episode_plan,indent=2))
    environment = task['runtime']['environment']
    station = environment['robot_station']
    assert station['key'] == 'tablecloth'
    baseline = REPO / 'arena_scene/v1_2/demo_sim/scene_room_baseline.usda'
    frame_to_world=np.eye(4);world_to_frame=np.eye(4);prepared_environment=None
    if episode_plan:
        from real2sim.production import task8_environment
        stage,args.session_name=task8_environment.open_scene(args.task,args.seed,episode_plan['layout_id'])
    else:
        # Diagnostic episode: one baseline restore before any physical actions.
        source = Usd.Stage.Open(str(baseline))
        layer = source.Flatten()
        stage = Usd.Stage.Open(layer)
        del source
    stage.RemovePrim('/World/Robot')
    robot = stage.DefinePrim('/World/Robot', 'Xform')
    robot.GetReferences().AddReference(str(args.robot))
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler('z', station['yaw_deg'], degrees=True).as_matrix()
    root[:2, 3] = station['xy_m']
    nominal_root=root.copy()
    if robot_draw:
        delta=np.asarray(robot_draw['base_delta_xyyaw_m_deg'])
        root[:2,3]+=delta[:2]
        root[:3,:3]=Rotation.from_euler('z',delta[2],degrees=True).as_matrix()@root[:3,:3]
    UsdGeom.Xformable(robot).AddTransformOp().Set(Gf.Matrix4d(root.T.tolist()))
    cloth_root = stage.DefinePrim('/World/Assets/Task69Items', 'Xform')
    cloth_root.GetReferences().AddReference(str(args.cloth_scene), '/World/Assets/Task69Items')
    for child in cloth_root.GetChildren():
        if child.GetName() not in ['DirtyTablecloth', 'Looks']:
            child.SetActive(False)
    cloth = stage.GetPrimAtPath('/World/Assets/Task69Items/DirtyTablecloth')
    assert cloth and cloth.IsA(UsdGeom.Mesh)
    UsdPhysics.CollisionAPI.Apply(cloth).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(cloth).CreateApproximationAttr('none')

    target_mesh_points={}
    def bounds(path):
        if path.startswith('/World/Assets/Task8Clear/'):
            if path not in target_mesh_points:
                target_mesh_points[path]=[(str(p.GetPath()),np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)) for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).GetPurposeAttr().Get() not in ('proxy','guide')]
            points=[]
            for mesh_path,local_points in target_mesh_points[path]:
                matrix=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(mesh_path)).ComputeLocalToWorldTransform(0)).T
                points.append(local_points@matrix[:3,:3].T+matrix[:3,3])
            if points:
                vertices=np.concatenate(points)
                vertices=vertices@world_to_frame[:3,:3].T+world_to_frame[:3,3]
                return np.array([vertices.min(0),vertices.max(0)])
        box = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(stage.GetPrimAtPath(path))
        box.Transform(Gf.Matrix4d(world_to_frame.T.tolist()))
        b=box.ComputeAlignedRange()
        return np.array([b.GetMin(), b.GetMax()], dtype=float)

    cloth_bounds = bounds(str(cloth.GetPath()))
    table_bounds = bounds('/World/Assets/CoffeeTable')
    rng = np.random.default_rng(args.seed)
    from real2sim.production.task8_initial_sampling import sample_assets, sample_centres
    if args.count is None:
        args.count = int(rng.integers(1, 4))
    targets = []
    assets = sample_assets(policy['asset_pool'], args.count, rng)
    # Read inactive inventory through an anonymous session layer. Flattening the
    # House omits those descendants, including external material bindings.
    inventory = Usd.Stage.Open(str(baseline))
    inventory.SetEditTarget(inventory.GetSessionLayer())
    for asset in set(assets):
        inventory.GetPrimAtPath(asset).SetActive(True)
    if is_book:
        # Choose only compositions that fit at native scale; no object shrinking.
        sizes_by_asset = {}
        for index, candidate in enumerate(policy['asset_pool']):
            # Flatten omits inactive inventory; compose a temporary reference.
            probe = stage.DefinePrim(f'/World/Assets/Task8Measure/Probe{index}', 'Xform')
            probe.GetReferences().AddReference(str(baseline), candidate)
            probe.SetActive(True)
            orientation = probe.GetAttribute('xformOp:orient')
            old_orientation = orientation.Get()
            orientation.Set(Gf.Quatf(1))
            bb = bounds(str(probe.GetPath()))
            sizes_by_asset[candidate] = bb[1] - bb[0]
            orientation.Set(old_orientation)
            probe.SetActive(False)
        stage.RemovePrim('/World/Assets/Task8Measure')
        for _ in range(32):
            try:
                sample_centres([sizes_by_asset[a] for a in assets], cloth_bounds, rng)
                break
            except ValueError:
                assets = sample_assets(policy['asset_pool'], args.count, rng)
        else:
            raise ValueError('No sampled book composition fits the cloth')
    # Actual assets retain their native geometry/physics, with only initial placement changed.
    for i, asset in enumerate(assets):
        inventory.GetPrimAtPath(asset).SetActive(True)
        path = f'/World/Assets/Task8Clear/Item{i}'
        prim = stage.DefinePrim(path, 'Xform')
        prim.GetReferences().AddReference(str(baseline), asset)
        prim.SetActive(True)
        if is_book:
            prim.GetAttribute('xformOp:orient').Set(Gf.Quatf(1))
        # Subtree references cannot resolve bindings outside the referenced root.
        # Preserve the delivered material relationships against the shared House.
        for original in Usd.PrimRange(inventory.GetPrimAtPath(asset)):
            copied = stage.GetPrimAtPath(path + str(original.GetPath())[len(asset):])
            for relation in original.GetRelationships():
                if relation.GetName().startswith('material:binding'):
                    mapped = [p.ReplacePrefix(Sdf.Path(asset), Sdf.Path(path)) if p.HasPrefix(Sdf.Path(asset)) else p for p in relation.GetTargets()]
                    copied.CreateRelationship(relation.GetName()).SetTargets(mapped)
        b = bounds(path)
        size = b[1] - b[0]
        if not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError(f'Invalid asset bounds: {asset}')
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(True)
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False]
        if not colliders:
            # Inactive shelf inventory assets ship with collision disabled.
            # Enable existing geometry before the episode, never a hidden support.
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
                    colliders.append(child)
            if not colliders:
                for child in Usd.PrimRange(prim):
                    if child.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(child).ComputeVisibility() != 'invisible':
                        UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                        UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
                        colliders.append(child)
            if not colliders:
                raise ValueError(f'Asset has no physical geometry: {asset}')
        targets.append(path)
    target_assets = dict(zip(targets, assets))
    before = [bounds(path) for path in targets]
    centres = sample_centres([b[1] - b[0] for b in before], cloth_bounds, rng)
    for path, b, desired in zip(targets, before, centres):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        old = xf.ComputeLocalToWorldTransform(0)
        delta = Gf.Matrix4d(1)
        delta.SetTranslate(Gf.Vec3d(*(desired - b.mean(0))))
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='episodeInitial').Set(old * delta * xf.ComputeParentToWorldTransform(0).GetInverse())
    # Clear the front row before reaching behind it; a non-overlapping object
    # can still obstruct the native gripper's approach to a farther target.
    targets.sort(key=lambda p: (bounds(p)[0, 0], -bounds(p).mean(0)[1]))
    initial_bounds = np.array([bounds(t) for t in targets])
    for b in initial_bounds:
        if np.any(b[0, :2] < cloth_bounds[0, :2]) or np.any(b[1, :2] > cloth_bounds[1, :2]):
            raise ValueError('Initial target footprint is not on cloth')

    arms = command_names('l') + command_names('r')
    body = ['idx02_body_joint2', 'idx01_body_joint1', 'idx11_head_joint1', 'idx12_head_joint2']
    grips = ['idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1']
    actnames = arms + body + grips
    command = np.r_[FAMILY4_STANDARD_ARM_RAD, [.55, .02, 0., .43634], [.7843981634, .7843981634]]
    nominal_command=command.copy()
    if robot_draw:command[:14]+=np.asarray(robot_draw['arm_delta_rad'])
    # Initial positions are state; this does not modify native drive gains or joint limits.
    for p in Usd.PrimRange(robot):
        if not p.IsA(UsdPhysics.Joint) or p.GetName() not in actnames:
            continue
        kind = 'linear' if p.IsA(UsdPhysics.PrismaticJoint) else 'angular'
        v = float(command[actnames.index(p.GetName())]) * (1 if kind == 'linear' else 180/np.pi)
        p.AddAppliedSchema('PhysicsJointStateAPI:' + kind)
        p.CreateAttribute(f'state:{kind}:physics:position', Sdf.ValueTypeNames.Float).Set(v)
        p.CreateAttribute(f'state:{kind}:physics:velocity', Sdf.ValueTypeNames.Float).Set(0.)
        attr = p.GetAttribute(f'drive:{kind}:physics:targetPosition')
        if attr and attr.Get() is not None:
            attr.Set(v)
    if episode_plan:
        stage,prepared_environment=task8_environment.prepare(stage,task,targets,str(cloth.GetPath()),episode_plan,out)
        frame_to_world=np.asarray(prepared_environment['frame_to_world'])
        world_to_frame=np.linalg.inv(frame_to_world)
    world_root=frame_to_world@root
    initial_path = out / 'initial_scene.usdc'
    # Flatten resolves external references for this immutable episode snapshot.
    stage.Flatten().Export(str(initial_path))
    ctx = omni.usd.get_context()
    ctx.open_stage(str(initial_path))
    while ctx.get_stage_loading_status()[2]:
        app.update()
    stage = ctx.get_stage()
    robot = stage.GetPrimAtPath('/World/Robot')
    carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
    contacts = RobotContactRecorder(stage)
    for path in targets:
        contacts.paths.add(path)
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    sim = SimulationContext(physics_dt=1/60, rendering_dt=1/20)
    sim.reset()
    art = SingleArticulation('/World/Robot')
    art.initialize()
    names = list(art.dof_names)
    indices = np.array([names.index(n) for n in actnames])
    q = art.get_joint_positions().copy()
    q[indices] = command
    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros(len(q)))
    models = {h: ArmKinematics(args.robot, names, h) for h in ['l', 'r']}
    cooked = capture_current_gripper_convexes(stage, out/'cooked_gripper.json')
    collision_model = CollisionModel(stage, names, world_root, cooked)
    from real2sim.production.task8_environment import is_base_support_contact
    # Generated layouts retain real floor collisions. Extend the planner's
    # existing canonical base-support recognition to their authored floor paths.
    base_support_floors={str(p.GetPath()) for p in stage.Traverse()
                        if str(p.GetPath()).startswith('/World/HouseTopologyCandidate/Shell/Floors/')
                        and p.HasAPI(UsdPhysics.CollisionAPI)
                        and p.GetAttribute('physics:collisionEnabled').Get() is not False}
    cameras = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(robot) if p.IsA(UsdGeom.Camera)}
    views = []
    for name in ['Head_Camera', 'Left_Camera', 'Right_Camera']:
        product = rep.create.render_product(cameras[name], (640, 400) if name == 'Head_Camera' else (424, 240))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach([product])
        views.append((name, annotator))
    (out / 'frames').mkdir()

    def configuration():
        result = {}
        for p in stage.Traverse():
            is_robot = str(p.GetPath()).startswith('/World/Robot')
            is_camera = p.IsA(UsdGeom.Camera)
            for a in p.GetAttributes():
                n = a.GetName()
                if is_camera or (is_robot and n.startswith(('physics:', 'physx', 'drive:'))):
                    if not any(k in n for k in ['targetPosition', 'targetVelocity', 'velocity', 'angularVelocity', 'physxContactReport']):
                        result[str(a.GetPath())] = str(a.Get())
        return result

    invariant = configuration()
    all_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith('/World/Robot') and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
    body_view=RigidPrim(all_bodies,reset_xform_properties=False,prepare_contact_sensors=False,disable_stablization=False)
    body_view.initialize()
    body_indices=[body_view.prim_paths.index(path) for path in all_bodies]
    body_velocities=[]
    # Resolve native mimic joints and settle initial placements before frame zero.
    # This is episode preparation; no object pose or velocity is assigned here.
    for _ in range(120):
        art.apply_action(ArticulationAction(joint_positions=command,joint_indices=indices))
        sim.step(render=False)
    # Warm all three annotators without advancing physics, so frame zero is
    # present in every camera stream rather than missing from delayed sensors.
    for _ in range(20):
        sim.render()
        if all(an.get_data().size for _,an in views):
            break
    else:
        raise RuntimeError('Native camera annotators did not become ready')
    contacts.events.clear()
    initial_joint_position=art.get_joint_positions().copy()
    initial_root_position,initial_root_orientation=art.get_world_pose()
    actual_arm=initial_joint_position[indices[:14]]
    actual_yaw=Rotation.from_quat(np.asarray(initial_root_orientation)[[1,2,3,0]]).as_euler('xyz',degrees=True)[2]
    actual_base=np.r_[np.asarray(initial_root_position)[:2],actual_yaw]
    expected_base=np.r_[world_root[:2,3],Rotation.from_matrix(world_root[:3,:3]).as_euler('xyz',degrees=True)[2]]
    nominal_world_root=frame_to_world@nominal_root
    nominal_base=np.r_[nominal_world_root[:2,3],Rotation.from_matrix(nominal_world_root[:3,:3]).as_euler('xyz',degrees=True)[2]]
    actual_bounds=np.array([bounds(t) for t in targets])
    initial_randomization={'axes':{
        'arm_initial_state':{'verified':bool(robot_draw and np.all(np.abs(actual_arm-nominal_command[:14])>1e-6) and np.max(np.abs(actual_arm-command[:14]))<.002),'requested':command[:14].tolist(),'nominal':nominal_command[:14].tolist(),'readback':actual_arm.tolist()},
        'base_initial_xy_yaw':base_initial_readback(actual_base,expected_base,nominal_base,sampled=bool(robot_draw)),
        'interaction_object_initial_pose':{'verified':bool(np.all(actual_bounds[:,0,:2]>=cloth_bounds[0,:2]) and np.all(actual_bounds[:,1,:2]<=cloth_bounds[1,:2])),'requested':initial_bounds.tolist(),'readback':actual_bounds.tolist()},
    },'source_sampler':policy['robot_initial_state_sampler'] if robot_draw else None,'complete':False,'physics_steps_before_readback':120}
    if prepared_environment:
        initial_randomization['axes'].update(task8_environment.readback(stage,prepared_environment))
        initial_randomization['complete']=all(a['verified'] for a in initial_randomization['axes'].values())
        if not initial_randomization['complete']:
            (out/'randomization_readback.json').write_text(json.dumps(initial_randomization,indent=2))
            raise RuntimeError('Fresh-physics randomization readback failed')
    (out/'randomization_readback.json').write_text(json.dumps(initial_randomization,indent=2))
    (out/'initial_conditions.json').write_text(json.dumps({'seed':args.seed,'target_assets':target_assets,'sampled_initial_bounds':initial_bounds.tolist(),'preparation_physics_steps':120,'robot_dof_names':names,'joint_position':initial_joint_position.tolist(),'actuated_joint_names':actnames,'command':command.tolist(),'target_bounds':[bounds(t).tolist() for t in targets],'object_pose_or_velocity_writes_during_settle':False},indent=2))
    def world_pose(path):
        return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T
    def world(path):
        # Planning remains in the catalog's canonical task frame.
        return world_to_frame@world_pose(path)
    states, velocities, poses, rotations, commands, target_bounds, phases = [], [], [], [], [], [], []
    events = []
    planning_recoveries=[]
    failure = None
    def tick(phase):
        contacts.frame = len(states)
        art.apply_action(ArticulationAction(joint_positions=command, joint_indices=indices))
        for _ in range(3):
            sim.step(render=False)
        states.append(art.get_joint_positions().copy())
        velocities.append(art.get_joint_velocities().copy())
        body_velocities.append(body_view.get_velocities()[body_indices].copy())
        mats = [world_pose(p) for p in all_bodies]
        poses.append([m[:3, 3] for m in mats])
        rotations.append([Rotation.from_matrix(m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)).as_quat()[[3,0,1,2]] for m in mats])
        commands.append(command.copy())
        target_bounds.append([bounds(t) for t in targets])
        phases.append(phase)
        f = len(states)-1
        if f % args.render_every == 0:
            sim.render(); sim.render()
            for name, an in views:
                image = an.get_data()
                if image.size:
                    Image.fromarray(image[:,:,:3]).save(out/'frames'/f'{f:05d}_{name}.jpg')
        if f % 100 == 0:
            if configuration() != invariant:
                raise RuntimeError('Shared robot/camera configuration changed')
            (out/'progress.json').write_text(json.dumps({'frame':f,'phase':phase,'target_centres':[bounds(t).mean(0).tolist() for t in targets]}))
            print('FRAME', f, phase, flush=True)
        for event in contacts.events.values():
            if event['frame'] != f or event['minimum_separation_m'] >= -.005:
                continue
            actors = [event['actor0'], event['actor1']]
            if all(a.startswith('/World/Robot/') for a in actors) and not any(all('/gripper_'+hand+'_' in a for a in actors) for hand in ('l','r')):
                raise RuntimeError(f'Self collision during {phase}: {actors}')
    def hold(seconds, phase):
        for _ in range(round(seconds*20)):
            tick(phase)
    def wrist(hand):
        return world(f'/World/Robot/arm_{hand}_end_link')
    def move(hand, centre, rotation, seconds, phase, next_centre=None, joint_goal=None):
        # Native pad midpoint is ~127 mm along wrist +Z; no gripper scale changes.
        local = np.array([0., 0., .127])
        model = models[hand]
        fixed = art.get_joint_positions().astype(np.float64)
        start = wrist(hand)
        transport = phase.endswith(':carry_left')
        released_target_obstacle = phase.endswith((':retreat', ':return_home'))
        # Empty-hand withdrawal may change elbow posture after release.
        # Search a fully checked joint path instead of forcing a vertical IK branch.
        joint_route = phase.endswith('approach') or transport or released_target_obstacle
        held_local = None
        if transport:
            held_bounds=bounds(target)
            corners=np.array([[x,y,z] for x in held_bounds[:,0] for y in held_bounds[:,1] for z in held_bounds[:,2]])
            held_local=(corners-start[:3,3])@start[:3,:3]
        nearby_bodies=[p for p in all_bodies if p in targets or np.linalg.norm(bounds(p).mean(0)[:2]-root[:2,3])<1.5]
        obstacle_bounds=[bounds(p) for p in nearby_bodies if p!=target]
        desired = np.array(centre) - np.asarray(rotation) @ local
        slerp = Slerp([0,1],Rotation.from_matrix([start[:3,:3],rotation]))
        seed = command[:7].copy() if hand=='l' else command[7:14].copy()
        offset = 0 if hand=='l' else 7
        if phase.endswith(':lift') or released_target_obstacle:
            # Physical closure may backdrive an arm joint under contact. Start
            # transport from measured state instead of jumping toward stale IK.
            measured = fixed[model.indices].copy()
            mismatch = float(np.max(np.abs(measured - seed)))
            if mismatch > .02:
                seed = measured
                command[offset:offset+7] = measured
                planning_recoveries.append({'phase':phase,'frame':len(states),'kind':'measured_lift_start','max_tracking_error_rad':mismatch})
        def solve_pose(dest, rot, initial, continuity=.004):
            prior = initial.copy()
            if args.task=="8.4" and "book_" not in phase:
                continuity=0.
            def residual(x):
                qc = fixed.copy(); qc[model.indices] = x
                m = model.forward(qc[None],root[None])[0]
                return np.r_[5*(m[:3,3]-dest),Rotation.from_matrix(rot@m[:3,:3].T).as_rotvec(),continuity*(x-prior)]
            fit = least_squares(residual,np.clip(initial,model.limits[:,0]+1e-6,model.limits[:,1]-1e-6),bounds=(model.limits[:,0]+1e-7,model.limits[:,1]-1e-7),max_nfev=180)
            return fit.x, float(np.linalg.norm(residual(fit.x)[:6]))
        approach_obstacles=[];protected_obstacles=[]
        for obstacle in nearby_bodies:
            include_active=(phase.endswith('approach') and 'handoff' not in phase) or released_target_obstacle
            if obstacle==target and not include_active:
                continue
            destination_shapes=approach_obstacles if obstacle==target else protected_obstacles
            for prim in Usd.PrimRange(stage.GetPrimAtPath(obstacle)):
                if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
                    destination_shapes.extend(collision_model.make_shapes(prim,world_pose(str(prim.GetPath()))))
        def collision_hits(qc,with_targets=False,max_hits=3,allow_target_pad=False,open_hand=False):
            if open_hand:
                qc=qc.copy()
                for joint_index,joint_name in enumerate(names):
                    if 'gripper_'+hand+'_' in joint_name:
                        qc[joint_index]+=float(open_hand)*(initial_joint_position[joint_index]-qc[joint_index])
            with_targets=with_targets or released_target_obstacle
            old=collision_model.static_shapes
            collision_model.static_shapes=old+protected_obstacles+(approach_obstacles if with_targets else [])
            try:
                hits=collision_model.check(qc,max_hits=32,allow_base_floor=True,pad_touch_tolerance_m=.00025,clearance_m=.003 if open_hand else 0.)
                hits=[hit for hit in hits if not is_base_support_contact(hit,base_support_floors)]
                if allow_target_pad:
                    hits=[hit for hit in hits if not (hit.get('b','').startswith(target+'/') and hit.get('a','').startswith(f'/World/Robot/gripper_{hand}_') and '_link4/' in hit.get('a',''))]
                return hits[:max_hits]
            finally:
                collision_model.static_shapes=old
        def clear(joints,with_targets=False,check_held=True,allow_target_pad=False):
            qc=fixed.copy();qc[model.indices]=joints
            if held_local is not None and check_held:
                transform=model.forward(qc[None],root[None])[0]
                corners=held_local@transform[:3,:3].T+transform[:3,3]
                low=corners.min(0);high=corners.max(0)
                if low[2]<table_bounds[1,2]+.025 or high[2]>1.30:
                    return False
                if any(np.all(high>other[0]-.008) and np.all(low<other[1]+.008) for other in obstacle_bounds):
                    return False
            return not collision_hits(qc,with_targets,max_hits=1,allow_target_pad=allow_target_pad)
        def command_step(joints):
            previous=command[offset:offset+7].copy()
            # Bound joint speed to 0.6 rad/s including segment transitions.
            count=max(1,int(np.ceil(np.max(np.abs(joints-previous))/.03)))
            for weight in np.linspace(0,1,count+1)[1:]:
                proposed=previous+weight*(joints-previous)
                if not clear(proposed,phase.endswith('approach')):
                    qc=fixed.copy();qc[model.indices]=proposed
                    hits=collision_hits(qc,phase.endswith('approach'))
                    (out/'rejected_motion.json').write_text(json.dumps({'phase':phase,'dof_names':names,'configuration':qc.tolist(),'hits':hits},indent=2))
                    raise RuntimeError(f'Collision predicted before {hand} command in {phase}: {hits}')
                command[offset:offset+7]=proposed
                tick(phase)
        if joint_route:
            found=None;detour_goals=[]
            diagnostics=[]
            for trial in range(1 if joint_goal is not None else 36):
                if trial%6==0:print('PLAN',phase,hand,trial,flush=True)
                trial_seed=seed if trial==0 else rng.uniform(model.limits[:,0],model.limits[:,1])
                if transport and trial%2 and next_centre is not None:
                    low=np.asarray(next_centre).reshape(-1,3)[-1]-np.asarray(rotation)@local
                    trial_seed,low_error=solve_pose(low,rotation,trial_seed,0.)
                    if low_error>.015 or not clear(trial_seed,check_held=False):
                        continue
                if args.task=='8.6' and hand=='l' and phase.endswith(':approach') and abs(rotation[2,2])<.2 and trial==1:
                    trial_seed=np.array([2.237228410350286,-.6194110377028877,.6577647610823597,1.8684822424935479,.5627610782648733,-.4311881818234261,.05767039352794542])
                if args.task=='8.6' and hand=='l' and phase.endswith(':approach') and trial==2:
                    trial_seed=np.array([2.3078973644595755,-1.517311387245936,-1.496342905724815,-1.7295166955405543,-1.8425348649394502,1.67091981565527,.08091706730562762])
                if phase.endswith('book_right_park_approach') and trial==1:
                    trial_seed=np.array(FAMILY4_STANDARD_ARM_RAD[7:])
                if args.task=='8.4' and phase.endswith('book_drag_approach') and hand=='l' and trial==1:
                    trial_seed=np.array([-2.139855482073432,1.5947771035552547,1.4230981362880464,-.6683919840487949,-1.46671209790855,-1.67685041069352,2.453339489636054])
                if args.task=='8.4' and phase.endswith(':approach') and hand=='l' and trial==1:
                    # A feasible native-arm posture found from edge-insertion planning.
                    # It is only an IK seed; the current pose and full path are rechecked.
                    trial_seed=np.array([-.596652,.722921,1.564566,-.738001,2.638739,-1.466873,3.042294])
                candidate,error=(np.asarray(joint_goal,dtype=float),0.) if joint_goal is not None else solve_pose(desired,rotation,trial_seed,0.)
                if error>.01 or not clear(candidate,True):
                    qc=fixed.copy();qc[model.indices]=candidate
                    diagnostics.append({'trial':trial,'stage':'endpoint','residual':error,'hits':collision_hits(qc,True) if error<=.01 else []})
                    continue
                if next_centre is not None:
                    preview=candidate.copy();reachable=True;start_goal=desired
                    for waypoint_index,waypoint in enumerate(np.asarray(next_centre).reshape(-1,3)):
                        goal=waypoint-np.asarray(rotation)@local
                        for weight in np.linspace(0,1,25)[1:]:
                            preview,error=solve_pose((1-weight)*start_goal+weight*goal,rotation,preview)
                            check_active=not is_book and phase.endswith(':approach') and waypoint_index==0
                            if error>.015 or not clear(preview,with_targets=check_active,check_held=not transport,allow_target_pad=check_active):
                                qc=fixed.copy();qc[model.indices]=preview
                                diagnostics.append({'trial':trial,'stage':'preview','goal':goal.tolist(),'weight':float(weight),'residual':error,'hits':collision_hits(qc,with_targets=check_active,allow_target_pad=check_active) if error<=.015 else []})
                                reachable=False;break
                        if not reachable:break
                        start_goal=goal
                    if not reachable:
                        continue
                    if transport:
                        # Reserve the native fully opened finger envelope at the
                        # lower pose before carrying beside already placed items.
                        released=fixed.copy();released[model.indices]=preview
                        release_hits=[]
                        for opening_fraction in (.25,.5,.75,1.):
                            release_hits=collision_hits(released,open_hand=opening_fraction)
                            if release_hits:break
                        if release_hits:
                            diagnostics.append({'trial':trial,'stage':'release_open','hits':release_hits})
                            continue
                if all(clear(seed+t*(candidate-seed),True) for t in np.linspace(0,1,max(30,int(np.max(abs(candidate-seed))/.025)))):
                    found=[candidate];break
                detour_goals.append(candidate)
            if found is None and detour_goals:
                from real2sim.production.joint_path_planner import plan_joint_path
                print('DETOUR',phase,hand,len(detour_goals),flush=True)
                for goal in sorted(detour_goals,key=lambda q:np.linalg.norm(q-seed))[:8]:
                    path=plan_joint_path(seed,goal,model.limits,lambda q:clear(q,True),rng,
                                         iterations=2200,seconds=90.)
                    if path is not None:
                        found=path[1:]
                        planning_recoveries.append({'phase':phase,'frame':len(states),'kind':'joint_detour','waypoints':[q.tolist() for q in path]})
                        break
            if found is None:
                with (out/'planning_failures.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase':phase,'hand':hand,'frame':len(states),'desired_pad':np.asarray(centre).tolist(),'rotation':np.asarray(rotation).tolist(),'trials':diagnostics,'detour_goals':len(detour_goals),'start_valid':clear(seed,True),'start_hits':collision_hits(fixed,True)})+'\n')
                raise RuntimeError(f'No collision-free approach for {hand} in {phase}')
            previous=seed.copy()
            for waypoint in found:
                count=max(round(seconds*20/len(found)),int(np.max(np.abs(waypoint-previous))/.012)+2)
                for t in np.linspace(0,1,count):
                    weight=10*t**3-15*t**4+6*t**5
                    command_step(previous+weight*(waypoint-previous))
                previous=waypoint.copy()
            hold(.5,phase)
            return
        def local_edge_clear(a,b,dest,rot):
            for weight in np.linspace(0,1,max(2,int(np.ceil(np.max(abs(b-a))/.02))+1)):
                joints=a+weight*(b-a)
                if not clear(joints):return False
                qc=fixed.copy();qc[model.indices]=joints
                m=model.forward(qc[None],root[None])[0]
                if np.linalg.norm(m[:3,3]-dest)>.03 or np.linalg.norm(Rotation.from_matrix(rot@m[:3,:3].T).as_rotvec())>.15:
                    return False
            return True
        # Correct small changes of IK branch caused by physical loading. The
        # complete joint edge must remain collision-free and near the grasp pose.
        for t in np.linspace(0,1,max(2,round(seconds*20))):
            t = 10*t**3-15*t**4+6*t**5
            dest = (1-t)*start[:3,3]+t*desired
            rot = slerp([t]).as_matrix()[0]
            seed,error=solve_pose(dest,rot,seed)
            previous=command[offset:offset+7].copy()
            if error>.025 or not local_edge_clear(previous,seed,dest,rot):
                recovered=None
                for trial in range(24):
                    spread=(.12,.25,.5,.8)[trial//6]
                    proposal=previous+rng.normal(0,spread,7)
                    candidate,candidate_error=solve_pose(dest,rot,proposal,0.)
                    if candidate_error<.015 and local_edge_clear(previous,candidate,dest,rot):
                        recovered=candidate
                        planning_recoveries.append({'phase':phase,'frame':len(states),'trial':trial,'max_joint_change_rad':float(np.max(abs(candidate-previous))),'pose_residual':candidate_error})
                        print('REPLAN',phase,trial,flush=True)
                        break
                if recovered is None:
                    qc=fixed.copy();qc[model.indices]=seed
                    hits=collision_hits(qc)
                    (out/'rejected_motion.json').write_text(json.dumps({'phase':phase,'dof_names':names,'configuration':qc.tolist(),'hits':hits,'pose_residual':error},indent=2))
                    raise RuntimeError(f'No safe local IK continuation in {phase}: residual={error:.4f}, hits={hits}')
                seed=recovered
            command_step(seed)
        hold(.3,phase)
    def grip(hand,closed,phase,open_angle=.7843981634):
        idx=18 if hand=='l' else 19
        initial=command[idx]
        for t in np.linspace(0,1,30):
            command[idx]=initial+t*((0. if closed else open_angle)-initial)
            tick(phase)
        hold(.5,phase)
    try:
        hold(3.,'settle')
        if args.inspect_only:
            return
        right_park=wrist('r').copy()
        left_cursor=cloth_bounds[1,1]+.010
        for i,target in enumerate(targets):
            b=bounds(target);centre=b.mean(0);size=b[1]-b[0]
            hand='l'
            lift_height=.12 if is_book else (size[2]+.04 if size[2]>.18 else .16)
            rot=np.diag([1.,-1.,-1.])
            if not is_book and size[0] > size[1] + .02:
                rot=Rotation.from_euler('z',90,degrees=True).as_matrix()@rot
            grasp=centre.copy()
            # Tall rigid containers are gripped near their upper half.
            if size[2]>.08:grasp[2]=b[0,2]+max(size[2]*.65,size[2]-.045)
            elif not is_book:grasp[2]=b[0,2]+min(size[2]-.004,.029)
            if is_book and size[2]<.025:
                # Create a real underside gap by friction-dragging the book to
                # the near table edge before pinching. Nothing is pose-written.
                paddle='r' if centre[1]<root[1,3] else 'l'
                dragrot=np.diag([1.,-1.,-1.]) if paddle=='r' else np.diag([-1.,1.,-1.])
                press=centre.copy();press[2]=b[1,2]+.0245
                overhang=min(.045,size[0]/2-.025)
                if overhang<.025:
                    raise RuntimeError('Book too narrow for a supported edge pinch')
                displacement=min(0.,table_bounds[0,0]-overhang-b[0,0])
                if paddle=='l' and displacement<-.17:
                    # Reapproach between short physical strokes so a far book
                    # does not force one elbow branch through the entire drag.
                    # Preview contact AND the complete stroke before descending.
                    for stroke in range(10):
                        current=bounds(target)
                        if current[0,0]<=table_bounds[0,0]-.025:break
                        press=current.mean(0);press[2]=current[1,2]+.0245
                        dx=-min(.055,max(.02,current[0,0]-(table_bounds[0,0]-overhang)))
                        destination=press+[dx,0,0]
                        grip(paddle,False,f'{i}:book_paddle_prepare',open_angle=.12)
                        prepared=False
                        for pitch in (0.,-15.,-25.,-40.):
                            for yaw in (0.,20.,-20.):
                                candidate=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('y',pitch,degrees=True).as_matrix()@np.diag([-1.,1.,-1.])
                                try:
                                    move(paddle,press-candidate[:,2]*.08,candidate,3.,f'{i}:book_drag_approach',next_centre=[press,destination,destination-candidate[:,2]*.04])
                                    dragrot=candidate;prepared=True;break
                                except RuntimeError as error:
                                    if not str(error).startswith('No collision-free approach'):raise
                            if prepared:break
                        if not prepared:
                            raise RuntimeError(f'No collision-free complete book stroke {stroke}')
                        move(paddle,press,dragrot,2.,f'{i}:book_surface_contact')
                        move(paddle,destination,dragrot,1.5,f'{i}:book_drag_to_edge')
                        hold(.3,f'{i}:book_edge_check')
                        measured=wrist(paddle)
                        pad=measured[:3,3]+measured[:3,:3]@np.array([0,0,.127])
                        move(paddle,pad-dragrot[:,2]*.04,dragrot,2.,f'{i}:book_paddle_clear')
                        if current[0,0]-bounds(target)[0,0]<.006:
                            raise RuntimeError('Book short friction stroke made insufficient physical progress')
                    if bounds(target)[0,0]>table_bounds[0,0]-.025:
                        raise RuntimeError('Book short strokes did not create a real underside gap')
                else:
                    prepared=False
                    for height,pitch in [( .16,0.),(.08,0.),(.08,-25.),(.06,-40.)]:
                        if pitch:
                            # The tilted paddle uses this opening at surface contact.
                            # Set it while clear of the table so the approach preview
                            # checks the same physical finger geometry as the drag.
                            grip(paddle,False,f'{i}:book_paddle_prepare',open_angle=.12)
                        candidate=Rotation.from_euler('y',pitch,degrees=True).as_matrix()@dragrot
                        try:
                            move(paddle,press+[0,0,height],candidate,3.,f'{i}:book_drag_approach',next_centre=press)
                            dragrot=candidate;prepared=True;break
                        except RuntimeError as error:
                            if not str(error).startswith('No collision-free approach'):raise
                    if not prepared:
                        raise RuntimeError(f'No supported-book preparation approach for {paddle}')
                    grip(paddle,False,f'{i}:book_paddle_close',open_angle=.12)
                    move(paddle,press,dragrot,2.,f'{i}:book_surface_contact')
                    # Keep the centre of mass on the table. The previous fixed 8cm
                    # overhang exceeds half the width of a native 15cm notebook.
                    overhang=min(.045,size[0]/2-.025)
                    if overhang<.025:
                        raise RuntimeError('Book too narrow for a supported edge pinch')
                    displacement=min(0.,table_bounds[0,0]-overhang-b[0,0])
                    drag_destination_rot=np.diag([-1.,1.,-1.]) if paddle=='l' and abs(dragrot[0,2])>.01 else dragrot
                    move(paddle,press+[displacement,0,0],drag_destination_rot,3.,f'{i}:book_drag_to_edge')
                    dragrot=drag_destination_rot
                    hold(.5,f'{i}:book_edge_check')
                    if bounds(target)[0,0]>table_bounds[0,0]-.025:
                        raise RuntimeError('Book friction drag did not create a real underside gap')
                    # The tilted right paddle must clear the thin book before
                    # parking; lifting 15cm here folds its wrist into the head.
                    move(paddle,press+[displacement,0,.07 if paddle=='r' else .15],dragrot,2.,f'{i}:book_paddle_clear')
                grip(paddle,False,f'{i}:book_open')
                if paddle=='r':
                    # Clear the receiving arm's edge-pinch/lift corridor by motion.
                    move('r',right_park[:3,3]+right_park[:3,:3]@np.array([0,0,.127]),right_park[:3,:3],3.,f'{i}:book_right_park_approach')
                b=bounds(target);centre=b.mean(0);size=b[1]-b[0];grasp=centre.copy()
                if b[0,2]<table_bounds[1,2]-.02 or centre[0]<table_bounds[0,0]+.015:
                    raise RuntimeError('Book lost stable tabletop support during edge preparation')
            if is_book:
                # Thin books require a real edge pinch. Failure remains a failed candidate.
                rot=Rotation.from_euler('z',-30,degrees=True).as_matrix()@Rotation.from_euler('y',10,degrees=True).as_matrix()@np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
                # Measured drag can undershoot the requested overhang. Keep the
                # lower pad outside the table even after physical arm deflection.
                grasp[0]=min(b[0,0]+.014,table_bounds[0,0]-.020)
                grasp[2]-=.004
            candidates=[rot]
            if is_book:
                candidates.append(Rotation.from_euler('z',-30,degrees=True).as_matrix()@rot)
            if not is_book:
                candidates.append(Rotation.from_euler('z',180,degrees=True).as_matrix()@rot)
            if not is_book and size[2]>.08:
                # Close across the narrow package dimension. The old oblique
                # side entry put the proximal finger into the broad face.
                side_rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
                if size[0]+.02<size[1]:
                    side_rotation=Rotation.from_euler('z',90,degrees=True).as_matrix()@side_rotation
                side_candidates=[]
                for yaw in (0.,-30.,30.,-60.,60.):
                    candidate=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@side_rotation
                    jaw_width=abs(candidate[0,0])*size[0]+abs(candidate[1,0])*size[1]
                    if jaw_width<=.10:side_candidates.append(candidate)
                # Far objects need an inclined wrist while the jaw still closes
                # across their narrow dimension. Endpoints and the full descent/
                # lift path are tested against the current accumulated scene.
                for pitch in (20.,35.):
                    for yaw in (-30.,0.,30.,-60.,60.):
                        candidate=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('y',pitch,degrees=True).as_matrix()@side_rotation
                        jaw_width=abs(candidate[0,0])*size[0]+abs(candidate[1,0])*size[1]
                        if jaw_width<=.10:side_candidates.append(candidate)
                candidates=side_candidates+candidates if size[2]>.18 else candidates+side_candidates
            left_ready=False;side_pick=False
            approaches=[]
            for candidate in candidates:
                approaches.append((candidate,None))
                if not is_book and abs(candidate[2,2])<.9 and size[2]<=.18:
                    # A shorter initial lift avoids folding a nearby grasp into
                    # the upper arm. The held-object planner then finds transport.
                    approaches.append((candidate,.08))
            for candidate,lift_override in approaches:
                try:
                    pregrasp=grasp-candidate[:,2]*.14 if is_book else grasp+[0,0,.15]
                    candidate_side=not is_book and abs(candidate[2,2])<.9
                    candidate_lift=.26 if candidate_side and size[2]>.18 else lift_height
                    if lift_override is not None:candidate_lift=lift_override
                    candidate_grasp=grasp.copy()
                    # Broad packages need a near-face clamp: their centre is
                    # deeper than the native distal pads can reach unobstructed.
                    if candidate_side and size[0]>.12 and size[1]<.10 and size[2]>.18:
                        candidate_grasp[0]=b[0,0]+.020
                        candidate_grasp[2]=b[0,2]+min(.20,size[2]-.04)
                        candidate_lift=.12
                    if candidate_side:pregrasp=candidate_grasp-candidate[:,2]*.10
                    move('l',pregrasp,candidate,3.,f'{i}:approach',next_centre=[candidate_grasp,candidate_grasp+[0,0,candidate_lift]])
                    if lift_override is not None:
                        planning_recoveries.append({'phase':f'{i}:approach','kind':'short_initial_side_lift','lift_m':candidate_lift})
                    grasp=candidate_grasp;rot=candidate;lift_height=candidate_lift;side_pick=candidate_side;left_ready=True;break
                except RuntimeError as error:
                    if not str(error).startswith('No collision-free approach'):
                        raise
            used_handoff=not left_ready
            if not left_ready:
                hand='r'
                # The right hand only lifts into the nearby transfer station.
                # The taller clearance is needed for a left-hand carry over the row.
                if size[2]>.18:lift_height=.16
                if size[2]>.08:
                    # Side-on lower clamp leaves the upper rim free for the left hand.
                    rot=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
                    grasp[2]=b[0,2]+size[2]*.35
                elif not is_book:
                    # Grip the near end from the side, leaving a separate far-end
                    # clamp site for the receiving left gripper.
                    angle=np.deg2rad(15.)
                    rot=np.array([[1.,0.,0.],[0.,-np.sin(angle),np.cos(angle)],[0.,-np.cos(angle),-np.sin(angle)]])
                    grasp[1]=centre[1]-.04
                    grasp[2]=b[0,2]+.028
                right_candidates=[(grasp.copy(),rot.copy(),'original')]
                if not is_book and size[2]>.08:
                    # Try a near-face clamp before reaching through a broad pack.
                    # All candidates preview native-finger collision, descent and lift.
                    for pitch in (0.,20.,-20.,35.,-35.,50.,65.):
                        for yaw in (0.,-10.,10.,-30.,30.):
                            candidate_rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('y',pitch,degrees=True).as_matrix()@rot
                            jaw_width=abs(candidate_rot[0,0])*size[0]+abs(candidate_rot[1,0])*size[1]
                            if jaw_width>.10:continue
                            candidate_grasp=grasp.copy()
                            axis=candidate_rot[:,2].copy();axis[2]=0.
                            axis/=np.linalg.norm(axis)
                            depth=abs(axis[0])*size[0]+abs(axis[1])*size[1]
                            if depth>.12:
                                candidate_grasp[:2]-=axis[:2]*(depth/2-.020)
                            right_candidates.append((candidate_grasp,candidate_rot,'native_near_face'))
                            if size[2]>.18:
                                # A low clamp can fold the right elbow into the
                                # torso. Search higher physical clamp sites on
                                # the same package without changing its pose.
                                for height_fraction in (.55,.70):
                                    higher_grasp=candidate_grasp.copy()
                                    higher_grasp[2]=b[0,2]+size[2]*height_fraction
                                    right_candidates.append((higher_grasp,candidate_rot,'native_near_face_height'))
                    if size[0]>.12 or size[1]>.12:
                        right_candidates=right_candidates[1:]+right_candidates[:1]
                ready=False
                for candidate_grasp,candidate_rot,kind in right_candidates:
                    pregrasp_height=max(.15,b[1,2]+.07-candidate_grasp[2])
                    entries=[candidate_grasp+[0,0,pregrasp_height],candidate_grasp-candidate_rot[:,2]*.08]
                    if size[2]>.18:entries.reverse()
                    for entry in entries:
                        try:
                            move('r',entry,candidate_rot,3.,f'{i}:approach',next_centre=[candidate_grasp,candidate_grasp+[0,0,lift_height]])
                            grasp=candidate_grasp;rot=candidate_rot;ready=True
                            planning_recoveries.append({'phase':f'{i}:approach','kind':kind,'hand':'right','grasp':grasp.tolist(),'rotation':rot.tolist()})
                            break
                        except RuntimeError as error:
                            if not str(error).startswith('No collision-free approach'):raise
                    if ready:break
                if not ready:raise RuntimeError(f'No collision-free approach for r in {i}:approach')
            move(hand,grasp,rot,2.,f'{i}:descend')
            grip(hand,True,f'{i}:grasp')
            # Closure can backdrive the wrist against the supported object.
            # Lift from that measured pose without simultaneously rotating back
            # toward the unloaded grasp command and sweeping a pad into the table.
            loaded=wrist(hand)
            rot=loaded[:3,:3].copy()
            loaded_pad=loaded[:3,3]+rot@np.array([0.,0.,.127])
            planning_recoveries.append({'phase':f'{i}:lift','frame':len(states),'kind':'measured_grasp_pose','pad':loaded_pad.tolist(),'rotation':rot.tolist()})
            move(hand,loaded_pad+[0,0,lift_height],rot,2.,f'{i}:lift')
            hold(.5,f'{i}:lift_check')
            if bounds(target)[0,2] < b[0,2]+.05:
                raise RuntimeError(f'Object {i} did not lift under physical grip')
            if hand=='r':
                handoff_ready=False
                for station_y in (-.04,.08,.16,.24):
                    transfer=np.array([grasp[0],root[1,3]+station_y,grasp[2]+.14])
                    try:
                        move('r',transfer,rot,3.,f'{i}:handoff_right')
                    except RuntimeError as error:
                        if not str(error).startswith('No collision-free approach'):raise
                        continue
                    live_bounds=bounds(target);live=live_bounds.mean(0)
                    nominal_left_rot=Rotation.from_euler('x',90,degrees=True).as_matrix() if size[2]>.08 else np.diag([-1.,1.,-1.])
                    left=live+np.array([0,0,.04]) if size[2]>.08 else live+np.array([0,.045,.012])
                    for yaw in (0.,-30.,30.,-60.,60.,90.):
                        candidate_rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@nominal_left_rot
                        footprint=live_bounds[1]-live_bounds[0]
                        if np.dot(np.abs(candidate_rot[:,0]),footprint)>.10:continue
                        for approach in (left-candidate_rot[:,2]*.08,left+[0,.12,.08]):
                            try:
                                move('l',approach,candidate_rot,3.,f'{i}:handoff_left_approach',next_centre=left)
                                left_rot=candidate_rot;handoff_ready=True
                                planning_recoveries.append({'phase':f'{i}:handoff_left_approach','kind':'coupled_transfer_station','station_y_m':station_y,'receiver_yaw_deg':yaw})
                                break
                            except RuntimeError as error:
                                if not str(error).startswith('No collision-free approach'):raise
                        if handoff_ready:break
                    if handoff_ready:break
                if not handoff_ready:raise RuntimeError(f'No reachable physical transfer station for object {i}')
                move('l',left,left_rot,2.,f'{i}:handoff_left')
                grip('l',True,f'{i}:handoff_left_grasp')
                # Open only enough to disengage until the fingers are separated.
                release_angle=min(.65,float(art.get_joint_positions()[names.index(grips[1])])+.17)
                grip('r',False,f'{i}:handoff_right_release',open_angle=release_angle)
                move('r',transfer+[0,0,-.10],rot,2.,f'{i}:handoff_right_down')
                grip('r',False,f'{i}:handoff_right_open')
                move('r',transfer+[0,-.16,-.10],rot,2.,f'{i}:handoff_right_clear')
                hold(.5,f'{i}:handoff_check')
                if bounds(target)[0,2] < .55:
                    raise RuntimeError(f'Object {i} failed physical right-to-left handoff')
                rot=left_rot;hand='l'
                if size[2]>.08:
                    pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    move('l',pad+[.04 if size[2]>.18 else .06,0,0],rot,1.5,f'{i}:handoff_left_extend')
                    pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    rot=Rotation.from_euler('z',60,degrees=True).as_matrix()@left_rot
                    move('l',pad,rot,2.,f'{i}:handoff_left_turn')
                    pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    if size[2]>.18:
                        # First clear the folded elbow, then lift the forearm above
                        # the neighboring can caps before crossing the row.
                        move('l',pad+[.03,0,0],rot,1.,f'{i}:handoff_left_clear_elbow')
                        pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                        move('l',pad+[0,0,.04],rot,1.5,f'{i}:handoff_left_clear_caps')
                        pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    move('l',pad+[.03 if size[2]>.18 else .06,0,0],rot,1.5,f'{i}:handoff_left_extend_forward')
                    pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    move('l',pad+[0,.43,0],rot,3.,f'{i}:handoff_left_stage')
                    pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                    rot=Rotation.from_euler('z',30,degrees=True).as_matrix()@rot
                    move('l',pad,rot,2.,f'{i}:handoff_left_forward')
            if is_book:
                pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                stage_pad=pad+np.array([.10,.20,0])
                stage_pad[1]=max(stage_pad[1],root[1,3]+.28)
                move('l',stage_pad,rot,2.,f'{i}:carry_stage')
                pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                rot=Rotation.from_euler('z',45,degrees=True).as_matrix()@rot
                move('l',pad,rot,2.,f'{i}:carry_turn')
            elif side_pick:
                # Plan the yaw change together with transport; a straight fixed-yaw
                # carry folds the forearm into the upper arm for this tall grasp.
                # Preserve inclination for a distant-object grasp instead of
                # forcing a level wrist and rotating the held object abruptly.
                if abs(rot[2,2])<.2:
                    rot=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
            elif not used_handoff:
                pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
                rot=Rotation.from_euler('z',60,degrees=True).as_matrix()@rot
                move('l',pad,rot,2.,f'{i}:carry_turn')
            current=bounds(target);pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
            place_size=np.maximum(current[1]-current[0],size)
            destination_y=left_cursor+place_size[1]/2+.005
            placement_x=centre[0] if used_handoff else current.mean(0)[0]
            destination_x=float(np.clip(placement_x,table_bounds[0,0]+place_size[0]/2+.01,table_bounds[1,0]-place_size[0]/2-.04))
            if used_handoff and size[2]>.08:
                destination_x=max(destination_x,table_bounds[0,0]+.15)
            if is_book:
                # Books share a compact temporary stack on the real left tabletop.
                # Every support is an already placed physical book, never a prop.
                destination_y=cloth_bounds[1,1]+.18
                destination_x=table_bounds[0,0]+.23
            support_z=max([table_bounds[1,2]]+[bounds(p)[1,2] for p in targets[:i]]) if is_book else table_bounds[1,2]
            destination=np.array([destination_x,destination_y,table_bounds[1,2]+.003+size[2]/2])
            delta=destination-current.mean(0)
            release_clearance=.045 if is_book else (.04 if used_handoff and size[2]>.08 else .003)
            carry_pad=pad+[delta[0],delta[1],0]
            placed_route=False
            placement_offsets=[(0.,0.)] if is_book else [(dx,dy) for dy in (0.,.015,.030) for dx in (0.,.05,-.05,.10,-.10)]
            if not is_book:
                # A third object need not extend the same row. Search the
                # remaining real tabletop left of the cloth in two dimensions;
                # the existing held-object, arm and release checks reject
                # occupied slots using the accumulated physical scene.
                min_x=table_bounds[0,0]+place_size[0]/2+.03
                max_x=table_bounds[1,0]-place_size[0]/2-.03
                min_y=cloth_bounds[1,1]+place_size[1]/2+.015
                max_y=table_bounds[1,1]-place_size[1]/2-.015
                if min_x<=max_x and min_y<=max_y:
                    placement_offsets.extend((x-destination_x,y-destination_y)
                        for y in np.linspace(min_y,max_y,3)
                        for x in np.linspace(min_x,max_x,5))
            placement_routes=[(dx,dy,dz) for dx,dy in placement_offsets for dz in ((0.,-.06) if i>=2 and not is_book else (0.,))]
            for extra_x,extra_y,extra_z in placement_routes:
                if current[0,2]+extra_z<table_bounds[1,2]+.025:continue
                if destination_y+extra_y-place_size[1]/2<cloth_bounds[1,1]+.005:
                    continue
                if destination_y+extra_y+place_size[1]/2>table_bounds[1,1]-.008:
                    continue
                if not table_bounds[0,0]+place_size[0]/2+.01 <= destination_x+extra_x <= table_bounds[1,0]-place_size[0]/2-.01:
                    continue
                candidate_pad=carry_pad+[extra_x,extra_y,extra_z]
                lower_pad=candidate_pad+[0,0,support_z+release_clearance-current[0,2]-extra_z]
                try:
                    move('l',candidate_pad,rot,3.,f'{i}:carry_left',next_centre=lower_pad)
                    placed_route=True
                    if extra_x or extra_y or extra_z:planning_recoveries.append({'phase':f'{i}:carry_left','kind':'placement_offset','dx_m':extra_x,'dy_m':extra_y,'transit_dz_m':extra_z})
                    break
                except RuntimeError as error:
                    if not str(error).startswith('No collision-free approach'):raise
            if not placed_route and side_pick:
                # Search a reachable tabletop destination together with wrist yaw.
                # Preserve pitch and predict the held footprint from the measured
                # grasp; the same live episode continues through every motion.
                held_rotation=wrist('l')[:3,:3]
                local_centre=(current.mean(0)-pad)@held_rotation
                corners=np.array([[x,y,z] for x in current[:,0] for y in current[:,1] for z in current[:,2]])
                local_corners=(corners-pad)@held_rotation
                for yaw in (-30.,30.,-60.,60.,-90.,90.,180.):
                    candidate_rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rot
                    projected=local_corners@candidate_rot.T
                    projected_size=np.ptp(projected,axis=0)
                    candidate_y=left_cursor+projected_size[1]/2+.005
                    if candidate_y+projected_size[1]/2>table_bounds[1,1]-.008:continue
                    for dx in (-.06,-.12,0.):
                        candidate_x=float(np.clip(destination_x+dx,table_bounds[0,0]+projected_size[0]/2+.01,table_bounds[1,0]-projected_size[0]/2-.04))
                        object_centre=np.array([candidate_x,candidate_y,current.mean(0)[2]])
                        candidate_pad=object_centre-candidate_rot@local_centre
                        lower_pad=candidate_pad.copy()
                        lower_pad[2]=support_z+release_clearance-projected[:,2].min()
                        try:
                            move('l',candidate_pad,candidate_rot,3.,f'{i}:carry_left',next_centre=lower_pad)
                            rot=candidate_rot;placed_route=True
                            planning_recoveries.append({'phase':f'{i}:carry_left','kind':'joint_destination_yaw_search','yaw_delta_deg':yaw,'destination_xy':[candidate_x,candidate_y]})
                            break
                        except RuntimeError as error:
                            if not str(error).startswith('No collision-free approach'):raise
                    if placed_route:break
            if not placed_route:raise RuntimeError(f'No reachable collision-free placement for object {i}')
            pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
            # An edge pinch places the lower finger beneath a thin book.
            # Release close above the surface so the finger cannot enter the table.
            release_clearance=.045 if is_book else (.04 if used_handoff and size[2]>.08 else .003)
            dz=support_z+release_clearance-bounds(target)[0,2]
            move('l',pad+[0,0,dz],rot,2.,f'{i}:lower')
            # A small opening can leave a can physically pinched: the previous
            # retreat then lifted it again and dropped it at the final opening.
            # Reserve finger clearance between destinations and release fully
            # while the object is at its support surface.
            grip('l',False,f'{i}:release')
            pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
            if is_book:
                move('l',pad-rot[:,2]*.12,rot,2.,f'{i}:book_withdraw')
                pad=wrist('l')[:3,3]+wrist('l')[:3,:3]@np.array([0,0,.127])
            # The object is already released: eight centimetres clears the
            # fingers without the unnecessary high withdrawal that folds the wrist.
            retreat_offsets=[np.array([0.,0.,.08]),np.array([0.,0.,.04]),
                             -rot[:,2]*.08+np.array([0.,0.,.04]),
                             -rot[:,2]*.12+np.array([0.,0.,.02])]
            for retreat_index,retreat_offset in enumerate(retreat_offsets):
                try:
                    move('l',pad+retreat_offset,rot,2.,f'{i}:retreat')
                    if retreat_index:
                        planning_recoveries.append({'phase':f'{i}:retreat','kind':'checked_empty_hand_retreat','offset_m':retreat_offset.tolist()})
                    break
                except RuntimeError as error:
                    if not str(error).startswith('No collision-free approach'):
                        raise
                    if retreat_index==len(retreat_offsets)-1:
                        # The empty hand can change wrist orientation. Use the
                        # existing chest-front goal through the same collision-
                        # checked planner when a fixed-orientation retreat fails.
                        home_q=art.get_joint_positions().astype(np.float64)
                        home_goal=FAMILY4_STANDARD_ARM_RAD[:7].copy()
                        home_q[models['l'].indices]=home_goal
                        home_pose=models['l'].forward(home_q[None],root[None])[0]
                        home_pad=home_pose[:3,3]+home_pose[:3,:3]@np.array([0.,0.,.127])
                        move('l',home_pad,home_pose[:3,:3],3.,f'{i}:retreat',joint_goal=home_goal)
                        planning_recoveries.append({'phase':f'{i}:retreat','kind':'checked_chest_front_empty_hand_route'})
            hold(1.,f'{i}:placed')
            for j in range(i+1):
                final=bounds(targets[j])
                support_heights=[table_bounds[1,2]]+([bounds(p)[1,2] for p in targets[:j]] if is_book else [])
                supported=min(abs(final[0,2]-height) for height in support_heights)<.015
                if final[0,1] <= cloth_bounds[1,1]+.005 or not supported or np.any(final[0,:2]<table_bounds[0,:2]) or np.any(final[1,:2]>table_bounds[1,:2]):
                    raise RuntimeError(f'Object {j} not stably supported left of cloth')
            events.append({'target':target,'placed_frame':len(states)-1,'hand':'left'})
            left_cursor=bounds(target)[1,1]+(.010 if is_book else .045)
        # Return empty arms to the user-selected chest-front joint pose.
        # Exact joint goals avoid an IK branch that places the elbow by the head.
        for hand,offset in [('r',7),('l',0)]:
            model=models[hand]
            goal=FAMILY4_STANDARD_ARM_RAD[offset:offset+7].copy()
            q=art.get_joint_positions().astype(np.float64)
            q[model.indices]=goal
            pose=model.forward(q[None],root[None])[0]
            centre=pose[:3,3]+pose[:3,:3]@np.array([0.,0.,.127])
            move(hand,centre,pose[:3,:3],3.,f'final_{hand}:return_home',joint_goal=goal)
        hold(2.,'final_settle')
        actual=art.get_joint_positions().astype(np.float64)
        home_error={hand:float(np.max(np.abs(actual[models[hand].indices]-FAMILY4_STANDARD_ARM_RAD[offset:offset+7]))) for hand,offset in [('l',0),('r',7)]}
        (out/'return_home_readback.json').write_text(json.dumps({'target':'native_chest_front_standby','max_joint_error_rad':home_error,'passed':max(home_error.values())<.08},indent=2))
        if max(home_error.values())>=.08:
            raise RuntimeError(f'Chest-front return tracking failed: {home_error}')
    except Exception as exc:
        failure=str(exc)
        print('FAILED',failure,flush=True)
    finally:
        cmds=np.asarray(commands)
        state=dict(schema=np.array('real2sim.full_state.v1'),capture_scope=np.array('all_active'),task_id=np.array(args.task),episode_uid=np.array(f'benchmark_{args.task}_{args.seed}'),fps=np.array(20.),frame_index=np.arange(len(states)),robot_initial_base_xyyaw_deg=expected_base,current_robot_base_xy_m=np.asarray(initial_root_position)[:2],current_robot_orient_wxyz=np.asarray(initial_root_orientation),robot_initial_joint_position_rad_or_m=initial_joint_position,prop_velocity_mps_radps=np.asarray(body_velocities),target_prim=np.array(targets[0]),target_prims=np.array(targets),robot_dof_names=np.array(names),robot_joint_position_rad_or_m=np.array(states),robot_joint_velocity_radps_or_mps=np.array(velocities),robot_root_xyz_m=world_root[:3,3],robot_root_wxyz=Rotation.from_matrix(world_root[:3,:3]).as_quat()[[3,0,1,2]],rigid_body_paths=np.array(all_bodies),prop_xyz_m=np.array(poses),prop_wxyz=np.array(rotations),target_bounds=np.array(target_bounds),target_bounds_frame=np.array('canonical_task_frame'),frame_to_world=frame_to_world,phase=np.array(phases),command_arm_rad=cmds[:,:14],command_waist_rad_m=cmds[:,14:16],command_head_rad=cmds[:,16:18],command_grip_rad=cmds[:,18:20])
        np.savez_compressed(out/'full_state.npz',**state)
        np.savez_compressed(out/'action.npz',arm=cmds[:,:14],waist=cmds[:,14:16],head=cmds[:,16:18],lgrip=(.7843981634-cmds[:,18])/.7843981634,rgrip=(.7843981634-cmds[:,19])/.7843981634,fps=np.array(20.))
        report={'environment_scope':'randomized_house_session' if episode_plan else 'canonical_house_default','layout_overlay_applied':bool(episode_plan),'planner_base_support_surfaces':sorted(base_support_floors),'frame_to_world':frame_to_world.tolist(),'bounds_frame':'canonical_task_frame','object_pose_or_velocity_writes':False,'task_id':args.task,'source_sheet_rows':task['source_sheet_rows'],'count':args.count,'seed':args.seed,'targets':targets,'target_assets':target_assets,'cloth_model':'initial static visible triangle mesh; cloth manipulation not simulated','cloth_bounds':cloth_bounds.tolist(),'table_bounds':table_bounds.tolist(),'initial_bounds':initial_bounds.tolist(),'camera_paths':cameras,'robot_asset':str(args.robot),'robot_camera_parameters_unchanged':configuration()==invariant,'runtime_target_pose_writes':False,'placed':events,'failure':failure,'physics_candidate_pass':not args.inspect_only and failure is None and len(events)==args.count,'owner_approved':False,'contact_error':contacts.error}
        from real2sim.production.git_provenance import source_revision
        try:
            report['source_revision'] = source_revision([Path(__file__), policy_path, REPO / 'real2sim/production/task8_initial_sampling.py', REPO / 'real2sim/production/task8_environment.py'])
        except (RuntimeError, OSError) as error:
            report['source_revision'] = {'error':str(error)}
            report['physics_candidate_pass'] = False
        (out/'episode.json').write_text(json.dumps({'benchmark_subtask':args.task,'source_sheet_rows':task['source_sheet_rows'],'frames':len(states),'fps':20,'source_revision':report['source_revision'],'owner_approved':False},indent=2))
        (out/'result.json').write_text(json.dumps(report,indent=2))
        (out/'contacts.json').write_text(json.dumps(list(contacts.events.values())))
        (out/'planning_recoveries.json').write_text(json.dumps(planning_recoveries,indent=2))
        (out/'shared_configuration.json').write_text(json.dumps(invariant,indent=2))
        print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    main()

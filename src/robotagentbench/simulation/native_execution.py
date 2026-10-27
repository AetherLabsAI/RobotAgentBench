"""Native Isaac execution in a dedicated worker; no task-scoring implementation."""
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from .admission import read
from .bootstrap import CAMERAS, camera_resolutions, valid_camera_frame, initialize_base_pose
from .shutdown import NativeResources


def execute_native(admitted, *, options, backend, lease, completion):
    """Run the preserved physics body, then attempt every owned cleanup stage.

    Call only from the dedicated native worker after launch and GPU admission.
    Task factories remain trusted original owners supplied by the backend.
    """
    args = options
    out = admitted.output
    request, manifest, plan = admitted.request, admitted.manifest, admitted.plan
    launch = admitted.launch
    is_openend = admitted.owner_kind == 'room'
    resolutions = camera_resolutions(plan['task_name'], plan['instruction'])
    app = sim = session = host = None
    collision_guard = None
    pending_native = None
    products, annotators = [], {}
    report = dict(status='initializing', task_name=plan['task_name'],
        rule_version=plan['rule_version'], maximum=plan['score_policy']['maximum'], control_hz=20,
        physics_backend='cpu', gpu_uuid=lease.uuid, owner_calls_models=False,
        action_source='internal_hold_smoke' if args.smoke else 'stdio_client',
        wall_clock_limit_s=None, source_layers_unchanged=None,
        native_shutdown_complete=False, cleanup_errors=[])
    report['gpu_admission'] = dict(lease.evidence)
    start = time.monotonic()
    def checkpoint():
        report['wall_s'] = time.monotonic() - start
        if collision_guard is not None:
            report['collision_guard'] = collision_guard.summary()
        (out / 'native_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    try:
        checkpoint()
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False,
            fast_shutdown=False, active_gpu=args.gpu, physics_gpu=0, width=640, height=480,
            sync_loads=True, limit_cpu_threads=8, extra_args=[
                '--/log/file='+str(out/'kit.log'), '--/app/userConfigPath='+str(out/'kit_user.json'),
                '--/app/tokens/cache='+str(out/'cache'), '--/app/tokens/data='+str(out/'data')]))
        report['startup_phase'] = 'simulation_app_started'
        checkpoint()
        import numpy as np
        import carb
        import omni.usd
        import omni.physx
        import omni.physics.tensors
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Gf
        from PIL import Image
        from robotagentbench.interfaces import robot_profiles as rp
        from .clock import NativeSceneClock
        from robotagentbench.runtime.action_driver import NativeDriver
        from robotagentbench.runtime.action_host import SessionHost, serve
        if args.coding_policy:
            from robotagentbench.runtime.coding_driver import NativeDriver
            from robotagentbench.runtime.coding_action_host import SessionHost, serve
        bindings = backend.load_native_bindings()
        Controller = bindings.mobile_controller
        def build_session(*arguments, **keywords):
            return backend.build_short_session(plan['task_name'], *arguments, **keywords)
        context = omni.usd.get_context()
        if not context.open_stage(manifest['scene']):
            raise RuntimeError('Could not load the prepared stage')
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                  for l in stage.GetUsedLayers() if l.realPath}
        report['source_layers_sha256'] = hashes
        NativeCollisionGuard = bindings.collision_guard
        collision_guard = NativeCollisionGuard(stage,
            out/'collision_guard.jsonl' if args.collision_guard_audit else None)
        settings = carb.settings.get_settings()
        for key, value in (('/physics/suppressReadback',False),('/physics/updateToUsd',True),
                           ('/physics/updateVelocitiesToUsd',True),('/physics/outputVelocitiesLocalSpace',False)):
            settings.set_bool(key, value)
        scene = stage.GetPrimAtPath('/World/PhysicsScene')
        if scene.GetAttribute('physxScene:enableGPUDynamics').Get() is not False or scene.GetAttribute('physxScene:timeStepsPerSecond').Get() != 60:
            raise ValueError('Scene-owned CPU/60 Hz startup required')
        # The launch declares base_link world pose, not Robot container pose.
        report['robot_base_initialization'] = initialize_base_pose(stage,launch['initial_base_xyyaw'])
        anchor = UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint'))
        if not anchor or anchor.GetJointEnabledAttr().Get() is False:
            raise ValueError('Source fixed initialization anchor missing')
        base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
        anchor.CreateLocalPos0Attr(Gf.Vec3f(base.ExtractTranslation()))
        anchor.CreateLocalRot0Attr(Gf.Quatf(base.ExtractRotationQuat()))
        episode = str(uuid4())
        if is_openend:
            session = backend.build_room(request, stage, manifest, out/'scoring',
                episode_id=episode, sensor=launch['sensor'], evaluation_mode=plan['task_name'])
        else:
            session = build_session(request, stage, manifest, out/'scoring', episode_id=episode, sensor=launch['sensor'])
        if not hasattr(session, 'ports'):
            raise RuntimeError('Short task is unavailable: '+str(session.result()))
        # Read only this admitted layout, before physical execution. This is
        # static geometry extraction, not a new physics world or an action.
        from robotagentbench.runtime.navigation import from_stage
        navigation = from_stage(stage, spawn=launch['initial_base_xyyaw'], output=out/'navigation')
        mobile = Controller(stage)
        robot_parameter_alignment = bindings.robot_alignment
        report['robot_parameter_alignment'] = robot_parameter_alignment.apply(stage)
        checkpoint()
        sid = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        sim = omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid):
            raise RuntimeError('Native physics attach failed')
        current_time = 0.
        def advance(dt):
            nonlocal current_time
            if collision_guard.ready and pending_native is not None:
                applied = collision_guard.filter(robot,names,pending_native,current_time)
                for name,value in applied.items():
                    targets[0,names.index(name)] = value
                robot.set_dof_position_targets(targets,indices)
            sim.simulate(dt, current_time)
            sim.fetch_results()
            current_time += dt
            return dict(events=[], docked_object_count=0, tray_dock_enabled=False)
        advance(1/60)
        view = omni.physics.tensors.create_simulation_view('numpy',sid)
        view.set_subspace_roots('/')
        robot = view.create_articulation_view('/World/Robot')
        if robot.count != 1:
            raise RuntimeError('Exactly one native robot required')
        names = list(robot.shared_metatype.dof_names)
        report['robot_native_force_verification'] = robot_parameter_alignment.verify_native_forces(robot, names)
        units, limits, driven = {}, {}, []
        for name in names:
            p = stage.GetPrimAtPath('/World/Robot/joints/'+name)
            scale = 1. if p.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
            units[name] = 'm' if scale == 1 else 'rad'
            if any(s.startswith('PhysicsDriveAPI:') for s in p.GetAppliedSchemas()):
                driven.append(name)
            lo, hi = p.GetAttribute('physics:lowerLimit').Get(),p.GetAttribute('physics:upperLimit').Get()
            if lo is not None and hi is not None:
                limits[name] = [float(lo)*scale,float(hi)*scale]
        targets = robot.get_dof_positions().copy()
        ready = read(launch['ready_pose'])['targets_rad']
        for name, value in ready.items():
            if not limits[name][0] <= value <= limits[name][1]:
                raise ValueError('Source ready pose out of limits: '+name)
            targets[0,names.index(name)] = value
        for inner, outer in rp.FOLLOWERS.items():
            targets[0,names.index(inner)] = -targets[0,names.index(outer)]
        indices = np.array([0],dtype=np.uint32)
        robot.set_dof_positions(targets.copy(), indices)
        robot.set_dof_velocities(np.zeros_like(targets), indices)
        robot.set_dof_position_targets(targets.copy(), indices)
        report['robot_initial_state_restores'] = 1
        adapter = rp.NativeAdapter(rp.MOBILE22,names,units,driven,
            {n:float(targets[0,names.index(n)]) for n in driven},limits,
            mimic_followers=rp.inspect_mimic_followers(stage))
        bodies = view.create_rigid_body_view([str(p.GetPath()) for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith('/World/Robot/')
            and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False])
        def snapshot():
            return b''.join(np.asarray(a).copy().tobytes() for a in (bodies.get_transforms(),
                bodies.get_velocities(),robot.get_link_transforms(),robot.get_dof_positions(),
                robot.get_dof_velocities(),robot.get_root_velocities()))
        clock = NativeSceneClock(stage,sim,lambda:current_time,snapshot)
        session.initialize(clock,view)
        report['status'] = 'settling'
        checkpoint()
        for i in range(240):
            robot.set_dof_position_targets(targets,indices)
            session.after_physics(advance(1/60))
            if i%60 == 59:
                report['settled_steps'] = i+1
                checkpoint()
        report['initial_acceptance'] = session.accept_initial()
        report['startup_phase'] = 'native_initial_accepted'
        checkpoint()
        if is_openend:
            session = backend.accept_room(session, episode_id=episode,
                instruction=plan['instruction'])
        report['robot_parameter_verification'] = robot_parameter_alignment.verify(stage, report['robot_parameter_alignment'])
        mobile.bind()
        report['robot_base_initialization']['native_after_settle'] = mobile.observe()
        collision_guard.bind(robot,names)
        report['startup_phase'] = 'control_guard_bound'
        checkpoint()
        for name,path in CAMERAS.items():
            if not stage.GetPrimAtPath(path):
                raise ValueError('Original robot camera missing: '+name)
            product = rep.create.render_product(path,resolutions[name])
            products.append(product)
            annotators[name] = []
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            annotators[name].append(rgb)
            stamp = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            annotators[name].append(stamp)
            rgb.attach([product]); stamp.attach([product])
        report['startup_phase'] = 'camera_products_attached'
        checkpoint()

        previous_ref = None
        capture_index = 0
        def capture():
            nonlocal previous_ref, capture_index
            from fractions import Fraction
            before, before_time = snapshot(),clock.current_time
            previous = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                if capture_index == 0:
                    report['startup_phase'] = 'first_render_start'
                    checkpoint()
                for attempt in range(60 if previous_ref is None else 1):
                    rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                    if previous_ref is None and attempt < 3:
                        continue
                    frames = {n:np.asarray(a[0].get_data()).copy() for n,a in annotators.items()}
                    stamps = {n:a[1].get_data() for n,a in annotators.items()}
                    if all(valid_camera_frame(n,f,resolutions) for n,f in frames.items()):
                        break
                if snapshot() != before or clock.current_time != before_time:
                    raise RuntimeError('Render advanced physics')
                refs = {Fraction(int(s['referenceTimeNumerator']),int(s['referenceTimeDenominator'])) for s in stamps.values()}
                if len(refs)!=1 or (previous_ref is not None and next(iter(refs))<=previous_ref):
                    raise RuntimeError('Missing, unsynchronized or stale render products')
                previous_ref = next(iter(refs))
                images = {}
                for name,frame in frames.items():
                    if not valid_camera_frame(name,frame,resolutions):
                        raise RuntimeError('Invalid camera frame: '+name)
                    buf = io.BytesIO(); Image.fromarray(frame[:,:,:3]).save(buf,format='PNG')
                    data = buf.getvalue()
                    (out/f'{capture_index:04d}_{name}.png').write_bytes(data)
                    images[name] = base64.b64encode(data).decode()
                capture_index += 1
                if capture_index == 1:
                    report['startup_phase'] = 'first_camera_images_written'
                    checkpoint()
                return images
            finally:
                settings.set('/app/player/playSimulations',previous)
        def apply(native):
            nonlocal pending_native
            pending_native = dict(native)
            # Recheck this target before every physical substep. Never first
            # write an unfiltered target, and do not freeze unrelated channels.
        driver = NativeDriver(session,adapter,mobile,
            measured_joints=lambda:dict(zip(names,map(float,robot.get_dof_positions()[0]))),
            apply_native_targets=apply,simulate_fetch=advance,capture_png=capture,
            context=admitted.context,navigation=navigation)
        report['startup_phase'] = 'first_host_observation'
        checkpoint()
        host = SessionHost(session,driver)
        report['startup_phase'] = 'host_observation_ready'
        checkpoint()
        if args.coding_policy:
            if args.smoke:
                raise ValueError('Use the coding smoke client for coding-policy verification')
            CodingHost = backend.wrap_coding_host
            host = CodingHost(host,dict(zip(names,map(float,targets[0]))),out)
            from robotagentbench.runtime.observation_host import ObservationHost
            host = ObservationHost(host, context=admitted.context, navigation=navigation)
        report['status'] = 'ready'
        checkpoint()
        if args.smoke:
            observation = host.observe()
            hold = [observation['joints'][n] for n in rp.get(rp.MOBILE22).order]+[0.,0.]
            reply = host.execute(dict(op='execute_actions',request_id=str(uuid4()),episode_id=episode,
                expected_step=0,actions=[hold[:] for _ in range(20)]))
            report['smoke'] = dict(actions_executed=reply['actions_executed'],
                step=reply['observation']['step'], simulation_time_s=reply['observation']['time_s'],
                physics_hz=clock.hz, final_base=reply['observation']['base_state'])
            host.close()
        else:
            serve(host,output_stream=completion)
        result = session.result()
        report['robot_native_drive_final_verification'] = robot_parameter_alignment.verify_native_forces(robot, names)
        report['robot_parameter_verification'] = robot_parameter_alignment.verify(stage, report['robot_parameter_alignment'])
        report.update(status='physics_completed', task_result=result, camera_captures=capture_index,
            full_task_success=result.get('success', False))
    except BaseException as exc:
        report.update(status='error', error=str(exc), traceback=traceback.format_exc())
        print(report['traceback'], file=sys.stderr, flush=True)
    finally:
        resources = NativeResources(app=app, simulation=sim, session=session, host=host,
            collision_guard=collision_guard, products=products, annotators=annotators,
            source_hashes=locals().get('hashes'), lease=lease)
        resources.shutdown(report, checkpoint)
    return report

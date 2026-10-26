"""Registry-driven native owner for the robot-only 22D ShortTask bridge.

Reuses the admitted scene owner, ShortTaskSession, native adapter and L1 base
controller. No model-provided code, paths, task poses or score overrides.
"""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from uuid import uuid4

HERE = Path(__file__).resolve().parent
CAMERAS = dict(head='/World/Robot/head_link2/Head_Camera',
    left='/World/Robot/gripper_l_base_link/Left_Camera',
    right='/World/Robot/gripper_r_base_link/Right_Camera')
# User-selected evaluation standard: full native frames for every task.
# Historical padded training families are not exceptions in this experiment.
CAMERA_RESOLUTIONS = dict(head=(1280,800), left=(848,480), right=(848,480))


PADDED_ATOMIC_INSTRUCTIONS = frozenset({
    'Turn the microwave start knob, confirm that heating has started, then release the knob and withdraw your hand.',
    'Wait for the microwave to finish heating, open the door, remove the food, and close the door.',
})


def camera_resolutions(task_name, instruction=None):
    return dict(CAMERA_RESOLUTIONS)


def valid_camera_frame(name, frame, resolutions=None):
    width, height = (CAMERA_RESOLUTIONS if resolutions is None else resolutions)[name]
    return frame.shape == (height,width,4) and str(frame.dtype) == 'uint8'


def initialize_base_pose(stage, initial_base_xyyaw):
    """Place the base frame, accounting for an already-authored local FK frame.

    This is pre-physics initialization only. Prepared Context scenes can keep
    the old Robot container transform while authoring a new base_link local
    transform. Assigning the requested BASE pose directly to Robot applies that
    displacement twice. Preserve the articulation's local transforms and solve
    the container transform so base_link has exactly the declared world pose.
    """
    from pxr import Gf, UsdGeom
    root=stage.GetPrimAtPath('/World/Robot')
    base=stage.GetPrimAtPath('/World/Robot/base_link')
    if not root or not base:raise ValueError('Robot and base frame required')
    if UsdGeom.Xformable(base).GetResetXformStack():
        raise ValueError('Base frame ignores Robot container; cannot initialize via container')
    cache=UsdGeom.XformCache()
    root_before=cache.GetLocalToWorldTransform(root)
    base_before=cache.GetLocalToWorldTransform(base)
    base_in_robot=base_before*root_before.GetInverse()
    x,y,yaw=initial_base_xyyaw
    desired=Gf.Matrix4d(1)
    desired.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(yaw)))
    desired.SetTranslateOnly(Gf.Vec3d(x,y,0))
    xf=UsdGeom.Xformable(root);reset=xf.GetResetXformStack()
    parent=Gf.Matrix4d(1) if reset else cache.GetLocalToWorldTransform(root.GetParent())
    xf.MakeMatrixXform().Set(base_in_robot.GetInverse()*desired*parent.GetInverse())
    xf.SetResetXformStack(reset)
    actual=UsdGeom.XformCache().GetLocalToWorldTransform(base)
    if not Gf.IsClose(actual,desired,1e-9):
        raise ValueError('Declared base world pose was not restored')
    return dict(method='base_world_frame_compensation_before_physics',
        requested_xyyaw=list(initial_base_xyyaw),
        base_world_before=list(map(list,base_before)),base_in_robot=list(map(list,base_in_robot)),
        base_world_after=list(map(list,actual)),source_joint_frames_unchanged=True)


def read(path):
    return json.loads(Path(path).read_text())


def permitted_astra_batch_output(path):
    """Explicitly approved batch storage, not an arbitrary output override."""
    root = Path('/mnt/agent-bench-data/lingjun/astra_all14_20261015')
    target = Path(path).resolve()
    return (root.is_dir() and root.resolve() == root
            and target != root and target.is_relative_to(root))


def admit_launch(launch, output):
    """Resolve trusted task plugins before GPU allocation; never infer readiness.

    The registry owns names, rules, parameters and scoring. This CPU owner does
    not reinterpret unavailable tasks or GPU-only previews as scored episodes.
    """
    from short_tasks import registry
    from short_tasks.common.contracts import TaskError
    from short_tasks.scene import verify_manifest, verify_scene_sources
    if not isinstance(launch, dict) or launch.get('schema') != 'openend.short_task.launch.v1':
        raise ValueError('Explicit prepared ShortTask launch required')
    request, manifest = read(launch['request']), read(launch['manifest'])
    if launch.get('task_name') != request.get('task_name'):
        raise ValueError('Launch task differs from request')
    plan = registry.resolve(request)
    if plan['task_name'] in registry.FAILURE_RECOVERY_TASKS:
        return registry.failure_recovery_adapter().admit_launch(launch, output)
    if plan['task_name'] in registry.ATOMIC_COFFEE_TASKS:
        return registry.atomic_coffee_adapter().admit_launch(launch, output)
    if not plan['available']:
        error = plan['unavailable']
        raise TaskError(error['message'], kind=error['kind'], reason=error['reason'])
    if registry.is_gpu_cloth_preview(plan):
        raise TaskError('GPU cloth preview needs its existing GPU scene owner; this owner uses CPU physics',
                        kind='load', reason='gpu_cloth_scene_owner_required')
    out = Path(output).resolve()
    task_root = registry.task_directory(plan['task_name'])
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    context_source = (plan['task_name'] == 'clear_table_trash'
        and out.is_relative_to(registry.task_directory('select_trash'))
        and out.parent.name == 'base_trash')
    context_source = context_source or (
        plan['task_name'] in ('stock_wooden_tray', 'sofa_to_coffee_table')
        and out.parent.name == 'base_source' and out.name == 'native_smoke'
        and any(out.is_relative_to(registry.task_directory(name))
                for name in ('select_sofa', 'select_drink', 'select_snack')))
    if out.exists() or not (out.is_relative_to(HERE) or out.is_relative_to(task_root.resolve()) or context_source or permits(out) or permitted_astra_batch_output(out)):
        raise ValueError('Fresh output must be inside the bridge or selected task directory')
    plan = registry.bind(plan, manifest)
    registry.verify_sources(plan)
    verify_scene_sources()
    verify_manifest(manifest)
    return request, manifest, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--smoke', action='store_true', help='20 hold rows; no model call')
    ap.add_argument('--coding-policy', action='store_true', help='Existing robot22 move_to tools')
    ap.add_argument('--collision-guard-audit', action='store_true',
        help='Write detailed reactive-guard evidence; normal runs keep aggregates only')
    ap.add_argument('--allow-idle-utilization', action='store_true',
        help='Per-run user approval for nonzero utilization only; requires zero memory and no listed process')
    ap.add_argument('--allow-shared-gpu', action='store_true',
        help='Explicit per-run approval to share this GPU; preserves owner lock and records occupancy')
    args = ap.parse_args()
    from short_tasks import registry
    if read(args.launch).get('task_name') in registry.FAILURE_RECOVERY_TASKS:
        registry.failure_recovery_adapter().exec_native()
    if read(args.launch).get('task_name') in registry.ATOMIC_COFFEE_TASKS:
        # A fresh interpreter preserves the frozen atomic physics/scorer imports;
        # its task-owned bridge uses the same pinned public robot22 wire contract.
        registry.atomic_coffee_adapter().exec_native()
    if args.allow_shared_gpu and args.allow_idle_utilization:
        ap.error('Shared GPU and idle-utilization modes are mutually exclusive')
    out = args.output.resolve()
    launch = read(args.launch)
    request, manifest, plan = admit_launch(launch, out)
    resolutions = camera_resolutions(plan['task_name'], plan['instruction'])
    gpu = subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')
    approved_idle_utilization = False
    if args.allow_idle_utilization:
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        process_monitor = subprocess.check_output(['nvidia-smi','pmon','-i',str(args.gpu),'-c','1'], text=True)
        rows = [line.split() for line in process_monitor.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        approved_idle_utilization = (int(gpu[1]) == 0 and gpu[0].strip() not in active
            and bool(rows) and all(row[1] == '-' for row in rows))
        if not approved_idle_utilization:
            raise RuntimeError('Approved idle-utilization exception requires zero memory and no listed process')
    existing_processes = None
    if args.allow_shared_gpu:
        existing_processes = subprocess.check_output(
            ['nvidia-smi','--id',str(args.gpu),'--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
             '--format=csv,noheader,nounits'], text=True).strip()
    if not args.allow_shared_gpu and (int(gpu[1]) > 32 or (int(gpu[2]) != 0 and not approved_idle_utilization)):
        raise RuntimeError('Selected GPU is occupied; no process may be displaced')
    out.mkdir(parents=True)
    # All artifacts/caches are owned by this task. No shared source is edited.
    lock = (HERE/('gpu-'+gpu[0].strip()+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu[0].strip()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['XDG_CACHE_HOME'] = str(out/'cache')
    os.environ['CUDA_CACHE_PATH'] = str(out/'cache/cuda')
    # Kit/native stdout is diagnostic, never JSON RPC or model evidence.
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    from .completion_stream import CompletionStream
    completion = CompletionStream(rpc)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    app = sim = session = host = None
    collision_guard = None
    pending_native = None
    products, annotators = [], {}
    report = dict(status='initializing', task_name=plan['task_name'],
        rule_version=plan['rule_version'], maximum=plan['score_policy']['maximum'], control_hz=20,
        physics_backend='cpu', gpu_uuid=gpu[0].strip(), owner_calls_models=False,
        action_source='internal_hold_smoke' if args.smoke else 'stdio_client',
        wall_clock_limit_s=None,
        source_layers_unchanged=None, native_shutdown_complete=False, cleanup_errors=[])
    report['gpu_admission'] = dict(memory_mib=int(gpu[1]), utilization_percent=int(gpu[2]),
        user_approved_idle_utilization=approved_idle_utilization,
        user_approved_shared_gpu=args.allow_shared_gpu, existing_processes=existing_processes)
    start = time.monotonic()
    def checkpoint():
        report['wall_s'] = time.monotonic()-start
        if collision_guard is not None:
            report['collision_guard'] = collision_guard.summary()
        (out/'native_report.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    # No total elapsed-time kill switch. The task ledger owns the existing
    # 120-second simulation-time no-progress deadline; client close still works.
    checkpoint()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False,
            fast_shutdown=False, active_gpu=args.gpu, physics_gpu=0, width=640, height=480,
            sync_loads=True, limit_cpu_threads=8, extra_args=[
                '--/log/file='+str(out/'kit.log'), '--/app/userConfigPath='+str(out/'kit_user.json'),
                '--/app/tokens/cache='+str(out/'cache'), '--/app/tokens/data='+str(out/'data')]))
        import numpy as np
        import carb
        import omni.usd
        import omni.physx
        import omni.physics.tensors
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Gf
        from PIL import Image
        from arena_api import robot_profiles as rp
        from arena_api.mobile_base_native import Controller
        from openend.clock import NativeSceneClock
        from short_tasks.runtime import build_session
        if plan['task_name'] == 'select_trash':
            from short_task_ContextReasoning.tasks.select_trash.native import build_session
        elif plan['task_name'] in ('select_sofa', 'select_drink', 'select_snack'):
            from short_task_ContextReasoning.tasks.select_shelf.native import build_session
        from .native_driver import NativeDriver
        from .host import SessionHost, serve
        if args.coding_policy:
            from .cap_v1.native_tools.native_driver import NativeDriver
            from .cap_v1.native_tools.host import SessionHost, serve
        context = omni.usd.get_context()
        if not context.open_stage(manifest['scene']):
            raise RuntimeError('Could not load the prepared stage')
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                  for l in stage.GetUsedLayers() if l.realPath}
        report['source_layers_sha256'] = hashes
        from .collision_guard.runtime import NativeCollisionGuard
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
        session = build_session(request, stage, manifest, out/'scoring', episode_id=episode, sensor=launch['sensor'])
        if not hasattr(session, 'ports'):
            raise RuntimeError('Short task is unavailable: '+str(session.result()))
        mobile = Controller(stage)
        from . import robot_parameter_alignment
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
        report['robot_parameter_verification'] = robot_parameter_alignment.verify(stage, report['robot_parameter_alignment'])
        mobile.bind()
        report['robot_base_initialization']['native_after_settle'] = mobile.observe()
        collision_guard.bind(robot,names)
        for name,path in CAMERAS.items():
            if not stage.GetPrimAtPath(path):
                raise ValueError('Original robot camera missing: '+name)
            product = rep.create.render_product(path,resolutions[name])
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            stamp = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            rgb.attach([product]); stamp.attach([product])
            products.append(product); annotators[name] = (rgb,stamp)
        previous_ref = None
        capture_index = 0
        def capture():
            nonlocal previous_ref, capture_index
            from fractions import Fraction
            before, before_time = snapshot(),clock.current_time
            previous = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
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
            apply_native_targets=apply,simulate_fetch=advance,capture_png=capture)
        host = SessionHost(session,driver)
        if args.coding_policy:
            if args.smoke:
                raise ValueError('Use the coding smoke client for coding-policy verification')
            from .cap_v1.native_tools.broker import CodingHost
            host = CodingHost(host,dict(zip(names,map(float,targets[0]))),out)
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
        report.update(status='error',error=str(exc),traceback=traceback.format_exc())
        print(report['traceback'],file=sys.stderr,flush=True)
    finally:
        for label,fn in [('host',lambda:host.close() if host else None),
                         ('session',lambda:session.close() if session else None),
                         ('physics',lambda:sim.detach_stage() if sim else None),
                         ('collision_guard',lambda:collision_guard.close() if collision_guard else None)]:
            try: fn()
            except BaseException as exc: report['cleanup_errors'].append(dict(operation=label,error=str(exc)))
        for pair in annotators.values():
            for annotator in pair:
                try: annotator.detach()
                except BaseException as exc: report['cleanup_errors'].append(dict(operation='camera',error=str(exc)))
        for product in products:
            try: product.destroy()
            except BaseException as exc: report['cleanup_errors'].append(dict(operation='render_product',error=str(exc)))
        if 'hashes' in locals():
            report['source_layers_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        checkpoint()
        if app:
            app.close()
        report['native_shutdown_complete'] = True
        if report['status'] == 'physics_completed':
            report['status'] = 'completed'
        checkpoint()
        lock.close()
    code = 0 if report['status']=='completed' and not report['cleanup_errors'] and report['source_layers_unchanged'] else 1
    if code == 0:
        completion.complete()
    rpc.flush(); sys.stderr.flush(); sys.stdout.flush()
    os._exit(code)


if __name__ == '__main__':
    main()

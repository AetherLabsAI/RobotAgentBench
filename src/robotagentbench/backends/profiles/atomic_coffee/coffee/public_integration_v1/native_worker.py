"""Isolated native owner. Same public robot22 wire, original atomic CPU physics."""
import argparse
import base64
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from . import adapter

CAMERAS = dict(head='/World/Robot/head_link2/Head_Camera',
    left='/World/Robot/gripper_l_base_link/Left_Camera',
    right='/World/Robot/gripper_r_base_link/Right_Camera')
CAMERA_RESOLUTIONS = dict(head=(1280,800), left=(848,480), right=(848,480))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--allow-idle-utilization', action='store_true')
    parser.add_argument('--coding-policy', action='store_true')
    parser.add_argument('--collision-guard-audit', action='store_true')
    args = parser.parse_args()
    if args.coding_policy or args.collision_guard_audit:
        parser.error('Atomic variants expose the existing raw robot22 host, not coding-policy or guard overrides')
    if args.allow_shared_gpu and args.allow_idle_utilization:
        parser.error('Mutually exclusive GPU admission modes')
    launch = json.loads(args.launch.read_text())
    request, manifest, plan = adapter.admit_launch(launch, args.output)
    gpu = subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-compute-apps=pid,process_name','--format=csv,noheader'], text=True).strip()
    monitor = subprocess.check_output(['nvidia-smi','pmon','-i',str(args.gpu),'-c','1'], text=True)
    rows = [s.split() for s in monitor.splitlines() if s.strip() and not s.lstrip().startswith('#')]
    active = apps or any(len(row)>1 and row[1]!='-' for row in rows)
    if not args.allow_shared_gpu and (active or int(gpu[1])>32 or (int(gpu[2])!=0 and
            not (args.allow_idle_utilization and int(gpu[1])==0 and not active))):
        raise RuntimeError('Selected GPU is occupied; no process stopped')
    out = adapter.owned(plan['task_name'], args.output, fresh=True)
    out.mkdir(parents=True)
    lock = open(adapter.ROOT/'short_tasks/agent_probe_v1'/('gpu-'+gpu[0].strip()+'.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu[0].strip(), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      XDG_CACHE_HOME=str(out/'cache'), CUDA_CACHE_PATH=str(out/'cache/cuda'))
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    from short_tasks.agent_probe_v1.completion_stream import CompletionStream
    completion = CompletionStream(rpc)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    start = time.monotonic()
    report = dict(status='initializing', task_name=plan['task_name'], physics_backend='cpu',
        physics_hz=240, control_hz=20, public_protocol='short_tasks.gpt.robot22.v1',
        score_policy='original_atomic_process', maximum=plan['score_policy']['maximum'],
        gpu=dict(uuid=gpu[0].strip(), memory_mib=int(gpu[1]), utilization_percent=int(gpu[2]),
                 shared=args.allow_shared_gpu, existing_processes=apps),
        full_actions_validated=False, model_run=False, cleanup_errors=[],
        camera_resolutions=CAMERA_RESOLUTIONS, camera_pose_and_lens_unchanged=True)
    app = world = session = host = physics = None
    products, annotators = [], {}

    def save():
        report['wall_s'] = time.monotonic()-start
        (out/'native_report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')

    try:
        save()
        # Frozen atomic imports only after SimulationApp; no native library preload.
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False,
            fast_shutdown=False, active_gpu=args.gpu, physics_gpu=0, width=640, height=480,
            sync_loads=True, limit_cpu_threads=8, extra_args=[
                '--/log/file='+str(out/'kit.log'), '--/app/userConfigPath='+str(out/'kit_user.json'),
                '--/app/tokens/cache='+str(out/'cache'), '--/app/tokens/data='+str(out/'data'),
                '--/app/tokens/documents='+str(out/'documents')]))
        import carb
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from arena_api import l1_scene, l1_fridge, physics_defaults, policy_execution as pe
        from arena_api import robot_collision_filters, finger_pad_material, finger_pad_contact
        from arena_api import microwave_physics, scene_obstacles, coffee_box_physics
        from arena_api.soft_body_runtime import EpisodePhysicsRuntime
        from arena_api.robot_initialization import apply as reset_robot
        from arena_randomization import shelf_supports
        from coffee_button_lighting_interference.audit import stage_issues, joint_issues
        from .robot_api.mobile_base_native import Controller
        from .host import SessionHost, serve
        from .driver import SourceDriver
        from short_tasks.runtime import build_session
        context = omni.usd.get_context()
        if not context.open_stage(manifest['scene']):
            raise RuntimeError('Cannot open admitted atomic scene')
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        policies = dict(l1_scene=l1_scene.apply(stage))
        fridge = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions',robot_collision_filters),('material',finger_pad_material),
                ('contact',finger_pad_contact),('microwave',microwave_physics),('shelf',shelf_supports),
                ('obstacles',scene_obstacles),('coffee',coffee_box_physics)]:
            policies[name] = module.apply(stage)
        physics_defaults.apply_robot(stage)
        issues = stage_issues(stage)
        session = build_session(request, stage, manifest, out/'scoring', episode_id=str(uuid4()))
        mobile = Controller(stage)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1/30, stage_units_in_meters=1.,
                      physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        physics = EpisodePhysicsRuntime(carb.settings.get_settings())
        physics.prepare(stage, world)
        session.lifecycle.before_physics_reset(world)
        world.reset()
        init = session.lifecycle.robot_initialization()
        (out/'robot_initialization.json').write_text(json.dumps(reset_robot(robot,stage,init),indent=2)+'\n')
        issues.extend(joint_issues(stage,robot))
        policies['runtime'] = physics_defaults.verify_runtime(world,robot)
        policies['episode_physics'] = physics.verify(world)
        session.initialize(world)
        indices = np.array([robot.dof_names.index(n) for n in init.hold_targets])
        values = np.array(list(init.hold_targets.values()))
        for step in range(session.worker.settle_steps):
            if step % 12 == 0:
                robot.apply_action(ArticulationAction(joint_positions=values,joint_indices=indices))
            world.step(render=False)
            fridge.after_physics_step()
            session.after_physics(fridge.report())
        metadata = dict(physics_issues=issues,
            self_collisions_enabled=root.GetAttribute('physxArticulation:enabledSelfCollisions').Get(),
            finger_pad_material_policy=policies['material'])
        session.accept_initial(metadata)
        mobile.bind()
        for name,path in CAMERAS.items():
            product = rep.create.render_product(path,CAMERA_RESOLUTIONS[name])
            products.append(product)
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            annotators[name] = annotator
        settings = carb.settings.get_settings()

        def capture():
            before_time = float(world.current_time)
            before = robot.get_joint_positions().copy()
            playing = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(4):
                    rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                result = {}
                for name,annotator in annotators.items():
                    pixels = np.asarray(annotator.get_data())
                    width,height = CAMERA_RESOLUTIONS[name]
                    if pixels.shape!=(height,width,4) or pixels.dtype!=np.uint8:
                        raise RuntimeError('Invalid native camera buffer')
                    stream = io.BytesIO()
                    Image.fromarray(pixels[:,:,:3]).save(stream,format='PNG')
                    result[name] = base64.b64encode(stream.getvalue()).decode('ascii')
                if float(world.current_time)!=before_time or not np.array_equal(before,robot.get_joint_positions()):
                    raise RuntimeError('Capture changed physics')
                return result
            finally:
                settings.set('/app/player/playSimulations',playing)

        driver = SourceDriver(session,robot,stage,mobile,capture)
        driver.fridge = fridge
        host = SessionHost(session,driver)
        report.update(status='ready',initial_acceptance=metadata['task_initial_acceptance'],policies=policies)
        save()
        if args.smoke:
            targets = driver.measured_targets()
            from .contract import ORDER
            host.execute(dict(op='execute_actions',request_id=str(uuid4()),episode_id=session.episode_id,
                expected_step=0,actions=[[targets[n] for n in ORDER[:20]]+[0.,0.] for _ in range(20)]))
            host.close()
        else:
            serve(host,output_stream=completion)
        if not session.closed:
            session.close()
        adapter.verify_sources(plan)
        adapter.bind(plan,manifest)
        report.update(status='completed',result=session.result(),native_owner_smoke=args.smoke)
    except BaseException as exc:
        report.update(status='invalid',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        save()  # Preserve the actual cause before slow native shutdown.
        if session is not None:
            session._fault(exc)
    finally:
        for label, callback in [('session',lambda: session and session.close()),
                ('annotators',lambda: [a.detach() for a in annotators.values()]),
                ('products',lambda: [p.destroy() for p in products]),
                ('physics',lambda: physics and physics.close()),
                ('world',lambda: world and world.clear()),('app',lambda: app and app.close())]:
            try:
                callback()
            except BaseException as exc:
                report['cleanup_errors'].append(label+': '+str(exc))
        report['native_shutdown_complete'] = not report['cleanup_errors']
        save()
    code = 0 if report['status']=='completed' and not report['cleanup_errors'] else 1
    if code == 0:
        completion.complete()
    rpc.flush(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    main()

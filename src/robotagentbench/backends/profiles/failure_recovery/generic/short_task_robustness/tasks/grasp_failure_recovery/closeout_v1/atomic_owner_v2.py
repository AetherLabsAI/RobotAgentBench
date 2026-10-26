"""Task-owned native robot22 owner; source physics and scoring are unchanged.

Setup and camera settings follow public_integration_v1/native_worker.py. The
only new state is the task-owned failed reset. No human actor or object helper.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
import traceback
from uuid import uuid4
from ..common import HERE, owned, digest, verify_sources, write_json
from .. import visual_diagnostics_id as visual_diagnostics


def stage_issues(stage):
    from pxr import UsdPhysics
    from arena_api.isaac_backend import contact_offset_issue
    issues = []
    for prim in stage.Traverse():
        issue = contact_offset_issue(prim.GetAttribute('physxCollision:contactOffset').Get(),
                                     prim.GetAttribute('physxCollision:restOffset').Get())
        if issue:
            issues.append(dict(issue, prim=str(prim.GetPath())))
        if prim.GetAttribute('physics:kinematicEnabled').Get() is True:
            parent = prim
            while parent and not parent.IsPseudoRoot():
                if parent.HasAPI(UsdPhysics.ArticulationRootAPI):
                    issues.append(dict(kind='kinematic_body_under_articulation', prim=str(prim.GetPath())))
                    break
                parent = parent.GetParent()
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int, choices=list(range(8)), required=True)
    p.add_argument('--gpu-uuid', help='Exactly one isolated device is required')
    p.add_argument('--allow-idle-2mib', '--allow-idle-gpu4-2mib', dest='allow_idle_gpu4_2mib', action='store_true',
                   help='Explicit per-run approval: no processes, 0%% utilization, <=2 MiB residue')
    a = p.parse_args()
    if socket.gethostname() not in ('VM-0-8-ubuntu', 'VM-0-10-ubuntu', 'VM-0-13-ubuntu', 'VM-0-11-ubuntu', 'VM-0-9-ubuntu'):
        raise RuntimeError('Only sim1/sim2 authorized')
    from .adapter import admit_initial
    initial_file, public_plan = admit_initial(a.initial)
    initial = json.loads(initial_file.read_text())
    visible_uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
    if len(visible_uuids) != 1 or (a.gpu_uuid is not None and visible_uuids != [a.gpu_uuid]):
        raise RuntimeError('This owner requires exactly one isolated GPU')
    a.gpu_uuid = visible_uuids[0]
    if initial.get('initialization_policy') == 'target_relative_empty_closed_grasp.v1':
        if initial.get('preparation_accepted') is not True or initial.get(
                'generation_reads_recorded_trajectory') is not False:
            raise ValueError('Procedural native probe requires an accepted trajectory-free CPU proposal')
    if initial.get('variant') not in ('shelf_drink', 'coffee_bag', 'coffee_cup', 'coffee_button'):
        raise ValueError('This hold-only diagnostic supports shelf/bag/cup/button adapters only; '
                         'the eight-case catalog is not a claim of native runtime support')
    if digest(initial['scene']) != initial['scene_sha256']:
        raise ValueError('Prepared scene changed')
    verify_sources(initial['source_layers'])
    verify_sources(initial['code_sources'])
    out = owned(a.output, fresh=True)
    gpu = subprocess.check_output(['nvidia-smi', '--id', a.gpu_uuid,
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--id', a.gpu_uuid,
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    if visible != [a.gpu_uuid]:
        raise RuntimeError('Container must expose exactly the selected device; no sharing or fallback')
    pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1'], text=True)
    rows = [s.split() for s in pmon.splitlines() if s.strip() and not s.lstrip().startswith('#')]
    memory_admission = 2 if a.allow_idle_gpu4_2mib else 1
    if apps or any(len(row) > 1 and row[1] != '-' for row in rows) or int(gpu[1]) > memory_admission or int(gpu[2]) != 0:
        raise RuntimeError('Selected GPU admission rejected; no process stopped: '+json.dumps(
            dict(memory_mib=int(gpu[1]),utilization=int(gpu[2]),allowed_residue_mib=memory_admission,
                 compute_processes=apps,graphics_inventory=rows)))
    out.mkdir(parents=True)
    lock = open(HERE/(socket.gethostname()+'-gpu'+str(a.gpu)+'-native.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
        XDG_CACHE_HOME=str(out/'cache'), CUDA_CACHE_PATH=str(out/'cache/cuda'))
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    from short_tasks.agent_probe_v1.completion_stream import CompletionStream
    completion = CompletionStream(rpc)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    started = time.monotonic()
    report = dict(status='initializing', variant=initial['variant'], task_name=public_plan['task_name'],
        control_hz=20, owner_calls_models=False, action_source='internal_robot22_hold_smoke' if a.smoke else 'stdio_client',
        physics_backend='cpu',
        source_api=initial['source_api'], source_checkpoint=initial['source_checkpoint'],
        source_is_recorded_failure=False, runtime_registered=False, full_recovery_actions_verified=False,
        gpu=dict(index=a.gpu, local_renderer_index=0, uuid=gpu[0].strip(), idle_memory_mib=int(gpu[1]), pmon=pmon,
                 approved_idle_residue_limit_mib=memory_admission),
        cleanup_errors=[])
    app = world = worker = physics = lifecycle = None
    annotators, products = [], []
    diagnostic_annotator = None
    report['visual_diagnostics'] = dict(observation_only=True, visual_acceptance=False,
        original_camera_and_renderer_unchanged=True, captures={})

    def save():
        report['wall_s'] = time.monotonic()-started
        write_json(out/'native_report.json', report)

    try:
        save()
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False,
            fast_shutdown=False, active_gpu=0, physics_gpu=0, width=640, height=480,
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
        from arena_api.task_runtime import WorkerContext
        from arena_randomization import shelf_supports
        from .registration import registration_for
        from arena_api.task_lifecycle import TaskLifecycle
        context = omni.usd.get_context()
        if not context.open_stage(initial['scene']):
            raise RuntimeError('Cannot open prepared scene')
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        policies = dict(l1_scene=l1_scene.apply(stage))
        fridge = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions', robot_collision_filters), ('material', finger_pad_material),
                ('contact', finger_pad_contact), ('microwave', microwave_physics), ('shelf', shelf_supports),
                ('obstacles', scene_obstacles), ('coffee', coffee_box_physics)]:
            policies[name] = module.apply(stage)
        physics_defaults.apply_robot(stage)
        issues = stage_issues(stage)
        scoring = out/'scoring'
        scoring.mkdir()
        registration, registration_receipt = registration_for(initial_file)
        write_json(out/'registration.json', registration_receipt)
        max_steps = registration.default_steps
        ctx = WorkerContext(str(uuid4()), initial['source_api'], scoring,
            initial['seed'], max_steps, {}, registration_receipt['task_parameters'])
        worker = registration.runtime_factory(ctx)
        lifecycle = TaskLifecycle(worker, ctx, registration, pe.b.CONTROL_DT)
        if lifecycle.prepare_scene(Path(initial['scene'])).resolve() != Path(initial['scene']).resolve():
            raise RuntimeError('Task initialization changed prepared scene identity')
        lifecycle.attach(stage)
        from ..public_mobile import Controller
        mobile = Controller(stage)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1/30, stage_units_in_meters=1.,
            physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        physics = EpisodePhysicsRuntime(carb.settings.get_settings())
        physics.prepare(stage, world)
        lifecycle.before_physics_reset(world)
        world.reset()
        init = lifecycle.robot_initialization()
        write_json(out/'robot_initialization.json', reset_robot(robot, stage, init))
        policies['runtime'] = physics_defaults.verify_runtime(world, robot)
        policies['episode_physics'] = physics.verify(world)
        lifecycle.initialize(world)
        indices = np.array([robot.dof_names.index(n) for n in init.hold_targets])
        values = np.array(list(init.hold_targets.values()))

        def physical_step(index):
            if index % pe.SUBSTEPS == 0:
                robot.apply_action(ArticulationAction(joint_positions=values, joint_indices=indices))
            world.step(render=False)
            fridge.after_physics_step()
            lifecycle.after_physics(fridge.report())

        for index in range(worker.settle_steps):
            physical_step(index)
        print('SETTLED', worker.settle_steps, flush=True)
        metadata = dict(physics_issues=issues,
            self_collisions_enabled=root.GetAttribute('physxArticulation:enabledSelfCollisions').Get(),
            finger_pad_material_policy=policies['material'])
        # Preserve a real settled image even if admission rejects the initial
        # state. The existing camera/render settings and physics remain intact.
        def capture_settled():
            nonlocal diagnostic_annotator
            camera_paths = dict(head='/World/Robot/head_link2/Head_Camera',
                left='/World/Robot/gripper_l_base_link/Left_Camera', right='/World/Robot/gripper_r_base_link/Right_Camera')
            for name, path in camera_paths.items():
                product = rep.create.render_product(path, (640, 480))
                products.append(product)
                ann = rep.AnnotatorRegistry.get_annotator('rgb')
                ann.attach([product]);annotators.append((name,ann))
                if name == 'head':
                    diagnostic_annotator = visual_diagnostics.attach_head_instance(rep, product)
            settings=carb.settings.get_settings()
            playing=settings.get('/app/player/playSimulations')
            before_time=float(world.current_time);before_joints=robot.get_joint_positions().copy()
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(4):
                    rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                for name,ann in annotators:
                    pixels=np.asarray(ann.get_data())
                    if pixels.shape!=(480,640,4) or pixels.dtype!=np.uint8:
                        raise RuntimeError('Invalid source camera image')
                    Image.fromarray(pixels[:,:,:3]).save(out/(name+'_settled.png'))
                measured = dict(zip(robot.dof_names, map(float, robot.get_joint_positions())))
                diagnostic = dict(inventory=visual_diagnostics.inventory(stage, measured),
                    renderer_instances=visual_diagnostics.instance_summary(diagnostic_annotator.get_data()))
                write_json(out/'visual_settled.json', diagnostic)
                report['visual_diagnostics']['captures']['settled'] = dict(
                    inventory=str(out/'visual_settled.json'),
                    renderer_instances=diagnostic['renderer_instances'])
                if float(world.current_time)!=before_time or not np.array_equal(before_joints,robot.get_joint_positions()):
                    raise RuntimeError('Image capture advanced physics')
            finally:
                settings.set('/app/player/playSimulations',playing)
        capture_settled()
        print('SETTLED_IMAGES_SAVED',flush=True)
        def measure_closed_hand(label):
            closure=initial.get('gripper_closure',initial.get('right_gripper_closure'))
            if initial['variant'] not in ('shelf_drink','coffee_bag') or not closure:
                return
            hand=closure.get('hand',initial.get('active_hand','r'))
            primary='idx41_gripper_l_outer_joint1' if hand=='l' else 'idx81_gripper_r_outer_joint1'
            measured=dict(zip(robot.dof_names,map(float,robot.get_joint_positions())))
            coordinates={n:measured[n] for n in closure['reset_coordinates']}
            # Reset-write roundoff is not a physical PD tracking tolerance.
            # Report real residuals rather than forcing q=0, changing physics,
            # or inventing a new closure/score threshold.
            requested={n:initial['robot_initialization']['positions'][n] for n in coordinates}
            if any(v!=0. for v in requested.values()):
                raise RuntimeError('The requested active-linkage reset is not native closed zero')
            report.setdefault('active_gripper_measurements',{})[label]=dict(hand=hand,
                requested_coordinates_rad=requested,coordinates_rad=coordinates,
                primary_tracking_residual_rad=coordinates[primary],
                physical_coordinates_forced=False)
        measure_closed_hand('settled')
        lifecycle.accept_initial(metadata, float(world.current_time))
        report['original_initial_acceptance'] = metadata['task_initial_acceptance']
        if 'recovery_initial_acceptance' in metadata:
            report['recovery_initial_acceptance'] = metadata['recovery_initial_acceptance']
        evidence, score = lifecycle.evidence, lifecycle.score
        report.update(status='initial_accepted', initial_score=score.score, policies=policies,
            measured_joint_positions=dict(zip(robot.dof_names, map(float, robot.get_joint_positions()))))
        save()
        if score.score != 0:
            raise RuntimeError('Empty-grasp initial state incorrectly earned score')
        from ..public_control import run as run_public_control
        run_public_control(stage=stage,world=world,robot=robot,mobile=mobile,lifecycle=lifecycle,
            init=init,plan=public_plan,fridge=fridge,products=products,annotators=annotators,
            app=app,out=out,report=report,completion=completion,smoke=a.smoke)
        verify_sources(initial['source_layers'])
        report.update(status='completed', original_native_initial_accepted=True,
            source_layers_unchanged=True, scored_recovery_trajectory=False, native_output=str(out))
    except BaseException as exc:
        report.update(status='invalid', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        save()
        for label, callback in [('reader', lambda: lifecycle.close(completed=report['status']=='completed') if lifecycle is not None else
                 worker.close() if worker is not None else None),
                ('diagnostic_annotator', lambda: diagnostic_annotator and diagnostic_annotator.detach()),
                ('annotators', lambda: [ann.detach() for _, ann in annotators]),
                ('products', lambda: [p.destroy() for p in products]),
                ('physics', lambda: physics and physics.close()),
                ('world', lambda: world and world.clear()), ('app', lambda: app and app.close())]:
            try:
                callback()
            except BaseException as exc:
                report['cleanup_errors'].append(label+': '+str(exc))
        if lifecycle is not None:
            report['final_lifecycle_evaluation'] = lifecycle.evaluation()
        save()
    print(json.dumps({k: report.get(k) for k in ('variant', 'status', 'error', 'initial_score', 'idle_score', 'wall_s')}), flush=True)
    code = 0 if report['status'] == 'completed' and not report['cleanup_errors'] else 1
    if code == 0:
        completion.complete()
    rpc.flush(); sys.stderr.flush(); sys.stdout.flush()
    os._exit(code)


if __name__ == '__main__':
    main()

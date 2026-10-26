"""Unscored native bag/combined actor test with the existing approach actions.

Not an adaptive recovery policy or a source-compatible full pour replay.
No robot pauses, generated recovery actions, target setters or new parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=('bag_only', 'bag_then_machine'), required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--authorized-gpu1-residual-1mib', action='store_true')
    args = parser.parse_args(argv)
    from .gpu_admission import admit
    gpu = admit(args.gpu, authorized_gpu1_residual_1mib=args.authorized_gpu1_residual_1mib)
    out, scene = args.output.resolve(), args.initial.resolve(strict=True)
    from short_task_robustness.tasks.coffee_bag_interference.build_preview_scene import HERE as BAG_SOURCE
    if not (scene.is_relative_to(HERE) or scene.is_relative_to(BAG_SOURCE.resolve())):
        raise ValueError('Task-owned or original bag-component initial required as read-only input')
    if out == HERE or not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Fresh task-local probe output required')
    cache = args.reuse_cache.resolve() if args.reuse_cache else out / 'cache'
    if cache == HERE or not cache.is_relative_to(HERE):
        raise ValueError('Task-local cache required')
    initial = json.loads((scene.parent / 'initial.json').read_text())
    if 'relocation_pair' not in initial:
        raise ValueError('Paired bag start/end required')
    if args.component == 'bag_then_machine' and initial.get('variant') != 'bag_then_machine':
        raise ValueError('Explicit combined initial required')
    if args.component == 'bag_only' and initial.get('approved_dynamic_body'):
        raise ValueError('Independent bag test must retain its original kinematic machine')
    source_hashes = dict(initial['source_layers'])
    source_hashes[str(scene)] = hashlib.sha256(scene.read_bytes()).hexdigest()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in source_hashes.items()):
        raise ValueError('Read-only initial or source layer changed')
    out.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      XDG_CACHE_HOME=str(cache), CUDA_CACHE_PATH=str(cache / 'cuda'))
    report = dict(status='starting', component=args.component, gpu_admission=gpu, physics='cpu_physx',
                  scored_episode=False, model_run=False, full_pour_verified=False,
                  robot_pose_commands=0, bag_pose_commands=0, machine_pose_commands=0, cleanup_errors=[])
    app = world = actor = bag_actor = reader = physics = encoder = rgb = product = None
    trace, times = [], []
    started = time.monotonic()

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False,
            default=lambda v: v.tolist() if hasattr(v, 'tolist') else list(v)) + '\n')

    def checkpoint():
        report['wall_s'] = time.monotonic() - started
        save('report.json', report)

    def timeout(*_):
        raise TimeoutError('Existing native preview 900-second watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    checkpoint()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False,
            fast_shutdown=False, active_gpu=args.gpu, physics_gpu=0, width=640, height=480,
            sync_loads=True, limit_cpu_threads=8,
            extra_args=['--/log/file=' + str(out / 'kit.log'), '--/app/userConfigPath=' + str(out / 'kit_user.json'),
                        '--/app/tokens/cache=' + str(cache), '--/app/tokens/data=' + str(out / 'data'),
                        '--/app/tokens/documents=' + str(out / 'documents')]))
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import PhysxSchema, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.types import ArticulationAction
        from arena_api import l1_scene, l1_fridge, physics_defaults, policy_execution as pe, robot_profiles as rp
        from arena_api import robot_collision_filters, finger_pad_material, finger_pad_contact
        from arena_api import microwave_physics, scene_obstacles, coffee_box_physics
        from arena_api.soft_body_runtime import EpisodePhysicsRuntime
        from arena_api.robot_initialization import apply as reset_robot
        from arena_api.task_runtime import RobotInitialization
        from arena_randomization import shelf_supports
        from evaluation.tasks.coffee_machine.isaac_evidence import CoffeeReader
        from evaluation.tasks.coffee_machine.specs import BAG
        from short_task_robustness.tasks.coffee_bag_interference.preview_actions import approach
        from short_task_robustness.tasks.coffee_bag_interference.training_domain import contains_origin
        from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import pose_matrix
        from .atomic_replay import StreamingVideo, compress_reader_output, validate_initial
        candidate = approach(initial)
        save('diagnostic_actions.json', candidate)
        context = omni.usd.get_context()
        if not context.open_stage(str(scene)):
            raise RuntimeError('Cannot open admitted initial')
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        policies = {'l1_scene': l1_scene.apply(stage)}
        controller = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions', robot_collision_filters), ('material', finger_pad_material),
                ('contact', finger_pad_contact), ('microwave', microwave_physics), ('shelf', shelf_supports),
                ('obstacles', scene_obstacles), ('coffee', coffee_box_physics)]:
            policies[name] = module.apply(stage)
        physics_defaults.apply_robot(stage)
        if args.component == 'bag_then_machine':
            from .combined_session import CombinedNativeSession
            from .isaac_evidence import MovingCoffeeReader
            actor = CombinedNativeSession(stage, initial)
            bag_actor = actor.bag
            reader = MovingCoffeeReader(stage, out / 'physical_evidence', 'combined-native-approach-probe')
        else:
            from .native_bag import NativeBagRelocation
            actor = NativeBagRelocation(stage, initial)
            bag_actor = actor
            reader = CoffeeReader(stage, out / 'physical_evidence', 'bag-native-approach-probe', '1-2')
        compress_reader_output(reader)
        save('bag_reservation.json', bag_actor.reservation)
        report['human_initial_absence'] = bag_actor.absence
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1/30, stage_units_in_meters=1.,
                      physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        physics = EpisodePhysicsRuntime(carb.settings.get_settings())
        physics.prepare(stage, world)
        world.reset()
        init = RobotInitialization(initial['robot_initialization']['positions'], initial['robot_initialization']['hold_targets'])
        save('robot_initialization.json', reset_robot(robot, stage, init))
        policies['runtime'] = physics_defaults.verify_runtime(world, robot)
        policies['episode_physics'] = physics.verify(world)
        save('policies.json', policies)
        reader.initialize(world)
        view = SimulationManager.get_physics_sim_view()
        view.initialize_kinematic_bodies()
        actor.bind(view, reader)
        native_robot = view.create_articulation_view('/World/Robot')
        target = view.create_rigid_body_view(BAG)
        names = tuple(init.hold_targets)
        indices = np.array([robot.dof_names.index(name) for name in names])

        def command(action):
            robot.apply_action(ArticulationAction(joint_positions=np.array([action[name] for name in names]), joint_indices=indices))

        def advance():
            world.step(render=False)
            controller.after_physics_step()
            reader.after_step()

        def poses():
            return {'/World/Robot/' + name: pose_matrix(row[:3], row[3:]).T for name, row in
                    zip(native_robot.shared_metatype.link_names, native_robot.get_link_transforms()[0])}

        report['status'] = 'settling'
        checkpoint()
        for step in range(round(initial['settle_seconds'] / pe.PHYSICS_DT)):
            if step % pe.SUBSTEPS == 0:
                command(init.hold_targets)
            advance()
        save('initial_acceptance.json', validate_initial(reader))
        # Same legal native JOINT20 plan/direct-target validation as the existing source probe.
        measured = dict(zip(robot.dof_names, robot.get_joint_positions().tolist()))
        measured = {name: measured[name] for name in rp.get(rp.JOINT20).joints}
        asset_limits, driven = {}, set()
        for name in robot.dof_names:
            prim = stage.GetPrimAtPath('/World/Robot/joints/' + name)
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
                raise RuntimeError('Unmapped native joint ' + name)
            lo, hi = (prim.GetAttribute('physics:' + key).Get() for key in ('lowerLimit', 'upperLimit'))
            if lo is not None and hi is not None:
                asset_limits[name] = [float(np.radians(v)) if prim.IsA(UsdPhysics.RevoluteJoint) else float(v) for v in (lo, hi)]
            if any(schema.startswith('PhysicsDriveAPI:') for schema in prim.GetAppliedSchemas()):
                driven.add(name)
        if driven != set(names) or driven != set(rp.get(rp.JOINT20).joints):
            raise RuntimeError('Exactly the original source 20 active drives required')
        previous, limits = pe.build_limits(measured, asset_limits, rp.JOINT20)
        for action in candidate['actions']:
            pe.validate(action, limits, rp.JOINT20)
        reader.begin_episode()
        actor.arm(robot_native_poses=poses())
        product = rep.create.render_product('/World/Robot/head_link2/Head_Camera', (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
        encoder = StreamingVideo(out / 'actor_probe.mp4')
        settings = carb.settings.get_settings()

        def capture():
            before = target.get_transforms().copy()
            playing = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations', False)
            try:
                for _ in range(4):
                    rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0., wait_for_render=True)
                image = np.asarray(rgb.get_data()).copy()
                if image.shape != (480, 640, 4) or not np.array_equal(before, target.get_transforms()):
                    raise RuntimeError('Invalid native no-step capture')
                encoder.write(image[:, :, :3])
                for name, eligible in [('initial', not times), ('trigger', bag_actor.start is not None),
                                       ('retired', bag_actor.retired)]:
                    path = out / (name + '.png')
                    if eligible and not path.exists():
                        Image.fromarray(image[:, :, :3]).save(path)
                times.append(float(world.current_time))
            finally:
                settings.set('/app/player/playSimulations', playing)

        capture()
        report.update(status='executing_approach', action_rows=len(candidate['actions']),
                      gripper_commands='unchanged recorded reset hold, existing independent approach diagnostic')
        checkpoint()
        for index, action in enumerate(candidate['actions']):
            measured = dict(zip(robot.dof_names, robot.get_joint_positions().tolist()))
            measured = {name: measured[name] for name in rp.get(rp.JOINT20).joints}
            interval = pe.plan_interval(action, measured, previous, limits, rp.JOINT20)
            for planned in interval['interpolated_substep_targets_rad']:
                measured = dict(zip(robot.dof_names, robot.get_joint_positions().tolist()))
                measured = {name: measured[name] for name in rp.get(rp.JOINT20).joints}
                direct, changes = pe.direct_gripper_substep(action, planned, measured, limits, rp.JOINT20)
                if changes or direct != action:
                    raise RuntimeError('No action rewrite allowed')
                command(action)
                actor.command_before_step(float(world.current_time))
                advance()
                event = actor.observe(robot_native_poses=poses())
                previous = dict(action)
            reader.frame()
            row = reader.latest
            trace.append(dict(time_s=row['time_s'], action_step=index, event=event,
                              bag_pose=row['poses'][BAG], bag_contacts=row['contacts'][BAG],
                              bag_human_completed=bag_actor.completed, bag_human_retired=bag_actor.retired))
            capture()
            if index % 20 == 0:
                report.update(control_step=index, bag_trigger_time_s=bag_actor.start,
                              bag_human_completed=bag_actor.completed, bag_human_retired=bag_actor.retired)
                checkpoint()
        row = reader.latest
        final_domain = contains_origin(row['poses'][BAG]['position_m'], bag_actor.reservation['training_domain'])
        report.update(status='approach_probe_completed', rows_executed=len(candidate['actions']),
                      bag_trigger_time_s=bag_actor.start, bag_human_completed=bag_actor.completed,
                      bag_human_retired=bag_actor.retired, bag_completion_evidence=bag_actor.completion_evidence,
                      final_in_training_domain=final_domain, final_bag_contacts=row['contacts'][BAG],
                      final_beans_in_bag=reader.beans_in_bag(), frame_count=len(times),
                      robot_continued=True, full_task_verified=False,
                      component_intervention_verified=bool(bag_actor.completed and bag_actor.retired),
                      complete_preview_verified=bool(bag_actor.completed and bag_actor.retired and final_domain))
        if args.component == 'bag_then_machine':
            report.update(completed_components=list(actor.sequence.completed),
                          machine_triggered=actor.push.start is not None, machine_retired=actor.push.retired,
                          combined_full_trigger_verified=actor.sequence.completed == ['bag_move', 'machine_push'])
    except BaseException as error:
        report.update(status='error', error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if bag_actor is not None:
            save('actor_events.json', dict(bag=bag_actor.events,
                combined=actor.events if args.component == 'bag_then_machine' else None))
        save('trace.json', trace)
        save('video_times.json', times)
        for label, close in [('actor', lambda: actor.close() if actor else None),
                ('encoder', lambda: encoder.close() if encoder else None),
                ('reader', lambda: reader.close() if reader else None),
                ('render', lambda: (rgb.detach(), product.destroy()) if rgb is not None else None),
                ('world', lambda: world.stop() if world else None),
                ('physics', lambda: physics.close() if physics else None)]:
            try:
                close()
            except BaseException as error:
                report['cleanup_errors'].append(dict(operation=label, error=str(error)))
        report['source_layers_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in source_hashes.items())
        checkpoint()
        if app:
            app.close()
        report['native_shutdown_complete'] = True
        checkpoint()
    return 0 if report['status'] == 'approach_probe_completed' else 1


if __name__ == '__main__':
    import sys
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

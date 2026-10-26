"""Unscored CPU palm-push diagnostics with explicit, distinct test modes.

Default: isolated mechanism with original legal reset hold commands at 20 Hz.
--atomic-source-replay: original source actions, explicitly approved opening
adaptation, actual native trigger only. Both modes are UNSCORED diagnostics,
not a registered API/VLA episode. Machine/cup/bag poses are never commanded,
constrained or repositioned. No fake holding/tilt facts enter the trigger.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent


class AtomicReplayCompleted(Exception):
    """Leave the isolated-mechanism branch after a real-trigger replay."""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--authorized-shared-gpu2', action='store_true')
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--atomic-source-replay', action='store_true',
                        help='User-approved 0.8-rad open adaptation; actual trigger, no forced intervention')
    parser.add_argument('--hd-presentation', action='store_true',
                        help='Requested native 1440x1080 delivery; same 4:3 camera, actions and physical clock')
    args = parser.parse_args(argv)
    if args.hd_presentation and not args.atomic_source_replay:
        parser.error('HD presentation is only defined for the complete source replay')
    from .atomic_replay import capture_resolution
    resolution = capture_resolution(args.hd_presentation)
    out = args.output.resolve()
    initial_path = args.initial.resolve()
    if out.exists() or out == HERE or not out.is_relative_to(HERE):
        raise ValueError('Fresh task-local diagnostic output required')
    if not initial_path.is_relative_to(HERE) or not initial_path.is_file():
        raise ValueError('Use a task-owned already audited initial scene')
    from .gpu_admission import admit
    gpu = admit(args.gpu, authorized_shared_gpu2=args.authorized_shared_gpu2)
    cache_dir = args.reuse_cache.resolve() if args.reuse_cache else out / 'cache'
    if cache_dir == HERE or not cache_dir.is_relative_to(HERE):
        raise ValueError('Cache must stay inside this task too')
    uuid = gpu['uuid']
    occupancy = gpu
    out.mkdir()
    (out / 'frames').mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      XDG_CACHE_HOME=str(cache_dir), CUDA_CACHE_PATH=str(cache_dir / 'cuda'))
    report = dict(status='starting', physical_backend='cpu_physx', render_gpu=args.gpu,
                  capture_resolution=list(resolution), hd_presentation=args.hd_presentation,
                  capture_is_native_resolution=True, camera_pose_and_lens_unchanged=True,
                  gpu_admission=occupancy, gpu_sharing_authorized=gpu['shared'],
                  diagnostic='isolated_palm_push_mechanism', task_trigger_exercised=False,
                  scored_episode=False, model_run=False, machine_pose_commands=0,
                  cup_pose_commands=0, bag_pose_commands=0, cache_directory=str(cache_dir), cleanup_errors=[])
    app = world = reader = session = physics_runtime = rgb = product = encoder = None
    started = time.monotonic()
    trace = []
    frame_times = []

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False,
                                          default=lambda v: v.tolist() if hasattr(v, 'tolist') else list(v)) + '\n')

    def checkpoint():
        report['wall_s'] = time.monotonic() - started
        save('report.json', report)

    def timeout(*_):
        raise TimeoutError('Existing native diagnostic 900-second watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    checkpoint()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', multi_gpu=False, fast_shutdown=False,
            active_gpu=args.gpu, physics_gpu=0, width=640, height=480, sync_loads=True, limit_cpu_threads=8,
            extra_args=['--/log/file=' + str(out / 'kit.log'), '--/app/userConfigPath=' + str(out / 'kit_user.json'),
                        '--/app/tokens/cache=' + str(cache_dir), '--/app/tokens/data=' + str(out / 'data'),
                        '--/app/tokens/documents=' + str(out / 'documents')]))
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.types import ArticulationAction
        from arena_api import l1_scene, l1_fridge, physics_defaults, policy_execution as pe
        from arena_api.bimanual import CONTROL_DT
        from arena_api import robot_collision_filters, finger_pad_material, finger_pad_contact
        from arena_api import microwave_physics, scene_obstacles, coffee_box_physics
        from arena_api.soft_body_runtime import EpisodePhysicsRuntime
        from arena_api.robot_initialization import apply as reset_robot
        from arena_api.task_runtime import RobotInitialization
        from arena_randomization import shelf_supports
        from evaluation.tasks.coffee_machine.specs import MACHINE_BODY, BAG, CUP, COUNTER
        from evaluation.tasks.coffee_machine.facts import COUNTER as ON_COUNTER, TRAY as ON_TRAY
        from evaluation.tasks.coffee_machine.isaac_evidence import under, UP_NORMAL_Z
        from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import pose_matrix as native_matrix
        from .isaac_evidence import MovingCoffeeReader
        from .native_push import NativePalmPush
        from .push_motion import PalmPushMotion, PUSH_DURATION_S
        initial = json.loads((initial_path.parent / 'initial.json').read_text())
        if args.atomic_source_replay:
            from .replay_actions import adapt
            candidate = adapt(initial)
            save('adapted_source_actions.json', candidate)
            report.update(diagnostic='atomic_source_real_trigger_replay', original_unmodified_replay=False)
        if initial['source_api'] != '1-2' or not initial['approved_dynamic_body']['all_other_physical_properties_equal']:
            raise ValueError('Use the approved independent dynamic atomic 1-2 layer')
        for path, digest in initial['source_layers'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source layer changed before diagnostic: ' + path)
        context = omni.usd.get_context()
        if not context.open_stage(str(initial_path)):
            raise RuntimeError('Cannot open independent initial scene')
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hashes = {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                  for layer in stage.GetUsedLayers() if layer.realPath}
        policies = dict(l1_scene=l1_scene.apply(stage))
        controller = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions', robot_collision_filters), ('material', finger_pad_material),
                ('contact', finger_pad_contact), ('microwave', microwave_physics), ('shelf', shelf_supports),
                ('obstacles', scene_obstacles), ('coffee', coffee_box_physics)]:
            policies[name] = module.apply(stage)
        physics_defaults.apply_robot(stage)
        if stage.GetPrimAtPath(MACHINE_BODY).GetAttribute('physics:kinematicEnabled').Get() is not False:
            raise RuntimeError('Existing runtime policies changed the approved dynamic body')
        session = NativePalmPush(stage, initial)
        report['human_initial_absence'] = session.absence
        reader = MovingCoffeeReader(stage, out / 'physical_evidence', 'coffee-machine-palm-diagnostic')
        if args.atomic_source_replay:
            from .atomic_replay import compress_reader_output
            compress_reader_output(reader)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1/30, stage_units_in_meters=1.,
                      physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        physics_runtime = EpisodePhysicsRuntime(carb.settings.get_settings())
        physics_runtime.prepare(stage, world)
        world.reset()
        init = RobotInitialization(initial['robot_initialization']['positions'], initial['robot_initialization']['hold_targets'])
        save('robot_initialization.json', reset_robot(robot, stage, init))
        policies['runtime'] = physics_defaults.verify_runtime(world, robot)
        policies['episode_physics'] = physics_runtime.verify(world)
        save('policies.json', policies)
        reader.initialize(world)
        view = SimulationManager.get_physics_sim_view()
        view.initialize_kinematic_bodies()
        session.bind(view, reader)
        target = view.create_rigid_body_view(MACHINE_BODY)
        native_robot = view.create_articulation_view('/World/Robot')
        indices = np.array([robot.dof_names.index(name) for name in init.hold_targets])
        values = np.array(list(init.hold_targets.values()))

        def command_robot():
            robot.apply_action(ArticulationAction(joint_positions=values, joint_indices=indices))

        def advance():
            world.step(render=False)
            controller.after_physics_step()
            reader.after_step()

        def robot_poses():
            return {'/World/Robot/' + name: native_matrix(row[:3], row[3:]).T for name, row in
                    zip(native_robot.shared_metatype.link_names, native_robot.get_link_transforms()[0])}

        def physical_row(phase):
            row = reader.latest
            supports = []
            palm_contacts = []
            robot_human = []
            for paths, samples in reader.contacts.pairs():
                for i in (0, 1):
                    if under(paths[i], MACHINE_BODY):
                        if under(paths[1-i], COUNTER) and any((1 if i == 0 else -1)*normal[2] > UP_NORMAL_Z for normal, _ in samples):
                            supports.append(list(paths))
                        if under(paths[1-i], session.human.links['hand.R']):
                            palm_contacts.append(dict(paths=list(paths), impulse=max(s for _, s in samples)))
            # Native headers include robot-human contacts even when the atomic
            # reader's tracked root set does not expand all robot-only pairs.
            from pxr import PhysicsSchemaTools
            from omni.physx.bindings._physx import ContactEventType
            for h in reader.interface.get_contact_report()[0]:
                if h.type == ContactEventType.CONTACT_LOST:
                    continue
                paths = [str(PhysicsSchemaTools.intToSdfPath(int(value))) for value in (h.actor0, h.actor1)]
                if ((under(paths[0], '/World/Robot') and under(paths[1], session.human.root))
                        or (under(paths[1], '/World/Robot') and under(paths[0], session.human.root))):
                    robot_human.append(paths)
            return dict(time_s=row['time_s'], phase=phase,
                machine_pose=row['poses'][MACHINE_BODY], machine_tilt_deg=reader.tilt(row['poses'][MACHINE_BODY]['quaternion_wxyz']),
                machine_counter_contacts=supports, palm_machine_contacts=palm_contacts, robot_human_contacts=robot_human,
                cup_pose=row['poses'][CUP], cup_tilt_deg=row['tilt_deg'][CUP], cup_contacts=row['contacts'][CUP],
                bag_contacts=row['contacts'][BAG], beans_in_bag=reader.beans_in_bag(),
                machine_pose_commands=0)

        report['status'] = 'settling'
        checkpoint()
        for step in range(round(initial['settle_seconds'] / pe.PHYSICS_DT)):
            if step % pe.SUBSTEPS == 0:
                command_robot()
            advance()
        initial_native = physical_row('settled_initial')
        report['settled_initial'] = initial_native
        if (not initial_native['machine_counter_contacts'] or ON_TRAY not in initial_native['cup_contacts']['supports']
                or ON_COUNTER not in initial_native['bag_contacts']['supports'] or initial_native['beans_in_bag'] != 100):
            raise RuntimeError('Dynamic source initial support/beans rejected')
        if args.atomic_source_replay:
            from .atomic_replay import run, StreamingVideo, validate_initial
            save('replay_initial_acceptance.json', validate_initial(reader))
            product = rep.create.render_product('/World/Robot/head_link2/Head_Camera', resolution)
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach([product])
            settings = carb.settings.get_settings()
            encoder = StreamingVideo(out / 'atomic_trigger_replay.mp4', resolution)

            def capture_replay():
                before = target.get_transforms().copy()
                playing = settings.get('/app/player/playSimulations')
                settings.set('/app/player/playSimulations', False)
                try:
                    for _ in range(4):
                        rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0., wait_for_render=True)
                    image = np.asarray(rgb.get_data()).copy()
                    if image.shape != (resolution[1], resolution[0], 4) or not np.array_equal(before, target.get_transforms()):
                        raise RuntimeError('Invalid native no-step capture')
                    encoder.write(image[:, :, :3])
                    if not frame_times:
                        Image.fromarray(image[:, :, :3]).save(out / 'initial.png')
                    Image.fromarray(image[:, :, :3]).save(out / 'latest.png')
                    if session.start is not None and not (out / 'trigger.png').exists():
                        Image.fromarray(image[:, :, :3]).save(out / 'trigger.png')
                    frame_times.append(float(world.current_time))
                finally:
                    settings.set('/app/player/playSimulations', playing)

            run(stage=stage, robot=robot, reader=reader, session=session, native_robot=native_robot,
                initial=initial, candidate=candidate, advance=advance, capture=capture_replay,
                physical_row=physical_row, checkpoint=checkpoint, report=report, trace=trace, save=save)
            report['frame_count'] = len(frame_times)
            raise AtomicReplayCompleted()
        admitted, admission = session._admit(robot_poses())
        report['route_admission'] = dict(admission, clear=admitted)
        if not admitted:
            raise RuntimeError('Live mechanical palm route rejected: ' + admission['reason'])
        fit, push = admission['palm_fit'], admission['proposal']
        motion = PalmPushMotion(fit['wrist_position_m'], push)
        orientation = np.asarray(fit['orientation'])
        product = rep.create.render_product('/World/Robot/head_link2/Head_Camera', (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
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
                Image.fromarray(image[:, :, :3]).save(out / 'frames' / ('frame_%05d.png' % len(frame_times)))
                frame_times.append(float(world.current_time))
            finally:
                settings.set('/app/player/playSimulations', playing)

        capture()
        # Isolated mechanical admission, NOT a forged task trigger. Position
        # the actor while absent, advance natively, then show it at that pose.
        session.orientation = orientation
        session._position_absent(motion.contact)
        advance()
        report['appearance'] = session.human.activate_on_trigger()
        time0 = float(world.current_time)
        baseline = target.get_transforms()[0].copy()
        report['status'] = 'mechanical_push'
        report['production_trigger_bypassed_for_unscored_mechanism_test'] = True
        checkpoint()
        retired = False
        # 1 second push + reverse retreat at the same mean speed, then the
        # original bag preview's 3.5 seconds of post-exit physics.
        for step in range(round((2 * PUSH_DURATION_S + 3.5) / CONTROL_DT)):
            command_robot()
            for _ in range(pe.SUBSTEPS):
                elapsed = float(world.current_time) - time0
                if elapsed < PUSH_DURATION_S:
                    sample = motion.sample(elapsed)
                    position = sample['wrist_position_m']
                    phase = 'push'
                elif elapsed < 2 * PUSH_DURATION_S:
                    fraction = (elapsed - PUSH_DURATION_S) / PUSH_DURATION_S
                    position = motion.contact + (1 - fraction) * motion.distance * motion.direction
                    phase = 'retreat'
                else:
                    phase = 'post_exit'
                    if not retired:
                        report['human_retirement'] = session.human.retire_after_exit()
                        retired = True
                if not retired:
                    matrix = np.eye(4)
                    matrix[:3, :3] = orientation
                    matrix[:3, 3] = position
                    session.human.command(matrix, 0.)
                advance()
                trace.append(physical_row(phase))
            capture()
            if step % 20 == 0:
                report['control_step'] = step
                checkpoint()
        final = target.get_transforms()[0]
        report.update(status='mechanical_preview_completed', planned_palm_travel_m=motion.distance,
                      direction_world_xy_unit=motion.direction.tolist(), initial_machine_pose=baseline.tolist(),
                      final_machine_pose=final.tolist(), measured_machine_horizontal_displacement_m=float(np.linalg.norm(final[:2] - baseline[:2])),
                      final=trace[-1], palm_contact_observed=any(row['palm_machine_contacts'] for row in trace),
                      maximum_machine_tilt_deg=max(row['machine_tilt_deg'] for row in trace),
                      maximum_cup_tilt_deg=max(row['cup_tilt_deg'] for row in trace),
                      robot_human_contact_observed=any(row['robot_human_contacts'] for row in trace),
                      human_retired=retired, frame_count=len(frame_times), robot_control_commands_continued_20hz=True,
                      robot_source_reset_hold_only=True, full_task_verified=False)
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-framerate', '20', '-i', str(out / 'frames/frame_%05d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'mechanical_preview.mp4')], check=True)
    except AtomicReplayCompleted:
        pass
    except BaseException as error:
        report.update(status='error', error=str(error), traceback=traceback.format_exc())
        if (out / 'frames/frame_00000.png').exists():
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-framerate', '20', '-i', str(out / 'frames/frame_%05d.png'),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out / 'partial_preview.mp4')])
    finally:
        signal.alarm(0)
        save('trace.json', trace)
        save('video_times.json', frame_times)
        for label, fn in [('actor', lambda: session.close() if session else None),
                ('encoder', lambda: encoder.close() if encoder else None),
                ('reader', lambda: reader.close() if reader else None),
                ('render', lambda: (rgb.detach(), product.destroy()) if rgb is not None else None),
                ('world', lambda: world.stop() if world else None),
                ('physics', lambda: physics_runtime.close() if physics_runtime else None)]:
            try:
                fn()
            except BaseException as error:
                report['cleanup_errors'].append(dict(operation=label, error=str(error)))
        if 'hashes' in locals():
            report['source_layers_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                                                   for path, digest in hashes.items())
        checkpoint()
        if app:
            app.close()
        report['native_shutdown_complete'] = True
        checkpoint()
    return 0 if report['status'] in ('mechanical_preview_completed', 'atomic_source_replay_completed') else 1


if __name__ == '__main__':
    import sys
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

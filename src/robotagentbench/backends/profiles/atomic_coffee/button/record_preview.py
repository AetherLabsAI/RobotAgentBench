"""Native idle illumination preview, NOT a button success or model evaluation.

Existing atomic 240/20 Hz, 640x480 head camera and 12.05 s lighting preview
capture contract. Original initial acceptance is mandatory before lights arm.
Only original reset-hold commands are sent; no button/target pose commands.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from uuid import uuid4

from .initial import task_output, digest


def idle_gpu(index, allow_idle_gpu4_residue=False):
    if index not in (0, 4):
        raise ValueError('This preview is authorized only for idle GPU0 or GPU4')
    if allow_idle_gpu4_residue and index != 4:
        raise ValueError('This run-specific residue exception is only for GPU4')
    row = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    # pmon includes graphics processes, not only compute contexts.
    monitor = subprocess.check_output(['nvidia-smi', 'pmon', '-i', str(index), '-c', '1'], text=True)
    process_lines = [line for line in monitor.splitlines() if line.strip() and not line.lstrip().startswith('#')
                     and len(line.split()) > 1 and line.split()[1] != '-']
    allowed_residue=2 if allow_idle_gpu4_residue else 0
    if int(row[1]) > allowed_residue or int(row[2]) != 0 or apps or process_lines:
        raise RuntimeError('Authorized GPU is not idle; no process stopped: ' + repr((row, apps, process_lines)))
    return dict(index=index, uuid=row[0].strip(), memory_mib=int(row[1]), utilization_percent=int(row[2]), pmon=monitor,
                compute_processes=apps, authorized_idle_gpu4_residue_2mib=allow_idle_gpu4_residue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 4), required=True)
    parser.add_argument('--allow-idle-gpu4-residue-2mib',action='store_true',
                        help='Explicit run-specific user approval; still reject all compute/graphics processes')
    args = parser.parse_args()
    initial_path = task_output(args.initial).resolve(strict=True)
    plan = json.loads((initial_path.parent/'initial.json').read_text())
    if plan['source_api'] != '1-3' or digest(initial_path) != plan['scene_sha256']:
        raise ValueError('Audited original atomic button initial required')
    for p, sha in plan['source_layers'].items():
        if digest(p) != sha:
            raise RuntimeError('Source layer changed: ' + p)
    out = task_output(args.output)
    if out.exists():
        raise ValueError('Fresh preview output required')
    gpu = idle_gpu(args.gpu,args.allow_idle_gpu4_residue_2mib)
    out.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      XDG_CACHE_HOME=str(out/'cache'), CUDA_CACHE_PATH=str(out/'cache/cuda'))
    report = dict(status='starting', gpu=gpu, physics_backend='cpu_physx',
                  preview_only=True, model_run=False, full_scored_replay=False,
                  cleanup_errors=[], frames=0, camera_parameter_writes=0,
                  original_physics_hz=240, original_control_hz=20)
    app = world = runtime = rgb = product = encoder = None
    start = time.monotonic()
    images = []

    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')

    def checkpoint():
        report['wall_s'] = time.monotonic()-start
        save('report.json', report)

    def timeout(*_):
        raise TimeoutError('Existing native 900-second diagnostic watchdog exhausted')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    checkpoint()
    try:
        # No USD/native physics import before SimulationApp.
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
        from pxr import PhysxSchema, UsdGeom
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.simulation_manager import SimulationManager
        from arena_api import l1_scene, l1_fridge, physics_defaults, policy_execution as pe
        from arena_api import robot_collision_filters, finger_pad_material, finger_pad_contact
        from arena_api import microwave_physics, scene_obstacles, coffee_box_physics
        from arena_api.soft_body_runtime import EpisodePhysicsRuntime
        from arena_api.robot_initialization import apply as reset_robot
        from arena_api.task_runtime import WorkerContext
        from arena_randomization import shelf_supports
        from .runtime import LightingButtonRuntime
        from .lighting import CAMERA
        from .audit import stage_issues, joint_issues
        context = omni.usd.get_context()
        if not context.open_stage(str(initial_path)):
            raise RuntimeError('Cannot open task-owned initial scene')
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hashes = {layer.realPath: digest(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath}
        policies = dict(l1_scene=l1_scene.apply(stage))
        fridge = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if root.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is not True:
            raise RuntimeError('Original robot self-collision setting required')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions', robot_collision_filters), ('material', finger_pad_material),
                ('contact', finger_pad_contact), ('microwave', microwave_physics), ('shelf', shelf_supports),
                ('obstacles', scene_obstacles), ('coffee', coffee_box_physics)]:
            policies[name] = module.apply(stage)
        physics_defaults.apply_robot(stage)
        issues = stage_issues(stage)
        worker = WorkerContext(str(uuid4()), 'coffee_button_lighting_interference', out, plan['seed'],
                               1340, {}, dict(source_episode=plan['source_episode'], random_layout=plan['role']=='random'))
        runtime = LightingButtonRuntime(worker)
        runtime.plan = plan
        runtime.appearance = plan['appearance']
        runtime.attach(stage)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1/30, stage_units_in_meters=1.,
                      physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        physics_runtime = EpisodePhysicsRuntime(carb.settings.get_settings())
        physics_runtime.prepare(stage, world)
        world.reset()
        init = runtime.robot_initialization()
        save('robot_initialization.json', reset_robot(robot, stage, init))
        issues.extend(joint_issues(stage, robot))
        policies['runtime'] = physics_defaults.verify_runtime(world, robot)
        policies['episode_physics'] = physics_runtime.verify(world)
        save('policies.json', policies)
        runtime.on_physics_ready(world)
        indices = np.array([robot.dof_names.index(n) for n in init.hold_targets])
        values = np.array(list(init.hold_targets.values()))

        def advance():
            world.step(render=False)
            fridge.after_physics_step()
            runtime.after_physics(fridge.report())

        def hold():
            robot.apply_action(ArticulationAction(joint_positions=values, joint_indices=indices))

        report['status'] = 'settling'; checkpoint()
        for index in range(runtime.settle_steps):
            if index % pe.SUBSTEPS == 0:
                hold()
            advance()
        metadata = dict(physics_issues=issues,
                        self_collisions_enabled=root.GetAttribute('physxArticulation:enabledSelfCollisions').Get(),
                        finger_pad_material_policy=policies['material'])
        runtime.validate_initial(metadata)
        report['initial_acceptance'] = metadata['task_initial_acceptance']
        save('initial_metadata.json', metadata)
        product = rep.create.render_product(CAMERA, (640,480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        encoder = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', '640x480', '-framerate', '20', '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'lighting_preview.mp4')],
            stdin=subprocess.PIPE)
        settings = carb.settings.get_settings()
        native_robot = SimulationManager.get_physics_sim_view().create_articulation_view('/World/Robot')
        saved_modes = set()

        def capture():
            before_time = float(world.current_time)
            before = native_robot.get_link_transforms().copy()
            playing = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations', False)
            try:
                for _ in range(4):
                    rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0., wait_for_render=True)
                pixels = np.asarray(rgb.get_data())[:,:,:3].copy()
                state = dict(runtime.lighting.cycle.state)
                # A legitimate brief flash may saturate the sensor. It is not a
                # missing buffer; shape/type still must match the native camera.
                if pixels.shape != (480,640,3) or pixels.dtype != np.uint8 or (pixels.std()<1 and not state['flash']):
                    raise ValueError('Invalid original head-camera frame')
                if float(world.current_time) != before_time or not np.array_equal(before, native_robot.get_link_transforms()):
                    raise RuntimeError('Rendering advanced the physical scene')
                row = dict(time_s=before_time, state=state, mean_rgb=float(pixels.mean()),
                           bright_fraction=float((pixels.min(axis=2)>250).mean()))
                cache=UsdGeom.XformCache()
                row['emitter_world_position_m']=list(cache.GetLocalToWorldTransform(
                    stage.GetPrimAtPath(runtime.lighting.emitter['path'])).ExtractTranslation())
                row['camera_world_position_m']=list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CAMERA)).ExtractTranslation())
                images.append(row)
                encoder.stdin.write(pixels.tobytes())
                if state['mode'] not in saved_modes:
                    Image.fromarray(pixels).save(out/('example_'+state['mode']+'.png'))
                    saved_modes.add(state['mode'])
                report['frames'] = len(images)
            finally:
                settings.set('/app/player/playSimulations', playing)

        evidence = runtime.collect_evidence()
        score = runtime.evaluate(evidence)
        report['status'] = 'recording'; checkpoint(); capture()
        for frame in range(240):
            hold()
            for _ in range(pe.SUBSTEPS):
                advance()
            evidence = runtime.collect_evidence()
            score = runtime.evaluate(evidence)
            capture()
            if (frame+1) % 20 == 0:
                checkpoint()
                print('PREVIEW_SECONDS', (frame+1)/20, flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoder failed')
        encoder = None
        save('frames.json', images)
        report.update(status='recorded', video='lighting_preview.mp4', duration_s=12.05,
                      illumination=runtime.lighting.snapshot(), measured_idle_score=score.score,
                      success=runtime.check_success(evidence, score), scoring_status='idle_preview_only',
                      source_layers_unchanged=all(digest(p)==sha for p,sha in hashes.items()))
    except BaseException as exc:
        report.update(status='error', error=str(exc), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        for name, action in [('encoder', lambda: (encoder.stdin.close(), encoder.wait()) if encoder else None),
                             ('lighting_and_reader', lambda: runtime.close() if runtime else None),
                             ('render_product', lambda: (rgb.detach(), product.destroy()) if rgb else None),
                             ('world_stop', lambda: world.stop() if world else None)]:
            try:
                action()
            except BaseException as exc:
                report['cleanup_errors'].append(name+': '+str(exc))
        checkpoint()
        if app:
            app.close()
        report['native_shutdown_complete'] = True
        checkpoint()
    print(json.dumps({k:report.get(k) for k in ('status','error','frames','wall_s','cleanup_errors')}), flush=True)
    return 0 if report['status']=='recorded' and not report['cleanup_errors'] else 1


if __name__ == '__main__':
    import sys
    result = main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(result)

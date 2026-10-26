"""Initial-state acceptance probe for API 1-1 … 1-4 (no model, no trajectory, no scoring).

Drives the registered task runtime (`api_runtime.CoffeeMachineRuntime`) through the same native reset, the same
public physics policies and the same settle as the public worker, then calls its own `validate_initial`. This tool
never writes a second acceptance rule, never moves a joint and never changes a physics parameter.

Roles:
  source  the layout of one packaged source episode (default: the fixed initial state = first packaged episode)
  random  the seeded layout of randomization.py (only APIs whose definition names a reproducible sampler)

Outputs (read by bind_reset.py, batch_queue.py, publish_results.py):
  manifest.json, initial-metadata.json, measured_checkpoint.json, physical_facts.json, audit.json,
  settled_<camera>.png
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
ROLES = ('source', 'random')
SCHEMA = 'arena.coffee_machine.initial_probe.v1'
AUDIT_SCHEMA = 'arena.coffee_machine.initial_probe_audit.v1'
APIS = ('1-1', '1-2', '1-3', '1-4')
PUBLIC_MODULES = ('task_runtime', 'task_lifecycle', 'policy_execution', 'isaac_worker', 'robot_initialization',
                  'contact_identity', 'scene_obstacles', 'finger_pad_material', 'finger_pad_contact',
                  'robot_collision_filters', 'l1_scene', 'l1_fridge', 'microwave_physics', 'coffee_box_physics',
                  'coffee_assets', 'soft_body_runtime', 'robot_profiles', 'bimanual', 'physics_defaults')


def source_files():
    return (sorted(TASK.glob('*.py')) + sorted(p for p in (TASK / 'vendor_producer').glob('*') if p.is_file())
            + sorted(p for p in (REPO / 'resources/shared_assets/coffee_v1').glob('*') if p.is_file())
            + [REPO / 'arena_api' / f'{name}.py' for name in PUBLIC_MODULES])


def code_hashes():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files() if p.is_file()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--api', required=True, choices=APIS)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--role', default='source', choices=ROLES)
    p.add_argument('--episode', help='packaged source episode (role source); default: the fixed initial state')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((TASK / 'artifacts').resolve()):
        raise ValueError('New task-owned output required')
    from evaluation.tasks.coffee_machine.registration import normalizer
    parameters = normalizer(args.api)(dict(random_layout=args.role == 'random',
                                           **({'source_episode': args.episode} if args.episode else {})))
    out.mkdir(parents=True)
    hashes = code_hashes()
    manifest = dict(schema=SCHEMA, episode_id=str(uuid4()), gpu=args.gpu, task_id=args.api, role=args.role, seed=args.seed,
                    task_parameters=parameters, diagnostic_only=True, model_used=False, source_sha256=hashes, status='starting')

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=list) + '\n')

    def save_manifest():
        save('manifest.json', manifest)

    save_manifest()
    try:
        usage = subprocess.check_output(['nvidia-smi', '--id', str(args.gpu), '--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True, timeout=150)
        processes = subprocess.check_output(['nvidia-smi', '--id', str(args.gpu)], text=True, timeout=150)
        if int(usage.split(',')[1].strip()) or 'No running processes found' not in processes:
            raise RuntimeError('GPU not idle; this probe never shares or stops another job: ' + usage)
        save('gpu_preflight.json', dict(gpu=args.gpu, usage=usage, processes=processes))
        uuid = subprocess.check_output(['nvidia-smi', '--id', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'],
                                       text=True, timeout=150).strip()
        if not uuid.startswith('GPU-'):
            raise RuntimeError('Invalid device UUID')
        manifest['gpu_uuid'] = uuid
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid          # as the public worker: never touch the other cards
        manifest['status'] = 'running'
        save_manifest()

        def complete(result):
            """Finalize from inside run_probe: SimulationApp.close() ends the process."""
            manifest.update(result)
            manifest['source_unchanged'] = code_hashes() == hashes
            manifest['status'] = 'completed' if manifest['accepted'] and manifest['source_unchanged'] else 'error'
            save_manifest()

        def fail(error):
            manifest['status'] = 'failed'
            manifest['error'] = repr(error)
            save_manifest()

        result = run_probe(args, parameters, manifest['episode_id'], out, save, complete, fail)
        if manifest['status'] == 'starting' or 'accepted' not in manifest:
            complete(result)
        if not manifest['source_unchanged']:
            raise RuntimeError('Code changed during the probe; a stable-version probe is required')
        if not manifest['accepted']:
            raise RuntimeError('Initial acceptance failed: ' + str(manifest.get('acceptance_error')))
    except BaseException as error:
        manifest['status'] = 'failed' if manifest['status'] != 'error' else 'error'
        manifest['error'] = repr(error)
        save_manifest()
        raise
    print(json.dumps(dict(output=str(out), api=args.api, role=args.role, status=manifest['status'], accepted=manifest['accepted'])))


def run_probe(args, parameters, episode_id, out, save, complete=lambda result: None, fail=lambda error: None):
    """Everything that needs Isaac. Imported here so --help stays CPU-only."""
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'multi_gpu': False,
                         'active_gpu': args.gpu, 'physics_gpu': 0, 'width': 640, 'height': 480,
                         'sync_loads': True, 'limit_cpu_threads': 8,
                         'extra_args': ['--/log/file=' + str(out / 'kit.log'), '--/app/tokens/cache=' + str(out / 'cache'),
                                        '--/app/tokens/data=' + str(out / 'data'), '--/app/tokens/documents=' + str(out / 'documents'),
                                        '--/app/userConfigPath=' + str(out / 'user.config.json')]})
    import numpy as np
    import carb
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdPhysics, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
    from PIL import Image
    from arena_api import l1_scene, l1_fridge, physics_defaults, policy_execution as pe
    from arena_api import robot_collision_filters, finger_pad_material, finger_pad_contact
    from arena_api import microwave_physics, scene_obstacles, coffee_box_physics
    from arena_api import robot_profiles as rp
    from arena_api.soft_body_runtime import EpisodePhysicsRuntime
    from arena_randomization import shelf_supports
    from arena_api.robot_initialization import apply as reset_robot
    from arena_api.task_runtime import WorkerContext
    from arena_api.isaac_backend import advance_fixed_steps, contact_offset_issue
    from arena_api.isaac_worker import CAMERA_PATHS
    from arena_api.camera_sampling import SampledCameraCapture
    from arena_api.schemas import Randomization
    from evaluation.tasks.coffee_machine.api_runtime import CoffeeMachineRuntime
    from evaluation.tasks.coffee_machine.registration import registration
    from evaluation.tasks.coffee_machine.specs import CUP, BAG

    physics_runtime = EpisodePhysicsRuntime(carb.settings.get_settings())
    # the flags must be the public worker's normalized ones, or the reset contract differs in `randomization`
    task = CoffeeMachineRuntime(WorkerContext(episode_id, args.api, out, args.seed, registration(args.api).max_steps,
                                              Randomization().model_dump(), dict(parameters)))
    world = controller = None
    rows = []
    acceptance_error = None
    try:
        scene = task.randomize(task.initialize(l1_scene.BASE))
        carb.settings.get_settings().set('/rtx/ecoMode/enabled', False)
        if not omni.usd.get_context().open_stage(str(scene)):
            raise RuntimeError('Could not open the initial-state scene')
        for _ in range(30):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        policies = {'l1_scene': l1_scene.apply(stage)}
        controller = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name, module in [('collisions', robot_collision_filters), ('material', finger_pad_material),
                             ('contact', finger_pad_contact), ('microwave', microwave_physics),
                             ('shelf', shelf_supports), ('obstacles', scene_obstacles), ('coffee', coffee_box_physics)]:
            policies[name] = module.apply(stage)
        task.attach(stage)
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
        physics_defaults.apply_robot(stage)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1 / 30, stage_units_in_meters=1.,
                      physics_prim_path='/World/PhysicsScene', backend='numpy', device='cpu')
        robot = world.scene.add(SingleArticulation('/World/Robot', name='arena_robot'))
        task.before_physics_reset(world)
        physics_runtime.prepare(stage, world, task=task)
        world.reset()
        save('robot_initialization.json', reset_robot(robot, stage, task.robot_initialization()))
        policies['runtime'] = physics_defaults.verify_runtime(world, robot)
        policies['episode_physics'] = physics_runtime.verify(world)
        save('policies.json', policies)
        task.on_physics_ready(world)
        init = task.robot_initialization()
        names = robot.dof_names
        indices = np.array([names.index(n) for n in init.hold_targets])
        values = np.array(list(init.hold_targets.values()))

        def action():
            robot.apply_action(ArticulationAction(joint_positions=values, joint_indices=indices))

        def after():
            controller.after_physics_step()
            task.after_physics(controller.report())
            rows.append(copy.deepcopy(task.reader.latest))

        products, rgbs, stamps = {}, {}, {}
        for name, path in CAMERA_PATHS.items():
            products[name] = rep.create.render_product(path, (640, 480))
            rgbs[name] = rep.AnnotatorRegistry.get_annotator('rgb')
            rgbs[name].attach([products[name]])
            stamps[name] = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            stamps[name].attach([products[name]])
        clock = _isaacsim_core_nodes.acquire_interface()

        def read_stamps():
            result = {}
            for name, stamp in stamps.items():
                d = stamp.get_data()
                n, den = int(d['referenceTimeNumerator']), int(d['referenceTimeDenominator'])
                result[name] = dict(numerator=n, denominator=den, simulation_time_s=float(clock.get_sim_time_at_time((n, den))))
            return result

        def render():
            settings = carb.settings.get_settings()
            previous = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations', False)
            try:
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0., wait_for_render=True)
            finally:
                settings.set('/app/player/playSimulations', previous)

        capture = SampledCameraCapture(world, lambda: {n: np.asarray(a.get_data()) for n, a in rgbs.items()},
                                       lambda f: len(f) == 3 and all(a.shape == (480, 640, 4) and a.dtype == np.uint8 for a in f.values()),
                                       read_stamps, capture_frame=render)

        def snap(tag):
            for name, data in capture.capture().items():
                Image.fromarray(data[..., :3]).save(out / (tag + '_' + name + '.png'))
            save(tag + '_capture.json', capture.last_evidence)

        advance_fixed_steps(world, action, task.settle_steps, pe.PHYSICS_DT, after_step=after)
        snap('settled')
        metadata = dict(physics_issues=issues, finger_pad_material_policy=policies['material'],
                        self_collisions_enabled=bool(PhysxSchema.PhysxArticulationAPI(root).GetEnabledSelfCollisionsAttr().Get()))
        try:
            task.validate_initial(metadata)
        except (ValueError, RuntimeError) as error:
            acceptance_error = str(error)
        profile = rp.get(rp.JOINT20)
        asset_limits = {}
        for name in names:
            prim = stage.GetPrimAtPath('/World/Robot/joints/' + name)
            if not prim or name not in profile.units:
                continue
            low, high = prim.GetAttribute('physics:lowerLimit').Get(), prim.GetAttribute('physics:upperLimit').Get()
            if low is None or high is None:
                continue
            asset_limits[name] = ([float(np.radians(low)), float(np.radians(high))] if profile.units[name] == 'rad' else [float(low), float(high)])
        measured = dict(zip(names, robot.get_joint_positions().tolist()))
        initial_targets, action_limits = pe.build_limits({n: measured[n] for n in profile.joints}, asset_limits, rp.JOINT20)
        metadata.update(pe.action_metadata(initial_targets, action_limits, rp.JOINT20))
        metadata['registered_evaluator_id'] = registration(args.api).evaluator_id
        metadata['finger_pad_material_policy'] = dict(policy_id=policies['material']['policy_id'])
        save('initial-metadata.json', metadata)
        save('measured_checkpoint.json', dict(api_id=args.api, role=args.role, task_parameters=parameters,
             joint_positions_native={n: float(measured[n]) for n in profile.joints}, all_measured_positions=measured,
             hold_targets=dict(init.hold_targets), measured=copy.deepcopy(task.reader.latest), plan=task.plan, source_scene=str(scene)))
        save('physical_facts.json', rows)
        acceptance = metadata.get('task_initial_acceptance', {})
        audit = dict(schema=AUDIT_SCHEMA, api_id=args.api, role=args.role, source_episode=task.plan['source_episode'],
                     passed=acceptance_error is None and bool(acceptance.get('accepted')), acceptance_error=acceptance_error,
                     acceptance=acceptance, evaluator_id=metadata['registered_evaluator_id'],
                     reset_contract_present='replay_reset_contract' in metadata, control_steps=0, settle_steps=task.settle_steps,
                     samples=len(rows), simulation_s=len(rows) * pe.PHYSICS_DT, physics_issues=issues,
                     episode_physics=policies['episode_physics'], layout_randomization=task.plan['randomization'],
                     physical_task_success_proven=False, scope='real initial-state acceptance only; visual review separate')
        save('audit.json', audit)
        result = dict(accepted=audit['passed'], acceptance_error=acceptance_error, samples=len(rows), audit=audit)
        complete(result)
        return result
    except BaseException as error:
        save('error.json', dict(error=repr(error), physics_parameters_modified=False))
        if rows:
            save('partial_facts.json', rows)
        fail(error)
        raise
    finally:
        try:
            task.close()
        except Exception as error:
            save('close_error.json', dict(error=repr(error)))
        finally:
            try:
                if controller:
                    controller.close()
                if world:
                    world.stop()
            finally:
                app.close()


if __name__ == '__main__':
    main()

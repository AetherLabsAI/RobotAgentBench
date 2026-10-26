"""Bounded Isaac main-thread worker; no policy, teleport, or task-success claim."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .image_recording import ImageRecorder
from .camera_sampling import SampledCameraCapture

from .isaac_backend import CAMERAS, advance_fixed_steps, contact_offset_issue, split_joint_state, unscored_step_budget
from .worker_wire import receive, send
from . import bimanual, robot_profiles as rp
from . import policy_execution as pe
from . import physics_defaults
from .physics_rate import PhysicsRate, SUPPORTED_HZ, validate_hz
from .soft_body_runtime import EpisodePhysicsRuntime
from . import robot_collision_filters
from . import microwave_physics
from . import scene_obstacles
from . import coffee_box_physics
from . import finger_pad_contact
from . import finger_pad_material
from . import l1_scene, l1_fridge
from arena_randomization import shelf_supports
from .obstacle_contacts import ContactLog
from .phase_timing import PhaseTiming
from .gripper_contact_feedback import create_for_mode as create_gripper_feedback

CAMERA_PATHS = {"head": "/World/Robot/head_link2/Head_Camera",
                "left": "/World/Robot/gripper_l_base_link/Left_Camera",
                "right": "/World/Robot/gripper_r_base_link/Right_Camera"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-id", choices=(pe.TASK_ID,), default=pe.TASK_ID)
    parser.add_argument('--runtime-task', help='Trusted registered task ID; never a module path')
    parser.add_argument('--episode-id')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--randomization-json', default='{}')
    parser.add_argument('--task-parameters-json', default='{}')
    parser.add_argument("--diagnostic-id", help="Opt-in trusted local read-only diagnostic ID")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--efficient-replay", action="store_true",
                        help="Opt-in notice-invalidated configuration checks; physics/observations unchanged")
    parser.add_argument('--async-observations', action='store_true',
                        help='Current RGB over memory IPC; bounded background PNG recording')
    parser.add_argument("--robot-profile", choices=(rp.JOINT20, rp.JOINT16), default=rp.DEFAULT)
    args = parser.parse_args()
    from .worker_diagnostics import validate_diagnostic, DiagnosticRecorder
    validate_diagnostic(args.diagnostic_id)
    profile = rp.get(args.robot_profile)
    evidence_keys = pe.evidence_keys(profile.id)
    sock = socket.socket(fileno=args.fd)
    # No requests means no physics stepping; abandoned clients release the GPU.
    sock.settimeout(300)
    app = None
    lock = None
    obstacle_contact_log = None
    gripper_feedback = None
    fridge_controller = None
    guard = None
    image_pool = None
    image_recorder = None
    latest_rgb = None
    camera_capture_log = None
    timing = PhaseTiming()
    task_runtime = None
    runtime_closed = False
    soft_body_runtime = None
    cloth_transitions = None
    diagnostic = None
    step_budget = pe.MAX_STEPS
    gripper_execution_mode = pe.DEFAULT_GRIPPER_MODE
    try:
        if not args.runtime_task:
            step_budget = unscored_step_budget(args.max_steps)
        if args.runtime_task:
            from .task_registry import default_registry
            from .task_runtime import WorkerContext
            from .task_lifecycle import TaskLifecycle
            from uuid import UUID
            task = default_registry().get(args.runtime_task)
            flags = json.loads(args.randomization_json)
            task.validate(args.max_steps, flags)
            parameters = task.resolve_parameters(json.loads(args.task_parameters_json))
            if task.runtime_factory is None or task.controller_id != args.task_id:
                raise ValueError('Registered task/controller mismatch')
            context = WorkerContext(str(UUID(args.episode_id)), task.task_id, args.output,
                                    args.seed, args.max_steps, flags, parameters)
            task_runtime = TaskLifecycle(task.runtime_factory(context), context, task, bimanual.CONTROL_DT)
            step_budget = args.max_steps
            gripper_execution_mode = task.gripper_execution_mode
        scene = args.scene.resolve(strict=True)
        source_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
        gpu_uuid = subprocess.check_output(["nvidia-smi", "--id", str(args.gpu),
            "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
            raise RuntimeError("Invalid GPU selection")
        lock = open(Path(tempfile.gettempdir()) / f"arena-{os.getuid()}-{gpu_uuid}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["OMP_NUM_THREADS"] = "4"
        print("ARENA_WORKER GPU", gpu_uuid, flush=True)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "multi_gpu": False,
            "active_gpu": args.gpu, "physics_gpu": 0, "width": 640, "height": 480,
            "sync_loads": True, "limit_cpu_threads": 8})
        import numpy as np
        import carb
        soft_body_runtime = EpisodePhysicsRuntime(carb.settings.get_settings())
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        from PIL import Image

        np.random.seed(args.seed)
        carb.settings.get_settings().set("/rtx/ecoMode/enabled", False)
        if task_runtime:
            scene = task_runtime.prepare_scene(scene).resolve(strict=True)
            source_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
        if not omni.usd.get_context().open_stage(str(scene)):
            raise RuntimeError("Cannot open scene")
        for _ in range(30):
            app.update()
        stage = omni.usd.get_context().get_stage()
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
            raise RuntimeError("Worker requires a meter-scale scene; refusing implicit unit conversion")
        stage.SetEditTarget(stage.GetSessionLayer())
        l1_policy = l1_scene.apply(stage)
        fridge_controller = l1_fridge.Controller(stage, physics_dt=pe.PHYSICS_DT)
        root = stage.GetPrimAtPath("/World/Robot")
        if not root or root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
            raise RuntimeError("Expected robot with self collisions explicitly enabled")
        applied_schema = not root.HasAPI(PhysxSchema.PhysxArticulationAPI)
        # Activate the stored
        # self-collision=True setting if its schema was missing. No false toggle.
        if applied_schema:
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        # Exact, versioned six-pair restoration; shoulder/assembly filters stay.
        # Session-only and before World.reset(), so fresh physics gets this policy.
        robot_collision_policy = robot_collision_filters.apply(stage)
        finger_material_policy = finger_pad_material.apply(stage)
        finger_pad_policy = finger_pad_contact.apply(stage)
        # Repair the known House mechanism conflict before PhysX parses the stage.
        microwave_policy = microwave_physics.apply(stage)
        # Selected variants already exist in the input scene. Invisible
        # replacements must not leave a collider behind.
        shelf_support_policy = shelf_supports.apply(stage)
        obstacle_policy = scene_obstacles.apply(stage)
        coffee_box_policy = coffee_box_physics.apply(stage)
        if task_runtime:
            task_runtime.attach(stage)

        def collision_flags():
            return {str(p.GetPath()): p.GetAttribute("physics:collisionEnabled").Get()
                    for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                    and str(p.GetPath()) not in fridge_controller.mutable_colliders}

        collision_before = collision_flags()
        initial_layers = {layer.identifier: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                          for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        issues = []
        for p in stage.Traverse():
            contact = p.GetAttribute("physxCollision:contactOffset").Get()
            rest = p.GetAttribute("physxCollision:restOffset").Get()
            offset_issue = contact_offset_issue(contact, rest)
            if offset_issue:
                issues.append({**offset_issue, "prim": str(p.GetPath())})
            if p.GetAttribute("physics:kinematicEnabled").Get() is True:
                parent = p
                while parent and not parent.IsPseudoRoot():
                    if parent.HasAPI(UsdPhysics.ArticulationRootAPI):
                        issues.append({"kind": "kinematic_body_under_articulation", "prim": str(p.GetPath())})
                        break
                    parent = parent.GetParent()
        for path in CAMERA_PATHS.values():
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                raise RuntimeError("Camera missing: " + path)
        physics_defaults.apply_robot(stage)
        world = World(physics_dt=pe.PHYSICS_DT, rendering_dt=1 / 30, stage_units_in_meters=1.0,
                      physics_prim_path="/World/PhysicsScene", backend="numpy", device="cpu")
        physics_rate = PhysicsRate()
        frame_index = 0
        contact_phase = "initialization"
        obstacle_contact_log = ContactLog(stage, args.output / "obstacle_contacts.jsonl",
            lambda: dict(simulation_time_s=float(world.current_time), control_step=frame_index,
                         phase=contact_phase))
        gripper_feedback = create_gripper_feedback(stage, pe.PHYSICS_DT, gripper_execution_mode)
        if gripper_execution_mode != 'direct_target.v1':
            raise ValueError('Worker only accepts unmodified native targets')
        robot = world.scene.add(SingleArticulation(prim_path="/World/Robot", name="arena_robot"))
        if task_runtime:
            task_runtime.before_physics_reset(world)
        # Select only after task-owned pre-reset geometry has been authored.
        # A cloth transition reserves GPU capability for the entire episode.
        soft_body_runtime.prepare(stage, world, task=task_runtime.task if task_runtime else None)
        world.reset()
        robot_initialization = task_runtime.robot_initialization() if task_runtime else None
        if robot_initialization is not None:
            from .robot_initialization import apply as apply_robot_initialization
            reset_receipt = apply_robot_initialization(robot, stage, robot_initialization)
            (args.output / 'robot_initialization.json').write_text(
                json.dumps(reset_receipt, indent=2, allow_nan=False) + '\n')
        if task_runtime:
            task_runtime.initialize(world)
            from .scene_transitions import for_task as scene_transitions_for_task
            cloth_transitions = scene_transitions_for_task(task_runtime.task,
                episode_id=task_runtime.context.episode_id, stage=stage)
        if args.diagnostic_id is not None:
            diagnostic = DiagnosticRecorder(args.diagnostic_id, stage, world, args.output)
        initialization_time = float(world.current_time)
        physics_receipt = physics_defaults.verify_runtime(world, robot)
        soft_body_receipt = soft_body_runtime.verify(world)
        names = robot.dof_names
        q0 = np.asarray(robot.get_joint_positions(), dtype=float)
        if not np.isfinite(q0).all():
            raise RuntimeError("Non-finite initial state")
        driven, units, asset_limits = [], {}, {}
        for i, name in enumerate(names):
            prim = stage.GetPrimAtPath("/World/Robot/joints/" + name)
            if prim and prim.IsA(UsdPhysics.RevoluteJoint):
                units[name] = "rad"
            elif prim and prim.IsA(UsdPhysics.PrismaticJoint):
                units[name] = "m"
            else:
                raise RuntimeError("Unmapped or unsupported robot joint: " + name)
            if any(s.startswith("PhysicsDriveAPI:") for s in prim.GetAppliedSchemas()):
                driven.append(i)
            low = prim.GetAttribute("physics:lowerLimit").Get()
            high = prim.GetAttribute("physics:upperLimit").Get()
            if low is not None and high is not None:
                asset_limits[name] = ([float(np.radians(low)), float(np.radians(high))]
                                      if units[name] == "rad" else [float(low), float(high)])
            native_value = float(np.degrees(q0[i])) if units[name] == "rad" else float(q0[i])
            native_unit = "deg" if units[name] == "rad" else "m"
            if low is not None and high is not None and not low - 1e-3 <= native_value <= high + 1e-3:
                issues.append({"kind": "initial_joint_outside_limit", "joint": name,
                               "initial_value": native_value, "limits": [low, high], "unit": native_unit})
        targets = q0[driven].copy()
        if robot_initialization is not None:
            for name, value in robot_initialization.hold_targets.items():
                targets[driven.index(names.index(name))] = value
        driven_indices = np.asarray(driven)
        products, annotators, camera_time_annotators = [], {}, {}
        for name, path in CAMERA_PATHS.items():
            product = rep.create.render_product(path, (640, 480))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            products.append(product)
            annotators[name] = annotator
            stamp = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            stamp.attach([product])
            camera_time_annotators[name] = stamp
        camera_clock = _isaacsim_core_nodes.acquire_interface()

        def camera_stamps():
            stamps = {}
            for name in CAMERAS:
                data = camera_time_annotators[name].get_data()
                numerator = int(data['referenceTimeNumerator'])
                denominator = int(data['referenceTimeDenominator'])
                stamps[name] = dict(numerator=numerator, denominator=denominator,
                    simulation_time_s=float(camera_clock.get_sim_time_at_time((numerator, denominator))))
            return stamps

        def capture_frame():
            # Like world.render(), suppress physics during app updates, but use
            # a capture-completion fence: ReferenceTime alone can advance while
            # an RGB annotator still exposes an older image buffer.
            settings = carb.settings.get_settings()
            previous = settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations', False)
            try:
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False,
                                      delta_time=0.0, wait_for_render=True)
            finally:
                settings.set('/app/player/playSimulations', previous)

        camera_capture = SampledCameraCapture(world,
            lambda: {name: np.asarray(annotators[name].get_data()) for name in CAMERAS},
            lambda frames: set(frames) == set(CAMERAS) and all(
                rgb.shape == (480, 640, 4) and rgb.dtype == np.uint8 for rgb in frames.values()),
            camera_stamps, capture_frame=capture_frame)
        camera_capture_log = (args.output / 'camera_capture.jsonl').open('x')

        def full_configuration_check():
            physics_defaults.verify_runtime(world, robot, expected_dt=physics_rate.dt)
            soft_body_runtime.verify(world, expected_dt=physics_rate.dt)
            if collision_flags() != collision_before:
                raise RuntimeError("Collision flags changed")
            if root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
                raise RuntimeError("Self collisions changed")
            robot_collision_filters.verify(stage)
            finger_pad_contact.verify(stage, finger_pad_policy["configuration"])
            finger_pad_material.verify(stage, finger_material_policy["configuration"])
            microwave_physics.verify(stage, microwave_policy["configuration"])
            scene_obstacles.verify(stage, obstacle_policy["configuration"])
            coffee_box_physics.verify(stage, coffee_box_policy["configuration"])

        if args.efficient_replay:
            from .physics_guard import PhysicsGuard
            guard = PhysicsGuard(stage, full_configuration_check)
            guard.check(force=True)
            if not args.async_observations:
                image_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="arena_png")
        if args.async_observations:
            image_recorder = ImageRecorder(args.output)

        def apply_targets():
            if image_recorder:
                image_recorder.check()
            if guard:
                guard.check()  # before EVERY physics step, including settle
            if gripper_feedback is not None:
                gripper_feedback.before_step()
            robot.apply_action(ArticulationAction(joint_positions=targets, joint_indices=driven_indices))

        def after_physics():
            if gripper_feedback is not None:
                gripper_feedback.after_step()
            if diagnostic is not None:
                diagnostic.sample(contact_phase)
            fridge_controller.after_physics_step()
            if task_runtime:
                task_runtime.after_physics(fridge_controller.report())
                if cloth_transitions is not None and task_runtime.phase == 'running':
                    cloth_transitions.after_physics(
                        episode_id=task_runtime.context.episode_id,
                        physics_step=int(world.current_time_step_index),
                        time_s=float(world.current_time))
            if guard:
                guard.check()
        # Task-local readers must still reject automatic helper/attachment events.
        after_physics.controller = fridge_controller

        def advance(count):
            # Never change physical policy or skip current-step contact reads.
            with timing.measure("configuration_pre"):
                if guard:
                    guard.check()
                else:
                    finger_pad_material.verify(stage, finger_material_policy["configuration"])
            with timing.measure("physics_and_reader"):
                advance_fixed_steps(world, apply_targets, count, physics_rate.dt, after_step=after_physics)
            with timing.measure("configuration_post"):
                if guard:
                    guard.check()
                else:
                    full_configuration_check()

        # Hold all native initial targets; existing questionable gripper state is
        # reported, not silently clamped/re-authored as part of API integration.
        contact_phase = "settle"
        settle_steps = task_runtime.settle_steps if task_runtime else physics_defaults.steps_for_seconds(0.5)
        advance(settle_steps)
        frame_index = 0
        contact_phase = "execution"
        if not set(profile.joints) <= {names[i] for i in driven}:
            raise RuntimeError("Policy action contains an undriven joint")
        measured = dict(zip(names, robot.get_joint_positions().tolist()))
        last_target, action_limits = pe.build_limits({n: measured[n] for n in profile.joints}, asset_limits, profile.id)
        native_adapter = rp.NativeAdapter(profile.id, names, units,
            [names[i] for i in driven], dict(zip([names[i] for i in driven], targets.tolist())), asset_limits,
            mimic_followers=rp.inspect_mimic_followers(stage))
        # Start from settled measured joints, without teleporting or editing drives.
        for name, target in last_target.items():
            targets[driven.index(names.index(name))] = target

        def observe():
            nonlocal latest_rgb
            before = float(world.current_time)
            with timing.measure("render_and_readback"):
                frames = camera_capture.capture()
            camera_capture_log.write(json.dumps(dict(frame_index=frame_index,
                **camera_capture.last_evidence), allow_nan=False) + '\n')
            camera_capture_log.flush()
            if guard:
                guard.check()  # rendering must not mutate physics configuration
            q = np.asarray(robot.get_joint_positions(), dtype=float)
            dq = np.asarray(robot.get_joint_velocities(), dtype=float)
            if not np.isfinite(q).all() or not np.isfinite(dq).all():
                raise RuntimeError("Non-finite simulated joint state")
            for name in CAMERAS:
                if frames[name].shape[:2] != (480, 640):
                    raise RuntimeError(f"Invalid camera frame {name}: {frames[name].shape}")
            def save_image(name):
                options = {"compress_level": 1} if image_pool else {}
                Image.fromarray(frames[name][..., :3]).save(
                    args.output / f"step{frame_index:04d}_{name}.png", **options)
            with timing.measure("image_snapshot_submit" if image_recorder else "image_encode_write_three_cameras"):
                if image_recorder:
                    # Detach renderer-owned memory before the next render.
                    if any(frames[k].dtype != np.uint8 for k in CAMERAS):
                        raise RuntimeError('Unexpected camera RGB type')
                    latest_rgb = {k: np.ascontiguousarray(frames[k][..., :3]).tobytes() for k in CAMERAS}
                    image_recorder.submit(frame_index, before, latest_rgb)
                elif image_pool:
                    # All three writes complete before returning observation
                    # or advancing physics. Lossless PNG; never drop frames.
                    list(image_pool.map(save_image, CAMERAS))
                else:
                    for name in CAMERAS:
                        save_image(name)
            fields = {"time_s": before, "time_basis": "simulation_time", "source": "isaac_physics",
                      **split_joint_state(names, units, q.tolist(), dq.tolist()),
                      "frame_index": frame_index,
                      "instruction": args.instruction or "Physical policy execution; no task scorer."}
            (args.output / f"step{frame_index:04d}.json").write_text(json.dumps(fields, indent=2) + "\n")
            if frame_index % 30 == 0:
                timing.save(args.output / "worker_timing.json", control_step=frame_index,
                            guard=None if guard is None else guard.report())
            return fields

        metadata = {"robot_profile": profile.id,
                    "native_hold_targets": dict(native_adapter.held),
                    "gripper_follower_modes": dict(native_adapter.follower_modes),
                    "gripper_execution_mode": gripper_execution_mode,
                    "native_follower_limits": {n: list(native_adapter.limits[n]) for n in rp.INNER},
                    "scene": str(scene), "scene_sha256": source_hash, "gpu_uuid": gpu_uuid,
                    "physics_dt_s": pe.PHYSICS_DT, "control_dt_s": bimanual.CONTROL_DT, "settle_steps": settle_steps,
                    "physics_defaults": physics_receipt,
                    "soft_body_runtime": soft_body_receipt,
                    "initialization_time_s": initialization_time, "physics_only_stepping": True,
                    "seed": args.seed, "randomization_applied": False,
                    "physics_issues": issues, "physics_configuration_validated": False,
                    "self_collision_schema_applied": applied_schema,
                    "l1_scene_policy": l1_policy,
                    "l1_fridge_policy": fridge_controller.report(),
                    "robot_collision_policy": robot_collision_policy,
                    "finger_pad_contact_policy": finger_pad_policy,
                    "finger_pad_material_policy": finger_material_policy,
                    "microwave_physics_policy": microwave_policy,
                    "scene_obstacle_policy": obstacle_policy,
                    "coffee_box_physics_policy": coffee_box_policy,
                    "shelf_support_policy": shelf_support_policy,
                    "obstacle_contact_evidence": {"file": "obstacle_contacts.jsonl",
                        "scope": "robot_vs_selected_static_obstacles", "threshold_N": 0.0,
                        "scoring_implemented": False},
                    "self_collisions_enabled": True, "collision_prim_count": len(collision_before),
                    "dof_names": names, "joint_units": units, "driven_joint_names": [names[i] for i in driven],
                    "model_inference": False, "task_success": None}
        if diagnostic is not None:
            metadata['diagnostic_recording'] = dict(id=args.diagnostic_id,
                file='diagnostic_samples.jsonl', scoring_implemented=False)
        metadata.update(execution_profile="efficient_replay" if guard else "exhaustive",
                        capture_requests_per_observation=1,
                        camera_sampling=dict(policy='sample_wait_v1', initialization_min_requests=4,
                            initialization_max_requests=60, freshness='capture_completion_and_reference_time',
                            capture_api='rep.orchestrator.step', wait_for_render=True, rt_subframes=1,
                            evidence_file='camera_capture.jsonl', stale_frame_policy='fail_episode'),
                        image_encoding={"format":"PNG", "lossless":True,
                                        "compression_level":1 if image_pool or image_recorder else 6,
                                        "parallel_cameras":bool(image_pool or image_recorder),
                                        "background_recording":bool(image_recorder)},
                        image_transport='owned_rgb_memory_ipc' if image_recorder else 'synchronous_png_files')
        metadata["gripper_force_feedback"] = dict(enabled=gripper_feedback is not None,
            mode=gripper_execution_mode, contact_reports_preserved=True)
        metadata.update(pe.action_metadata(last_target, action_limits, profile.id), task_id=args.task_id,
                        target_initialization="measured_after_native_settle; PD control only")
        metadata["max_episode_steps"] = step_budget
        metadata['physics_rate_switch'] = dict(
            supported=task_runtime is None and not soft_body_runtime.gpu_dynamics and gripper_feedback is None,
            reason='Requires unscored rigid whole-scene execution; scored task clocks and GPU soft bodies remain fixed',
            supported_hz=list(SUPPORTED_HZ), scope='entire_physics_scene',
            boundary='before_next_control_action', control_hz=bimanual.CONTROL_HZ,
            default_hz=240, resets_state=False)
        if not task_runtime and args.max_steps is not None:
            metadata["action_spec"]["max_episode_steps"] = step_budget
        if task_runtime:
            task_runtime.accept_initial(metadata, float(world.current_time))
            metadata.update(runtime_task_id=args.runtime_task, max_episode_steps=step_budget,
                            task_parameters=context.task_parameters,
                            randomization_applied=json.loads(args.randomization_json))
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        initial = observe()
        def send_observation(payload):
            if task_runtime:
                payload = dict(payload, task_evaluation=task_runtime.evaluation())
            if latest_rgb is None:
                return send(sock, payload)
            return send(sock, payload, images=latest_rgb)
        send_observation({"ok": True, "observation": initial, "metadata": metadata})
        print("ARENA_WORKER READY", flush=True)
        control_started = perf_counter()
        while True:
            try:
                command = receive(sock)
            except (socket.timeout, EOFError):
                print("ARENA_WORKER idle timeout or owner disconnected", flush=True)
                break
            if command == {"op": "close"}:
                break
            if command == {"op": "keepalive"}:
                # Stay in the socket wait loop: no world.step, render, observe,
                # target update, contact query, scoring or simulation time.
                send(sock, {'ok': True, 'op': 'keepalive',
                            'frame_index': frame_index, 'time_s': float(world.current_time)})
                continue
            if set(command) not in ({"op", "targets"}, {"op", "targets", "physics_hz"}) or command["op"] != "step":
                raise RuntimeError("Unsupported worker command")
            if task_runtime and (task_runtime.evaluation()['terminated'] or task_runtime.evaluation()['truncated']):
                send(sock, {'ok': False, 'error': 'episode_finished'})
                continue  # close remains available; no further physics
            if frame_index >= step_budget:
                raise RuntimeError("Execution step budget reached")
            command_path = args.output / f"command{frame_index + 1:04d}.json"
            def measured_joints():
                q = dict(zip(names, robot.get_joint_positions().tolist()))
                return {n: q[n] for n in profile.joints}

            native_adapter.expand(command["targets"])  # validate before the first physical step
            pe.validate(command['targets'], action_limits, profile.id)
            switch = None
            if 'physics_hz' in command:
                validate_hz(command['physics_hz'])
                if not metadata['physics_rate_switch']['supported']:
                    raise ValueError('This episode does not support physics-rate switching')
                full_configuration_check()
                command_path.write_text(json.dumps(dict(
                    phase='physics_rate_switch_pending', physics_hz=command['physics_hz'],
                    time_s=float(world.current_time), previous_rate=physics_rate.receipt(),
                    targets=command['targets']), allow_nan=False) + '\n')
                before_q = robot.get_joint_positions().copy()
                before_dq = robot.get_joint_velocities().copy()
                from pxr import Usd
                def set_native_dt(dt):
                    # Author only the session-layer clock, not root-layer render FPS.
                    with Usd.EditContext(stage, stage.GetSessionLayer()):
                        world.get_physics_context().set_physics_dt(
                            dt, max(1, round(world.get_rendering_dt() / dt)))
                switch = physics_rate.switch(world, command['physics_hz'],
                    setter=set_native_dt, helper=fridge_controller)
                if (not np.array_equal(before_q, robot.get_joint_positions())
                        or not np.array_equal(before_dq, robot.get_joint_velocities())):
                    raise RuntimeError('Rate switch changed robot state before action execution')
                switch['robot_q_dq_unchanged'] = True
                full_configuration_check()
                if guard:
                    guard.check(force=True)
            execution = pe.plan_interval(command["targets"], measured_joints(), last_target, action_limits,
                                         profile.id, physics_hz=physics_rate.hz)
            execution['physics_rate'] = physics_rate.receipt()
            execution['physics_rate_switch'] = switch
            execution = pe.wire_evidence(execution, profile.id)
            execution["native_substep_targets"] = []
            execution["gripper_surface_stop"] = []
            execution["gripper_execution_mode"] = gripper_execution_mode
            execution["gripper_follower_modes"] = dict(native_adapter.follower_modes)
            command_path.write_text(json.dumps({"execution": execution}, indent=2, allow_nan=False) + "\n")
            for index, interpolated in enumerate(execution[evidence_keys["interpolated_substep_targets_rad"]]):
                measured = measured_joints()
                subtarget, changes = pe.direct_gripper_substep(
                    command["targets"], interpolated, measured, action_limits, profile.id)
                contact_evidence = {'mode': gripper_execution_mode,
                                    'contact_stop_applied': False, 'measured_lead_applied': False}
                execution["gripper_surface_stop"].append(contact_evidence)
                execution[evidence_keys["measured_before_substep_rad"]].append(measured)
                execution[evidence_keys["substep_targets_rad"]].append(subtarget)
                execution["interventions"].extend(changes)
                command_path.write_text(json.dumps({"execution": execution}, indent=2, allow_nan=False) + "\n")
                expanded = native_adapter.expand(subtarget)
                targets[:] = [expanded[name] for name in native_adapter.driven]
                execution["native_substep_targets"].append(expanded)
                advance(1)
                last_target = dict(subtarget)
                execution["applied_substeps"] += 1
                measured_after = measured_joints()
                execution[evidence_keys["measured_after_substep_rad"]].append(measured_after)
                execution[evidence_keys["tracking_error_after_substep_rad"]].append(pe.tracking_errors(measured_after, last_target, profile.id))
                command_path.write_text(json.dumps({"execution": execution}, indent=2, allow_nan=False) + "\n")
            frame_index += 1
            if task_runtime:
                task_runtime.after_control(float(world.current_time))
            fields = observe()
            # Commands are saved as evidence but not returned as observations.
            command_path.write_text(json.dumps({("targets_rad" if profile.id == rp.JOINT16 else "targets_native"): command["targets"],
                "execution": execution, "observation": fields}, allow_nan=False) + "\n")
            send_observation({"ok": True, "observation": fields, "execution": execution})
        control_elapsed = perf_counter()-control_started
        if image_recorder:
            with timing.measure('image_recording_drain'):
                image_recorder.close()
        if guard:
            guard.check(force=True)
        else:
            full_configuration_check()
        timing.save(args.output / "worker_timing.json", control_step=frame_index,
                    guard=None if guard is None else guard.report(), completed=True,
                    control_loop_wall_seconds=control_elapsed,
                    image_recording=None if image_recorder is None else image_recorder.report())
        finger_pad_contact.verify(stage, finger_pad_policy["configuration"])
        finger_pad_material.verify(stage, finger_material_policy["configuration"])
        microwave_physics.verify(stage, microwave_policy["configuration"])
        scene_obstacles.verify(stage, obstacle_policy["configuration"])
        coffee_box_physics.verify(stage, coffee_box_policy["configuration"])
        world.stop()
        obstacle_contact_log.close()
        for annotator in annotators.values():
            annotator.detach()
        for annotator in camera_time_annotators.values():
            annotator.detach()
        for product in products:
            product.destroy()
        unchanged = all(hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest() == initial_layers[layer.identifier]
                        for layer in stage.GetUsedLayers() if layer.identifier in initial_layers)
        if not unchanged or hashlib.sha256(scene.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Source asset changed on disk")
        if task_runtime:
            final_evaluation = task_runtime.close(True)
            runtime_closed = True
            (args.output / 'task_evaluation.json').write_text(
                json.dumps(final_evaluation, indent=2, allow_nan=False) + '\n')
        if diagnostic is not None:
            diagnostic.close(completed=True)
        (args.output / "closed.json").write_text(json.dumps({"steps": frame_index,
            "final_physics_rate": physics_rate.receipt(),
            "image_recording":None if image_recorder is None else image_recorder.report(),
            "source_layers_unchanged": True, "collision_flags_unchanged": True,
            "collision_flags_guard_scope": "all except logged original L1 fridge helper scope",
            "l1_scene_policy": l1_policy,
            "l1_fridge_policy": fridge_controller.report(),
            "robot_collision_policy_unchanged": True,
            "finger_pad_contact_policy_unchanged": True,
            "finger_pad_contact_policy_id": finger_pad_policy["policy_id"],
            "finger_pad_material_policy_unchanged": True,
            "finger_pad_material_policy_id": finger_material_policy["policy_id"],
            "microwave_physics_policy_unchanged": True,
            "microwave_physics_policy_id": microwave_policy["policy_id"],
            "scene_obstacle_policy_unchanged": True,
            "scene_obstacle_policy_id": obstacle_policy["policy_id"],
            "coffee_box_physics_policy_unchanged": True,
            "coffee_box_physics_policy_id": coffee_box_policy["policy_id"],
            "obstacle_contact_counts": obstacle_contact_log.counts,
            "robot_collision_policy_id": robot_collision_policy["policy_id"]}) + "\n")
        print("ARENA_WORKER CLOSED", frame_index, flush=True)
    except BaseException:
        traceback.print_exc()
        try:
            send(sock, {"ok": False, "error": "worker_failed"})
        except (OSError, ValueError):
            pass
        raise
    finally:
        if cloth_transitions is not None:
            try:
                (args.output / 'cloth_transitions.json').write_text(
                    json.dumps(cloth_transitions.report(), indent=2, allow_nan=False) + '\n')
            except Exception:
                traceback.print_exc()
            finally:
                cloth_transitions.close()
        if diagnostic is not None and not diagnostic.closed:
            try:
                diagnostic.close(completed=False)
            except Exception:
                traceback.print_exc()
        if task_runtime and not runtime_closed:
            try:
                failed = task_runtime.close(False)
                (args.output / 'task_evaluation.json').write_text(json.dumps(failed, allow_nan=False) + '\n')
            except Exception:
                traceback.print_exc()
        if camera_capture_log:
            camera_capture_log.close()
        sock.close()
        if obstacle_contact_log:
            obstacle_contact_log.close()
        if gripper_feedback:
            gripper_feedback.close()
        if fridge_controller:
            fridge_controller.close()
        if guard:
            guard.close()
        if image_pool:
            image_pool.shutdown(wait=True)
        try:
            if image_recorder and not image_recorder.closed:
                image_recorder.close()
        finally:
            try:
                if app:
                    try:
                        if soft_body_runtime:
                            soft_body_runtime.close()
                    finally:
                        app.close()
            finally:
                if lock:
                    lock.close()


if __name__ == "__main__":
    main()

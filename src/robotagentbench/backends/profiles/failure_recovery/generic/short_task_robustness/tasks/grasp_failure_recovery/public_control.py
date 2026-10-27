"""Reuse the full-source robot22 host and driver with original physics/scoring."""
import base64
import io
from pathlib import Path
from uuid import uuid4


def run(*, stage, world, robot, mobile, lifecycle, init, plan, fridge,
        products, annotators, app, out, report, completion, smoke):
    import carb
    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import UsdPhysics
    from isaacsim.core.utils.types import ArticulationAction
    from arena_api import policy_execution as pe
    from . import public_profiles as rp
    from .public_driver import NativeDriver
    from .public_host import SessionHost, serve
    from .public_session import RecoverySession
    from .common import write_json
    names = list(robot.dof_names)
    units, driven, limits, held = {}, [], {}, {}
    for name in names:
        prim = stage.GetPrimAtPath('/World/Robot/joints/'+name)
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        scale, units[name] = (np.pi/180, 'rad') if angular else (1., 'm')
        limits[name] = [float(prim.GetAttribute('physics:'+side+'Limit').Get())*scale
                        for side in ('lower', 'upper')]
        kind = 'angular' if angular else 'linear'
        if prim.HasAPI(UsdPhysics.DriveAPI, kind):
            driven.append(name)
            original = prim.GetAttribute('drive:'+kind+':physics:targetPosition').Get()
            held[name] = init.hold_targets[name] if name in init.hold_targets else float(original)*scale
    adapter = rp.NativeAdapter(rp.MOBILE22, names, units, driven, held, limits,
                              mimic_followers=rp.inspect_mimic_followers(stage))
    session = RecoverySession(lifecycle, world, plan['instruction'])
    mobile.bind()
    stamps = []
    for product in products:
        stamp = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
        stamp.attach([product])
        stamps.append(stamp)
    camera_resolutions = dict(head=(1280,800), left=(848,480), right=(848,480))
    capture_index = 0
    previous_ref = None
    bodies = mobile.simulation_view.create_rigid_body_view([
        str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
        and not str(p.GetPath()).startswith('/World/Robot/')
        and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False])

    def snapshot():
        return b''.join(np.asarray(x).copy().tobytes() for x in (
            robot.get_joint_positions(), robot.get_joint_velocities(),
            mobile.view.get_root_transforms(), mobile.view.get_root_velocities(),
            bodies.get_transforms(), bodies.get_velocities()))

    def capture():
        nonlocal capture_index, previous_ref
        from fractions import Fraction
        settings = carb.settings.get_settings()
        before, now = snapshot(), float(world.current_time)
        playing = settings.get('/app/player/playSimulations')
        settings.set('/app/player/playSimulations', False)
        try:
            # Original diagnostic uses four render-only subframes; no tick is
            # added to either the World or the source scorer.
            for _ in range(4):
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False,
                    delta_time=0., wait_for_render=True)
            refs = set()
            for stamp in stamps:
                value = stamp.get_data()
                refs.add(Fraction(int(value['referenceTimeNumerator']), int(value['referenceTimeDenominator'])))
            if len(refs) != 1 or (previous_ref is not None and next(iter(refs)) <= previous_ref):
                raise RuntimeError('Stale or unsynchronized public camera frames')
            images = {}
            for name, ann in annotators:
                pixels = np.asarray(ann.get_data())
                width, height = camera_resolutions[name]
                if pixels.shape != (height,width,4) or pixels.dtype != np.uint8:
                    raise RuntimeError('Standard native source camera frame missing')
                buffer = io.BytesIO()
                Image.fromarray(pixels[:,:,:3]).save(buffer, format='PNG')
                data = buffer.getvalue()
                (out/f'{capture_index:04d}_{name}.png').write_bytes(data)
                images[name] = base64.b64encode(data).decode()
            if float(world.current_time) != now or snapshot() != before:
                raise RuntimeError('Public observation advanced physics')
            previous_ref = next(iter(refs))
            capture_index += 1
            return images
        finally:
            settings.set('/app/player/playSimulations', playing)

    def apply(targets):
        robot.apply_action(ArticulationAction(joint_positions=np.asarray([targets[n] for n in driven]),
            joint_indices=np.asarray([names.index(n) for n in driven])))

    def advance(dt):
        if abs(dt-pe.PHYSICS_DT) > 1e-12:
            raise RuntimeError('Original physical timestep must not change')
        world.step(render=False)
        fridge.after_physics_step()
        return fridge.report()

    host = None
    completed = False
    try:
        driver = NativeDriver(session, adapter, mobile,
            measured_joints=lambda: dict(zip(names, map(float, robot.get_joint_positions()))),
            apply_native_targets=apply, simulate_fetch=advance, capture_png=capture)
        host = SessionHost(session, driver)
        report.update(status='ready', robot22_contract=host.contract,
                      public_observation_truth_filtered=True)
        write_json(out/'native_report.json', report)
        if smoke:
            hold = [init.positions[name] for name in rp.get(rp.MOBILE22).order]+[0.,0.]
            reply = host.execute(dict(op='execute_actions',request_id=str(uuid4()),
                episode_id=session.episode_id,expected_step=0,actions=[hold[:] for _ in range(20)]))
            if reply['actions_executed'] != 20 or reply['observation']['step'] != 20 or \
                    lifecycle.score.score != 0 or lifecycle.reason is not None:
                raise RuntimeError('Robot22 hold smoke unexpectedly scored, terminated or lost actions')
            report['smoke'] = dict(actions_executed=reply['actions_executed'],
                step=reply['observation']['step'], simulation_time_s=reply['observation']['time_s'],
                physics_hz=session.clock.hz, source_reset_targets_used_without_clipping=True,
                dimension=22, base_velocity_channels_executed=True,
                final_base=reply['observation']['base_state'])
        else:
            serve(host, output_stream=completion)
        host.close()
        completed = True
        report.update(task_result=session.result(), public_robot22_smoke_verified=smoke,
            camera_captures=capture_index, full_recovery_actions_verified=False,
            runtime_registered=True)
    except BaseException as error:
        session._fault(error)
        raise
    finally:
        if host is not None:
            host.close()
        elif not session.closed:
            session.close()
        for stamp in stamps:
            stamp.detach()
    return completed

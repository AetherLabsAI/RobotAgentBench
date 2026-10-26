"""Real CPU PhysX + synthetic activation inputs. No robot action/score/video.

Run in the existing isolated, no-GPU, read-only-source diagnostic container.
Only /pilot output is writable on the host. This is not initial-state acceptance.
"""
import sys
from pathlib import Path
ROOT = Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
sys.path[:0] = [str(ROOT), '/home/lingjun/robot_agent_arena/api-protocol']
import json, hashlib, traceback
import omni.kit.app
OUT = Path('/pilot/native_result.json')
report = dict(status='started', gpu_devices=[], pose_velocity_writes=0, robot_actions=0,
              activation_source='synthetic_existing_activation_output_fixture',
              native_physics_steps=0, control_hz=20, full_task_success=False,
              initial_acceptance=False, rows=[])
sim = None
owner = None
try:
    if OUT.exists(): raise RuntimeError('Unique native output required')
    import numpy as np
    import omni.physx, omni.physics.tensors
    from pxr import Usd, UsdPhysics, UsdUtils
    from coffee_startup_v1 import CoffeeSceneStartup
    from coffee_hotwake_v1 import NativeSceneClock
    manifest = json.loads((ROOT/'coffee_initialized_rooms_v1/layout03/composition.json').read_text())
    stage = Usd.Stage.Open(manifest['scene'])
    stage.SetEditTarget(stage.GetSessionLayer())
    source_hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath}
    scene = stage.GetPrimAtPath('/World/PhysicsScene')
    assert scene.GetAttribute('physxScene:timeStepsPerSecond').Get() == 60
    assert scene.GetAttribute('physxScene:enableGPUDynamics').Get() is False
    owner = CoffeeSceneStartup()
    report['startup'] = owner.prepare(stage, manifest, episode_id='native-hotwake-smoke', instance_id='coffee')
    sim = omni.physx.get_physx_simulation_interface()
    sid = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    assert sim.attach_stage(sid)
    current_time = 0.
    sim.simulate(1/60, current_time); sim.fetch_results(); current_time += 1/60
    report['native_physics_steps'] += 1
    view = omni.physics.tensors.create_simulation_view('numpy', sid)
    view.set_subspace_roots('/')
    robot = view.create_articulation_view('/World/Robot')
    assert robot.count == 1
    paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
             and not str(p.GetPath()).startswith('/World/Robot/')
             and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
    bodies = view.create_rigid_body_view(paths)
    assert sorted(map(str, bodies.prim_paths)) == sorted(paths)

    def snapshot():
        return b''.join(np.asarray(a).copy().tobytes() for a in (
            bodies.get_transforms(), bodies.get_velocities(), robot.get_link_transforms(),
            robot.get_dof_positions(), robot.get_dof_velocities(), robot.get_root_velocities()))

    clock = NativeSceneClock(stage, sim, lambda: current_time, snapshot)
    cycle = owner.enable_hotwake(clock)
    owner.on_physics_ready(clock, simulation_view=view)
    controller_id, animation_id = id(owner.service.controller), id(owner.service.animation)
    # No manipulation helper exists in this runner; no docking/assistance code is called.
    no_helpers = dict(events=[], docked_object_count=0, tray_dock_enabled=False)
    for label, active in [('far', False), ('coffee_sector_hit_fixture', True), ('outside_exit_fixture', False)]:
        for _ in range(20):
            plan = cycle.begin_control(dict(active=active, state='ACTIVE' if active else 'DORMANT'),
                                       command_vw=[0., 0.], base_velocity=list(map(float, robot.get_root_velocities()[0])))
            for _ in range(plan['substeps']):
                sim.simulate(plan['physics_dt_s'], current_time); sim.fetch_results()
                current_time += plan['physics_dt_s']; report['native_physics_steps'] += 1
                owner.after_physics(no_helpers)
            cycle.finish_control(completed_substeps=plan['substeps'])
            report['rows'].append(dict(label=label, **plan, time_s=current_time,
                                       camera_contacts=owner.source.last_camera_contacts,
                                       contact_pairs=len(owner.service.latest['sample'].contact_pairs),
                                       bean_count=owner.service.latest['device']['bean_count']))
    report['transitions'] = cycle.transitions
    assert [t['to_hz'] for t in cycle.transitions] == [240, 60], 'Expected two measured stopped clock switches'
    assert id(owner.service.controller) == controller_id and id(owner.service.animation) == animation_id
    assert owner.bridge is None, 'Wake must not start scoring or certify initial state'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in source_hashes.items())
    report.update(status='native_clock_and_evidence_smoke_passed', source_layers_unchanged=True,
                  same_device=True, actual_contact_collection=True, scoring_started=False,
                  source_sha256=source_hashes)
except Exception:
    report.update(status='failed', error=traceback.format_exc())
finally:
    print('COFFEE_HOTWAKE_RESULT', report['status'], report.get('error', ''), flush=True)
    try:
        if owner is not None: owner.close()
        if sim is not None: sim.detach_stage()
        OUT.write_text(json.dumps(report, indent=2, default=list) + '\n')
    except Exception:
        report['status'] = 'failed'
        print('COFFEE_HOTWAKE_CLEANUP_OR_OUTPUT_ERROR', traceback.format_exc(), flush=True)
    finally:
        omni.kit.app.get_app().post_quit(0 if report['status'] != 'failed' else 1)

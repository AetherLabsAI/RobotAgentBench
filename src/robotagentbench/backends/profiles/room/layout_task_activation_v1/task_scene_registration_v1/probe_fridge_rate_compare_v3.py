"""Matched 60/240 Hz native diagnostic through the existing room clock, without a robot trajectory.

External torque is a declared test input, never task success evidence. No
asset, pose, velocity, drive, collision, or physics-setting overrides.
"""
import os
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
sys.path[:0] = [str(ROOT), '/home/lingjun/robot_agent_arena/api-protocol']
import numpy as np
import omni.kit.app
import omni.physx
import omni.physics.tensors
from pxr import Usd, UsdUtils, PhysicsSchemaTools
from multi_task_room_v2 import MultiTaskRoom
from coffee_hotwake_v1 import NativeSceneClock
from room_fridge_helper_v1 import FRIDGE

out = Path('/pilot')
test_hz = int(os.environ['FRIDGE_TEST_HZ'])
assert test_hz in (60, 240)
report = dict(status='started', full_task_success=False, control_hz=20,
              physics_hz=test_hz, test='isolated_fridge_plus_room_floor_external_torque_not_robot_action',
              rows=[], contacts={}, pose_velocity_writes=0, source_parameter_overrides=0)
owner = sim = subscription = None
time = 0.
try:
    (out/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    manifest = json.loads((ROOT/'room_demo_v6/composition.json').read_text())
    stage = Usd.Stage.OpenMasked(manifest['scene'], Usd.StagePopulationMask([
        FRIDGE, '/World/Background/Shell/Floor', '/World/PhysicsScene', '/World/PhysicsMaterials']))
    stage.SetEditTarget(stage.GetSessionLayer())
    hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
              for l in stage.GetUsedLayers() if l.realPath}
    request = json.loads((ROOT/'multi_task_episode_v1.json').read_text())
    request['tasks'] = []
    sensor = json.loads((ROOT.parent/'hotwake_probe_inputs.json').read_text())['sensor']
    owner = MultiTaskRoom(stage, manifest, out/'room', 'fridge-floor-60hz', request=request, sensor=sensor)
    physics = stage.GetPrimAtPath('/World/PhysicsScene')
    assert physics.GetAttribute('physxScene:timeStepsPerSecond').Get() == 60
    assert physics.GetAttribute('physxScene:enableGPUDynamics').Get() is False
    sim = omni.physx.get_physx_simulation_interface()
    def contact(headers, points):
        for h in headers:
            if not h.num_contact_data: continue
            colliders = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in [h.collider0, h.collider1]]
            if not any(p.startswith(FRIDGE+'/right_door/') for p in colliders): continue
            key = ' | '.join(sorted(colliders))
            row = report['contacts'].setdefault(key, dict(count=0, first_s=time))
            row.update(count=row['count']+1, last_s=time)
    subscription = sim.subscribe_contact_report_events(contact)
    sid = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    assert sim.attach_stage(sid)
    sim.simulate(1/60, time); sim.fetch_results(); time += 1/60
    view = omni.physics.tensors.create_simulation_view('numpy', sid)
    view.set_subspace_roots('/')
    door = view.create_rigid_body_view(FRIDGE+'/right_door')
    assert door.count == 1
    def snapshot(): return door.get_transforms().copy().tobytes()+door.get_velocities().copy().tobytes()
    clock = NativeSceneClock(stage, sim, lambda: time, snapshot)
    owner.initialize(clock, view)
    helpers = dict(events=[], docked_object_count=0, tray_dock_enabled=False)
    for _ in range(240):
        sim.simulate(clock.get_physics_dt(), time); sim.fetch_results(); time += clock.get_physics_dt()
        owner.after_physics(helpers)
    # Keep diagnostic in pre-control native stepping; no fake control action.
    report['room_control_dispatch_exercised'] = False
    report['torque_input_hz'] = 20
    owner._switch_clock(test_hz)  # existing room clock path, not a new solver setting
    indices = np.array([0], dtype=np.uint32)
    def advance(phase, frames, torque=0., until=None):
        global time
        for _ in range(frames):
            plan = dict(substeps=test_hz//20, physics_dt_s=clock.get_physics_dt())
            assert abs(plan['physics_dt_s']-1/test_hz) < 1e-12
            for _ in range(plan['substeps']):
                if torque:
                    door.apply_forces_and_torques_at_position(None, np.array([[0.,0.,torque]], dtype=np.float32), None, indices, True)
                sim.simulate(plan['physics_dt_s'], time); sim.fetch_results(); time += plan['physics_dt_s']
                owner.after_physics(helpers)
                ctrl = owner.fridge_helper.controller
                report['rows'].append(dict(time_s=time, phase=phase, torque_nm=torque,
                    angle_deg=ctrl.ns['_fridge_pose']('right')[2], zone=ctrl.ns['FR_LATCH']['zone']))
            if until and report['rows'][-1]['zone'] == until: return True
        return False
    advance('settle', 20)
    report['opening_triggered'] = advance('opening_external_torque', 20, 40., 'opening')
    report['open_reached'] = advance('helper_open', 100, until='open')
    if report['open_reached']:
        advance('open_hold', 20)
        report['closing_triggered'] = advance('closing_external_torque', 20, -150., 'closing')
        if report['closing_triggered']: report['closed_reached'] = advance('helper_close', 100, until='latch')
    report['helper'] = owner.environment_helpers()['refrigerator']
    report['transitions'] = owner.transitions
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    report.update(status='diagnostic_completed', source_layers_unchanged=True, source_sha256=hashes)
except Exception:
    report.update(status='error', error=traceback.format_exc())
finally:
    if owner: owner.close()
    subscription = None
    if sim: sim.detach_stage()
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FRIDGE_RATE_RESULT', report['status'], report.get('error', ''), flush=True)
    omni.kit.app.get_app().post_quit(1 if report['status']=='error' else 0)

"""Real USD geometry + synthetic sensor inputs; NOT a native physical run."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from evaluation.tasks.coffee_machine import isaac_evidence as old
from .scene_service import SceneCoffeeService, CoffeeEvidenceBridge, Substep
from .native_source import preflight, NativeCoffeeSource


def check(value, name, checks):
    if not value: raise AssertionError(name)
    checks[name] = True


def rejected(call, name, checks):
    try:
        call()
    except (ValueError, RuntimeError, KeyError):
        checks[name] = True
    else:
        raise AssertionError(name)


def run(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    stage = Usd.Stage.Open(manifest['scene'])
    files = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
             for l in stage.GetUsedLayers() if l.realPath}
    checks = {}
    service = SceneCoffeeService(stage, manifest, 'cpu-fixture', 'coffee-fixture')
    roles = service.roles
    cache = UsdGeom.XformCache()
    poses = {}
    for path in (*service.objects, roles['button'], *old.HANDS.values()):
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        q = m.ExtractRotationQuat()
        poses[path] = dict(position_m=list(m.ExtractTranslation()), quaternion_wxyz=[q.GetReal(), *q.GetImaginary()])
    original = deepcopy(poses)
    beans = np.array([list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
                      for p in service.binding.resolved['bean_bodies']])
    floor = '/World/Background/Shell/Floor'
    pre = preflight(service, floor)
    if not pre['ready']:
        rejected(lambda: NativeCoffeeSource(service, None, floor=floor),
                 'unprepared_native_scene_rejected_before_physics_binding', checks)
    pair = lambda a, b, n=(0., 0., 1.): ((a, b), ((n, .01),))
    support = dict(cup=roles['counter'], bag=roles['counter'])
    held = set()
    press = False
    ticks = 0
    time_s = 0.
    bridge = None
    controller_id, animation_id = id(service.controller), id(service.animation)

    def tick(score=False, dt=old.PHYSICS_DT):
        nonlocal ticks, time_s
        contacts = []
        for role in ('cup', 'bag'):
            if support[role]: contacts.append(pair(roles[role], support[role]))
            if role in held:
                contacts += [pair(roles[role], old.PADS['r'][0], (1., 0., 0.)),
                             pair(roles[role], old.PADS['r'][1], (-1., 0., 0.))]
        if press: contacts.append(pair(old.PADS['r'][0], roles['button']))
        ticks += 1
        time_s += dt
        service.advance(Substep(time_s, deepcopy(poses), beans.copy(), tuple(contacts)), floor=floor, physics_dt=dt)
        if score: return bridge.collect_and_score()

    def frames(n):
        result = None
        for _ in range(n):
            for _ in range(old.SUBSTEPS - 1): tick()
            result = tick(True)
        return result

    def expected_phase(phase):
        check(bridge.process.phase == phase, 'entered_' + phase, checks)
        check(id(service.controller) == controller_id and id(service.animation) == animation_id,
              'same_device_at_' + phase, checks)

    # Authored geometry does not establish support; test fixtures explicitly supply contacts.
    for _ in range(3): tick(dt=1/60)
    rejected(lambda: CoffeeEvidenceBridge(service, accepted_binding=service.identity, evidence_source='cpu_synthetic_fixture'),
             'idle_60hz_cannot_start_score', checks)
    for _ in range(old.SUBSTEPS * 3): tick()
    rejected(lambda: SceneCoffeeService(stage, manifest, 'another-episode', 'another-instance'),
             'duplicate_machine_device_rejected', checks)
    bridge = CoffeeEvidenceBridge(service, accepted_binding=service.identity, evidence_source='cpu_synthetic_fixture')
    first = bridge.collect_and_score()
    check(first['score'] == 0 and not first['physics_verified'], 'initial_no_credit_no_physics_claim', checks)
    rejected(bridge.collect_and_score, 'duplicate_control_frame_rejected', checks)
    before = service.controller.elapsed
    last = service.latest['sample']
    rejected(lambda: service.advance(replace(last, time_s=last.time_s+1/60), floor=floor, physics_dt=1/60),
             'unfinished_score_refuses_downgrade', checks)
    rejected(lambda: service.advance(last, floor=floor), 'duplicate_physics_frame_rejected', checks)
    rejected(lambda: service.advance(replace(last, time_s=last.time_s + 2 * old.PHYSICS_DT), floor=floor),
             'missing_physics_frame_rejected', checks)
    rejected(lambda: service.advance(replace(last, time_s=last.time_s + old.PHYSICS_DT,
                                            contact_pairs=None), floor=floor), 'missing_contacts_rejected', checks)
    rejected(lambda: service.advance(replace(last, time_s=last.time_s + old.PHYSICS_DT,
                                            bean_positions=np.empty((0, 3))), floor=floor), 'missing_beans_rejected', checks)
    check(service.controller.elapsed == before, 'invalid_inputs_do_not_advance_device', checks)
    empty = service._contacts((), floor)[0]
    check(not empty[roles['cup']]['supports'], 'geometry_nearness_is_not_support', checks)
    rejected(lambda: service._contacts((((roles['cup'], roles['counter']), (((0., 0., 1.), 0.),)),), floor),
             'zero_impulse_cannot_establish_support', checks)
    single = service._contacts((pair(roles['cup'], old.PADS['r'][0], (1., 0., 0.)),), floor)[1]
    check(not single[roles['cup']]['r'], 'single_finger_cannot_establish_grasp', checks)
    service.window['robot_button'] = True
    check(service.facts(0, '1-1').robot_machine_contact and not service.facts(0, '1-3').robot_machine_contact
          and service.facts(0, '1-4').robot_machine_contact, 'phase_specific_button_contact_rule_preserved', checks)
    service.window['robot_button'] = False

    # 1-1 grasp, then place and release. These poses are fixture inputs, never USD/robot motion.
    support['cup'] = None; held.add('cup')
    poses[roles['cup']]['position_m'][2] += .08
    frames(4)
    check(bridge.process.summary()['score'] == 2, 'contact_and_relative_motion_grasp', checks)
    held.clear(); support['cup'] = roles['cup_support']
    poses[roles['cup']]['position_m'] = list(service.machine.Transform(Gf.Vec3d(*service.binding.contract.cup)))
    frames(23); expected_phase('1-2')
    # 1-2 physical beans are represented by measured-position fixtures, not simulated here.
    support['bag'] = None; held.add('bag'); poses[roles['bag']]['position_m'][2] += .08
    frames(4)
    hopper = service.binding.contract.hopper
    centre = service.machine.Transform(Gf.Vec3d(0, (hopper.floor_y + hopper.rim_y)/2, hopper.center_z))
    beans = np.tile(np.asarray(centre), (100, 1))
    frames(4)
    held.clear(); support['bag'] = roles['counter']; poses[roles['bag']] = deepcopy(original[roles['bag']])
    frames(23); expected_phase('1-3')
    # 1-3 two real-controller press edges, animation accounting from the actual cup geometry.
    press = True
    poses[roles['button']]['position_m'] = list(service.machine.Transform(Gf.Vec3d(0, -.004, 0)))
    frames(1)
    check(service.controller.dispensing, 'controller_started', checks)
    frames(155)
    press = False; poses[roles['button']] = deepcopy(original[roles['button']]); frames(1)
    press = True
    poses[roles['button']]['position_m'] = list(service.machine.Transform(Gf.Vec3d(0, -.004, 0))); frames(1)
    press = False; poses[roles['button']] = deepcopy(original[roles['button']]); frames(24)
    expected_phase('1-4')
    brewed = service.animation.counts()['in_cup']
    check(25 <= brewed * .004096 <= 60, 'animated_volume_in_original_band', checks)
    check(bridge.process.completed[2]['result']['final_counts']['valid_button_events'] == 2,
          'held_button_not_repeated_start_stop', checks)
    # 1-4 still the same cup/content; no prefill and no device reset at the boundary.
    held.add('cup'); support['cup'] = None
    point = Gf.Vec3d(*service.binding.contract.cup); point[1] += .08; point[2] += .30
    poses[roles['cup']]['position_m'] = list(service.machine.Transform(point))
    for _ in range(30):
        result = frames(1)
        if result['ended']: break
    check(result['success'] and result['score'] == 21, 'synthetic_four_phase_score_21', checks)
    identity = id(service.controller)
    tick(dt=1/60)
    check(id(service.controller) == identity and service.consumer is bridge and bridge.process.ended,
          'idle_downgrade_after_terminal_preserves_device_and_score', checks)
    check(service.animation.counts()['in_cup'] == brewed, 'no_prefill_or_reset_during_retrieve', checks)
    check(result['evidence_source'] == 'cpu_synthetic_fixture' and not result['runtime_registered']
          and not result['physics_verified'], 'provenance_explicit', checks)
    rejected(lambda: CoffeeEvidenceBridge(service, accepted_binding=service.identity, evidence_source='cpu_synthetic_fixture'),
             'reattach_cannot_reset_score', checks)
    check(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in files.items()),
          'source_layers_unchanged', checks)
    animation_prims = list(Usd.PrimRange(stage.GetPrimAtPath(service.animation_root)))
    check(not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                  or 'Particle' in p.GetTypeName() for p in animation_prims), 'animation_has_no_physics_or_particles', checks)
    return dict(layout_id=manifest['layout_id'], kind='actual_usd_geometry_synthetic_sensor_inputs',
                checks=checks, native_preflight=pre, score=result['score'], brewed_volume_units=brewed,
                physical_run=False, runtime_registered=False, source_sha256=files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(Path(__file__).resolve().parent) or out.exists():
        raise ValueError('New output under this task directory required')
    rows = [run(m) for m in args.manifest]
    result = dict(kind='CPU_USD_component_tests_not_native_physics', layouts=rows,
                  checks_passed=True, native_physics_verified=False)
    out.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__': main()

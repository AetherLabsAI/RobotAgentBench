"""CPU/USD tests of actual room startup; never claim native physics success."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from coffee_startup_v1 import CoffeeSceneStartup, CAMERA_VALUES, equal_value
from evaluation.tasks.coffee_machine.camera_contacts import CAMERA_COLLIDERS


def stage_snapshot(stage):
    return {str(p.GetPath()): dict(type=p.GetTypeName(), schemas=p.GetAppliedSchemas(),
             attrs={a.GetName(): str(a.Get()) for a in p.GetAttributes()},
             rels={r.GetName(): list(map(str, r.GetTargets())) for r in p.GetRelationships()})
            for p in stage.Traverse()}


def run(path):
    manifest = json.loads(Path(path).read_text())
    stage = Usd.Stage.Open(manifest['scene'])
    files = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
             for l in stage.GetUsedLayers() if l.realPath}
    before = stage_snapshot(stage)
    original_target = stage.GetEditTarget().GetLayer()
    owner = CoffeeSceneStartup()
    receipt = owner.prepare(stage, manifest, episode_id='startup-cpu-probe', instance_id='coffee')
    checks = {}

    def check(ok, name):
        if not ok: raise AssertionError(name)
        checks[name] = True

    def rejected(call, name):
        try: call()
        except (RuntimeError, ValueError): checks[name] = True
        else: raise AssertionError(name)

    check(receipt['instrumentation_preflight']['ready'], 'instrumentation_preflight_ready')
    expected_reports = 4 + sum(p.HasAPI(UsdPhysics.RigidBodyAPI)
                              for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')))
    check(len(receipt['contact_report_bodies']) == expected_reports, 'all_coffee_and_robot_reports_present')
    check(stage.GetEditTarget().GetLayer() == original_target, 'caller_edit_target_preserved')
    after = stage_snapshot(stage)
    reports = set(receipt['contact_report_bodies'])
    for path, data in before.items():
        actual = after[path]
        new_rels = set(actual['rels']) - set(data['rels'])
        api_rels = set()
        if path in CAMERA_COLLIDERS: api_rels.add('physics:simulationOwner')
        if path in reports: api_rels.add('physxContactReport:reportPairs')
        check(data['type'] == actual['type']
              and all(actual['rels'].get(n) == v for n, v in data['rels'].items())
              and new_rels <= api_rels and all(not actual['rels'][n] for n in new_rels),
              'types_and_existing_relationships_unchanged_only_empty_api_defaults_added')
        allowed = set(CAMERA_VALUES) if path in CAMERA_COLLIDERS else set()
        if path in reports: allowed.add('physxContactReport:threshold')
        for name, value in data['attrs'].items():
            if name not in allowed and actual['attrs'].get(name) != value:
                raise AssertionError('Unrelated value changed: ' + path + '.' + name)
        # Existing properties including poses/velocities/drives remain exactly equal.
        expected_schemas = set(data['schemas'])
        if path in CAMERA_COLLIDERS:
            expected_schemas.update(('PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI', 'PhysxCollisionAPI'))
        if path in reports: expected_schemas.add('PhysxContactReportAPI')
        if set(actual['schemas']) != expected_schemas:
            raise AssertionError('Unexpected schema: ' + path + str(actual['schemas']))
    check(True, 'all_existing_unrelated_properties_and_schemas_unchanged')
    for path in CAMERA_COLLIDERS:
        p = stage.GetPrimAtPath(path)
        check(all(equal_value(p.GetAttribute(n).Get(), v) for n, v in CAMERA_VALUES.items()),
              'camera_exact_original_configuration')
    check(all(p == '/World/OpenendedCoffeeVisual' or p.startswith('/World/OpenendedCoffeeVisual/')
              for p in set(after) - set(before)), 'only_visual_new_prims')
    check(not any('Particle' in p.GetTypeName() for p in stage.Traverse()), 'no_particle_liquid')
    device = owner.service.controller
    saved = stage.GetSessionLayer().ExportToString()
    rejected(lambda: owner.prepare(stage, manifest, episode_id='x', instance_id='y'), 'same_owner_no_reset')
    rejected(lambda: CoffeeSceneStartup().prepare(stage, manifest, episode_id='x', instance_id='y'),
             'second_owner_cannot_replace_device')
    check(saved == stage.GetSessionLayer().ExportToString() and owner.service.controller is device,
          'failed_duplicate_preserves_existing_session_and_device')
    rejected(lambda: owner.attach_score(accepted_binding=owner.service.identity), 'no_native_no_score')
    rejected(lambda: owner.after_physics({}), 'no_native_no_facts')
    rejected(lambda: owner.on_physics_ready(SimpleNamespace(get_physics_dt=lambda: 1/60)),
             '60hz_rejected_before_native_import')
    check(owner.clock_check(1/240)['compatible'], 'existing_coffee_240hz_contract_recognized')
    scene = stage.GetPrimAtPath('/World/PhysicsScene')
    hz = scene.GetAttribute('physxScene:timeStepsPerSecond').Get()
    check(hz == 60, 'existing_room_60hz_unchanged')
    # Actual USD/session clock adapter, synthetic time/flush/state (NOT PhysX).
    from coffee_hotwake_v1 import NativeSceneClock
    fixture_time = [0.]
    flushes = []
    clock = NativeSceneClock(stage, SimpleNamespace(flush_changes=lambda: flushes.append(True)),
                             lambda: fixture_time[0], lambda: b'explicit-synthetic-snapshot')
    cycle = owner.enable_hotwake(clock)
    plans = []
    for active in (False, True, False):
        plan = cycle.begin_control(dict(active=active, state='ACTIVE' if active else 'DORMANT'),
                                   command_vw=[0, 0], base_velocity=[0]*6)
        for _ in range(plan['substeps']): fixture_time[0] += plan['physics_dt_s']
        cycle.finish_control(completed_substeps=plan['substeps'])
        plans.append(plan)
    check([p['physics_hz'] for p in plans] == [60, 240, 60], 'hotwake_session_clock_sequence')
    check([p['substeps'] for p in plans] == [3, 12, 3], 'hotwake_20hz_substep_counts')
    check(abs(fixture_time[0]-.15) < 1e-9, 'hotwake_time_not_recomputed_from_step_count')
    check(len(flushes) == 2 and owner.service.controller is device, 'hotwake_keeps_device_and_flushes_only_on_change')
    check(owner.bridge is None and owner.service.consumer is None, 'wake_never_auto_scores_or_accepts_initial_state')
    owner.close(); owner.close()
    check(owner.service.closed, 'close_idempotent_service_sealed')
    rejected(lambda: owner.prepare(stage, manifest, episode_id='x', instance_id='y'), 'closed_owner_not_reused')

    # One separate stage, restored between negative cases; no good-stage contamination.
    s = Usd.Stage.Open(manifest['scene'])
    clean_session = s.GetSessionLayer().ExportToString()
    for kind in ('camera_conflict', 'mesh_changed', 'scaled_camera', 'report_conflict', 'missing_floor'):
        s.GetSessionLayer().ImportFromString(clean_session)
        with Usd.EditContext(s, s.GetSessionLayer()):
            p = s.GetPrimAtPath(CAMERA_COLLIDERS[-1])
            if kind == 'camera_conflict':
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(False)
            elif kind == 'mesh_changed':
                points = p.GetAttribute('points').Get(); points[0] += Gf.Vec3f(.001, 0, 0)
                p.GetAttribute('points').Set(points)
            elif kind == 'scaled_camera':
                p.GetAttribute('xformOp:scale').Set(Gf.Vec3f(2))
            elif kind == 'report_conflict':
                from pxr import PhysxSchema
                PhysxSchema.PhysxContactReportAPI.Apply(s.GetPrimAtPath(owner.service.roles['cup'])).CreateThresholdAttr(1.)
            else:
                s.GetPrimAtPath(owner.floor).SetActive(False)
        original = s.GetSessionLayer().ExportToString()
        rejected(lambda: CoffeeSceneStartup().prepare(s, manifest, episode_id='x', instance_id='y'), kind + '_rejected')
        check(original == s.GetSessionLayer().ExportToString(), kind + '_atomic_no_partial_edits')
    check(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in files.items()),
          'all_source_layer_hashes_unchanged')
    return dict(layout=manifest['layout_id'], checks=checks, preparation=receipt,
                authored_physics_hz=hz, clock_compatible=False, native_physics_run=False,
                full_task_success=False, source_sha256=files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent) or output.exists():
        raise ValueError('Unique output inside scene startup directory required')
    result = dict(kind='CPU_USD_startup_checks_only', layouts=[run(m) for m in args.manifest],
                  native_physics_verified=False, runtime_registered=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({r['layout']: len(r['checks']) for r in result['layouts']}))


if __name__ == '__main__': main()

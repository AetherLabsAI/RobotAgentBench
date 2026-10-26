"""CPU USD composition audit, explicitly NOT native physical acceptance."""
import hashlib
import json
from pathlib import Path
from .build_initial import HERE, MACHINE_BODY, physical_properties


def check(scene):
    from pxr import Usd
    from .combined_session import CombinedNativeSession
    scene = Path(scene).resolve(strict=True)
    if not scene.is_relative_to(HERE):
        raise ValueError('Task-owned combined initial required')
    initial = json.loads((scene.parent / 'initial.json').read_text())
    if initial.get('variant') != 'bag_then_machine':
        raise ValueError('Not a combined initial')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(scene) != initial['initial_layer_sha256']:
        raise ValueError('Combined initial changed')
    if any(digest(p) != h for p, h in initial['source_layers'].items()):
        raise ValueError('Referenced source layer changed')
    stage = Usd.Stage.Open(str(scene))
    stage.SetEditTarget(stage.GetSessionLayer())
    before = physical_properties(stage)
    session = CombinedNativeSession(stage, initial)
    try:
        checks = dict(
            machine_dynamic=stage.GetPrimAtPath(MACHINE_BODY).GetAttribute('physics:kinematicEnabled').Get() is False,
            one_bag_and_100_beans=initial['joint_initialization']['translated_body_count'] == 101,
            bag_beans_rigidly_translated=initial['joint_initialization']['translation_matrix_max_error'] <= 1e-9,
            both_humans_initially_absent=all(a.human.appearance_state == 'waiting' for a in (session.bag, session.push)),
            all_human_colliders_initially_disabled=all(
                stage.GetPrimAtPath(p + '/Collision').GetAttribute('physics:collisionEnabled').Get() is False
                for a in (session.bag, session.push) for p in a.human.links.values()),
            separate_human_roots=session.bag.human.root != session.push.human.root,
            source_physical_properties_unchanged=physical_properties(stage) == before,
            initial_file_unchanged=digest(scene) == initial['initial_layer_sha256'],
            referenced_layers_unchanged=all(digest(p) == h for p, h in initial['source_layers'].items()),
            only_bag_eligible=session.sequence.eligible_component == 'bag_move',
            no_robot_credit_during_bag_interference=not session.robot_process_scoring_enabled,
        )
        if not all(checks.values()):
            raise RuntimeError('Combined CPU composition rejected: ' + repr(checks))
        return dict(variant='bag_then_machine', seed=initial['seed'], checks=checks,
                    cpu_composition_accepted=True, native_actor_topology_constructed=True,
                    native_physics_run=False, native_initial_acceptance=False,
                    full_combination_verified=False, scoring_verified=False,
                    scene=str(scene), initial_layer_sha256=initial['initial_layer_sha256'])
    finally:
        session.close()


if __name__ == '__main__':
    import sys
    print(json.dumps(check(sys.argv[1]), indent=2))

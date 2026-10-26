"""Compose the approved bag pair and dynamic machine in ONE task-owned initial."""
import hashlib
import json
from pathlib import Path
from .build_initial import build as build_machine_initial, physical_properties


def build(output, *, seed, role):
    if role != 'random':
        raise ValueError('Combined interference requires the existing paired/random bag initializer')
    plan = build_machine_initial(output, seed=seed, role=role)
    from pxr import Usd
    from short_task_robustness.tasks.coffee_bag_interference.joint_initialization import author_pair
    from short_task_robustness.tasks.coffee_bag_interference.route_options import reserve_routes
    from .geometry import inspect_routes
    scene = Path(plan['scene'])
    stage = Usd.Stage.Open(str(scene))
    before = physical_properties(stage)
    try:
        joint = author_pair(stage, plan)
        if physical_properties(stage) != before:
            raise RuntimeError('Bag paired initialization changed a physical property')
        plan.update(variant='bag_then_machine', joint_initialization=joint,
                    relocation_pair=joint['pair'],
                    randomization_scope='unchanged producer machine/cup/yaw; original paired bag/100-bean translation',
                    push_routes=inspect_routes(stage, seed), combined_native_verified=False,
                    compatible_recorded_source_replay=False)
        # The existing NativeBagRelocation recognizes this field in both solo
        # and combined sessions. Old frozen initials without it stay legacy.
        plan['relocation_options'] = reserve_routes(stage, plan, joint['reservation'])
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
               for p, digest in plan['source_layers'].items()):
            raise RuntimeError('Referenced source layer changed during combined preparation')
        stage.GetRootLayer().Save()
        plan['initial_layer_sha256'] = hashlib.sha256(scene.read_bytes()).hexdigest()
        (scene.parent / 'initial.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
        return plan
    except Exception as error:
        (scene.parent / 'combined_initialization_failure.json').write_text(json.dumps(dict(
            variant='bag_then_machine', accepted=False, seed=seed, reason=str(error)), indent=2) + '\n')
        raise

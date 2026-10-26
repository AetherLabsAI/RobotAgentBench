"""Read-only source/scene/native action admission before the GPU replay."""
import hashlib
import json
from pathlib import Path
from pxr import Usd, UsdPhysics
import numpy as np
from arena_api import policy_execution as pe, robot_profiles as rp
from .build_initial import HERE
from .replay_actions import adapt


def preflight(scene):
    scene = Path(scene).resolve(strict=True)
    if not scene.is_relative_to(HERE):
        raise ValueError('Task-owned independent scene required')
    initial = json.loads((scene.parent / 'initial.json').read_text())
    if hashlib.sha256(scene.read_bytes()).hexdigest() != initial['initial_layer_sha256']:
        raise ValueError('Initial layer changed')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in initial['source_layers'].items()):
        raise ValueError('Referenced source layer changed')
    stage = Usd.Stage.Open(str(scene))
    candidate = adapt(initial)
    names = list(rp.NATIVE_JOINTS)
    units, limits, driven = {}, {}, []
    for name in names:
        p = stage.GetPrimAtPath('/World/Robot/joints/' + name)
        units[name] = 'rad' if p.IsA(UsdPhysics.RevoluteJoint) else 'm' if p.IsA(UsdPhysics.PrismaticJoint) else None
        if units[name] is None:
            raise ValueError('Unsupported native joint')
        if any(s.startswith('PhysicsDriveAPI:') for s in p.GetAppliedSchemas()):
            driven.append(name)
        lo, hi = (p.GetAttribute('physics:' + k).Get() for k in ('lowerLimit', 'upperLimit'))
        if lo is not None and hi is not None:
            limits[name] = [float(np.radians(v)) if units[name] == 'rad' else float(v) for v in (lo,hi)]
    # Existing source physical initialization includes the native passive joints.
    defaults = dict(initial['robot_initialization']['positions'])
    defaults.update(initial['robot_initialization']['hold_targets'])
    if set(driven) != set(rp.get(rp.JOINT20).joints):
        raise ValueError('Source original 20 active drives do not match producer executor')
    public_limits = {n: limits[n] for n in rp.get(rp.JOINT20).joints}
    for action in candidate['actions']:
        pe.validate(action, public_limits, rp.JOINT20)
    result = dict(source_episode=candidate['source_episode'], source_plan_sha256=candidate['source_plan_sha256'],
        source_scene_exact=True, source_hashes_unchanged=True, native_action_limits_passed=True,
        rows=len(candidate['actions']), changed_open_commands=candidate['changed_commands_count'],
        original_unmodified_replay=False, source_file_modified=False, all_other_values_unchanged=True,
        executor='original_atomic_producer_20_active_drives', unified_native_adapter_verified=False,
        newer_adapter_incompatibility='original source inner fingers have no explicit drive or Mimic API',
        physics_hz=240, control_hz=20, native_physics_run=False, real_trigger_verified=False)
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(preflight(sys.argv[1]), indent=2))

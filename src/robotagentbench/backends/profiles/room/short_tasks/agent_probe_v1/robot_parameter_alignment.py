"""Approved API robot baseline, applied only in the episode session layer.

2026-10-13: align robot drives, pad material/contact and robot solver counts.
Never change the task clock, pose, limits, collision filters, geometry or scoring.
Source: api-protocol baseline USD + its shared pre-physics initialization.
"""
import hashlib
import math
from pathlib import Path

from arena_api import robot_drive_defaults

POLICY = 'short_tasks.public_robot_drives.task_owned_clock.v2'
ROOT = '/World/Robot'
FIELDS = ('stiffness', 'damping', 'maxForce')
DRIVES = robot_drive_defaults.DRIVES


def _near(actual, expected):
    return isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)


def _scene_clock(stage):
    scene = stage.GetPrimAtPath('/World/PhysicsScene')
    if not scene:
        raise ValueError('Missing task-owned PhysicsScene')
    return {a.GetName(): str(a.Get()) for a in scene.GetAttributes()
            if a.GetName().startswith(('physics:', 'physxScene:'))}


def configuration(stage):
    result = {}
    for name, (axis, *_) in DRIVES.items():
        p = stage.GetPrimAtPath(ROOT + '/joints/' + name)
        if not p or 'PhysicsDriveAPI:' + axis not in p.GetAppliedSchemas():
            raise ValueError('Missing active baseline drive: ' + name)
        if p.GetAttribute(f'drive:{axis}:physics:type').Get() != 'force':
            raise ValueError('Unexpected drive type: ' + name)
        result[name] = {f: p.GetAttribute(f'drive:{axis}:physics:{f}').Get() for f in FIELDS}
    return result


def _dependencies():
    from arena_api import finger_pad_material, finger_pad_contact, physics_defaults
    # Fail closed if a concurrent API update changes the approved baseline.
    expected = {'physics:staticFriction': 2.2, 'physics:dynamicFriction': 2.,
                'physics:restitution': 0., 'physxMaterial:compliantContactStiffness': 0.,
                'physxMaterial:compliantContactDamping': 0., 'physxMaterial:frictionCombineMode': 'max'}
    if any(finger_pad_material.PARAMETERS.get(k) != v for k, v in expected.items()):
        raise ValueError('Shared finger material baseline changed; review required')
    if (physics_defaults.POSITION_ITERATIONS, physics_defaults.VELOCITY_ITERATIONS) != (32, 4):
        raise ValueError('Shared robot solver baseline changed; review required')
    return finger_pad_material, finger_pad_contact, physics_defaults


def apply(stage):
    """Call once before PhysX attachment, never save referenced source layers."""
    from pxr import Usd
    material, contact, solver = _dependencies()
    before = configuration(stage)
    # Known legacy force15 is migrated; unexplained drive differences are rejected.
    for name, (_, *expected) in DRIVES.items():
        for field, wanted in zip(FIELDS, expected):
            actual = before[name][field]
            if not robot_drive_defaults.accepted_source(name, field, actual):
                raise ValueError(f'Unreviewed drive value: {name}.{field}={actual}')
    clock = _scene_clock(stage)
    layer = stage.GetSessionLayer()
    original = layer.ExportToString()
    changes = []
    try:
        with Usd.EditContext(stage, layer):
            for name, (axis, *expected) in DRIVES.items():
                for field, wanted in zip(FIELDS, expected):
                    if not _near(before[name][field], wanted):
                        attr = stage.GetPrimAtPath(ROOT + '/joints/' + name).GetAttribute(f'drive:{axis}:physics:{field}')
                        if not attr.Set(wanted):
                            raise RuntimeError('Could not align drive: ' + str(attr.GetPath()))
                        changes.append(dict(joint=name, field=field, before=before[name][field], after=wanted))
            # Reuse the exact shared policies; apply_robot does NOT change dt.
            solver.apply_robot(stage)
            material_receipt = material.apply(stage)
            contact_receipt = contact.apply(stage)
        if _scene_clock(stage) != clock:
            raise RuntimeError('Robot alignment unexpectedly changed task scene physics')
        receipt = dict(policy_id=POLICY, application_layer='session', drives=configuration(stage),
            drive_changes=changes, solver_position_iterations=32, solver_velocity_iterations=4,
            material=material_receipt, contact=contact_receipt, scene_physics_before=clock,
            task_physics_frequency_changed=False, initial_pose_changed=False,
            collision_guard_changed=False, source_assets_modified=False,
            implementation_sha256={str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                   for m in (material, contact, solver, robot_drive_defaults)})
        verify(stage, receipt)
        return receipt
    except Exception:
        layer.ImportFromString(original)
        raise


def verify(stage, receipt):
    """Read-only check; task-specific scene clock changes remain allowed later."""
    material, contact, _ = _dependencies()
    values = configuration(stage)
    for name, (_, *expected) in DRIVES.items():
        for field, wanted in zip(FIELDS, expected):
            if not _near(values[name][field], wanted):
                raise RuntimeError('Robot drive changed: ' + name + '.' + field)
    root = stage.GetPrimAtPath(ROOT)
    for field, expected in [('solverPositionIterationCount', 32), ('solverVelocityIterationCount', 4)]:
        if root.GetAttribute('physxArticulation:' + field).Get() != expected:
            raise RuntimeError('Robot solver changed: ' + field)
    material.verify(stage, receipt['material']['configuration'])
    contact.verify(stage, receipt['contact']['configuration'])
    return dict(verified=True, policy_id=POLICY)


def verify_native_forces(robot, names):
    """Verify native gains and caps, including USD degree/radian conversion."""
    return robot_drive_defaults.verify_native(robot, names, include_unchanged=True)

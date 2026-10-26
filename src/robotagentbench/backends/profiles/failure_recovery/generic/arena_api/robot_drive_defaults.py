"""Public G1 drive defaults approved after shelf/fridge tests, 2026-10-14.

USD angular gains are per degree; PhysX tensor gains are per radian.
Torques stay N.m. Only session layers are authored, never source assets.
"""
import math

POLICY = 'arena.robot_drives.head500_50_50.arm60_30.v1'
ROOT = '/World/Robot/joints/'
FIELDS = ('stiffness', 'damping', 'maxForce')
LEGACY_DRIVES = {
    'idx01_body_joint1': ('linear', 2000000., 50000., 5000000.),
    'idx02_body_joint2': ('angular', 2000000., 50000., 5000000.),
    'idx11_head_joint1': ('angular', 500000., 50000., 5000000.),
    'idx12_head_joint2': ('angular', 500000., 50000., 5000000.),
    **{f'idx{base+i}_arm_{side}_joint{i}': ('angular', 80000., 2500., 10000.)
       for side, base in (('l', 20), ('r', 60)) for i in range(1, 8)},
    'idx41_gripper_l_outer_joint1': ('angular', 500., 100., 80.),
    'idx81_gripper_r_outer_joint1': ('angular', 500., 100., 80.),
}
DRIVES = {
    **LEGACY_DRIVES,
    'idx11_head_joint1': ('angular', 500., 50., 50.),
    'idx12_head_joint2': ('angular', 500., 50., 50.),
    **{f'idx{base+i}_arm_{side}_joint{i}': ('angular', 80000., 2500., 60. if i <= 4 else 30.)
       for side, base in (('l', 20), ('r', 60)) for i in range(1, 8)},
}
# Only these sixteen drives are newly authorized for retuning. Public API and
# Openend must not inherit ShortTask's older, separately approved gripper migration.
CHANGED_DRIVES = {name: values for name, values in DRIVES.items() if values != LEGACY_DRIVES[name]}

def near(a, b):
    return isinstance(a, (float, int)) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)

def accepted_source(name, field, value):
    index = FIELDS.index(field) + 1
    return (near(value, DRIVES[name][index]) or near(value, LEGACY_DRIVES[name][index])
            or ('gripper_' in name and field == 'maxForce' and near(value, 15.)))

def apply(stage):
    """Validate known old/new values, then author this policy before PhysX attach."""
    from pxr import Usd
    edits = []
    for name, (axis, *values) in CHANGED_DRIVES.items():
        prim = stage.GetPrimAtPath(ROOT + name)
        if not prim or prim.GetAttribute(f'drive:{axis}:physics:type').Get() != 'force':
            raise ValueError('Missing force drive: ' + name)
        for field, wanted in zip(FIELDS, values):
            attr = prim.GetAttribute(f'drive:{axis}:physics:{field}')
            value = attr.Get()
            if not accepted_source(name, field, value):
                raise ValueError(f'Unreviewed drive value: {name}.{field}={value}')
            if not near(value, wanted):
                edits.append((attr, wanted, dict(joint=name, field=field, before=value, after=wanted)))
    layer = stage.GetSessionLayer(); before = layer.ExportToString()
    try:
        with Usd.EditContext(stage, layer):
            for attr, value, _ in edits:
                if not attr.Set(value):
                    raise RuntimeError('Could not set drive: ' + str(attr.GetPath()))
        verify(stage)
    except Exception:
        layer.ImportFromString(before)
        raise
    return dict(policy_id=POLICY, changes=[r for _, _, r in edits], application_layer='session',
                source_assets_modified=False, angular_gain_units='USD_per_degree', torque_units='N.m')

def verify(stage):
    for name, (axis, *values) in CHANGED_DRIVES.items():
        prim = stage.GetPrimAtPath(ROOT + name)
        for field, wanted in zip(FIELDS, values):
            if not near(prim.GetAttribute(f'drive:{axis}:physics:{field}').Get(), wanted):
                raise RuntimeError('Robot drive changed: ' + name + '.' + field)
    return dict(verified=True, policy_id=POLICY)

def verify_native(robot, names, *, include_unchanged=False):
    arrays = dict(stiffness=robot.get_dof_stiffnesses()[0], damping=robot.get_dof_dampings()[0],
                  maxForce=robot.get_dof_max_forces()[0])
    values = {}
    for name, (axis, *expected) in (DRIVES if include_unchanged else CHANGED_DRIVES).items():
        i = names.index(name); values[name] = {}
        for field, usd_value in zip(FIELDS, expected):
            wanted = usd_value * (180 / math.pi if axis == 'angular' and field != 'maxForce' else 1.)
            actual = float(arrays[field][i]); values[name][field] = actual
            if not near(actual, wanted):
                raise RuntimeError(f'Native drive mismatch: {name}.{field}={actual}, expected {wanted}')
    return dict(verified=True, policy_id=POLICY, values=values,
                maximum_drive_forces={n: v['maxForce'] for n, v in values.items()})

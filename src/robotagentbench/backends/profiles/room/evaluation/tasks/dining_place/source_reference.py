"""Audited L2 motion reference; deliberately not a runnable API recording.

No network access, USD mutation, physics tuning or scoring. Legacy commanded
gripper closure is NOT evidence of a held plate. Historical runtime added a
release controller, so even the complete NPZ is not the complete executed motion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESOURCE_DIR = Path(__file__).parent / 'resources' / 'task108'
SCHEMA = 'arena.dining_place.l2_reference.v1'
ARM_NAMES = tuple(f'idx{offset+i}_arm_{side}_joint{i}'
                  for side, offset in [('l', 20), ('r', 60)] for i in range(1, 8))
GRIP_NAMES = ('idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1')


def load_reference(directory=RESOURCE_DIR):
    import numpy as np
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['schema'] != SCHEMA or manifest['target_task_id'] != '10.2'
            or manifest['source_destination']['body_path'] != '/World/Assets/CoffeeTable'):
        raise ValueError('Reference identity/destination mismatch')
    if set(manifest['files']) != {'raw.npz', 'baseline.npz', 'candidate.npz', 'historical_metrics.npz'}:
        raise ValueError('Incomplete reference bundle')
    for name, row in manifest['files'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Reference must be self-contained')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Changed reference file: ' + name)
    with np.load(directory / 'candidate.npz', allow_pickle=False) as npz:
        motion = {key: npz[key].copy() for key in npz.files}
    validate_motion(motion)
    if len(motion['arm']) != manifest['sequence']['candidate_frames']:
        raise ValueError('Candidate length changed')
    return manifest, motion


def validate_motion(motion):
    import numpy as np
    if set(motion) != {'fps', 'arm', 'waist', 'head', 'base_xyyaw', 'lgrip', 'rgrip'}:
        raise ValueError('Unknown legacy motion fields')
    n = len(motion['arm'])
    if n < 2 or np.asarray(motion['fps']).shape != () or float(motion['fps']) != 20:
        raise ValueError('Expected audited 20 Hz recording')
    shapes = {'arm': (n, 14), 'waist': (n, 2), 'head': (n, 2),
              'base_xyyaw': (n, 3), 'lgrip': (n,), 'rgrip': (n,)}
    for key, shape in shapes.items():
        if motion[key].shape != shape or not np.isfinite(motion[key]).all():
            raise ValueError('Invalid motion field ' + key)
    for key in ('lgrip', 'rgrip'):
        if ((motion[key] < 0) | (motion[key] > 1)).any():
            raise ValueError('Unknown gripper encoding')
    # Legacy odometry is not the USD base transform. Never silently discard
    # navigation or changing torso/head from an unsupported source episode.
    tolerances = {'base_xyyaw': [0, 0, 0], 'waist': [1e-4, 2e-6], 'head': [1e-4, 1e-4]}
    for key, tolerance in tolerances.items():
        if (np.ptp(motion[key], axis=0) > np.asarray(tolerance)).any():
            raise ValueError('Cannot treat moving ' + key + ' as stationary')


def initial_reference(motion):
    """Source joint commands, not measured reset joint positions."""
    import numpy as np
    validate_motion(motion)
    arms = dict(zip(ARM_NAMES, map(float, motion['arm'][0])))
    fixed = {'idx01_body_joint1': {'value': float(motion['waist'][0, 1]), 'unit': 'm'},
             'idx02_body_joint2': {'value': float(motion['waist'][0, 0]), 'unit': 'rad'},
             'idx11_head_joint1': {'value': float(motion['head'][0, 0]), 'unit': 'rad'},
             'idx12_head_joint2': {'value': float(motion['head'][0, 1]), 'unit': 'rad'}}
    return dict(arm_targets_rad=arms, stationary_joint_targets=fixed,
        gripper_command_reference_rad={name: .62 * (1 - float(motion[key][0]))
                                      for name, key in zip(GRIP_NAMES, ('lgrip', 'rgrip'))},
        gripper_measured_reset_position=None,
        fixed_field_peak_to_peak={key: np.ptp(motion[key], axis=0).tolist()
                                 for key in ('waist', 'head', 'base_xyyaw')},
        base_odometry_not_world_pose=motion['base_xyyaw'][0].tolist(),
        initial_holding_required=True, initial_holding_verified=False,
        warning='Do not initialize a closed gripper around an intersecting plate. Calibrate actual aperture and verify bilateral contacts.')


def command_reference(motion, control_hz=30):
    """Time-preserving candidate targets, no hidden tail or legacy helper.

    Samples are interval-end targets (frame zero belongs to reset). Arms use
    linear interpolation; legacy gripper commands use zero-order hold. The last
    source frame occupies its original final 1/20 s interval. Tiny stationary
    torso/head recording noise is explicitly collapsed to frame zero in setup.
    """
    import numpy as np
    validate_motion(motion)
    if type(control_hz) is not int or control_hz != 30:
        raise ValueError('Current common controller is 30 Hz; do not change it here')
    duration = len(motion['arm']) / 20
    count = round(duration * control_hz)
    if abs(count / control_hz - duration) > 1e-9:
        raise ValueError('Source duration does not end on current control grid')
    times = np.arange(1, count + 1) / control_hz
    old_times = np.arange(len(motion['arm'])) / 20
    values = {name: np.interp(times, old_times, motion['arm'][:, i])
              for i, name in enumerate(ARM_NAMES)}
    indices = np.minimum(np.floor(times * 20 + 1e-9).astype(int), len(old_times) - 1)
    for name, key in zip(GRIP_NAMES, ('lgrip', 'rgrip')):
        values[name] = .62 * (1 - motion[key][indices].astype(float))
    return dict(schema='arena.dining_place.command_reference.v1',
        task_id='10.2', execution_ready=False, physics_validated=False,
        source_fps=20, control_hz=control_hz, duration_s=duration,
        control_steps=count, sample_times_s=times.tolist(),
        targets=[{name: float(row[i]) for name, row in values.items()} for i in range(count)],
        arm_resampling='linear at interval end', gripper_resampling='zero-order hold',
        gripper_mapping='outer_joint_rad = 0.62 * (1 - legacy_closure)',
        helper_tail_included=False, appended_hold_s=0,
        notice='Reference only. Requires accepted dining reset, explicit release tail and current physics/API verification.')

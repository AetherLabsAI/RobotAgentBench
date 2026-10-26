"""Repair robot commands for native cloth diagnostics without rewriting state.

The default preserves every command after the first approach. Optional clearance
probes change only robot commands. Neither is a successful episode or a fully
randomized production generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def repair_approach(action, phases, robot, fps):
    """Bypass the Cartesian IK branch that folds link 6 into link 4.

    Keep the approach endpoint, duration, gripper state and remaining action.
    This is bounded joint interpolation, subject to native physics validation.
    """
    names = action['joint_names'].tolist()
    original = action['joint_target']
    ids = np.flatnonzero(phases == 'L_friction_approach')
    if not len(ids) or ids[0] == 0 or not np.array_equal(ids, np.arange(ids[0], ids[-1]+1)):
        raise ValueError('Expected one contiguous approach after an initial hold')
    start, stop = int(ids[0]), int(ids[-1])+1
    columns = [names.index(n) for n in robot.arm_names('L')]
    q = original.copy()
    fraction = np.arange(1, stop-start+1)/(stop-start)
    fraction = fraction*fraction*(3-2*fraction)
    q[start:stop, columns] = ((1-fraction[:, None])*q[start-1, columns]
                              + fraction[:, None]*q[stop-1, columns])
    speed = np.abs(np.diff(q[start-1:stop, columns], axis=0)).max()*fps
    if speed > .65:
        raise ValueError(f'Approach exceeds 0.65 rad/s: {speed:.3f}')
    for j, name in enumerate(names):
        if name in robot.by_name and 'limits' in robot.by_name[name]:
            lo, hi = robot.by_name[name]['limits']
            if np.any((q[start:stop, j] < lo) | (q[start:stop, j] > hi)):
                raise ValueError('Approach exceeds native joint limit: '+name)
    commands = dict(joint_target=q, joint_names=np.asarray(names),
                    time_s=np.arange(len(q))/fps, phase=phases.copy(), fps=fps)
    audit = dict(original_approach=[start, stop], replacement_frames=stop-start,
                 unchanged_suffix_frames=len(original)-stop,
                 max_joint_speed_rad_s=float(speed),
                 repair='smooth joint interpolation between unchanged approach endpoints',
                 diagnostic_only=True, eligible_for_production=False)
    return commands, audit


def adjust_scrape_height(commands, robot, lift_m):
    """Change only robot clearance; cloth state and material are untouched."""
    if not 0 <= lift_m <= .005:
        raise ValueError('Diagnostic scrape adjustment must be 0..5 mm')
    names = commands['joint_names'].tolist()
    phases = commands['phase']
    q = commands['joint_target'].copy()
    touch = np.flatnonzero(phases == 'L_friction_touch')
    clear = np.flatnonzero(phases == 'L_friction_clear')
    if not len(touch) or not len(clear):
        raise ValueError('Missing first touch/clear phases')
    first, end = int(touch[0]), int(clear[-1])+1
    errors = []
    for frame in range(first, end):
        fraction = min(1., (frame-first+1)/20, (end-frame-1)/20)
        height = lift_m*fraction*fraction*(3-2*fraction)
        values = dict(zip(names, q[frame]))
        pose = robot.fk(values)['gripper_l_base_link']
        solved, error = robot.solve(values, 'L', pose[:3, 3]+[0, 0, height],
                                     np.zeros(3), pose[:3, :3], max_steps=100)
        if error > .0003:
            raise ValueError(f'Unreachable adjusted scrape: {error:.6f} m')
        q[frame] = [solved[n] for n in names]
        errors.append(error)
    return {**commands, 'joint_target': q}, dict(surface_lift_m=lift_m,
        adjusted_frames=[first, end], max_ik_residual_m=max(errors))


def main():
    from tablecloth_robot_kinematics import Robot
    from tablecloth_scene_binding import station_xyyaw
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--robot', type=Path, required=True, help='Exported native FK metadata JSON')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--surface-lift-mm', type=float, default=0.)
    p.add_argument('--contact-probe', action='store_true',
                   help='Initialize the robot before contact and run only a short material/contact diagnostic')
    args = p.parse_args()
    meta = json.loads((args.reference/'trial.json').read_text())
    robot = Robot(json.loads(args.robot.read_text()), station_xyyaw(meta['benchmark_task_id']))
    with np.load(args.reference/'action.npz') as z:
        action = dict(z)
    with np.load(args.reference/'full_state.npz') as z:
        phases = z['phase']
    commands, audit = repair_approach(action, phases, robot, int(meta['fps']))
    if args.surface_lift_mm:
        commands, audit['scrape_adjustment'] = adjust_scrape_height(commands, robot, args.surface_lift_mm/1000)
        audit['unchanged_suffix_frames'] = len(phases)-audit['scrape_adjustment']['adjusted_frames'][1]
    if args.contact_probe:
        # No saved intermediate cloth or velocity is consumed by the native
        # replay: its cloth still starts from independent initial geometry.
        touch = np.flatnonzero(phases == 'L_friction_touch')
        clear = np.flatnonzero(phases == 'L_friction_clear')
        first, stop = int(touch[0])+14, int(clear[-1])+1
        hold = int(meta['fps'])
        commands['joint_target'] = np.concatenate((np.repeat(commands['joint_target'][first:first+1], hold, axis=0),
                                                   commands['joint_target'][first+1:stop]))
        commands['phase'] = np.r_[np.repeat('contact_probe_settle', hold), phases[first+1:stop]]
        commands['time_s'] = np.arange(len(commands['phase']))/int(meta['fps'])
        audit['contact_probe'] = dict(initial_robot_command_frame=first, original_command_stop=stop,
                                      cloth_initialization='independent original geometry, no saved intermediate state')
        audit['unchanged_suffix_frames'] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f, **commands)
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

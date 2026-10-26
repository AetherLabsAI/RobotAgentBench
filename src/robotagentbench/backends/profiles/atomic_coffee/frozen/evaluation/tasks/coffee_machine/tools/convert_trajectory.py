"""Format conversion of one source episode's executed joint-target plan to the public joint20 recording.

Nothing is generated, smoothed, retimed or appended. The producer's plan already is one row per 20 Hz control step;
its executor kept every stationary joint at `initial_targets` and wrote, per row, the 7 joints of the active arm
(`q`) and the active gripper (`opening`) (run_pour.py:289-291, run_two_press.py, cup_place/run_selected.py:169-170).
This tool writes exactly those targets under the public joint names.

No value is changed (user 2026-09-30: action values may not be modified in any way). A producer target outside the
public interface (gripper targets are accepted in [0, pi/4], the joint limit) makes the trajectory unconvertible: the
tool refuses with the offending rows listed, and the replay is recorded as not run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from arena_api import bimanual as b, policy_execution as pe, robot_profiles as rp
from evaluation.tasks.coffee_machine.specs import TASK, EPISODES, spec, episodes

SCHEMA = 'arena.coffee_machine.unbound_candidate.v1'
GRIP_MIN, GRIP_MAX = 0., math.pi / 4
ARMS = dict(left=(b.LEFT_ARM, b.GRIPPERS[0]), right=(b.RIGHT_ARM, b.GRIPPERS[1]))


def convert(api, episode):
    spec(api)
    if episode not in episodes(api):
        raise ValueError('Not a packaged source episode of ' + api)
    folder = EPISODES / api / episode
    record = json.loads((folder / 'producer_plan.source.json').read_text())
    path = folder / 'producer_plan.json'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != record['sha256']:
        raise ValueError('Producer plan changed: ' + str(path))
    plan = json.loads(path.read_text())
    manifest = json.loads((folder / 'episode.json').read_text())
    joints = rp.get(rp.JOINT20).joints
    targets = {n: float(plan['initial_targets'][n]) for n in joints}
    if set(plan['initial_targets']) != set(joints):
        raise ValueError('Producer initial targets do not cover the public joint20 profile')
    arm = plan.get('active_arm', 'right')
    if arm != manifest['facts'].get('arm', arm) or arm not in ARMS:
        raise ValueError('Active arm mismatch')
    arm_joints, gripper = ARMS[arm]
    if len(b.GRIPPERS) != 2 or 'gripper_l' not in b.GRIPPERS[0] or 'gripper_r' not in b.GRIPPERS[1]:
        raise ValueError('Unexpected public gripper order')
    actions, outside = [], []
    for index, row in enumerate(plan['steps']):
        if len(row['q']) != 7:
            raise ValueError('Expected 7 arm joints per row')
        for name, value in zip(arm_joints, row['q']):
            targets[name] = float(value)
        opening = float(row['opening'])
        if not GRIP_MIN <= opening <= GRIP_MAX:
            outside.append((index, opening))
        targets[gripper] = opening
        # rows repeat the stationary torso / other arm; the executor never wrote them, they must equal the initial targets
        for key, name in (('bodylift', 'idx01_body_joint1'), ('waist', 'idx02_body_joint2')):
            if key in row and abs(float(row[key]) - targets[name]) > 1e-9:
                raise ValueError(f'Row {index}: moving {key} is not part of the producer executor')
        if 'left_q' in row and arm == 'right' and max(abs(float(v) - targets[n]) for n, v in zip(b.LEFT_ARM, row['left_q'])) > 1e-9:
            raise ValueError(f'Row {index}: moving left arm is not part of the producer executor')
        rp.values(targets, rp.JOINT20)
        actions.append(dict(targets))
    if outside:
        raise ValueError(f'{api} {episode}: {len(outside)} of {len(plan["steps"])} rows command the gripper outside the public interface '
                         f'[0, pi/4] (largest {max(v for _, v in outside):.4f} rad, first row {outside[0][0]}); action values are not modified, '
                         'so this trajectory cannot be replayed as recorded')
    phases = []
    for index, row in enumerate(plan['steps']):
        if not phases or phases[-1]['phase'] != row.get('phase'):
            phases.append(dict(phase=row.get('phase'), first_step=index))
    return dict(
        schema=SCHEMA, task_id=api, source_episode=episode,
        source=dict(path=str(path.relative_to(TASK)), sha256=sha, producer_file=record['source'], rows=len(plan['steps']), rate_hz=20,
                    active_arm=arm, producer_status_label=manifest['producer_status_label'], phases=phases),
        execution_profile=pe.spec(rp.JOINT20), actions=actions,
        adaptation=dict(method='one public action per producer row, no resampling', duration_s=len(actions) * b.CONTROL_DT,
                        generated_rows=0, final_hold_steps=0,
                        stationary_joints='producer initial_targets, as in the producer executor',
                        gripper_mapping='producer radians, unchanged; a target outside [0, pi/4] refuses the conversion',
                        values_modified=False),
        reset_binding_required=True, physical_acceptance=False, success=None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--api', required=True)
    p.add_argument('--episode', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to((TASK / 'artifacts').resolve()):
        raise ValueError('New task-owned output required')
    data = convert(a.api, a.episode)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(data, f, indent=1, allow_nan=False)
    print(json.dumps(dict(output=str(out), api=a.api, episode=a.episode, actions=len(data['actions']), adaptation=data['adaptation'])))


if __name__ == '__main__':
    main()

"""Explicitly approved 0.8-rad open-command adaptation; never source edits."""
import hashlib
import json
import math
from arena_api import bimanual as b, robot_profiles as rp, policy_execution as pe
from evaluation.tasks.coffee_machine.specs import EPISODES, episodes


def adapt(initial):
    episode = initial['source_episode']
    if initial['role'] != 'source' or episode not in episodes('1-2'):
        raise ValueError('Replay needs the exact packaged atomic source initial state')
    folder = EPISODES / '1-2' / episode
    path = folder / 'producer_plan.json'
    receipt = json.loads((folder / 'producer_plan.source.json').read_text())
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt['sha256']:
        raise ValueError('Source plan checksum mismatch')
    source = json.loads(raw)
    base = dict(source['initial_targets'])
    manifest = json.loads((folder / 'episode.json').read_text())
    # The source reset is a measured snapshot, not byte-identical to issued
    # drive targets. Verify the actual packaged reset instead of inventing a
    # zero tracking-error requirement or rewriting the recorded commands.
    if initial['robot_initialization'] != manifest['robot_initialization']:
        raise ValueError('Initialization does not match this packaged source episode')
    arm = source.get('active_arm', 'right')
    joints, grip = (b.LEFT_ARM, b.GRIPPERS[0]) if arm == 'left' else (b.RIGHT_ARM, b.GRIPPERS[1])
    if arm not in ('left', 'right'):
        raise ValueError('Unknown source arm')
    actions, changes, phases = [], [], []
    for index, row in enumerate(source['steps']):
        if len(row['q']) != 7:
            raise ValueError('Source arm dimensions changed')
        target = dict(base)
        target.update(zip(joints, map(float, row['q'])))
        opening = float(row['opening'])
        if math.pi / 4 < opening <= 0.8:
            changes.append(dict(step=index, joint=grip, source=opening, adapted=math.pi / 4))
            opening = math.pi / 4
        if not 0 <= opening <= math.pi / 4:
            raise ValueError('Unapproved source command needs modification')
        target[grip] = opening
        for key, name in (('bodylift', 'idx01_body_joint1'), ('waist', 'idx02_body_joint2')):
            if key in row and abs(float(row[key]) - base[name]) > 1e-9:
                raise ValueError('Moving stationary joint is not part of source executor')
        rp.values(target, rp.JOINT20)
        actions.append(target)
        phases.append(row.get('phase'))
    return dict(schema='arena.coffee_machine.authorized_adapted_replay.v1',
                source_episode=episode, source_plan_sha256=digest,
                source_plan_path=str(path), active_arm=arm, actions=actions, phases=phases,
                changed_commands=changes, changed_commands_count=len(changes),
                only_change='active-gripper open-command samples in (pi/4, 0.8] -> pi/4, including transition sample',
                arm_and_other_values_unchanged=True, source_file_modified=False,
                generated_rows=0, resampled=False, appended_hold_rows=0,
                control_hz=20, execution_profile=pe.spec(rp.JOINT20),
                original_unmodified_replay=False, scored_episode=False)

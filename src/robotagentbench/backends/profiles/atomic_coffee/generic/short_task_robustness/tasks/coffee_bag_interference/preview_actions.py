"""Explicitly authorized diagnostic, NOT a converted/repaired source replay.

Read only the source arm targets; keep both grippers and stationary joints at
their recorded reset values. Original plans and public interface stay intact.
"""
import hashlib
import json
from evaluation.tasks.coffee_machine.specs import EPISODES
from arena_api import bimanual,robot_profiles


def approach(initial):
    folder=EPISODES/'1-2'/initial['source_episode']
    receipt=json.loads((folder/'producer_plan.source.json').read_text())
    path=folder/'producer_plan.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=receipt['sha256']:raise ValueError('Source plan changed')
    source=json.loads(path.read_text());base=dict(initial['robot_initialization']['hold_targets'])
    actions=[]
    for row in source['steps'][:200]:
        target=dict(base)
        target.update({joint:float(value) for joint,value in zip(bimanual.RIGHT_ARM,row['q'])})
        robot_profiles.values(target,robot_profiles.JOINT20)
        actions.append(target)
    return dict(schema='arena.coffee_bag.diagnostic_approach.v1',actions=actions,
        original_plan_sha256=digest,source_arm_targets_unchanged=True,original_gripper_commands_not_replayed=True,
        grippers='recorded reset hold targets',rate_hz=20,scored_episode=False,
        authorized_by_user=True,original_plan_modified=False)

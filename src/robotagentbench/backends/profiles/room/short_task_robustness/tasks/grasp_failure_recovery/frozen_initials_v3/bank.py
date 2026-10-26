"""Training-row-only portable selection; no scene/data/solver imports."""
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from ..frozen_initials_v1.bank import checksum,validate as base_validate


def validate(value):
    base_validate(value)
    if value.get('selection_revision')!='remaining_exact_training_v3':raise ValueError('Explicit v3 training-only bank required')
    for variant in ('shelf_drink','coffee_bag','sofa'):
        episodes=set()
        for p in value['cases'][variant]['profiles']:
            pr=p['training_provenance'];hand=p['active_hand'];q=p['robot_initialization']['positions']
            if not pr['active_arm_copied_exactly'] or not pr['body_head_and_inactive_arm_from_same_training_row']:
                raise ValueError('No adapted/mirrored source arms in this bank')
            if [q[f'idx{2 if hand=="l" else 6}{i}_arm_{hand}_joint{i}'] for i in range(1,8)]!=pr['measured_active_arm_q_rad']:
                raise ValueError('Source active arm changed')
            if any(q[n]!=v for n,v in pr['source_state18_rad_or_m'].items()):raise ValueError('Source body/head/other arm changed')
            if pr['episode_index'] in episodes:raise ValueError('Adjacent frames are not distinct source poses')
            episodes.add(pr['episode_index'])
            if not p['validation']['protected_self_collision_checked']:raise ValueError('Original protected-pair check required')
            if not p['fit']['result']['accepted_cpu']:raise ValueError('A rejected raw pose cannot enter this bank')
            if p['validation']['native'] is not False:raise ValueError('Do not inherit original/native successes into a new pose')
            if variant=='coffee_bag' and hand!='r':raise ValueError('Original coffee operating hand required')
    return value


def load(path):return validate(json.loads(Path(path).read_text()))


def choose(bank,variant,seed,eligible):
    validate(bank)
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 seed required')
    candidates=[p for p in bank['cases'][variant]['profiles'] if p['id'] in eligible]
    if not candidates:raise ValueError('No frozen source pose covers this unchanged sampled scene; no IK/seed/target fallback')
    key=int.from_bytes(hashlib.sha256(f'exact-source-v3:{variant}:{seed}'.encode()).digest(),'big')
    return deepcopy(random.Random(key).choice(candidates))

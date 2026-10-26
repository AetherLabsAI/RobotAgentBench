"""Portable offline-adapted sofa contract; never imports IK, USD or datasets."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random


def checksum(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='content_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(value):
    if value.get('schema')!='failure_recovery.offline_adapted_sofa_pool.v1':raise ValueError('Explicit adapted sofa schema required')
    if value.get('content_sha256')!=checksum(value):raise ValueError('Adapted pool checksum mismatch')
    episodes=set();ids=set()
    for p in value['profiles']:
        pr=p['training_provenance'];q=p['robot_initialization']['positions'];h=p['robot_initialization']['hold_targets']
        if p['variant']!='sofa' or p['active_hand'] not in ('l','r'):raise ValueError('Sofa and independently checked hand required')
        if pr['episode_index'] in episodes or p['id'] in ids:raise ValueError('Distinct training episodes and profile IDs required')
        episodes.add(pr['episode_index']);ids.add(p['id'])
        if p['provenance_kind']!='training_active_arm_adapted_offline_NOT_raw_training_pose' or pr['active_arm_copied_exactly'] is not False:
            raise ValueError('Adapted arms must never be labeled raw training poses')
        hand=p['active_hand'];names={f'idx{2 if hand=="l" else 6}{i}_arm_{hand}_joint{i}' for i in range(1,8)}
        source=pr['source_state18_rad_or_m'];delta=pr['active_arm_delta_rad']
        if len(source)!=18 or set(delta)!=names:raise ValueError('Exact source18 and active-arm-only delta required')
        if len(q)!=34 or not set(h)<=set(q) or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in [*q.values(),*h.values()]):
            raise ValueError('Finite native34 required')
        if any(q[n]!=v for n,v in h.items()):raise ValueError('Hold/reset mismatch')
        if any(q[n]!=v for n,v in source.items() if n not in names):raise ValueError('Non-active source coordinates changed')
        if any(not math.isclose(q[n]-source[n],delta[n],rel_tol=0,abs_tol=1e-12) for n in names):raise ValueError('Incorrect source delta')
        if not p['cpu_accepted'] or not p['fit']['result']['accepted_cpu'] or not p['fit']['ik']['converged']:
            raise ValueError('Rejected fits cannot enter runtime selection')
        if not p['validation']['protected_self_collision_checked']:raise ValueError('Original protected-pair check required')
        if p['validation']['native'] is not False:raise ValueError('No inherited native acceptance')
    if value.get('online_ik_required') is not False or value.get('training_dataset_required_at_startup') is not False:
        raise ValueError('No runtime IK or training dependency allowed')
    return value


def load(path):return validate(json.loads(Path(path).read_text()))


def select(value,seed,eligible):
    validate(value)
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 seed required')
    candidates=[p for p in value['profiles'] if p['id'] in eligible]
    if not candidates:raise ValueError('No frozen adapted sofa covers this original room; no IK/seed/target fallback')
    key=int.from_bytes(hashlib.sha256(f'offline-adapted-sofa-v1:{seed}'.encode()).digest(),'big')
    return deepcopy(random.Random(key).choice(candidates))

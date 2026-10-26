"""Small, deterministic, immutable pose-bank contract. No USD/IK imports."""
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path

SCHEMA='failure_recovery.frozen_pose_bank.v1'
CASES=('shelf_drink','coffee_bag','sofa','table_waste_tipped_cup','table_waste_empty_grasp')


def checksum(value):
    body={k:v for k,v in value.items() if k!='content_sha256'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(value):
    if value.get('schema')!=SCHEMA or set(value.get('cases',{}))!=set(CASES):
        raise ValueError('Exact five-case frozen-bank schema required')
    if value.get('content_sha256')!=checksum(value):raise ValueError('Frozen-bank checksum changed')
    ids=[]
    for variant,case in value['cases'].items():
        for p in case['profiles']:
            ids.append(p['id'])
            if p['variant']!=variant or p.get('active_hand') not in ('l','r'):
                raise ValueError('Explicit variant and independently solved hand required')
            q=p['robot_initialization']['positions'];holds=p['robot_initialization']['hold_targets']
            if len(q)!=34 or not set(holds)<=set(q) or any(type(v) not in (int,float) or not math.isfinite(v) for v in (*q.values(),*holds.values())):
                raise ValueError('Complete finite native34 snapshot required')
            if any(v!=q[n] for n,v in holds.items()):raise ValueError('Reset and hold snapshot must agree')
            if p.get('cpu_accepted') is not True:raise ValueError('Do not export rejected solver diagnostics')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate frozen pose identity')
    return value


def load(path):return validate(json.loads(Path(path).read_text()))


def select(value,variant,seed,*,hand=None,miss_sign=None):
    validate(value)
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 episode seed required')
    if variant not in CASES:raise ValueError('Unknown recovery case')
    candidates=[p for p in value['cases'][variant]['profiles'] if
        (hand is None or p['active_hand']==hand) and
        (miss_sign is None or miss_sign in p.get('supported_miss_signs',[]))]
    if not candidates:raise ValueError('No frozen pose coverage; offline generation required; no online IK fallback')
    key=int.from_bytes(hashlib.sha256(f'frozen-pose-v1:{variant}:{seed}'.encode()).digest(),'big')
    return deepcopy(random.Random(key).choice(candidates))

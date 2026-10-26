"""Portable authored-side-cup bank; no USD, IK or training imports."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from ..frozen_initials_v1.bank import checksum


def validate(value):
    if value.get('schema')!='failure_recovery.authored_tipped_open.bank.v1' or value.get('content_sha256')!=checksum(value):
        raise ValueError('Unmodified authored tipped-open bank required')
    if value['physical_history_required'] or value['native_physics_tested'] or value['runtime_training_dataset_required'] or value['online_ik_calls']!=0:
        raise ValueError('No recorded drop, native success, runtime dataset or IK may be inherited')
    ids=set();episodes=set()
    for p in value['profiles']:
        pr=p['training_provenance'];q=p['robot_initialization']['positions'];holds=p['robot_initialization']['hold_targets']
        if p['id'] in ids or pr['episode_index'] in episodes:raise ValueError('Distinct profiles and source episodes required')
        ids.add(p['id']);episodes.add(pr['episode_index'])
        if p['variant']!='table_waste_tipped_cup' or p['active_hand'] not in ('l','r'):raise ValueError('Explicit tipped-cup hand required')
        if len(q)!=34 or not set(holds)<=set(q) or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in (*q.values(),*holds.values())):
            raise ValueError('Finite complete native34 required')
        if any(q[n]!=v for n,v in holds.items()):raise ValueError('Reset/hold mismatch')
        if len(pr['source_state18_rad_or_m'])!=18 or any(q[n]!=v for n,v in pr['source_state18_rad_or_m'].items()):raise ValueError('Source arm/body/head changed')
        opening=p['gripper_opening']['open_coordinates']
        if len(opening)!=8 or not any(opening.values()) or any(q[n]!=v for n,v in opening.items()):raise ValueError('Existing native-open linkage required')
        if not p['authored_failure_initial'] or p['physical_history_required'] or p['physical_failure_generation_verified']:
            raise ValueError('Authored start must not claim a physical drop replay')
        if not p['cpu_accepted'] or not p['fit']['accepted_cpu'] or p['validation']['native']:
            raise ValueError('Only CPU-accepted, not yet native-certified configurations')
    return value


def load(path):return validate(json.loads(Path(path).read_text()))


def choose(value,seed):
    validate(value)
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 seed required')
    if not value['profiles']:raise ValueError('No checked configuration; no solver fallback')
    key=int.from_bytes(hashlib.sha256(f'authored-tipped-open:{seed}'.encode()).digest(),'big')
    return deepcopy(random.Random(key).choice(value['profiles']))

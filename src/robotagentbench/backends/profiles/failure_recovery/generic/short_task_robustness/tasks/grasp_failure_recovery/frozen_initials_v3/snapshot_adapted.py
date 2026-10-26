"""Freeze completed, CPU-accepted profiles from an ongoing offline compiler.

The snapshot is explicitly partial, immutable, and not a native acceptance.
It never replaces the compiler's progress or final output.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from ..common import HERE,owned,digest,write_json
from .geometry import SOURCES
from .adapted_bank import checksum,validate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compile-key',required=True);p.add_argument('--key',required=True);a=p.parse_args()
    for key in (a.compile_key,a.key):
        if not key.replace('_','').isalnum():raise ValueError('Simple own key required')
    progress=HERE/'frozen_initials_v3/compile'/a.compile_key/'profiles_progress.json'
    data=json.loads(progress.read_text());profiles=deepcopy(data['profiles'])
    if not profiles:raise ValueError('No completed CPU-accepted profile to freeze')
    inventory=HERE/'frozen_initials_v3/mine/sofa_training58_v1/inventory.json'
    out=owned(HERE/'frozen_initials_v3/compile'/a.key,fresh=True);out.mkdir(parents=True)
    value=dict(schema='failure_recovery.offline_adapted_sofa_pool.v1',variant='sofa',profiles=profiles,
        requested=8,accepted=len(profiles),counts=data['counts'],partial_snapshot=True,
        source_progress_sha256=digest(progress),inventory_sha256=digest(inventory),
        source_initial_sha256=digest(HERE/SOURCES['sofa']),
        code_sha256={str(Path(__file__)):digest(__file__),str(HERE/'frozen_initials_v3/adapt_sofa.py'):digest(HERE/'frozen_initials_v3/adapt_sofa.py')},
        original_target_unchanged=True,physics_unchanged=True,native_physics_tested=False,
        runtime_registered=False,online_ik_required=False,training_dataset_required_at_startup=False)
    value['content_sha256']=checksum(value);validate(value);write_json(out/'pool.json',value)
    print(json.dumps(dict(count=len(profiles),partial_snapshot=True,pool_sha256=digest(out/'pool.json'))),flush=True)

"""Fast sofa startup from an offline-adapted pool; no optimizer or dataset reads."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
from ..common import HERE,owned,digest,physics,finish,write_json
from ..frozen_initials_v1.cup import base_matrix
from . import adapted_bank
from .geometry import Gate,existing


def build(pool_path,output,seed,*,profile_id=None):
    from pxr import Usd
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    path=owned(pool_path);bank=adapted_bank.load(path);out=owned(output,fresh=True);out.mkdir(parents=True)
    source_stage,record,source=existing('sofa')
    if digest(source)!=bank['source_initial_sha256']:raise ValueError('Frozen source scene context changed')
    stage=Usd.Stage.CreateNew(str(out/'initial.usda'));stage.GetRootLayer().subLayerPaths=[os.path.relpath(record['scene'],out)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));before=physics(stage);gate=Gate(stage,record)
    checks={};stations={}
    for profile in bank['profiles']:
        if profile_id is not None and profile['id']!=profile_id:continue
        index=profile['fit']['station']['candidate_index']
        station=next((r for r in gate.bases if r['candidate_index']==index),None)
        if station is None:checks[profile['id']]=dict(accepted_cpu=False,reason='station_not_in_original_domain');continue
        stations[profile['id']]=station
        checks[profile['id']]=gate.check(profile['robot_initialization'],profile['active_hand'],base=base_matrix(station['initial_base_xyyaw']))
    write_json(out/'frozen_pose_eligibility.json',checks)
    profile=adapted_bank.select(bank,seed,{k for k,v in checks.items() if v['accepted_cpu']})
    station=stations[profile['id']];base_receipt=initialize_base_pose(stage,station['initial_base_xyyaw'])
    record.update(robot_initialization=deepcopy(profile['robot_initialization']),active_hand=profile['active_hand'],
        gripper_closure=profile['gripper_closure'],initial_base_xyyaw=station['initial_base_xyyaw'],robot_base_initialization=base_receipt,
        frozen_profile_id=profile['id'],training_pose_provenance=profile['training_provenance'],
        initialization_policy='offline_adapted_sofa_frozen_bank.v1',provenance_kind=profile['provenance_kind'],
        seed=seed,room_seed=record['seed'],fresh_full_room_generated=False,preparation_accepted=True,
        new_episode_native_acceptance_inherited=False,source_motion_read=False,source_checkpoint=None,
        online_ik_calls=0,generation_reads_recorded_trajectory=False,runtime_training_dataset_required=False,
        object_relocated_for_ik=False,seed_retried=False,procedural_grasp=checks[profile['id']],
        code_sources={str(p):digest(p) for p in (Path(__file__),path,Path(adapted_bank.__file__),HERE/'frozen_initials_v3/geometry.py')})
    return finish(stage,out,record,before)


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--profile-id');a=p.parse_args();started=time.monotonic();calls=[];blocked=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if path.startswith('/training_dataset') or path.endswith(('.npz','.parquet')):
                blocked.append(path);raise RuntimeError('Frozen runtime cannot read training/trajectory files')
    def trace(frame,event,arg):
        if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py')) or
            (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
            calls.append(frame.f_code.co_name);raise RuntimeError('No startup IK allowed')
    sys.addaudithook(guard);sys.setprofile(trace)
    try:value=build(a.pool,a.output,a.seed,profile_id=a.profile_id)
    finally:sys.setprofile(None)
    write_json(owned(a.output)/'runtime_dependency_audit.json',dict(optimizer_calls=calls,blocked_training_or_trajectory_read_attempts=blocked,
        training_dataset_mounted=False,generation_wall_seconds=time.monotonic()-started,fresh_full_room_generated=False))
    print(json.dumps(dict(profile=value['frozen_profile_id'],seed=a.seed,online_ik_calls=0)),flush=True)

"""Task-local no-IK startup, original object draws plus eligible frozen poses."""
import argparse
import json
import os
from pathlib import Path
import time
from copy import deepcopy
from ..common import HERE,owned,physics,finish,digest,write_json
from . import bank as contract,geometry


def sofa_context(out,base_key):
    from pxr import Usd,UsdGeom
    from ..expansion_v2.tipped_initial import load_base
    from ..procedural_grasp import complete_closed_reset
    if not base_key:raise ValueError('Explicit fresh original sofa base preparation required')
    _,launch,manifest,bound,ready=load_base(base_key,'sofa_to_coffee_table')
    rows=[r for r in manifest['objects'] if r.get('task_id')=='sofa_to_coffee_table']
    if bound['asset']!='remote_tv' or len(rows)!=1 or rows[0]['id']!='remote_tv':raise ValueError('Original TV remote target required')
    out.mkdir(parents=True);stage=Usd.Stage.CreateNew(str(out/'initial.usda'));stage.GetRootLayer().subLayerPaths=[os.path.relpath(manifest['scene'],out)]
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    initialize_base_pose(stage,launch['initial_base_xyyaw'])
    initial,_=complete_closed_reset(stage,dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad']),'l')
    record=dict(variant='sofa',seed=manifest['seed'],target=rows[0]['path'],
        robot_initialization=initial,original_launch=launch,original_bindings=bound,
        maximum_score=bound['score_policy']['maximum'],
        completion_policy='same_target_released_in_permitted_coffee_table_object_region',code_sources={})
    return stage,record,physics(stage)


def build(variant,seed,output,bank_path,*,coffee_context=None,base_key=None):
    out=owned(output,fresh=True)
    path=owned(bank_path);bank=contract.load(path)
    if not bank['cases'][variant]['profiles']:raise ValueError('No compatible exact source pose accepted for this variant; legacy pose not silently substituted')
    started=time.monotonic()
    stage,record,before=(sofa_context(out,base_key) if variant=='sofa' else geometry.compose(variant,out,seed,coffee_context=coffee_context))
    record['room_seed']=record['seed'];record['seed']=seed
    gate=geometry.Gate(stage,record);checks={};stations={}
    for profile in bank['cases'][variant]['profiles']:
        if variant=='sofa':
            from ..frozen_initials_v1.cup import base_matrix
            index=profile['fit']['station']['candidate_index']
            row=next((r for r in gate.bases if r['candidate_index']==index),None)
            if row is None:checks[profile['id']]=dict(accepted_cpu=False,reason='original_station_index_not_available_in_new_room');continue
            stations[profile['id']]=row
            checks[profile['id']]=gate.check(profile['robot_initialization'],profile['active_hand'],base=base_matrix(row['initial_base_xyyaw']))
        else:checks[profile['id']]=gate.check(profile['robot_initialization'],profile['active_hand'])
    eligible={key for key,value in checks.items() if value['accepted_cpu']}
    write_json(out/'frozen_pose_eligibility.json',dict(seed=seed,checks=checks,eligible=sorted(eligible),
        online_ik_calls=0,target_redrawn=False,robot_q_interpolated=False))
    try:profile=contract.choose(bank,variant,seed,eligible)
    except ValueError:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(seed=seed,variant=variant,reason='no_source_pose_covers_original_draw',
            online_ik_fallback=False,seed_retried=False,target_relocated=False));raise
    record.update(robot_initialization=deepcopy(profile['robot_initialization']),active_hand=profile['active_hand'],
        gripper_closure=profile['gripper_closure'],frozen_profile_id=profile['id'],training_pose_provenance=profile['training_provenance'],
        initialization_policy='exact_training_bank_original_object_sampler.v3',preparation_accepted=True,
        source_checkpoint=None,generation_reads_recorded_trajectory=False,runtime_training_dataset_required=False,
        source_motion_read=False,online_ik_calls=0,seed_retried=False,object_relocated_for_ik=False,
        new_episode_native_acceptance_inherited=False,procedural_grasp=checks[profile['id']])
    if variant=='sofa':
        from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
        station=stations[profile['id']];record['initial_base_xyyaw']=station['initial_base_xyyaw']
        record['robot_base_initialization']=initialize_base_pose(stage,record['initial_base_xyyaw'])
        record['station_policy']=gate.station_domain
    from . import inspect
    from .. import common,procedural_grasp,shelf_checkpoint
    from ..expansion_v2 import camera_projection_complete,empty_grasp
    files=[Path(__file__),path,*[Path(m.__file__) for m in (contract,geometry,inspect,common,procedural_grasp,shelf_checkpoint,camera_projection_complete,empty_grasp)]]
    record.setdefault('code_sources',{}).update({str(p):digest(p) for p in files})
    value=finish(stage,out,record,before)
    if variant!='sofa':
        from ..visual_preload import apply
        value=apply(stage,out,value)
    value['generation_wall_seconds']=time.monotonic()-started;write_json(out/'initial.json',value)
    return value


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('shelf_drink','coffee_bag','sofa'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--bank',required=True);p.add_argument('--coffee-context');p.add_argument('--base-key');a=p.parse_args()
    calls=[];blocked_reads=[]
    def read_guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if path.startswith('/training_dataset') or path.endswith(('.npz','.parquet')):
                blocked_reads.append(path);raise RuntimeError('Runtime training/trajectory reads forbidden')
    sys.addaudithook(read_guard)
    def audit(frame,event,arg):
        if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py')) or
            (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
            calls.append(frame.f_code.co_name);raise RuntimeError('No online IK permitted in frozen startup')
    sys.setprofile(audit)
    try:value=build(a.variant,a.seed,a.output,a.bank,coffee_context=a.coffee_context,base_key=a.base_key)
    finally:sys.setprofile(None)
    write_json(owned(a.output)/'runtime_dependency_audit.json',dict(optimizer_calls=calls,training_dataset_mounted=False,
        fresh_full_room_generated=False,original_object_sampler_used=True,blocked_training_or_trajectory_read_attempts=blocked_reads))
    print(json.dumps(dict(seed=value['seed'],profile=value['frozen_profile_id'],generation_wall_seconds=value['generation_wall_seconds'])))

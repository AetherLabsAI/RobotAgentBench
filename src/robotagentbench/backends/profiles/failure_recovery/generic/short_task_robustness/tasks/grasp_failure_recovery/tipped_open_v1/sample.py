"""Choose frozen post-lift pose, native-open hand, place side cup beneath it."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
from ..common import HERE,owned,digest,physics,finish,write_json
from ..frozen_initials_v1.cup import context
from . import bank,geometry


def build(base_key,bank_path,output,seed):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from short_tasks.agent_probe_v1.collision_guard.stage_model import from_stage
    from short_tasks.agent_probe_v1.collision_guard.reactive_gate import ReactiveGate
    from ..expansion_v2.operating_station import RoomGate
    path=owned(bank_path);value=bank.load(path);profile=bank.choose(value,seed)
    definition=HERE/'tipped_open_v1/definition.json'
    if digest(definition)!=value['definition_sha256']:raise ValueError('Authored initial contract changed')
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out);before=physics(c['stage'])
    base=c['delta'].T@np.asarray(profile['base_canonical_column'])
    xyyaw=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
    base_receipt=initialize_base_pose(c['stage'],xyyaw);initial=deepcopy(profile['robot_initialization'])
    primary='idx41_gripper_l_outer_joint1' if profile['active_hand']=='l' else 'idx81_gripper_r_outer_joint1'
    if initial['positions'][primary]!=c['ready']['targets_rad'][primary]:raise ValueError('Existing native ready/open configuration changed')
    gate=RoomGate(c['stage'],c['manifest']);gate.obstacles=[o for o in gate.obstacles if not(o[0]==c['target'] or o[0].startswith(c['target']+'/'))]
    protected=ReactiveGate(from_stage(c['stage']));result=geometry.check(c,initial,profile['active_hand'],xyyaw,gate,protected)
    if not result['accepted_cpu']:
        c['stage'].GetRootLayer().Save();write_json(out/'rejection.json',dict(result,profile_id=profile['id'],seed=seed,
            seed_retried=False,robot_pose_replaced=False,online_ik_fallback=False));raise ValueError('Selected authored initial rejected: '+str(result['reason']))
    binding=c['bound']['runtime_components'][0]['parameters']
    from .. import common,poses,procedural_grasp
    from ..frozen_initials_v1 import cup
    from ..expansion_v2 import operating_station,camera_projection_complete,arm_first_visible_cup_v4
    sources=[Path(__file__),path,definition,Path(bank.__file__),Path(geometry.__file__),*c['references']]
    sources += [Path(m.__file__) for m in (common,poses,procedural_grasp,cup,operating_station,camera_projection_complete,arm_first_visible_cup_v4)]
    record=dict(variant='table_waste_tipped_cup',seed=seed,room_seed=c['manifest']['seed'],fresh_full_room_generated=False,
        target=c['target'],destination=binding['destination_path'],category='cup',asset='paper_cup',
        failure_kind='authored_open_hand_above_tipped_cup',initialization_contract='authored_tipped_open.v1',
        original_bindings=c['bound'],original_launch=c['launch'],original_request=c['launch']['request'],
        robot_initialization=initial,active_hand=profile['active_hand'],gripper_opening=profile['gripper_opening'],
        initial_base_xyyaw=xyyaw,robot_base_initialization=base_receipt,
        maximum_score=c['bound']['score_policy']['maximum'],completion_policy='same_cup_released_inside_correct_bin',
        frozen_profile_id=profile['id'],training_pose_provenance=profile['training_provenance'],
        preparation_accepted=True,original_geometry_checker_passed=True,cpu_initial_checks=result,
        original_target_world=c['original'].tolist(),proposed_target_world=result['proposed_target_world'],
        authored_failure_initial=True,cup_side_pose_authored=True,physical_history_required=False,
        physical_failure_generation_verified=False,source_grasp_lift_release_replayed=False,
        source_motion_read=False,source_checkpoint=None,runtime_training_dataset_required=False,online_ik_calls=0,
        seed_retried=False,robot_joint_interpolation=False,new_episode_native_acceptance_inherited=False,
        code_sources={str(p):digest(p) for p in sources})
    final=finish(c['stage'],out,record,before)
    final['layout_scope']='original_full_room_authored_side_cup_below_frozen_open_hand'
    write_json(out/'initial.json',final);return final


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();start=time.monotonic();calls=[];blocked=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if path.startswith('/training_dataset') or path.endswith(('.npz','.parquet')):blocked.append(path);raise RuntimeError('Runtime source data reads forbidden')
    def trace(frame,event,arg):
        if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py')) or
            (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
            calls.append(frame.f_code.co_name);raise RuntimeError('Runtime IK forbidden')
    sys.addaudithook(guard);sys.setprofile(trace)
    try:v=build(a.base_key,a.bank,a.output,a.seed)
    finally:sys.setprofile(None)
    write_json(owned(a.output)/'runtime_dependency_audit.json',dict(optimizer_calls=calls,training_dataset_mounted=False,
        blocked_training_or_trajectory_reads=blocked,physical_drop_replayed=False,generation_wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(profile=v['frozen_profile_id'],seed=a.seed,preparation_accepted=True,physical_settle_verified=False)),flush=True)

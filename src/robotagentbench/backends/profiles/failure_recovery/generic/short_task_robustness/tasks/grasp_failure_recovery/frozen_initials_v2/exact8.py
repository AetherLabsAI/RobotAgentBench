"""Approved eight exact training arm/body snapshots; no IK or dataset at runtime."""
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..common import HERE,owned,digest,physics,finish,write_json
from ..frozen_initials_v1 import bank as contract,cup as helpers
from ..expansion_v2.empty_grasp import draw

IDS=('train_ep1450_f70_l','train_ep1472_f70_l','train_ep1465_f80_l','train_ep1491_f70_l',
     'train_ep1474_f71_l','train_ep1481_f80_l','train_ep1463_f80_l','train_ep1476_f70_l')
DEFAULT_BANK=HERE/'frozen_initials_v2/compile/cup_training_exact8_v6/bank_v2.json'


def checked_bank(path=DEFAULT_BANK):
    bank=contract.load(path);profiles=bank['cases']['table_waste_empty_grasp']['profiles']
    if len(profiles)!=8 or {p['id'] for p in profiles}!=set(IDS):raise ValueError('Only the approved exact eight training profiles')
    if bank.get('runtime_training_dataset_required') is not False or bank.get('online_ik_fallback') is not False:
        raise ValueError('Portable bank without runtime solver required')
    for p in profiles:
        pr=p['training_provenance']
        if p['active_hand']!='l' or not pr['active_arm_copied_exactly'] or not pr['body_head_and_inactive_arm_from_same_measured_row']:
            raise ValueError('Exact same-row training arm/body snapshot required')
        q=p['robot_initialization']['positions']
        if [q[f'idx2{i}_arm_l_joint{i}'] for i in range(1,8)]!=pr['measured_active_arm_q_rad']:
            raise ValueError('Training arm altered')
        if len(p['supported_miss_signs'])!=1:raise ValueError('Use independently checked placement side only')
    return bank


def selection(bank,seed):
    # Select the approved posture first; cup draw follows its already checked side.
    # No seed retries, mirrored poses, joint jitter or runtime optimization.
    profile=contract.select(bank,'table_waste_empty_grasp',seed)
    request=draw('table_waste_empty_grasp',seed)
    request['miss_sign']=profile['supported_miss_signs'][0]
    request['side_policy']='selected_frozen_profile_prevalidated_side'
    return profile,request


def build(base_key,output,*,seed=None,bank_path=None):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..expansion_v2.catalog import definition
    start=time.monotonic();out=owned(output,fresh=True);out.mkdir(parents=True)
    path=owned(bank_path or DEFAULT_BANK);bank=checked_bank(path)
    c=helpers.context(base_key,out);stage=c['stage'];before=physics(stage)
    seed=c['manifest']['seed'] if seed is None else seed
    profile,request=selection(bank,seed)
    base=c['delta'].T@np.asarray(profile['base_canonical_column'])
    if not np.isclose(base[2,3],0.,atol=1e-7):raise ValueError('Frozen station/current floor mismatch')
    xyyaw=[float(base[0,3]),float(base[1,3]),math.atan2(base[1,0],base[0,0])]
    base_receipt=initialize_base_pose(stage,xyyaw);initial=deepcopy(profile['robot_initialization'])
    result=helpers.place_and_check(c,initial,profile['active_hand'],xyyaw,request)
    if not result['accepted_cpu']:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(result,seed=seed,profile=profile['id'],
            online_ik_fallback=False,seed_retried=False,offline_bank_extension_required=True))
        raise ValueError('Exact training posture rejected in actual room: '+str(result['room_rejection']))
    from .. import common,procedural_grasp,poses
    from ..expansion_v2 import empty_grasp,arm_first_visible_cup_v4,camera_projection_complete,operating_station
    from short_tasks.tasks.robot_pose_alignment_v1 import geometry
    sources=[Path(__file__),Path(contract.__file__),Path(helpers.__file__),path,*c['references']]
    sources += [Path(m.__file__) for m in (common,procedural_grasp,poses,empty_grasp,arm_first_visible_cup_v4,
        camera_projection_complete,operating_station,geometry)]
    record=dict(variant='table_waste_empty_grasp',seed=seed,room_seed=c['manifest']['seed'],target=c['target'],
        active_hand=profile['active_hand'],robot_initialization=initial,initial_base_xyyaw=xyyaw,
        robot_base_initialization=base_receipt,original_launch=c['launch'],original_bindings=c['bound'],
        original_target_world=c['original'].tolist(),proposed_target_world=result['proposed_target_world'],
        maximum_score=c['bound']['score_policy']['maximum'],completion_policy=definition('table_waste_empty_grasp')['completion'],
        preparation_accepted=True,original_geometry_checker_passed=True,
        initialization_policy='exact8_training_native34_then_continuous_analytic_cup_placement.v6',
        frozen_profile_id=profile['id'],frozen_template_validation=profile['validation'],
        training_pose_provenance=profile['training_provenance'],new_episode_native_acceptance_inherited=False,
        procedural_grasp=result,source_motion_read=False,source_checkpoint=None,runtime_training_dataset_required=False,
        online_ik_calls=0,robot_joint_interpolation=False,independent_joint_jitter=False,
        arm_changed_after_cup_placement=False,cup_translation_only=True,original_cup_placement_region_preserved=True,
        seed_retried=False,posture_selected_before_cup=True,code_sources={str(p):digest(p) for p in sources})
    value=finish(stage,out,record,before);value.update(layout_scope='full_original_room_with_fresh_continuous_cup_draw',
        generation_wall_seconds=time.monotonic()-start)
    write_json(out/'initial.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int);a=p.parse_args();v=build(a.base_key,a.output,seed=a.seed)
    print(json.dumps({k:v[k] for k in ('seed','active_hand','frozen_profile_id','online_ik_calls','generation_wall_seconds')}))

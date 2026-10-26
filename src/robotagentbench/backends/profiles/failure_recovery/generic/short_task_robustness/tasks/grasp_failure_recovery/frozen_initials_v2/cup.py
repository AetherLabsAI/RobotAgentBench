"""Fast v2 reset: frozen source posture, then continuous analytic cup position."""
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

DEFAULT_BANK=HERE/'frozen_initials_v2/compile/cup_training_adapt_v5/bank_v2.json'


def build(base_key,output,*,seed=None,bank_path=None):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..expansion_v2.catalog import definition
    start=time.monotonic();out=owned(output,fresh=True);out.mkdir(parents=True)
    c=helpers.context(base_key,out);stage=c['stage'];before=physics(stage)
    seed=c['manifest']['seed'] if seed is None else seed
    path=owned(bank_path or DEFAULT_BANK)
    if not path.is_relative_to(HERE/'frozen_initials_v2/compile'):raise ValueError('Only task-owned offline-compiled banks')
    bank=contract.load(path)
    if bank.get('version')!='training_expansion_v2' or bank.get('runtime_training_dataset_required') is not False:
        raise ValueError('Explicit portable training expansion bank required')
    request=draw('table_waste_empty_grasp',seed)
    profile=contract.select(bank,'table_waste_empty_grasp',seed,miss_sign=request['miss_sign'])
    base=c['delta'].T@np.asarray(profile['base_canonical_column'])
    if not np.isclose(base[2,3],0.,atol=1e-7):raise ValueError('Frozen station/current floor mismatch')
    xyyaw=[float(base[0,3]),float(base[1,3]),math.atan2(base[1,0],base[0,0])]
    base_receipt=initialize_base_pose(stage,xyyaw);initial=deepcopy(profile['robot_initialization'])
    result=helpers.place_and_check(c,initial,profile['active_hand'],xyyaw,request)
    if not result['accepted_cpu']:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(result,seed=seed,profile=profile['id'],
            online_ik_fallback=False,seed_retried=False,offline_bank_extension_required=True))
        raise ValueError('Frozen source posture rejected in actual room: '+str(result['room_rejection']))
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
        initialization_policy='frozen_source_native34_then_continuous_analytic_cup_placement.v2',
        frozen_profile_id=profile['id'],frozen_template_validation=profile['validation'],
        training_pose_provenance=profile.get('training_provenance'),new_episode_native_acceptance_inherited=False,
        procedural_grasp=result,source_motion_read=False,source_checkpoint=None,runtime_training_dataset_required=False,
        online_ik_calls=0,robot_joint_interpolation=False,independent_joint_jitter=False,
        arm_changed_after_cup_placement=False,cup_translation_only=True,original_cup_placement_region_preserved=True,
        seed_retried=False,code_sources={str(p):digest(p) for p in sources})
    value=finish(stage,out,record,before);value.update(layout_scope='full_original_room_with_fresh_continuous_cup_draw',
        generation_wall_seconds=time.monotonic()-start)
    write_json(out/'initial.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int);p.add_argument('--bank');a=p.parse_args();v=build(a.base_key,a.output,seed=a.seed,bank_path=a.bank)
    print(json.dumps({k:v[k] for k in ('seed','active_hand','frozen_profile_id','online_ik_calls','generation_wall_seconds')}))

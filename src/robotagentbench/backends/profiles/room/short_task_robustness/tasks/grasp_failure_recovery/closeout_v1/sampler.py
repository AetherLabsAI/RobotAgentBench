"""Existing object samplers plus approved frozen robot banks, no runtime IK."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from ..common import HERE,CURRENT,owned,digest,physics,finish,write_json

def room_base(variant,seed,out):
    from ..expansion_v2.catalog import base_request
    from ..expansion_v2.prepare_base import NAMESPACE
    from short_tasks.agent_probe_v1 import prepare_task
    from short_tasks.scene import verify_scene_sources
    from openend.tasks.layout_repair_v1 import ready_envelope
    key='recovery5_'+hashlib.sha256(str(out).encode()).hexdigest()[:20]
    request=base_request(variant,seed,sofa_asset='remote_tv' if variant=='sofa' else None,
        waste_categories=None if variant=='sofa' else ['cup'])
    reference=ready_envelope.REFERENCE/'navigation_result.json'
    calibration=CURRENT/'short_tasks/agent_probe_v1/completion_trash43_v1/native/launch.json'
    write_json(out/'base_ready.json',json.loads(reference.read_text())['navigation_posture'])
    write_json(out/'base_sensor.json',dict(sensor=json.loads(calibration.read_text())['sensor']))
    verify_scene_sources()
    prepare_task.prepare(request,out/'base_ready.json',out/'base_sensor.json',NAMESPACE/key/'base')
    write_json(out/'base_receipt.json',dict(key=key,seed=seed,original_request=request,full_source_only=True))
    return key

def atomic(variant,seed,out,*,diagnostic_profile_id=None):
    from ..frozen_initials_v3 import bank,geometry
    from ..procedural_initial import compose_shelf
    from ..procedural_coffee import compose
    from ..visual_preload import apply
    cfg_path=HERE/'procedural_config.json';cfg=json.loads(cfg_path.read_text());cfg['config_path']=str(cfg_path)
    bank_path=HERE/'frozen_initials_v3/banks/remaining_exact_training_v1/bank.json';pool=bank.load(bank_path)
    stage,record,before=(compose_shelf(out,seed,cfg) if variant=='shelf_drink' else compose(out,seed,cfg))
    gate=geometry.Gate(stage,record)
    checks={p['id']:gate.check(p['robot_initialization'],p['active_hand']) for p in pool['cases'][variant]['profiles']}
    write_json(out/'frozen_pose_eligibility.json',checks)
    eligible={k for k,v in checks.items() if v['accepted_cpu']}
    if diagnostic_profile_id is None:selected=bank.choose(pool,variant,seed,eligible)
    else:
        if diagnostic_profile_id not in eligible:raise ValueError('Requested frozen profile rejected in its original fit context')
        selected=next(p for p in pool['cases'][variant]['profiles'] if p['id']==diagnostic_profile_id)
    record.update(robot_initialization=deepcopy(selected['robot_initialization']),active_hand=selected['active_hand'],
        gripper_closure=selected['gripper_closure'],frozen_profile_id=selected['id'],training_pose_provenance=selected['training_provenance'],
        initialization_policy='exact_training_bank_original_object_sampler.v3',preparation_accepted=True,
        generation_reads_recorded_trajectory=False,runtime_training_dataset_required=False,source_motion_read=False,
        source_checkpoint=None,online_ik_calls=0,seed_retried=False,object_relocated_for_ik=False,
        new_episode_native_acceptance_inherited=False,procedural_grasp=checks[selected['id']])
    record['code_sources'].update({str(p):digest(p) for p in (Path(__file__),bank_path,cfg_path)})
    return apply(stage,out,finish(stage,out,record,before))

def sofa(base_key,seed,out):
    from ..frozen_initials_v3.fast import sofa_context
    from ..frozen_initials_v3 import adapted_bank
    from ..frozen_initials_v3.geometry import Gate
    from ..frozen_initials_v1.cup import base_matrix
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    path=HERE/'frozen_initials_v3/compile/sofa_offline_adapt_v3/pool.json';pool=adapted_bank.load(path)
    stage,record,before=sofa_context(out,base_key);gate=Gate(stage,record);checks={};stations={}
    for p in pool['profiles']:
        idx=p['fit']['station']['candidate_index'];station=next((r for r in gate.bases if r['candidate_index']==idx),None)
        if station is None:checks[p['id']]=dict(accepted_cpu=False,reason='station_not_in_original_domain');continue
        stations[p['id']]=station;checks[p['id']]=gate.check(p['robot_initialization'],p['active_hand'],base=base_matrix(station['initial_base_xyyaw']))
    write_json(out/'frozen_pose_eligibility.json',checks)
    p=adapted_bank.select(pool,seed,{k for k,v in checks.items() if v['accepted_cpu']});station=stations[p['id']]
    record.update(robot_initialization=deepcopy(p['robot_initialization']),active_hand=p['active_hand'],gripper_closure=p['gripper_closure'],
        initial_base_xyyaw=station['initial_base_xyyaw'],robot_base_initialization=initialize_base_pose(stage,station['initial_base_xyyaw']),
        frozen_profile_id=p['id'],training_pose_provenance=p['training_provenance'],provenance_kind=p['provenance_kind'],
        initialization_policy='offline_adapted_sofa_frozen_bank.v1',room_seed=record['seed'],seed=seed,
        preparation_accepted=True,new_episode_native_acceptance_inherited=False,source_motion_read=False,source_checkpoint=None,
        online_ik_calls=0,generation_reads_recorded_trajectory=False,runtime_training_dataset_required=False,
        object_relocated_for_ik=False,seed_retried=False,procedural_grasp=checks[p['id']],
        code_sources={str(p):digest(p) for p in (Path(__file__),path,Path(adapted_bank.__file__),HERE/'frozen_initials_v3/geometry.py')})
    return finish(stage,out,record,before)

def build(variant,seed,out):
    out=owned(out,fresh=True)
    if variant in ('shelf_drink','coffee_bag'):return atomic(variant,seed,out)
    metadata=out.parent/'base_generation';metadata.mkdir(parents=True,exist_ok=False)
    key=room_base(variant,seed,metadata)
    if variant=='sofa':return sofa(key,seed,out)
    if variant=='table_waste_empty_grasp':
        from ..frozen_initials_v2.exact8 import build as cup
        return cup(key,out,seed=seed)
    if variant=='table_waste_tipped_cup':
        from ..tipped_open_v1.sample import build as tipped
        return tipped(key,HERE/'tipped_open_v1/compile/elevated_open_v2/bank.json',out,seed)
    raise ValueError('Only the five approved variants')

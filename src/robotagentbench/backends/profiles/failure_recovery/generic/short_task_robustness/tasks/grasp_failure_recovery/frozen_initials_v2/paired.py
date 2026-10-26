"""Offline source-pose selection; validate each natural pose's two miss sides."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,digest,physics
from ..frozen_initials_v1.bank import load,checksum,validate
from ..frozen_initials_v1.cup import context,place_and_check
from .compile import diverse_order


def build(base_key,inventory,output):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..poses import place
    from ..expansion_v2.operating_station import RoomGate
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out);before=physics(c['stage'])
    data=json.loads(Path(inventory).read_text());old=HERE/'frozen_initials_v1/configurations/bank_v1.json'
    bank=deepcopy(load(old));case=bank['cases']['table_waste_empty_grasp'];original=deepcopy(case['profiles'])
    accepted=[];attempts=[];gate=RoomGate(c['stage'],c['manifest'])
    for hand in ('l','r'):
        used_episodes=set();chosen=[]
        rows=[r for r in data['rows'] if r.get('reset_accepted') and r['active_hand']==hand and r['vertical_envelope_separation_m']==0]
        for row in diverse_order(rows):
            if row['episode_index'] in used_episodes:continue
            if any(np.array_equal(row['arm_q_rad'],r['arm_q_rad']) for r in chosen):continue
            base=c['delta'].T@np.asarray(row['base_canonical_column']);xyyaw=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
            initialize_base_pose(c['stage'],xyyaw);valid_signs=[];all_checks=[]
            for sign in (-1,1):
                checks=[]
                for gap in (.020001,.035,.049999):
                    request=dict(gap_m=gap,miss_sign=sign,orientation_jitter=False,axis='robot_local_y',offline_probe=True)
                    result=place_and_check(c,row['robot_initialization'],hand,xyyaw,request,gate=gate);checks.append(result)
                    if not result['accepted_cpu']:break
                all_checks.append(dict(miss_sign=sign,checks=checks))
                if len(checks)==3 and all(r['accepted_cpu'] for r in checks):valid_signs.append(sign)
                place(c['stage'],c['target'],c['original'])
            attempts.append(dict(id=row['id'],hand=hand,signs=all_checks));write_json(out/'attempts.json',attempts)
            if not valid_signs:continue
            used_episodes.add(row['episode_index']);chosen.append(row)
            accepted.append(dict(id=row['id'],variant='table_waste_empty_grasp',active_hand=hand,
                robot_initialization=deepcopy(row['robot_initialization']),base_canonical_column=row['base_canonical_column'],
                initial_base_xyyaw=xyyaw,supported_miss_signs=valid_signs,asset_id='paper_cup',cpu_accepted=True,
                validation=dict(cpu=True,native=False,random_revalidation_required=True,gap_probes_m=[.020001,.035,.049999],
                    whole_target_in_original_camera_frustum=True),source_motion_needed=False,
                maximum_score=original[0]['maximum_score'],completion_policy=original[0]['completion_policy'],
                source_record=str(inventory),source_record_sha256=digest(inventory),
                training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],benchmark_task_id=row['benchmark_task_id'],
                    row=row['source_row'],measured_active_arm_q_rad=row['arm_q_rad'],active_arm_copied_exactly=True,
                    body_head_and_inactive_arm_from_same_measured_row=True,active_gripper_closed_empty=True,
                    inactive_gripper='current_native_ready_linkage',source_success_not_inferred=True),
                gripper_closure=row['gripper_closure'],anchor='canonical_actual_coffee_table_geometry'))
            print(json.dumps(dict(accepted=row['id'],hand=hand,hand_count=len(chosen),signs=valid_signs)),flush=True)
            if len(chosen)==8:break
    if physics(c['stage'])!=before:raise ValueError('Physical parameters changed')
    if not accepted:raise ValueError('No training poses passed')
    case['profiles']=original+accepted;case['hand_coverage']=sorted({p['active_hand'] for p in case['profiles']})
    bank.update(version='training_expansion_v2',previous_bank_sha256=digest(old),runtime_training_dataset_required=False,
        online_ik_fallback=False,training_expansion=dict(requested_per_hand=8,
            accepted_per_hand={h:sum(p['active_hand']==h for p in accepted) for h in ('l','r')},
            source_inspection=str(inventory),source_inspection_sha256=digest(inventory),
            source_code_sha256=digest(__file__),native_physics_tested=False))
    bank['content_sha256']=checksum(bank);validate(bank);write_json(out/'bank_v2.json',bank)
    c['stage'].GetRootLayer().Save()
    summary=dict(added_training_profiles=len(accepted),total_cup_profiles=len(case['profiles']),
        hands={h:sum(p['active_hand']==h for p in case['profiles']) for h in ('l','r')},attempts=len(attempts),
        training_episode_count=len({p['training_provenance']['episode_index'] for p in accepted}),bank_sha256=digest(out/'bank_v2.json'),
        physical_parameters_unchanged=True,online_ik_calls=0,native_physics_tested=False)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.base_key,a.inventory,a.output)

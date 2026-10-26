"""Offline additional training postures at existing legal operating stations."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,digest,physics
from ..frozen_initials_v1.bank import load,checksum,validate
from ..frozen_initials_v1.cup import context,base_matrix,hand_geometry,place_and_check
from ..expansion_v2.arm_first_visible_cup_v4 import cup_after_hand,inside_region
from .compile import diverse_order


def build(base_key,inventory,output,bank_path):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..poses import place
    from ..procedural_grasp import Kinematics
    from ..expansion_v2.operating_station import RoomGate,proposals
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out);before=physics(c['stage'])
    bank=deepcopy(load(bank_path));case=bank['cases']['table_waste_empty_grasp']
    data=json.loads(Path(inventory).read_text());attempts=[];added=[]
    bases,domain=proposals(c['stage'],c['manifest'],'clear_table_trash',c['launch']);write_json(out/'station_domain.json',domain)
    gate=RoomGate(c['stage'],c['manifest']);r=c['region']
    xy=np.column_stack((r['u'],r['v']))@np.asarray([(r['start']+r['end'])/2,(r['left']+r['right'])/2])
    centre=(np.r_[xy,0.,1.]@c['delta'])[:2]
    for hand in ('l','r'):
        existing=[p for p in case['profiles'] if p['active_hand']==hand and p.get('training_provenance')]
        count=len(existing);used={p['training_provenance']['episode_index'] for p in existing}
        if count>=8:continue
        rows=[v for v in data['rows'] if v.get('reset_accepted') and v['active_hand']==hand and v['vertical_envelope_separation_m']==0]
        for row in diverse_order(rows):
            if row['episode_index'] in used:continue
            initial=row['robot_initialization'];template=c['delta'].T@np.asarray(row['base_canonical_column'])
            initialize_base_pose(c['stage'],[float(template[0,3]),float(template[1,3]),float(np.arctan2(template[1,0],template[0,0]))])
            model=Kinematics(c['stage'],hand,initial['positions']);tips,_=hand_geometry(c['stage'],model,initial['positions'])
            local=(np.c_[tips,np.ones(len(tips))]@np.linalg.inv(model.base).T)[:,:3]
            viable=[]
            for station in bases:
                b=base_matrix(station['initial_base_xyyaw']);actual=local@b[:3,:3].T+b[:3,3]
                for sign in (-1,1):
                    request=dict(gap_m=.035,miss_sign=sign,orientation_jitter=False,axis='robot_local_y',offline_probe=True)
                    shift,_=cup_after_hand(c['cup'],actual,b[:3,:3],request);moved=c['cup']+shift
                    loc=np.linalg.inv(b)@np.r_[(moved.min(0)+moved.max(0))/2,1.]
                    side='l' if loc[1]>0 else 'r' if loc[1]<0 else None
                    if side!=hand or not inside_region(np.concatenate((moved,c['visual']+shift)),c['delta'],c['region']):continue
                    viable.append((float(np.linalg.norm(moved.mean(0)[:2]-centre)),station,sign))
            found=None
            for _,station,sign in sorted(viable,key=lambda v:v[0]):
                xyyaw=station['initial_base_xyyaw'];initialize_base_pose(c['stage'],xyyaw);checks=[]
                for gap in (.020001,.035,.049999):
                    request=dict(gap_m=gap,miss_sign=sign,orientation_jitter=False,axis='robot_local_y',offline_probe=True)
                    result=place_and_check(c,initial,hand,xyyaw,request,gate=gate);checks.append(result)
                    if not result['accepted_cpu']:break
                place(c['stage'],c['target'],c['original'])
                attempts.append(dict(id=row['id'],hand=hand,station=station,miss_sign=sign,checks=checks))
                write_json(out/'attempts.json',attempts)
                if len(checks)==3 and all(v['accepted_cpu'] for v in checks):found=(station,sign);break
            if not found:continue
            station,sign=found;xyyaw=station['initial_base_xyyaw'];used.add(row['episode_index']);count+=1
            profile=dict(id=row['id']+'_station',variant='table_waste_empty_grasp',active_hand=hand,
                robot_initialization=deepcopy(initial),base_canonical_column=(np.linalg.inv(c['delta'].T)@base_matrix(xyyaw)).tolist(),
                initial_base_xyyaw=xyyaw,supported_miss_signs=[sign],asset_id='paper_cup',cpu_accepted=True,
                validation=dict(cpu=True,native=False,random_revalidation_required=True,gap_probes_m=[.020001,.035,.049999],
                    whole_target_in_original_camera_frustum=True),source_motion_needed=False,
                maximum_score=case['profiles'][0]['maximum_score'],completion_policy=case['profiles'][0]['completion_policy'],
                source_record=str(inventory),source_record_sha256=digest(inventory),
                training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],benchmark_task_id=row['benchmark_task_id'],
                    row=row['source_row'],measured_active_arm_q_rad=row['arm_q_rad'],active_arm_copied_exactly=True,
                    body_head_and_inactive_arm_from_same_measured_row=True,active_gripper_closed_empty=True,
                    inactive_gripper='current_native_ready_linkage',source_success_not_inferred=True),
                gripper_closure=row['gripper_closure'],operating_station=station,anchor='canonical_actual_coffee_table_geometry')
            case['profiles'].append(profile);added.append(profile)
            print(json.dumps(dict(accepted=profile['id'],hand=hand,hand_training_count=count)),flush=True)
            if count>=8:break
    if physics(c['stage'])!=before:raise ValueError('Physical parameters changed')
    bank['training_expansion']['accepted_per_hand']={h:sum(p['active_hand']==h and bool(p.get('training_provenance')) for p in case['profiles']) for h in ('l','r')}
    bank['station_extension']=dict(previous_bank_sha256=digest(bank_path),source_code_sha256=digest(__file__),
        existing_source_station_sector_unchanged=True,new_profiles=len(added),native_physics_tested=False)
    bank['content_sha256']=checksum(bank);validate(bank);write_json(out/'bank_v2.json',bank)
    c['stage'].GetRootLayer().Save()
    summary=dict(added_training_profiles=len(added),total_cup_profiles=len(case['profiles']),
        hands={h:sum(p['active_hand']==h for p in case['profiles']) for h in ('l','r')},attempts=len(attempts),
        bank_sha256=digest(out/'bank_v2.json'),physical_parameters_unchanged=True,online_ik_calls=0,native_physics_tested=False)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--inventory',required=True)
    p.add_argument('--output',required=True);p.add_argument('--bank',required=True)
    a=p.parse_args();build(a.base_key,a.inventory,a.output,a.bank)

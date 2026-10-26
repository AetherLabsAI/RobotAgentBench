"""OFFLINE ONLY: training-seeded native IK at confirmed fixed cup orientation.

Raw training right-arm snapshots collide in the empty-grasp geometry. Preserve
those rejections; never label these adapted postures as exact recorded poses.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,digest,physics,audit_reset
from ..frozen_initials_v1.bank import load,checksum,validate
from ..frozen_initials_v1.cup import context,base_matrix,hand_geometry,place_and_check
from ..expansion_v2.arm_first_visible_cup_v4 import cup_after_hand
from .compile import diverse_order


def build(base_key,inventory,output,bank_path):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..poses import place
    from ..procedural_grasp import Kinematics
    from ..expansion_v2.operating_station import RoomGate
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out);before=physics(c['stage'])
    data=json.loads(Path(inventory).read_text());bank=deepcopy(load(bank_path));case=bank['cases']['table_waste_empty_grasp']
    hand='r';template=next(p for p in case['profiles'] if p['active_hand']==hand and not p.get('training_provenance'))
    base=c['delta'].T@np.asarray(template['base_canonical_column'])
    xyyaw=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
    initialize_base_pose(c['stage'],xyyaw);gate=RoomGate(c['stage'],c['manifest'])
    initial=template['robot_initialization'];reference=Kinematics(c['stage'],hand,initial['positions'])
    tips,frames=hand_geometry(c['stage'],reference,initial['positions']);goal0=frames[reference.tool].copy()
    request=dict(gap_m=.035,miss_sign=1,orientation_jitter=False,axis='robot_local_y',offline_probe=True)
    shift,_=cup_after_hand(c['cup'],tips,reference.base[:3,:3],request);reference_cup=(c['cup']+shift).mean(0)
    rows=list(diverse_order([r for r in data['rows'] if r.get('reset_accepted') and r['active_hand']==hand]))
    r=c['region'];goals=[]
    for u in (.1,.3,.5,.7,.9):
        for v in (.25,.5,.75):
            xy=np.column_stack((r['u'],r['v']))@np.asarray([r['start']+(r['end']-r['start'])*u,r['left']+(r['right']-r['left'])*v])
            xy=(np.r_[xy,0.,1.]@c['delta'])[:2];goals.append((float(np.linalg.norm(xy-reference_cup[:2])),xy))
    accepted=[];attempts=[];used=set()
    for _,xy in sorted(goals,key=lambda x:x[0]):
        goal=goal0.copy();goal[:2,3]+=xy-reference_cup[:2]
        seed_rows=[];episode_seen=set()
        for row in rows:
            if row['episode_index'] in used or row['episode_index'] in episode_seen:continue
            seed_rows.append(row);episode_seen.add(row['episode_index'])
            if len(seed_rows)==4:break
        for row in seed_rows:
            candidate=deepcopy(initial)
            for name,val in zip(reference.names,row['arm_q_rad']):candidate['positions'][name]=val;candidate['hold_targets'][name]=val
            model=Kinematics(c['stage'],hand,candidate['positions']);q,fit=model.solve(goal)
            candidate['positions']=q
            for name in model.names:candidate['hold_targets'][name]=q[name]
            checks=[]
            if fit['converged']:
                audit_reset(c['stage'],candidate)
                for gap in (.020001,.035,.049999):
                    req=dict(request,gap_m=gap);check=place_and_check(c,candidate,hand,xyyaw,req,gate=gate);checks.append(check)
                    if not check['accepted_cpu']:break
                place(c['stage'],c['target'],c['original'])
            attempts.append(dict(source_id=row['id'],goal=goal.tolist(),ik=fit,checks=checks));write_json(out/'attempts.json',attempts)
            if not(fit['converged'] and len(checks)==3 and all(s['accepted_cpu'] for s in checks)):continue
            profile=deepcopy(template);profile.update(id=f'trainseed_ep{row["episode_index"]}_f{row["frame"]}_r_adapt{len(accepted)+1}',
                robot_initialization=candidate,cpu_accepted=True,source_record=str(inventory),source_record_sha256=digest(inventory),
                validation=dict(cpu=True,native=False,random_revalidation_required=True,gap_probes_m=[.020001,.035,.049999],
                    whole_target_in_original_camera_frustum=True),
                training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],benchmark_task_id=row['benchmark_task_id'],
                    row=row['source_row'],measured_active_arm_q_rad=row['arm_q_rad'],active_arm_copied_exactly=False,
                    training_seeded_offline_ik=True,ik=fit,fixed_orientation='unchanged_accepted_geometry_v5_right_tool_orientation',
                    actual_frozen_arm_q_rad=[q[n] for n in model.names],active_gripper_closed_empty=True,
                    non_active_coordinates='unchanged_accepted_ready_reset',source_success_not_inferred=True))
            case['profiles'].append(profile);accepted.append(profile);used.add(row['episode_index'])
            print(json.dumps(dict(accepted=profile['id'],right_added=len(accepted),offline_ik_evaluations=fit['evaluations'])),flush=True);break
        if len(accepted)>=8:break
    if physics(c['stage'])!=before:raise ValueError('Physical parameters changed')
    bank['training_expansion']['accepted_per_hand']={h:sum(p['active_hand']==h and bool(p.get('training_provenance')) for p in case['profiles']) for h in ('l','r')}
    bank['offline_right_adaptation']=dict(previous_bank_sha256=digest(bank_path),source_code_sha256=digest(__file__),
        added=len(accepted),offline_ik_attempts=len(attempts),runtime_ik_required=False,native_physics_tested=False,
        workspace_region_unchanged=True,fixed_grasp_orientation_unchanged=True,original_base_station_unchanged=True)
    bank['content_sha256']=checksum(bank);validate(bank);write_json(out/'bank_v2.json',bank);c['stage'].GetRootLayer().Save()
    summary=dict(added_right_profiles=len(accepted),total_cup_profiles=len(case['profiles']),
        hands={h:sum(p['active_hand']==h for p in case['profiles']) for h in ('l','r')},
        bank_sha256=digest(out/'bank_v2.json'),physical_parameters_unchanged=True,online_ik_calls=0,native_physics_tested=False)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--inventory',required=True)
    p.add_argument('--output',required=True);p.add_argument('--bank',required=True)
    a=p.parse_args();build(a.base_key,a.inventory,a.output,a.bank)

"""Offline exact elevated source arms + native-open hand + authored side cup."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..common import HERE,owned,digest,write_json,physics
from ..frozen_initials_v1.cup import context
from ..frozen_initials_v1.bank import checksum
from .geometry import opened_initial,check


def compile_pool(base_key,inventory,output):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from short_tasks.agent_probe_v1.collision_guard.stage_model import from_stage
    from short_tasks.agent_probe_v1.collision_guard.reactive_gate import ReactiveGate
    from ..expansion_v2.operating_station import RoomGate
    from ..poses import place
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out);before=physics(c['stage'])
    inventory=owned(inventory);data=json.loads(inventory.read_text());spec=HERE/'tipped_open_v1/definition.json'
    bank_path=HERE/'frozen_initials_v2/compile/cup_training_exact8_v6/bank_v2.json'
    accepted_bank=json.loads(bank_path.read_text());templates={}
    for profile in accepted_bank['cases']['table_waste_empty_grasp']['profiles']:templates.setdefault(profile['active_hand'],profile)
    gate=RoomGate(c['stage'],c['manifest'])
    # Omit stale old cup only in cached computational obstacles. Final source
    # checker still checks the same dynamic cup at its new authored position.
    gate.obstacles=[o for o in gate.obstacles if not(o[0]==c['target'] or o[0].startswith(c['target']+'/'))]
    protected=ReactiveGate(from_stage(c['stage']));profiles=[];used=set();vectors=set();attempts=[];counts=Counter()
    minima={}
    for row in data['candidates']:
        if row['frame']==row['source_row']['first_minimum_frame']:
            primary='idx41_gripper_l_outer_joint1' if row['active_hand']=='l' else 'idx81_gripper_r_outer_joint1'
            minima[row['episode_index']]=row['command'][primary]
    rows=sorted(data['candidates'],key=lambda r:(r['frame']-r['source_row']['close_transition_frame'],r['episode_index']))
    for row in rows:
        hand=row['active_hand'];primary='idx41_gripper_l_outer_joint1' if hand=='l' else 'idx81_gripper_r_outer_joint1'
        if row['episode_index'] in used or hand not in templates:continue
        if row['frame']<=row['source_row']['close_transition_frame']:continue
        template=templates[hand];base=c['delta'].T@np.asarray(template['base_canonical_column'])
        xyyaw=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
        initialize_base_pose(c['stage'],xyyaw);place(c['stage'],c['target'],c['original'])
        try:
            initial,opening=opened_initial(c,row)
            result=check(c,initial,hand,xyyaw,gate,protected)
            attempts.append(dict(id=row['id'],result=result));counts['accepted' if result['accepted_cpu'] else result['reason']]+=1
            if not result['accepted_cpu']:continue
            names=[f'idx{2 if hand=="l" else 6}{i}_arm_{hand}_joint{i}' for i in range(1,8)]
            vector=tuple(initial['positions'][n] for n in names)
            if vector in vectors:continue
            vectors.add(vector);used.add(row['episode_index'])
            profiles.append(dict(id=row['id']+'_opened_authored_side_cup',variant='table_waste_tipped_cup',active_hand=hand,
                robot_initialization=initial,gripper_opening=opening,base_canonical_column=template['base_canonical_column'],
                initial_base_xyyaw=xyyaw,cpu_accepted=True,fit=result,
                training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],row=row['source_row'],
                    source_state18_rad_or_m={n:v for n,v in row['measured_state'].items() if any(s in n for s in ('_arm_','_body_','_head_'))},
                    arm_body_head_copied_exactly=True,physical_source_grasp_proven=False,
                    source_phase='after_source_close_command_geometrically_elevated_not_a_failure_history'),
                authored_failure_initial=True,physical_history_required=False,physical_failure_generation_verified=False,
                validation=dict(cpu=True,native=False,random_revalidation_required=True),
                maximum_score=c['bound']['score_policy']['maximum'],completion_policy='same_cup_released_inside_correct_bin'))
            write_json(out/'profiles_progress.json',dict(profiles=profiles,counts=dict(counts)))
            print('ACCEPTED',row['id'],hand,len(profiles),flush=True)
            if len(profiles)>=8:break
        except ValueError as exc:
            attempts.append(dict(id=row['id'],error=str(exc)));counts['reset_or_geometry_exception']+=1
    if physics(c['stage'])!=before:raise ValueError('Physical parameters changed')
    value=dict(schema='failure_recovery.authored_tipped_open.bank.v1',profiles=profiles,counts=dict(counts),
        definition_sha256=digest(spec),inventory_sha256=digest(inventory),station_template_sha256=digest(bank_path),
        source_base_key=base_key,online_ik_calls=0,runtime_training_dataset_required=False,
        physical_history_required=False,native_physics_tested=False,runtime_registered=False,
        code_sha256={str(Path(__file__)):digest(__file__),str(HERE/'tipped_open_v1/geometry.py'):digest(HERE/'tipped_open_v1/geometry.py')})
    value['content_sha256']=checksum(value)
    write_json(out/'attempts.json',attempts);write_json(out/'bank.json',value)
    c['stage'].GetRootLayer().Save()
    print(json.dumps(dict(accepted=len(profiles),counts=dict(counts))),flush=True)
    if not profiles:raise ValueError('No elevated/open source pose passed; original targets/geometry/physics were not relaxed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compile_pool(a.base_key,a.inventory,a.output)


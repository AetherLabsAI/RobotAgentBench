"""Freeze distinct exact training episodes only, never adjusted or mirrored arms."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..common import HERE,owned,digest,write_json,physics,finish
from .geometry import Gate,compose,existing


def compile_pool(variant,inventory,out,seeds,coffee_context=None):
    inventory=owned(inventory);data=json.loads(inventory.read_text());out=owned(out,fresh=True);out.mkdir(parents=True)
    profiles=[];episodes=set();attempts=[];counts=Counter();qseen=set()
    # Review all source hands independently. Adjacent frames never count twice.
    source_rows=sorted(data['candidates'],key=lambda r:(abs(r['frame']-r['source_row']['close_transition_frame']),r['episode_index'],r['frame']))
    contexts=[None] if variant=='sofa' else seeds
    for seed in contexts:
        if variant=='sofa':stage,record,source=existing(variant);before=physics(stage);source_path=str(source);gate=Gate(stage,record)
        else:
            path=out/f'source_seed{seed}';stage,record,before=compose(variant,path,seed,coffee_context=coffee_context);source_path=str(path/'initial.usda')
            gate=Gate(stage,record)
        context_examined=0
        for row in source_rows:
            if row['episode_index'] in episodes:continue
            try:
                initial,closure=gate.source_initial(row);hand=row['active_hand']
                if variant=='coffee_bag' and hand!='r':raise ValueError('Coffee bag must keep original right operating hand')
                if variant=='sofa':station,result=gate.sofa_station(initial,hand)
                else:station=None;result=gate.check(initial,hand)
                counts['accepted' if result['accepted_cpu'] else result['reason']]+=1
                attempts.append(dict(id=row['id'],source_seed=seed,result=result,station=station))
                context_examined+=1
                if not result['accepted_cpu']:continue
                vector=tuple(initial['positions'][n] for n in row['measured_state'] if '_arm_' in n or '_body_' in n or '_head_' in n)
                if vector in qseen:continue
                qseen.add(vector);episodes.add(row['episode_index'])
                profile=dict(id=row['id'],variant=variant,active_hand=hand,robot_initialization=initial,cpu_accepted=True,
                    validation=dict(cpu=True,native=False,random_revalidation_required=True,protected_self_collision_checked=True),
                    supported_miss_signs=result['supported_miss_signs'],source_motion_needed=False,
                    training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],
                        active_arm_copied_exactly=True,body_head_and_inactive_arm_from_same_training_row=True,
                        measured_active_arm_q_rad=[row['measured_state'][f'idx{2 if hand=="l" else 6}{i}_arm_{hand}_joint{i}'] for i in range(1,8)],
                        row=row['source_row'],state_fill_metadata_preserved=row['source_row'].get('mapping')),
                    fit=dict(source_seed=seed,source_initial=source_path,station=station,result=result),
                    gripper_closure=closure,maximum_score=record['maximum_score'],completion_policy=record['completion_policy'])
                profiles.append(profile)
                write_json(out/'profiles_progress.json',dict(profiles=profiles,counts=dict(counts)))
                print('ACCEPTED',variant,profile['id'],hand,'count',len(profiles),flush=True)
                if len(profiles)>=8:break
            except ValueError as exc:
                counts['native_reset_rejected']+=1;attempts.append(dict(id=row['id'],source_seed=seed,error=str(exc)))
            if len(attempts)%100==0:
                write_json(out/'progress.json',dict(examined=len(attempts),counts=dict(counts)));print(variant,len(attempts),dict(counts),flush=True)
        write_json(out/f'attempts_seed{seed}.json',attempts)
        if physics(stage)!=before:raise ValueError('Physical attributes changed in offline bank selection')
        if len(profiles)>=8:break
    value=dict(schema='failure_recovery.exact_training_case_pool.v3',variant=variant,profiles=profiles,
        requested=8,accepted=len(profiles),counts=dict(counts),source_episode_count=len(episodes),
        hands={h:sum(p['active_hand']==h for p in profiles) for h in ('l','r')},
        inventory_sha256=digest(inventory),source_code_sha256={str(Path(__file__)):digest(__file__),
            str(HERE/'frozen_initials_v3/geometry.py'):digest(HERE/'frozen_initials_v3/geometry.py')},
        native_physics_tested=False,online_ik_calls=0,training_data_required_at_startup=False,
        runtime_registered=False,original_targets_not_moved=True)
    write_json(out/'pool.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('profiles','source_code_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('shelf_drink','coffee_bag','sofa'),required=True)
    p.add_argument('--inventory',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[301,302,303,304,305,306,307,308])
    p.add_argument('--coffee-context');a=p.parse_args()
    compile_pool(a.variant,a.inventory,a.output,a.seeds,a.coffee_context)

"""OFFLINE ONLY: adapt source active arms, keeping original fixed grasp direction.

This is deliberately separate from exact-training banks and startup. No target,
joint limits, physics, source frames, or non-active coordinates are rewritten.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..common import HERE, owned, digest, write_json, physics, audit_reset
from ..procedural_grasp import Kinematics, collider_points
from ..frozen_initials_v1.cup import base_matrix
from ..expansion_v2.empty_grasp import draw, fingertip_goal, support_clear_goal
from .geometry import existing, Gate


def adapt(inventory, output):
    inventory=owned(inventory); data=json.loads(inventory.read_text())
    out=owned(output,fresh=True); out.mkdir(parents=True)
    stage,record,source=existing('sofa'); before=physics(stage); gate=Gate(stage,record)
    cfg_path=HERE/'expansion_v2/configurations/sofa_directions_v1.json'
    cfg=json.loads(cfg_path.read_text())
    reference=Rotation.from_euler('z',record['original_launch']['initial_base_xyyaw'][2]).as_matrix()
    # Try the already legal station first, then the unchanged original grid.
    bases=sorted(gate.bases,key=lambda r:np.linalg.norm(np.asarray(r['initial_base_xyyaw'])-record['initial_base_xyyaw']))
    def source_order(row):
        # Offline scheduling only: start with the template's already feasible
        # hand and source non-active configuration, without changing any row.
        active={f'idx{2 if row["active_hand"]=="l" else 6}{i}_arm_{row["active_hand"]}_joint{i}' for i in range(1,8)}
        error=sum((v-record['robot_initialization']['positions'][n])**2
            for n,v in row['measured_state'].items() if n not in active and any(s in n for s in ('_arm_','_body_','_head_')))
        return row['active_hand']!=record['active_hand'],error,abs(row['frame']-row['source_row']['close_transition_frame']),row['episode_index']
    rows=sorted(data['candidates'],key=source_order)
    profiles=[]; episodes=set(); counts=Counter(); attempts=[]; seen=set()
    for row in rows:
        if row['episode_index'] in episodes:continue
        try:initial,closure=gate.source_initial(row)
        except ValueError as exc:
            counts['source_reset_rejected']+=1;attempts.append(dict(id=row['id'],reason=str(exc)));continue
        hand=row['active_hand']; rotation=reference@np.asarray(cfg['fixed_tool_rotation_base'][hand])
        request=draw('sofa',row['episode_index']+1)
        for station in bases:
            base=base_matrix(station['initial_base_xyyaw'])
            center=(gate.points.min(0)+gate.points.max(0))/2
            side=(np.linalg.inv(base)@np.r_[center,1.])[1]
            if ('l' if side>0 else 'r')!=hand:
                counts['target_side_hand_mismatch']+=1;continue
            model=Kinematics(stage,hand,initial['positions']);model.base=base
            reason=gate.room.check(model,initial['positions'],omit_active_arm=True)
            if reason:
                counts[reason]+=1;continue
            frames=model.frames(initial['positions']); inverse=np.linalg.inv(frames[model.tool])
            tips=np.concatenate([(inverse@frames[body]@local.T).T[:,:3]
                for branch in ('inner','outer')
                for _,body,local,_ in collider_points(stage,f'/World/Robot/gripper_{hand}_{branch}_link4')])
            hand_points=np.concatenate([(inverse@frames[body]@local.T).T[:,:3]
                for _,body,local in gate.room.robot if body.startswith('/World/Robot/gripper_'+hand+'_')])
            goal,_=fingertip_goal(gate.points,tips,rotation,base[:3,:3],request)
            goal,height=support_clear_goal(goal,gate.points,tips,hand_points,base[:3,:3],maximum_gap=.05)
            if goal is None:
                counts[height['reason']]+=1;continue
            q,ik=model.solve(goal)
            if not ik['converged']:
                counts['ik_not_converged']+=1;continue
            adapted=deepcopy(initial);adapted['positions']=q
            for name in model.names:adapted['hold_targets'][name]=q[name]
            audit_reset(stage,adapted)
            result=gate.check(adapted,hand,base=base)
            counts['accepted' if result['accepted_cpu'] else result['reason']]+=1
            attempts.append(dict(id=row['id'],station=station['candidate_index'],ik=ik,result=result))
            if not result['accepted_cpu']:continue
            vector=tuple(q[n] for n in model.names)
            if vector in seen:continue
            if any(q[n]!=v for n,v in initial['positions'].items() if n not in model.names):
                raise ValueError('Offline adaptation changed a non-active coordinate')
            seen.add(vector);episodes.add(row['episode_index'])
            profiles.append(dict(id=row['id']+'_offline_ik',variant='sofa',active_hand=hand,
                robot_initialization=adapted,gripper_closure=closure,cpu_accepted=True,
                supported_miss_signs=result['supported_miss_signs'],source_motion_needed=False,
                provenance_kind='training_active_arm_adapted_offline_NOT_raw_training_pose',
                training_provenance=dict(episode_index=row['episode_index'],frame=row['frame'],row=row['source_row'],
                    active_arm_copied_exactly=False,offline_ik_adapted=True,
                    source_state18_rad_or_m={n:v for n,v in row['measured_state'].items() if any(s in n for s in ('_arm_','_body_','_head_'))},
                    active_arm_delta_rad={n:q[n]-initial['positions'][n] for n in model.names},
                    non_active_coordinates_unchanged=True),
                fit=dict(station=station,result=result,ik=ik,request=request,height=height,
                    goal_tcp_world=goal.tolist(),fixed_orientation_configuration_sha256=digest(cfg_path)),
                validation=dict(cpu=True,native=False,random_revalidation_required=True,
                    protected_self_collision_checked=True),maximum_score=record['maximum_score'],completion_policy=record['completion_policy']))
            print('ACCEPTED',row['id'],hand,len(profiles),flush=True)
            write_json(out/'profiles_progress.json',dict(profiles=profiles,counts=dict(counts)))
            break
        if len(profiles)>=8:break
        write_json(out/'progress.json',dict(last_source_id=row['id'],examined=len(attempts),counts=dict(counts)))
    if physics(stage)!=before:raise ValueError('Physical properties changed')
    value=dict(schema='failure_recovery.offline_adapted_sofa_pool.v1',variant='sofa',profiles=profiles,
        requested=8,accepted=len(profiles),hands={h:sum(p['active_hand']==h for p in profiles) for h in ('l','r')},
        counts=dict(counts),inventory_sha256=digest(inventory),source_initial_sha256=digest(source),
        code_sha256={str(Path(__file__)):digest(__file__),str(HERE/'frozen_initials_v3/geometry.py'):digest(HERE/'frozen_initials_v3/geometry.py')},
        original_target_unchanged=True,physics_unchanged=True,native_physics_tested=False,
        runtime_registered=False,online_ik_required=False,training_dataset_required_at_startup=False)
    from .adapted_bank import checksum,validate
    value['content_sha256']=checksum(value);validate(value)
    write_json(out/'attempts.json',attempts);write_json(out/'pool.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('profiles','code_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();adapt(a.inventory,a.output)

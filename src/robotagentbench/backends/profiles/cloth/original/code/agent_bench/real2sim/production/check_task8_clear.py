"""Fail-closed physical evidence checks for the continuous Task8 clear producer."""
from pathlib import Path
import json
import numpy as np


def score(task_id, directory):
    directory=Path(directory)
    report=json.loads((directory/'result.json').read_text())
    with np.load(directory/'full_state.npz',allow_pickle=False) as z:
        state={k:z[k] for k in z.files}
    with np.load(directory/'action.npz',allow_pickle=False) as z:
        action={k:z[k] for k in z.files}
    events=json.loads((directory/'contacts.json').read_text())
    b=state['target_bounds'];n=len(state['frame_index']);fps=float(state['fps'])
    targets=state['target_prims'].astype(str).tolist()
    phases=state.get('phase',np.full(n,'')).astype(str)
    if b.shape != (n,len(targets),2,3) or n<2 or fps<=0:
        raise ValueError('Malformed aligned target state')
    if len(set(targets))!=len(targets) or targets!=report['targets']:
        raise ValueError('Target inventory differs from episode definition')
    gates={'identity':str(state['task_id'])==task_id==report['task_id'],
           'complete':report['failure'] is None and len(report['placed'])==len(targets),
           'finite_state':bool(np.isfinite(b).all() and np.isfinite(state['robot_joint_position_rad_or_m']).all()),
           'aligned_action_state':all(len(action[k])==n for k in ['arm','head','waist','lgrip','rgrip']),
           'phase_evidence':len(phases)==n and bool(np.any(phases!='')),
           'unaltered_robot_camera':report['robot_camera_parameters_unchanged'] is True,
           'no_runtime_target_pose_writes':report['runtime_target_pose_writes'] is False,
           'contact_capture_available':report['contact_error'] is None,
           'git_source_recorded':bool(report.get('source_revision',{}).get('git_commit'))}
    for a,s in [('arm','command_arm_rad'),('waist','command_waist_rad_m'),('head','command_head_rad')]:
        gates['aligned_action_state'] = bool(gates['aligned_action_state'] and np.array_equal(action[a],state[s]))
    for index,name in enumerate(['lgrip','rgrip']):
        gates['aligned_action_state'] = bool(gates['aligned_action_state'] and np.allclose(action[name],(.7843981634-state['command_grip_rad'][:,index])/.7843981634,rtol=0,atol=1e-9))
    cloth=np.array(report['cloth_bounds']);table=np.array(report['table_bounds'])
    initial=np.array(report['initial_bounds']);tail=b[-max(2,round(fps)):]
    table_tolerance=.003 if task_id=='8.4' else .015
    supported={i for i in range(len(targets)) if np.all(np.abs(tail[:,i,0,2]-table[1,2])<table_tolerance)}
    if task_id=='8.4':
        contact_pairs={frozenset([e.get('actor0'),e.get('actor1')]) for e in events if e.get('impulse_norm_sum_ns',0)>.0001}
        for _ in targets:
            for i,target in enumerate(targets):
                for j in list(supported):
                    if i==j or frozenset([target,targets[j]]) not in contact_pairs:continue
                    vertical=np.all(abs(tail[:,i,0,2]-tail[:,j,1,2])<.015) and np.all(tail[:,i,0,2]>tail[:,j,0,2]+.003)
                    overlap=np.all(np.minimum(tail[:,i,1,:2],tail[:,j,1,:2])-np.maximum(tail[:,i,0,:2],tail[:,j,0,:2])>.005)
                    if vertical and overlap:supported.add(i)
    for i,target in enumerate(targets):
        release=np.flatnonzero(phases==f'{i}:release')
        retreat=np.flatnonzero(np.isin(phases,[f'{i}:retreat',f'{i}:book_withdraw',f'{i}:open_clear']))
        # A stable final pose alone can hide a delayed release followed by a
        # drop. Once release finishes, withdrawing the hand must not lift the
        # object off its support again.
        gates[f'{i}:released_before_retreat']=bool(len(release) and len(retreat) and retreat[0]>release[-1] and np.max(b[retreat,i,0,2])<=b[release[-1],i,0,2]+.025)
        gates[f'{i}:initially_on_cloth']=bool(np.all(initial[i,0,:2]>=cloth[0,:2]) and np.all(initial[i,1,:2]<=cloth[1,:2]) and abs(initial[i,0,2]-cloth[1,2])<.015)
        gates[f'{i}:lifted']=bool(np.max(b[:,i,0,2])>=initial[i,0,2]+.05)
        gates[f'{i}:never_dropped_below_table']=bool(np.min(b[:,i,0,2])>=table[1,2]-.05)
        gates[f'{i}:left_of_cloth']=bool(np.all(tail[:,i,0,1]>cloth[1,1]+.005))
        gates[f'{i}:inside_table']=bool(np.all(tail[:,i,0,:2]>=table[0,:2]) and np.all(tail[:,i,1,:2]<=table[1,:2]))
        gates[f'{i}:supported']=i in supported
        centres=tail[:,i].mean(1)
        gates[f'{i}:settled']=bool(np.max(np.linalg.norm(np.diff(centres,axis=0),axis=1)*fps)<.025)
        # No fingertip (link4) contact gate: at 240 Hz a proximal-link grasp
        # is a valid physical grasp; lifted/released gates cover the outcome.
        final_contacts=[e for e in events if e.get('frame',-1)>=n-round(fps) and e.get('impulse_norm_sum_ns',0)>.001 and any(target in str(e.get(k,'')) for k in ['actor0','actor1']) and any(str(e.get(k,'')).startswith('/World/Robot/') for k in ['actor0','actor1'])]
        gates[f'{i}:released_from_robot']=not final_contacts
    severe=[e for e in events if e.get('minimum_separation_m',0)<-.005 and all(str(e.get(k,'')).startswith('/World/Robot/') for k in ['actor0','actor1']) and not any(all('/gripper_'+hand+'_' in str(e.get(k,'')) for k in ['actor0','actor1']) for hand in ('l','r'))]
    gates['no_severe_robot_self_collision']=not severe
    return {'schema':'real2sim.task8_clear_score.v1','task_id':task_id,'ok':all(gates.values()),'gates':gates,'failed_gates':[k for k,v in gates.items() if not v],'owner_approved':False}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_id')
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=score(args.task_id,args.directory)
    print(json.dumps(result,indent=2))
    return 0 if result['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())

"""L1 continuous controls -> OpenEnd Action22, 20 Hz zero-order hold.

No recorded state is an action. No 60 Hz targets may be applied between rows.
Navigation anchors are provenance only: a runner must anchor at its measured
stopped pose via the existing navigation controller, never replay old poses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def convert(log, info):
    n=len(log['native_target'])
    if n<1 or info['control_hz']!=60:raise ValueError('Expected nonempty 60 Hz source')
    if not np.allclose(log['timestamp'],(np.arange(n)+1)/60,rtol=0,atol=1e-8):
        raise ValueError('Missing or nonuniform source control timestamps')
    action_names=info['action_names']
    if len(action_names)!=22 or action_names[-2:]!=['base_forward_m_s','base_yaw_rate_rad_s']:
        raise ValueError('Unexpected Action22 contract')
    dofs=log['dof_names'].tolist()
    if len(set(dofs))!=len(dofs):raise ValueError('Duplicate DOF names')
    order=[dofs.index(k) for k in action_names[:20]]
    all_actions=np.column_stack((log['native_target'][:,order],log['base_action_body']))
    if all_actions.shape!=(n,22) or not np.isfinite(all_actions).all():
        raise ValueError('Invalid action values')
    ix=np.arange(0,n,3)
    phases=log['phase'][ix]
    if set(phases)!=set(log['phase']):raise ValueError('Sampling dropped an entire phase')
    modes=log['nav_constraint_mode'][ix]
    if not set(modes)<=set(('disabled','hold','drive','turn')):raise ValueError('Unknown navigation mode')
    # Ignore neither a translated/rotated base nor an out-of-contract primitive.
    a=all_actions[ix]
    if np.any((np.abs(a[:,-2])>1e-6)&(np.abs(a[:,-1])>1e-6)):
        raise ValueError('Source violates exclusive drive/turn contract')
    phase_changes=np.flatnonzero(np.r_[True,log['phase'][1:]!=log['phase'][:-1]])
    delayed=[dict(phase=str(log['phase'][i]),source_frame=int(i),
                  applied_control=int((i+2)//3),delay_s=float(((i+2)//3*3-i)/60))
             for i in phase_changes]
    return dict(action=a,action_names=np.asarray(action_names),control_hz=np.asarray([20]),
                timestamp_start_s=np.arange(len(ix))/20,source_frame=ix,phase=phases,
                nav_constraint_mode=modes,base_navigation_active=log['base_navigation_active'][ix]),dict(
        source_frames=n,source_control_hz=60,output_frames=len(ix),control_hz=20,
        source_duration_s=n/60,output_duration_s=len(ix)/20,
        final_interval_extension_s=len(ix)/20-n/60,phase_boundaries=delayed,
        sampling='source rows 0,3,6,...; hold each Action22 for 0.05 s; no inner-step target update',
        physical_parameters_changed=False,recorded_states_used_as_actions=False,
        old_navigation_anchor_poses_are_commands=False,
        physical_replay_completed=False,task_success=None)


def prepare(source, output):
    source=Path(source);output=Path(output)
    info=json.loads((source/'sim_info.json').read_text())
    log=np.load(source/'recording_60hz.npz',allow_pickle=False)
    data,report=convert(log,info)
    report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source.iterdir() if p.is_file()}
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'actions_20hz.npz',**data)
    report['output_sha256']=hashlib.sha256((output/'actions_20hz.npz').read_bytes()).hexdigest()
    (output/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();print(json.dumps(prepare(args.source,args.output),indent=2))

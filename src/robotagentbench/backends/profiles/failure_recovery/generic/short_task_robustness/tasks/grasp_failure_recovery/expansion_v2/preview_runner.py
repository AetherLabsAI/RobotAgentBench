"""Render a labeled static diagnostic with the existing task-owned renderer.

Only sim1/2, existing readonly fleet lease, one isolated idle GPU, original
renderer 2MiB/zero-process guard. Never starts physics, scores, or model calls.
"""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io
from .runner import command


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial-key',required=True)
    p.add_argument('--key',required=True)
    p.add_argument('--gpu',type=int,choices=range(8),required=True)
    p.add_argument('--camera',choices=('head','left','right'),default='head')
    a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):
        raise ValueError('Only this task on sim1/2')
    if any(not k.replace('_','').isalnum() for k in (a.key,a.initial_key)):
        raise ValueError('Simple new keys required')
    initial=io.HOST/'expansion_v2/candidates'/a.initial_key/'initial.json'
    value=json.loads(initial.read_text())
    if value['variant'] not in ('sofa','table_waste_empty_grasp','table_waste_tipped_cup'):
        raise ValueError('Only the three new recovery initial diagnostics')
    lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(a.gpu)+'.lock')
    with lease.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not io.idle(a.gpu):raise RuntimeError('Existing renderer idle guard rejected; no process stopped')
        uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        cmd=command('sofa',value['seed'],a.key,sofa_asset='remote_tv',gpu_uuid=uuid)
        ix=cmd.index('-m')
        cmd[ix+1:]=['grasp_failure_recovery.render_ready_fleet','--initial',
            str(io.GUEST/'expansion_v2/candidates'/a.initial_key/'initial.json'),
            '--output',str(io.GUEST/'expansion_v2/previews'/a.key),
            '--gpu',str(a.gpu),'--gpu-uuid',uuid,'--variants',value['variant'],'--camera',a.camera]
        run=io.HOST/'expansion_v2/commands'/a.key
        run.mkdir(parents=True,exist_ok=False)
        io.save(run/'command.json',dict(command=cmd,started=time.time(),
            phase='static_pose_diagnostic',physics_advanced=False,scoring_executed=False,
            initial_preparation_accepted=value['preparation_accepted'],
            source_rejection=value.get('procedural_grasp',{}).get('room_rejection')))
        with (run/'stdout.log').open('x') as log:
            rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        io.save(run/'exit.json',dict(returncode=rc,ended=time.time()))
        raise SystemExit(rc)


if __name__=='__main__':main()

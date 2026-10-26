"""Remaining distinct starts, existing lease per lane, one native app per case."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=('a','b'),required=True);a=p.parse_args()
    expected={'a':'VM-0-8-ubuntu','b':'VM-0-10-ubuntu'}
    if io.HERE!=io.HOST or socket.gethostname()!=expected[a.lane]:raise ValueError('Wrong review lane/host')
    root=io.HOST/'tipped_open_v1';plan=json.loads((root/'review_remaining_v1/plan.json').read_text())
    rows=[r for r in plan['cases'] if r.get('lane')==a.lane]
    receipt=root/'review_remaining_v1'/('lane_'+a.lane+'.json')
    # Wait without changing an existing native job or taking an occupied GPU.
    gpu=6;lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_6.lock')
    io.save(receipt,dict(status='waiting_for_original_lease',cases=[r['number'] for r in rows],updated=time.time()))
    with lease.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for row in rows:
            initial=root/'samples'/row['sample_key']/'initial.json'
            while not initial.exists():
                io.save(receipt,dict(status='waiting_for_checked_cpu_initial',number=row['number'],updated=time.time()))
                time.sleep(5)
            value=json.loads(initial.read_text())
            if value['frozen_profile_id']!=row['profile'] or value['seed']!=row['seed'] or not value['preparation_accepted']:
                raise ValueError('Requested distinct profile does not match actual prepared input')
            run=root/'native_commands'/row['native_key'];run.mkdir(parents=True,exist_ok=False)
            if not io.idle(gpu):
                io.save(run/'admission_failure.json',dict(error='Original idle guard rejected; no process stopped',number=row['number']))
                raise RuntimeError('GPU not idle after lease acquisition')
            uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            cmd=command('sofa',value['room_seed'],'tipreview_'+row['native_key'],sofa_asset='remote_tv',gpu_uuid=uuid)
            cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.tipped_open_v1.native_worker',
                '--initial',str(io.GUEST/initial.relative_to(io.HOST)),
                '--output',str(io.GUEST/'tipped_open_v1/native'/row['native_key']),
                '--gpu',str(gpu),'--gpu-uuid',uuid]
            start=time.time();io.save(run/'command.json',dict(command=cmd,number=row['number'],profile=row['profile'],
                host=socket.gethostname(),gpu=gpu,uuid=uuid,started=start,physics_backend='cpu',model_called=False,processes_stopped=[]))
            io.save(receipt,dict(status='native_running',number=row['number'],native_key=row['native_key'],updated=start))
            with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            io.save(run/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start))
            if rc:raise RuntimeError('Native case failed; retained report and did not skip it as a success')
    io.save(receipt,dict(status='lane_finished',cases=[r['number'] for r in rows],updated=time.time()))

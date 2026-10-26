"""One original native app, isolated idle GPU, existing shared lease, sim1-5."""
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
    p=argparse.ArgumentParser();p.add_argument('--sample-key',required=True);p.add_argument('--key',required=True);p.add_argument('--gpu',type=int,choices=range(8),required=True);a=p.parse_args()
    allowed=('VM-0-8-ubuntu','VM-0-10-ubuntu','VM-0-13-ubuntu','VM-0-11-ubuntu','VM-0-9-ubuntu')
    if io.HERE!=io.HOST or socket.gethostname() not in allowed:raise ValueError('Own recovery task on user-authorized sim1-5 only')
    for key in (a.key,a.sample_key):
        if not key.replace('_','').isalnum():raise ValueError('Simple own keys required')
    initial=io.HOST/'tipped_open_v1/samples'/a.sample_key/'initial.json';v=json.loads(initial.read_text())
    if v['initialization_contract']!='authored_tipped_open.v1' or not v['preparation_accepted']:raise ValueError('Accepted authored start required')
    run=io.HOST/'tipped_open_v1/native_commands'/a.key;run.mkdir(parents=True,exist_ok=False)
    lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(a.gpu)+'.lock')
    start=time.time()
    try:
        with lease.open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not io.idle(a.gpu):raise RuntimeError('Existing idle guard rejected; no process stopped')
            uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            cmd=command('sofa',v['room_seed'],'tipopen_native_'+a.key,sofa_asset='remote_tv',gpu_uuid=uuid)
            cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.tipped_open_v1.native_worker',
                '--initial',str(io.GUEST/initial.relative_to(io.HOST)),
                '--output',str(io.GUEST/'tipped_open_v1/native'/a.key),'--gpu',str(a.gpu),'--gpu-uuid',uuid]
            io.save(run/'command.json',dict(command=cmd,started=start,host=socket.gethostname(),gpu=a.gpu,uuid=uuid,
                physics_backend='cpu',model_called=False,processes_stopped=[],shared_sources_readonly=True))
            with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            io.save(run/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start));raise SystemExit(rc)
    except Exception as exc:
        io.save(run/'admission_failure.json',dict(error=type(exc).__name__+': '+str(exc),model_called=False,wall_seconds=time.time()-start));raise

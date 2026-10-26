"""Task-owned sim1/2 initial-state diagnostics, with the existing GPU lease."""
import argparse
import fcntl
import json
import socket
import subprocess
import time
from pathlib import Path
from .. import sim2_runner as io
from ..expansion_v2.runner import command


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True,help='Path relative to this recovery task')
    p.add_argument('--key',required=True);p.add_argument('--gpu',type=int,choices=range(8),required=True)
    p.add_argument('--mode',choices=('native','static'),default='native');a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Only task on sim1/2')
    if not a.key.replace('_','').isalnum():raise ValueError('Fresh simple key required')
    initial=(io.HOST/a.initial).resolve(strict=True)
    if not initial.is_relative_to(io.HOST/'frozen_initials_v2/batch'):raise ValueError('Only actual frozen-v2 CPU candidates')
    value=json.loads(initial.read_text())
    if value['variant']!='table_waste_empty_grasp' or not value['preparation_accepted']:raise ValueError('Accepted cup proposal required')
    lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(a.gpu)+'.lock')
    run=io.HOST/'frozen_initials_v2/probe_commands'/a.key;run.mkdir(parents=True,exist_ok=False)
    with lease.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not io.idle(a.gpu):raise RuntimeError('Original no-process/0-util/2MiB guard rejected; no job stopped')
        uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        cmd=command('sofa',value['seed'],'probe2_'+a.key,sofa_asset='remote_tv',gpu_uuid=uuid)
        args=['--initial',str(io.GUEST/initial.relative_to(io.HOST)),
            '--output',str(io.GUEST/'frozen_initials_v2/probes'/a.key),'--gpu',str(a.gpu),'--gpu-uuid',uuid]
        if a.mode=='static':
            module='grasp_failure_recovery.render_ready_fleet';args+=['--variants','table_waste_empty_grasp']
        else:module='grasp_failure_recovery.frozen_initials_v2.native_probe_v2'
        cmd[cmd.index('-m')+1:]=[module,*args]
        start=time.time();io.save(run/'command.json',dict(command=cmd,started=start,mode=a.mode,
            initial_sha256=__import__('hashlib').sha256(initial.read_bytes()).hexdigest(),
            source_tree='full_only',physics_backend='cpu',model_called=False,shared_sources_writable=False))
        with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        io.save(run/'exit.json',dict(returncode=rc,ended=time.time(),wall_seconds=time.time()-start));raise SystemExit(rc)


if __name__=='__main__':main()

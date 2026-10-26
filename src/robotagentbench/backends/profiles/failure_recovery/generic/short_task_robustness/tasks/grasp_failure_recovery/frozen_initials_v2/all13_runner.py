"""Task-only all-profile review with existing exclusive GPU leases."""
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
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=('sim1','sim2'),required=True)
    p.add_argument('--gpu',type=int,choices=(6,),required=True);a=p.parse_args()
    expected={'sim1':'VM-0-8-ubuntu','sim2':'VM-0-10-ubuntu'}
    if io.HERE!=io.HOST or socket.gethostname()!=expected[a.lane]:raise ValueError('Wrong host')
    plan=io.HOST/'frozen_initials_v2/all13_review_v1/plan.json'
    rows=json.loads(plan.read_text())['lanes'][a.lane]
    for row in rows:
        initial=(io.HOST/row['initial']).resolve(strict=True)
        if not initial.is_relative_to(io.HOST/'frozen_initials_v2/batch'):raise ValueError('Frozen candidates only')
        if json.loads(initial.read_text())['frozen_profile_id']!=row['profile']:raise ValueError('Profile mismatch')
    run=io.HOST/'frozen_initials_v2/all13_review_v1/commands'/a.lane
    run.mkdir(parents=True,exist_ok=False)
    lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_6.lock')
    started=time.time()
    try:
        with lease.open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not io.idle(a.gpu):raise RuntimeError('Original idle admission rejected; no job stopped')
            uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            cmd=command('sofa',401,'all13_review_v1_'+a.lane,sofa_asset='remote_tv',gpu_uuid=uuid)
            cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.frozen_initials_v2.all13_worker',
                '--plan',str(io.GUEST/plan.relative_to(io.HOST)),'--lane',a.lane,
                '--output',str(io.GUEST/'frozen_initials_v2/all13_review_v1'/a.lane),
                '--gpu',str(a.gpu),'--gpu-uuid',uuid]
            io.save(run/'command.json',dict(command=cmd,started=started,source_tree='full_only',physics_backend='cpu'))
            with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            io.save(run/'exit.json',dict(returncode=rc,ended=time.time(),wall_seconds=time.time()-started))
            raise SystemExit(rc)
    except Exception as exc:
        io.save(run/'admission_failure.json',dict(error=type(exc).__name__+': '+str(exc),ended=time.time()))
        raise


if __name__=='__main__':main()

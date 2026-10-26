"""CPU-only sim1/2 launcher, full source readonly, all writes task-owned."""
import argparse
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('compile','sample','batch'))
    p.add_argument('--key',required=True);p.add_argument('--base-key');p.add_argument('--seed',type=int)
    p.add_argument('--seeds',type=int,nargs='+');a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Only this task on sim1/2')
    if not a.key.replace('_','').isalnum():raise ValueError('Simple new output key required')
    run=io.HOST/'frozen_initials_v1/commands'/a.key;run.mkdir(parents=True,exist_ok=False)
    cmd=command('sofa',401,'frozen_'+a.key,sofa_asset='remote_tv');ix=cmd.index('-m')
    module='compile' if a.mode=='compile' else 'cup' if a.mode=='sample' else 'batch'
    if a.mode=='compile':
        if (io.HOST/'frozen_initials_v1/configurations/bank_v1.json').exists():raise ValueError('Bank already frozen; do not overwrite it')
        if (io.HOST/'frozen_initials_v1/offline/cup_left_v1/initial.json').exists():module='finalize'
    args=[]
    if a.mode!='compile':
        if not a.base_key or not a.base_key.replace('_','').isalnum():raise ValueError('Explicit full-source base key required')
        args=['--base-key',a.base_key,'--output',str(io.GUEST/'frozen_initials_v1/samples'/a.key)]
        if a.mode=='sample' and a.seed is not None:args+=['--seed',str(a.seed)]
        if a.mode=='batch':
            if not a.seeds:raise ValueError('Explicit random seeds required')
            args+=['--seeds',*map(str,a.seeds)]
    cmd[ix+1:]=['grasp_failure_recovery.frozen_initials_v1.'+module,*args]
    started=time.time();io.save(run/'command.json',dict(command=cmd,started=started,mode=a.mode,
        physics_advanced=False,model_called=False,shared_source_writable=False))
    with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json',dict(returncode=rc,ended=time.time(),wall_seconds=time.time()-started))
    raise SystemExit(rc)


if __name__=='__main__':main()

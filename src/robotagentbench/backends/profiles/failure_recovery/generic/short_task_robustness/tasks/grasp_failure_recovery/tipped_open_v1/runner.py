"""Full-source CPU wrapper; exact own outputs and no training mount at startup."""
import argparse
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('sample','tests'));p.add_argument('--key',required=True)
    p.add_argument('--base-key',default='tipped_cup_base401_r3');p.add_argument('--bank-key',default='elevated_open_v2')
    p.add_argument('--seed',type=int);a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Own task on sim1/2 only')
    for key in (a.key,a.base_key,a.bank_key):
        if not key.replace('_','').isalnum():raise ValueError('Simple own keys required')
    run=io.HOST/'tipped_open_v1/commands'/a.key;run.mkdir(parents=True,exist_ok=False)
    cmd=command('sofa',401,'tipopen_'+a.key,sofa_asset='remote_tv')
    if a.mode=='sample':
        if type(a.seed) is not int or not 1<=a.seed<2**32:raise ValueError('Explicit positive uint32 seed required')
        args=['grasp_failure_recovery.tipped_open_v1.sample','--base-key',a.base_key,'--seed',str(a.seed),
            '--bank',str(io.GUEST/'tipped_open_v1/compile'/a.bank_key/'bank.json'),
            '--output',str(io.GUEST/'tipped_open_v1/samples'/a.key)]
    else:args=['grasp_failure_recovery.tipped_open_v1.test_initial']
    cmd[cmd.index('-m')+1:]=args
    start=time.time();io.save(run/'command.json',dict(command=cmd,started=start,source_tree='full_only',physics_backend='not_started',training_dataset_mounted=False))
    with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start));raise SystemExit(rc)

"""Task-local fast sample; original objects first, frozen row second, no training mount."""
import argparse
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command
from .runner import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('shelf_drink','coffee_bag','sofa'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--key',required=True);p.add_argument('--bank-key',required=True)
    p.add_argument('--base-key');a=p.parse_args()
    for key in (a.key,a.bank_key,a.base_key):
        if key and not key.replace('_','').isalnum():raise ValueError('Simple own keys required')
    args=['--variant',a.variant,'--seed',str(a.seed),'--output',str(io.GUEST/'frozen_initials_v3/samples'/a.key),
        '--bank',str(io.GUEST/'frozen_initials_v3/banks'/a.bank_key/'bank.json')]
    if a.variant=='coffee_bag':
        context_key=a.key+'_objects'
        cmd=['python3','-B','-m','grasp_failure_recovery.frozen_initials_v3.pool_runner','compose_coffee',
            '--key',context_key,'--seeds',str(a.seed)]
        rc=subprocess.run(cmd).returncode
        if rc:raise SystemExit(rc)
        args+=['--coffee-context',str(io.GUEST/'frozen_initials_v3/contexts'/context_key)]
    elif a.variant=='sofa':
        key=a.base_key
        if key is None:
            key=a.key+'_room';root=io.HOST/'frozen_initials_v3/commands'/key;root.mkdir(parents=True,exist_ok=False)
            cmd=command('sofa',a.seed,key,sofa_asset='remote_tv');start=time.time();io.save(root/'command.json',dict(command=cmd,started=start))
            with (root/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            io.save(root/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start))
            if rc:raise SystemExit(rc)
        args+=['--base-key',key]
    raise SystemExit(run('fast',args,a.key))

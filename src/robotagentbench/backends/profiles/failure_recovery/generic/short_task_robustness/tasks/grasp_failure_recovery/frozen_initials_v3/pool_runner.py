import argparse
import socket
import subprocess
import time
from .. import sim2_runner as io
from .runner import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('compose_coffee','compile'));p.add_argument('--key',required=True)
    p.add_argument('--variant',choices=('shelf_drink','coffee_bag','sofa'));p.add_argument('--inventory-key')
    p.add_argument('--coffee-context-key');p.add_argument('--seeds',type=int,nargs='+',default=[301,302,303,304,305,306,307,308]);a=p.parse_args()
    for k in (a.key,a.inventory_key,a.coffee_context_key):
        if k and not k.replace('_','').isalnum():raise ValueError('Simple fresh keys required')
    if a.mode=='compose_coffee':
        if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Own task only')
        root=io.HOST/'frozen_initials_v3/commands'/a.key;root.mkdir(parents=True,exist_ok=False)
        cmd=io.command('coffee_bag','frozen_initials_v3.compose_coffee',['--output',str(io.GUEST/'frozen_initials_v3/contexts'/a.key),
            '--seeds',*map(str,a.seeds)],'frozen3_'+a.key)
        start=time.time();io.save(root/'command.json',dict(command=cmd,source='unchanged_full_project_original_atomic_sampler',
            executable_sources_not_mixed=True,gpu_used=False,started=start))
        with (root/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        io.save(root/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start));raise SystemExit(rc)
    if not a.variant or not a.inventory_key:raise ValueError('Explicit case and mined inventory required')
    args=['--variant',a.variant,'--inventory',str(io.GUEST/'frozen_initials_v3/mine'/a.inventory_key/'inventory.json'),
        '--output',str(io.GUEST/'frozen_initials_v3/compile'/a.key),'--seeds',*map(str,a.seeds)]
    if a.coffee_context_key:args+=['--coffee-context',str(io.GUEST/'frozen_initials_v3/contexts'/a.coffee_context_key)]
    raise SystemExit(run('compile_pool',args,a.key))

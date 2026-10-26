"""Existing full-source CPU/native container recipes, task-local artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import subprocess
import time
from .. import sim2_runner as io
from .catalog import APIS

def run(api,seed,key,profile=None):
    if not key.replace('_','').isalnum():raise ValueError('Own simple key')
    root=io.HOST/'shelf_continuous_v2/commands'/key;root.mkdir(parents=True,exist_ok=False)
    args=['--seed',str(seed),'--output',str(io.GUEST/'shelf_continuous_v2/initials'/key)]
    if api:args+=['--api',api]
    if profile:args+=['--diagnostic-profile',profile]
    cmd=io.command('shelf_drink','shelf_continuous_v2.generate',args,'continuous_'+key)
    io.save(root/'command.json',dict(command=cmd,source='full_only',online_ik=False))
    start=time.time()
    with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
        rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err).returncode
    io.save(root/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start))
    return rc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--api',choices=APIS);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--key',required=True);p.add_argument('--diagnostic-profile');a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Exact task on sim1/2 only')
    raise SystemExit(run(a.api,a.seed,a.key,a.diagnostic_profile))

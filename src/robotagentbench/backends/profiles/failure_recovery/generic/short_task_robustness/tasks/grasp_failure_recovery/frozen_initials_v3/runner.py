"""Full-source, task-owned CPU diagnostics; never changes existing public pins."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command
from ..frozen_initials_v2.runner import DATASET


def run(module,args,key,*,training=False):
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Only own task on sim1/2')
    if not key.replace('_','').isalnum():raise ValueError('Simple fresh key required')
    run=io.HOST/'frozen_initials_v3/commands'/key;run.mkdir(parents=True,exist_ok=False)
    cmd=command('sofa',401,'frozen3_'+key,sofa_asset='remote_tv')
    if training:cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')]=['--mount',f'type=bind,src={DATASET},dst=/training_dataset,readonly']
    cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.frozen_initials_v3.'+module,*args]
    started=time.time();io.save(run/'command.json',dict(command=cmd,training_readonly=training,source='full_only',gpu_used=False,started=started))
    with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json',dict(returncode=rc,wall_seconds=time.time()-started,ended=time.time()));return rc


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('catalogue','mine'));p.add_argument('--key',required=True)
    p.add_argument('--task-indices',type=int,nargs='+');p.add_argument('--cover-all-phases',action='store_true');a=p.parse_args()
    args=[a.mode,'--dataset','/training_dataset','--output',str(io.GUEST/'frozen_initials_v3'/a.mode/a.key)]
    if a.task_indices:args+=['--task-indices',*map(str,a.task_indices)]
    if a.cover_all_phases:args+=['--cover-all-phases']
    raise SystemExit(run('training',args,a.key,training=True))


if __name__=='__main__':main()

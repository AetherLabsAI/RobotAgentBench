"""Retire this task's superseded idle candidate waiter, never active workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from .. import sim2_runner as io

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Own task sim1/2')
    expected=['python3','-B','-m','grasp_failure_recovery.closeout_v1.candidate_native']
    actual=Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if actual!=expected:raise ValueError('Exact own superseded waiter required')
    # Only nvidia-smi inventory children are allowed; no Docker/Isaac execution.
    children=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[1])!=a.pid:continue
            pid=int(path.parent.name);cmd=(path.parent/'cmdline').read_bytes().decode().replace('\0',' ')
            if not cmd.startswith('nvidia-smi '):raise ValueError('An active worker is a child; refuse interruption: '+cmd)
            children.append(dict(pid=pid,command=cmd))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    root=io.HOST/'closeout_v1/candidate_native_v2'
    for claim in (root/'claims').glob('*.json'):
        key=claim.stem;d=json.loads(claim.read_text())
        if d['host']==socket.gethostname() and not (root/key/'exit.json').is_file():raise ValueError('Own claimed native attempt is still running')
    os.kill(a.pid,signal.SIGTERM)
    io.save(io.HOST/'shelf_continuous_v2'/('retired_waiter_'+socket.gethostname()+'.json'),
        dict(pid=a.pid,command=actual,children=children,reason='superseded_idle_waiter_holding_task_gpu_leases',
             native_or_model_processes_stopped=False,old_results_preserved=True,when=time.time()))

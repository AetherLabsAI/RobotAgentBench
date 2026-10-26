"""One isolated application per accepted v3 sample; unchanged original probe."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io


def run(key,gpu,run_key=None):
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Own task on sim1/2 only')
    if not key.replace('_','').isalnum():raise ValueError('Simple sample key required')
    initial=io.HOST/'frozen_initials_v3/samples'/key/'initial.json';v=json.loads(initial.read_text())
    audit=json.loads((initial.parent/'runtime_dependency_audit.json').read_text())
    if v['variant'] not in ('shelf_drink','coffee_bag') or v['initialization_policy']!='exact_training_bank_original_object_sampler.v3':
        raise ValueError('Only new first-two frozen CPU samples')
    if not v['preparation_accepted'] or not v['procedural_grasp']['accepted_cpu'] or v['new_episode_native_acceptance_inherited']:
        raise ValueError('Fresh CPU acceptance required')
    if audit['optimizer_calls'] or audit['blocked_training_or_trajectory_read_attempts'] or audit['training_dataset_mounted']:
        raise ValueError('Startup runtime audit failed')
    run_key=key if run_key is None else run_key
    if not run_key.replace('_','').isalnum():raise ValueError('Simple run key required')
    output=io.HOST/'frozen_initials_v3/native_commands'/run_key;output.mkdir(parents=True,exist_ok=False)
    lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(gpu)+'.lock')
    with lease.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not io.idle(gpu):raise RuntimeError('Original idle admission rejected; no job stopped')
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        label='frozen3_native_'+run_key
        cmd=io.command(v['variant'],'native_procedural',[
            '--initial',str(io.GUEST/initial.relative_to(io.HOST)),
            '--output',str(io.GUEST/'frozen_initials_v3/native'/run_key),
            '--gpu',str(gpu),'--gpu-uuid',uuid,'--allow-idle-2mib'],label,gpu=True)
        cmd[cmd.index('--gpus')+1]='device='+uuid;cmd[2:2]=['--hostname',socket.gethostname()]
        mounts=[]
        for name,target in (('kit_cache','/isaac-sim/kit/cache'),('kit_data','/isaac-sim/kit/data')):
            path=io.HOST/'execution'/label/name;path.mkdir(parents=True,exist_ok=True)
            mounts+=['--mount',f'type=bind,src={path},dst={target}']
        cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')]=mounts
        start=time.time();io.save(output/'command.json',dict(command=cmd,started=start,physics_backend='cpu',model_called=False))
        with (output/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        io.save(output/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start));return rc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--gpu',type=int,choices=range(8),required=True);p.add_argument('--run-key')
    a=p.parse_args()
    try:rc=run(a.key,a.gpu,a.run_key)
    except Exception as exc:
        receipt=io.HOST/'frozen_initials_v3/native_commands'/(a.run_key or a.key)/'admission_failure.json'
        if receipt.parent.exists():io.save(receipt,dict(error=type(exc).__name__+': '+str(exc),model_called=False,native_physics_started=False))
        raise
    raise SystemExit(rc)

"""Task-owned native smoke, one idle sim1/2 GPU, existing lease/2 MiB flag."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Own exact task on sim1/2')
    if not a.key.replace('_','').isalnum():raise ValueError('Simple own initial key')
    initial=io.HOST/'shelf_continuous_v2/initials'/a.key/'initial.json'
    if not initial.is_file():raise ValueError('Accepted initial must exist')
    locks=[];chosen=None
    for gpu in range(8):
        path=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(gpu)+'.lock')
        lock=path.open('r')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if not io.idle(gpu):lock.close();continue
        chosen=gpu;locks.append(lock);break
    if chosen is None:raise RuntimeError('No isolated idle GPU; no sharing/stopping')
    root=io.HOST/'shelf_continuous_v2/native_commands'/a.key;root.mkdir(parents=True,exist_ok=False)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(chosen),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cmd=io.command('shelf_drink','shelf_continuous_v2.owner',[
        '--initial',str(io.GUEST/initial.relative_to(io.HOST)),
        '--output',str(io.GUEST/'shelf_continuous_v2/native'/a.key),
        '--gpu',str(chosen),'--gpu-uuid',uuid,'--allow-idle-2mib','--smoke'],'continuous_native_'+a.key,gpu=True)
    cmd[cmd.index('--gpus')+1]='device='+uuid;cmd[2:2]=['--hostname',socket.gethostname()]
    mounts=[]
    for sub,dest in [('kit_cache','/isaac-sim/kit/cache'),('kit_data','/isaac-sim/kit/data')]:
        folder=io.HOST/'shelf_continuous_v2/native_cache'/a.key/sub;folder.mkdir(parents=True,exist_ok=True)
        mounts+=['--mount',f'type=bind,src={folder},dst={dest}']
    cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')]=mounts
    io.save(root/'command.json',dict(command=cmd,gpu=chosen,uuid=uuid,source='full_only',model_calls=0,started=time.time()))
    start=time.time()
    with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
        rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err).returncode
    io.save(root/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start))
    for lock in locks:lock.close()
    raise SystemExit(rc)

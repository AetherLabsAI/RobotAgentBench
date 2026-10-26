"""First genuinely idle leased device for explicitly pending review cases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',type=int,nargs='+',choices=(3,4,5,6),default=[3,5]);args=parser.parse_args()
    hostname=socket.gethostname();allowed=('VM-0-8-ubuntu','VM-0-10-ubuntu','VM-0-13-ubuntu','VM-0-11-ubuntu','VM-0-9-ubuntu')
    if io.HERE!=io.HOST or hostname not in allowed:raise ValueError('Exact recovery task on sim1-5 only')
    root=io.HOST/'tipped_open_v1';review=root/'review_remaining_v1';queue=review/'flex_queue'
    claims=queue/'claims';claims.mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads((review/'plan.json').read_text())['cases'] if r['number'] in args.cases]
    for row in rows:
        if (root/'native_commands'/row['native_key']/'command.json').exists():
            raise ValueError('Original native attempt already started; do not duplicate')
    while any(not (claims/f"case{r['number']}.json").exists() for r in rows):
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        for line in query.splitlines():
            index,memory,util=map(lambda x:int(x.strip()),line.split(','))
            if memory>2 or util!=0:continue
            lease=Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(hostname+'_'+str(index)+'.lock')
            if not lease.exists():continue
            with lease.open('r') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if not io.idle(index):continue
                selected=None
                for row in rows:
                    claim=claims/f"case{row['number']}.json"
                    try:fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
                    except FileExistsError:continue
                    with os.fdopen(fd,'w') as stream:json.dump(dict(number=row['number'],host=hostname,gpu=index,claimed=time.time()),stream)
                    selected=row;break
                if selected is None:return
                initial=root/'samples'/selected['sample_key']/'initial.json';value=json.loads(initial.read_text())
                if value['frozen_profile_id']!=selected['profile'] or not value['preparation_accepted']:raise ValueError('Profile mismatch')
                key=f"remaining_flex_case{selected['number']}_{hostname.replace('-','_')}_gpu{index}_v1"
                io.save(queue/f"dispatch{selected['number']}.json",dict(number=selected['number'],native_key=key,host=hostname,gpu=index,
                    previous_native_key=selected['native_key'],reason='fixed-lane fresh idle check rejected; input and native worker unchanged'))
                uuid=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                run=root/'native_commands'/key;run.mkdir(parents=True,exist_ok=False)
                cmd=command('sofa',value['room_seed'],'tipflex_'+key,sofa_asset='remote_tv',gpu_uuid=uuid)
                cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.tipped_open_v1.native_worker',
                    '--initial',str(io.GUEST/initial.relative_to(io.HOST)),
                    '--output',str(io.GUEST/'tipped_open_v1/native'/key),'--gpu',str(index),'--gpu-uuid',uuid]
                start=time.time();io.save(run/'command.json',dict(command=cmd,profile=selected['profile'],started=start,
                    host=hostname,gpu=index,uuid=uuid,physics_backend='cpu',processes_stopped=[],model_called=False))
                with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
                io.save(run/'exit.json',dict(returncode=rc,wall_seconds=time.time()-start))
                if rc:raise RuntimeError('Native review failed; case retained, not counted as a successful review')
        if any(not (claims/f"case{r['number']}.json").exists() for r in rows):
            io.save(queue/(hostname+'.json'),dict(status='waiting_for_idle_lease_no_process_stopped',updated=time.time()))
            time.sleep(10)


if __name__=='__main__':main()

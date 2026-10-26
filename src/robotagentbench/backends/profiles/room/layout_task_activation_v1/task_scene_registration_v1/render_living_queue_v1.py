"""Launch just the two requested static previews, only on currently empty GPUs."""
import subprocess,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;run=HERE/'other_layouts_v4'
pending=[3,4];active={};results=[];deadline=time.monotonic()+600
while pending or active:
    for gpu,(process,layout,log) in list(active.items()):
        code=process.poll()
        if code is not None:
            log.close();results.append(dict(layout=layout,gpu=gpu,exit_code=code));del active[gpu]
            print('FINISHED',layout,code,flush=True)
    if pending and time.monotonic()<deadline:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        for line in output.splitlines():
            gpu,used=map(int,line.split(','))
            if gpu in active or used>=100 or not pending:continue
            layout=pending.pop(0)
            log=(run/f'render{layout:02d}.log').open('x')
            command=['docker','run','--rm','--name',f'arena-living-random-layout{layout:02d}-v1','--gpus',f'device={gpu}',
              '--network','host','--user','0:0','-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y',
              '-v','/home/lingjun:/home/lingjun','-v','/mnt/agent-bench-data/lingjun:/mnt/agent-bench-data/lingjun:ro',
              '--entrypoint','/isaac-sim/python.sh','b272c84fbf8a',str(HERE/'render_other_living_v1.py'),'--run',str(run),'--layout',str(layout)]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(process,layout,log);print('STARTED',layout,'GPU',gpu,flush=True)
    if pending and time.monotonic()>=deadline and not active:break
    (run/'render_queue.json').write_text(json.dumps(dict(pending=pending,active={k:v[1] for k,v in active.items()},results=results),indent=2))
    if pending or active:time.sleep(2)
print('QUEUE_END',json.dumps(dict(pending=pending,results=results)),flush=True)

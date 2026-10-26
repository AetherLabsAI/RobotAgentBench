import subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent;run=here/'other_layouts_v4'
for attempt in range(300):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    free=[int(row.split(',')[0]) for row in rows.splitlines() if int(row.split(',')[1])<100]
    if free:
        gpu=free[0];print('RENDER_SMALL_TABLE_GPU',gpu,flush=True)
        with (run/'detail_fix.log').open('x') as log:
            code=subprocess.call(['docker','run','--rm','--name','arena-small-table-camera-fix-v1','--gpus',f'device={gpu}','--network','host','--user','0:0','-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-v','/home/lingjun:/home/lingjun','-v','/mnt/agent-bench-data/lingjun:/mnt/agent-bench-data/lingjun:ro','--entrypoint','/isaac-sim/python.sh','b272c84fbf8a',str(here/'render_small_table_fix_v1.py'),'--run',str(run),'--layout','0'],stdout=log,stderr=subprocess.STDOUT)
        raise SystemExit(code)
    time.sleep(2)
raise SystemExit('No free GPU within bounded wait')

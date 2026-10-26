"""User-authorized GPU1 reclamation; never broadly kill by process name."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from gpu_selection import inspect

def process_map(text):
    result={}
    for line in text.splitlines():
        if not line.strip():continue
        uuid,pid=[v.strip() for v in line.split(',')]
        result.setdefault(int(pid),set()).add(uuid)
    return result

def processes():
    return process_map(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True))

def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.rsplit(')',1)[1].split()[19]  # Linux starttime, guards PID reuse.
    except FileNotFoundError:return None

def eligible(mapping,pid,uuid):
    return mapping.get(pid)=={uuid} and pid>1 and pid!=os.getpid()

def reserve_gpu1(policy,directory,episode,progress):
    if policy['allowed_indices']!=[1] or policy['selection']!='fixed_gpu1_reclaim':
        raise ValueError('Reclamation is authorized for GPU1 only')
    lock=(directory/'gpu1_render.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('GPU1 已有本网站的渲染任务；不终止正在进行的渲染')
    try:
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:raise RuntimeError('GPU1 identity unavailable')
        audit=[]
        mapping=processes()
        targets={p:identity(p) for p,ids in mapping.items() if uuid in ids}
        if any(not eligible(mapping,p,uuid) for p in targets):
            raise RuntimeError('GPU1 占用进程同时关联其他 GPU；为避免影响其他卡，未终止')
        def send(pid,start,sig):
            if start is None or identity(pid)!=start or not eligible(processes(),pid,uuid):return
            try:fd=os.pidfd_open(pid)
            except ProcessLookupError:return
            try:
                if identity(pid)!=start:return
                audit.append(dict(pid=pid,starttime=start,gpu_uuid=uuid,signal=sig.name,time=time.time()))
                (episode/'gpu_reclaim.json').write_text(json.dumps(audit,indent=2))
                try:signal.pidfd_send_signal(fd,sig)
                except ProcessLookupError:return
                except PermissionError:
                    if identity(pid)!=start or not eligible(processes(),pid,uuid):return
                    helper=directory/'gpu1_signal_helper.py'
                    # Utility-only NVIDIA access rechecks all PID memberships;
                    # this helper performs no GPU compute or rendering.
                    subprocess.run(['docker','run','--rm','--pid=host','--network=none','--user','0:0',
                        '--gpus','all','-e','NVIDIA_DRIVER_CAPABILITIES=utility',
                        '-v',str(helper)+':/gpu1_signal_helper.py:ro',
                        '--entrypoint','/usr/bin/python3','b272c84fbf8a',
                        '/gpu1_signal_helper.py',str(pid),start,uuid,str(int(sig))],
                        check=True,capture_output=True,text=True,timeout=30)
            finally:os.close(fd)
        if targets:
            progress(dict(stage='gpu',message=f'GPU1：正在停止 {len(targets)} 个已核实占用进程'))
            for p,start in targets.items():send(p,start,signal.SIGTERM)
            deadline=time.monotonic()+8
            while time.monotonic()<deadline and any(identity(p)==s for p,s in targets.items() if s is not None):time.sleep(.5)
            for p,start in targets.items():send(p,start,signal.SIGKILL)
        progress(dict(stage='gpu',message='GPU1：等待显存释放并复核空闲状态'))
        for _ in range(30):
            available=inspect(policy)
            if available:return available[0],lock
            time.sleep(1)
        raise RuntimeError('GPU1 尚未释放或存在无法识别的图形占用；未切换其他卡，请查看占用记录')
    except BaseException:
        lock.close();raise

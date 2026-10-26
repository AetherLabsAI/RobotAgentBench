"""Stream real render events; retain the complete process output in a log."""
import json
import queue
import subprocess
import threading
import time

def stream(command,log_path,progress,timeout=600):
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    lines=queue.Queue()
    def read():
        try:
            for line in proc.stdout:lines.put(line)
        finally:
            proc.stdout.close()
            lines.put(None)
    threading.Thread(target=read,daemon=True).start()
    deadline=time.monotonic()+timeout
    tail=[]
    try:
        with log_path.open('w') as log:
            while True:
                line=lines.get(timeout=max(.01,deadline-time.monotonic()))
                if line is None:break
                log.write(line);log.flush();tail=(tail+[line])[-25:]
                if line.startswith('ROOM_PROGRESS '):progress(json.loads(line[len('ROOM_PROGRESS '):]))
            code=proc.wait(timeout=max(.01,deadline-time.monotonic()))
        if code:raise RuntimeError('场景已保留，但渲染未完成：'+''.join(tail)[-1800:])
    except (queue.Empty,subprocess.TimeoutExpired):
        proc.terminate()
        raise RuntimeError('渲染连接等待超时；远程状态需核实，场景和日志已保留')

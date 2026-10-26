"""Privileged, exact-PID GPU1 signal only. No arbitrary command execution."""
import os
import signal
import subprocess
import sys
from pathlib import Path

def main(args):
    if args==['--check']:
        assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
        print('exact-PID helper available');return
    pid,start,uuid,sig=args
    pid=int(pid);sig=int(sig)
    if pid<=1 or sig not in (9,15):raise ValueError('Invalid exact-PID signal')
    actual=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual!=uuid:raise ValueError('GPU1 identity changed')
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    devices={r.split(',')[0].strip() for r in rows.splitlines() if r.strip() and int(r.split(',')[1])==pid}
    if devices!={uuid}:raise ValueError('PID is not exclusively attached to GPU1')
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:
        if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]!=start:
            raise ValueError('PID identity changed')
        signal.pidfd_send_signal(fd,sig)
    except (FileNotFoundError,ProcessLookupError):return
    finally:os.close(fd)

if __name__=='__main__':main(sys.argv[1:])

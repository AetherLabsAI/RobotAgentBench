"""Cooperative native-owner GPU admission; never displace existing processes."""
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import re
import stat
import subprocess


@dataclass
class GpuLease:
    index: int
    uuid: str
    evidence: dict
    lock: object
    closed: bool = False

    def configure_environment(self, output):
        out = Path(output)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.uuid
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
        os.environ['CUDA_CACHE_PATH'] = str(out / 'cache/cuda')

    def close(self):
        if not self.closed:
            self.lock.close()
            self.closed = True


def acquire_gpu(index, *, workspace, allow_idle_utilization=False, allow_shared_gpu=False, query=None):
    if type(index) is not int or index < 0:
        raise ValueError('An explicit nonnegative GPU index is required')
    if type(allow_shared_gpu) is not bool or type(allow_idle_utilization) is not bool:
        raise ValueError('GPU exceptions require explicit boolean approvals')
    if allow_shared_gpu and allow_idle_utilization:
        raise ValueError('Shared GPU and idle-utilization modes are mutually exclusive')
    query = query or subprocess.check_output

    def inspect():
        fields = query(['nvidia-smi', '--id', str(index),
                        '--query-gpu=uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if len(fields) != 3 or not re.fullmatch(r'GPU-[a-fA-F0-9-]{8,}', fields[0].strip()):
            raise RuntimeError('Unexpected GPU identity or occupancy response')
        uuid = fields[0].strip()
        memory, utilization = int(fields[1]), int(fields[2])
        if memory < 0 or not 0 <= utilization <= 100:
            raise RuntimeError('Invalid GPU occupancy response')
        approved_idle = False
        if allow_idle_utilization:
            active = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                            '--format=csv,noheader,nounits'], text=True)
            monitor = query(['nvidia-smi', 'pmon', '-i', str(index), '-c', '1'], text=True)
            rows = [line.split() for line in monitor.splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
            approved_idle = (memory == 0 and uuid not in active and bool(rows)
                             and all(len(row) > 1 and row[1] == '-' for row in rows))
            if not approved_idle:
                raise RuntimeError('Approved idle-utilization exception requires zero memory and no listed process')
        existing = None
        if allow_shared_gpu:
            existing = query(['nvidia-smi', '--id', str(index),
                              '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                              '--format=csv,noheader,nounits'], text=True).strip()
        if not allow_shared_gpu and (memory > 32 or (utilization != 0 and not approved_idle)):
            raise RuntimeError('Selected GPU is occupied; no process may be displaced')
        return uuid, dict(memory_mib=memory, utilization_percent=utilization,
                          user_approved_idle_utilization=approved_idle,
                          user_approved_shared_gpu=allow_shared_gpu, existing_processes=existing)

    uuid, _ = inspect()
    lock_path = workspace.lock_root / ('gpu-' + uuid + '.lock')
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        lock = os.fdopen(descriptor, 'a')
    except BaseException:
        os.close(descriptor)
        raise
    try:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise RuntimeError('GPU owner lock must be a regular file')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed_uuid, evidence = inspect()  # Recheck occupancy after acquiring the owner lock.
        if confirmed_uuid != uuid:
            raise RuntimeError('Selected GPU identity changed during admission')
        return GpuLease(index, uuid, evidence, lock)
    except BaseException:
        lock.close()
        raise

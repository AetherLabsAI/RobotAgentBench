"""Conservative availability checks and project-local advisory locks."""
import fcntl
import subprocess


def candidates(gpus, apps, policy):
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    available = []
    for line in gpus.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) != 4:
            continue
        try:
            index, uuid, memory, utilization = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        except ValueError:
            continue  # Unknown telemetry never means idle.
        if (index in policy['allowed_indices'] and uuid not in busy
                and memory <= policy['max_memory_used_mib']
                and utilization <= policy['max_utilization_percent']):
            available.append(dict(index=index, uuid=uuid, memory_mib=memory, utilization=utilization))
    return sorted(available, key=lambda gpu: (gpu['memory_mib'], gpu['utilization'], gpu['index']))


def inspect(policy):
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    return candidates(gpus, apps, policy)


def reserve(policy, directory):
    for gpu in inspect(policy):
        lock = (directory / f"gpu{gpu['index']}_render.lock").open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        try:
            if any(fresh['uuid'] == gpu['uuid'] and fresh['index'] == gpu['index']
                   for fresh in inspect(policy)):
                return gpu, lock
        except BaseException:
            lock.close()
            raise
        lock.close()
    raise RuntimeError('No available GPU among allowed devices; no process stopped')

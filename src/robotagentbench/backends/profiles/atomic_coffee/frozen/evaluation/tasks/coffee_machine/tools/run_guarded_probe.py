"""Reserve one idle allowed GPU and run one initial-state probe of the coffee machine family.

Same guard as the other task families: never shares a card, never stops another job, refuses reserved cards
(gpu_policy). The reservation file is namespaced for this family.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.tasks.coffee_machine.tools import gpu_policy
from evaluation.tasks.coffee_machine.tools.initial_probe import APIS, ROLES, TASK

RESERVATION = '/tmp/arena-coffee-machine-{uuid}.reservation'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api', required=True, choices=APIS)
    p.add_argument('--role', default='source', choices=ROLES)
    p.add_argument('--episode')
    a = p.parse_args(argv)
    out = a.output.resolve()
    record = out.with_suffix('.preflight.json')
    gpu_policy.check(a.gpu)
    if out.exists() or record.exists() or not out.is_relative_to((TASK / 'artifacts').resolve()):
        raise ValueError('Fresh task output required')
    if os.environ.get('OMNI_KIT_ACCEPT_EULA') != 'YES':
        raise RuntimeError('Existing Isaac startup environment required')
    metadata = dict(gpu=a.gpu, seed=a.seed, api=a.api, role=a.role, episode=a.episode, started_at=time.time(), status='preflight')

    def save():
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(metadata, indent=2) + '\n')

    save()
    try:
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid', '--format=csv,noheader'],
                                       text=True, timeout=150).strip()
        if not uuid.startswith('GPU-'):
            raise RuntimeError('Invalid device UUID')
        metadata['uuid'] = uuid
        save()
        with open(RESERVATION.format(uuid=uuid), 'a+') as reservation:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                             '--format=csv'], text=True, timeout=150)
            metadata['process_query'] = query
            save()
            if uuid in query:
                raise RuntimeError('Device occupied; no worker launched')
            with open(f'/tmp/arena-{os.getuid()}-{uuid}.lock', 'a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            metadata['status'] = 'running'
            save()
            command = [sys.executable, '-m', 'evaluation.tasks.coffee_machine.tools.initial_probe', '--gpu', str(a.gpu),
                       '--seed', str(a.seed), '--api', a.api, '--role', a.role, '--output', str(out)]
            if a.episode:
                command += ['--episode', a.episode]
            with out.with_suffix('.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            metadata.update(returncode=result.returncode, status='completed' if result.returncode == 0 else 'error')
    except BaseException as error:
        metadata.update(status='error', error=repr(error))
        raise
    finally:
        metadata['finished_at'] = time.time()
        save()
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()

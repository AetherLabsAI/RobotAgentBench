"""One explicit replay on an idle allowed GPU, with its own public service and provenance.

The service is the unmodified public `arena_api` with the Isaac backend; the scene is the public L1 baseline, the
task runtime authors the layout of the trajectory's source episode. A random layout is never used for a replay.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from evaluation.tasks.coffee_machine.tools import gpu_policy, interpreters
from arena_api.l1_scene import BASE

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
RESERVATION = '/tmp/arena-coffee-machine-{uuid}.reservation'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--api', required=True)
    a = p.parse_args(argv)
    out = a.output.resolve()
    gpu_policy.check(a.gpu)
    if out.exists() or not out.is_relative_to((TASK/'artifacts').resolve()):
        raise ValueError('Fresh task output required')
    out.mkdir(parents=True)
    meta = dict(gpu=a.gpu, port=a.port, api=a.api, seed=a.seed, status='preflight', started_at=time.time())

    def save():
        (out/'job.json').write_text(json.dumps(meta, indent=2)+'\n')

    save()
    service = None
    files = (sorted((REPO/'arena_api').glob('*.py')) + sorted(TASK.glob('*.py'))
             + sorted(p for p in (TASK/'vendor_producer').glob('*') if p.is_file())
             + sorted(p for p in (REPO/'resources/shared_assets/coffee_v1').glob('*') if p.is_file()))
    hashes = lambda: {str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    baseline = hashes()
    meta['source_sha256'] = baseline
    try:
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True, timeout=150).strip()
        if not uuid.startswith('GPU-'):
            raise RuntimeError('Invalid GPU UUID')
        meta['uuid'] = uuid
        with open(RESERVATION.format(uuid=uuid), 'a+') as reservation:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                             '--format=csv'], text=True, timeout=150)
            meta['process_query'] = query
            save()
            if uuid in query:
                raise RuntimeError('GPU occupied; no worker launched')
            with open(f'/tmp/arena-{os.getuid()}-{uuid}.lock', 'a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', a.port))
            env = dict(os.environ)
            # The operator must explicitly provide the already accepted startup environment.
            if env.get('OMNI_KIT_ACCEPT_EULA') != 'YES':
                raise RuntimeError('Existing Isaac startup environment required')
            with (out/'server.log').open('x') as log:
                service = subprocess.Popen(
                    [interpreters.API, '-m', 'arena_api', '--host', '127.0.0.1', '--port', str(a.port),
                     '--backend', 'isaac', '--isaac-python', interpreters.ISAAC,
                     '--scene', str(BASE), '--output', str(out/'episodes'), '--gpu', str(a.gpu),
                     '--isaac-init-timeout', '900'],
                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=REPO)
                meta.update(service_pid=service.pid, status='starting_service')
                save()
                for _ in range(30):
                    if service.poll() is not None:
                        raise RuntimeError('Service exited')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/healthz', timeout=2) as r:
                            if json.load(r)['status'] == 'ok':
                                break
                    except OSError:
                        time.sleep(1)
                else:
                    raise RuntimeError('Service not ready')
                if hashes() != baseline:
                    raise RuntimeError('Sources changed before replay')
                meta['status'] = 'running'
                save()
                with (out/'runner.log').open('x') as log:
                    result = subprocess.run(
                        [interpreters.API, '-m', 'evaluation.tasks.coffee_machine.tools.run_replay',
                         '--trajectory', str(a.trajectory.resolve()), '--env-url', f'http://127.0.0.1:{a.port}',
                         '--output', str(out/'run'), '--seed', str(a.seed), '--api', a.api],
                        stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env)
                meta['runner_returncode'] = result.returncode
                meta['source_unchanged'] = hashes() == baseline
                report = json.loads((out/'run/report.json').read_text())
                meta['report'] = report
                meta['status'] = ('completed' if report['run_status'] == 'completed' and meta['source_unchanged']
                                  else 'error')
    except BaseException as error:
        meta.update(status='error', error=repr(error))
        raise
    finally:
        if service is not None:
            service.terminate()
            try:
                service.wait(timeout=75)
            except subprocess.TimeoutExpired:
                meta['service_shutdown_pending'] = True
            meta['service_returncode'] = service.returncode
        meta['finished_at'] = time.time()
        save()
    print(json.dumps({k: meta.get(k) for k in ('status', 'gpu', 'uuid', 'runner_returncode', 'source_unchanged')}))


if __name__ == '__main__':
    main()

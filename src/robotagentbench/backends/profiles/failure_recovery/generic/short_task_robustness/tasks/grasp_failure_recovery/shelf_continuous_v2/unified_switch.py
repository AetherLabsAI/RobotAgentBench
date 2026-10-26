"""Stage/test/publish the shelf task's existing local extension, not the registry.

Only this exact task's installed directory is writable during publication.
The old five-task global-registration candidate is intentionally not used.
"""
import argparse
import difflib
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..closeout_v1.candidate_runner import command as composed_command, CANON, PACKAGE

ROOT = io.PROJECT/'short_tasks/agent_probe_v1/deployment_generic_20261010/source'
TARGET = ROOT/'short_task_robustness/tasks/grasp_failure_recovery'
AUDIT = io.HOST/'shelf_continuous_v2/public_switch_v2'
ADAPTER = 'public_adapter.py'
RECORD = 'public_component_sources.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage():
    if (AUDIT/'stage.json').exists():
        raise ValueError('Immutable stage exists')
    old = json.loads((TARGET/RECORD).read_text())
    for group, base in (('shared_sources', ROOT), ('task_sources', TARGET)):
        for name, expected in old[group].items():
            if sha(base/name) != expected:
                raise ValueError('Existing pinned source changed: '+name)
    for name, expected in old['task_sources'].items():
        if sha(io.HOST/name) != expected:
            raise ValueError('Working v1 source changed: '+name)
    candidate = AUDIT/'candidate'/ADAPTER
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError('Uploaded task adapter candidate required')
    files = set(old['task_sources'])
    # Include current task helper packages used by FK/geometry and their imports.
    for folder in ('expansion_v2', 'frozen_initials_v1', 'frozen_initials_v2',
                   'frozen_initials_v3', 'shelf_continuous_v2'):
        files.update(str(p.relative_to(io.HOST)) for p in (io.HOST/folder).glob('*.py'))
    files.update(('shelf_continuous_v2/robot_bank.json', 'shelf_drinks_v1/__init__.py',
                  'shelf_drinks_v1/catalog.py'))
    record = dict(old, task_sources={})
    for name in sorted(files):
        source = candidate if name == ADAPTER else io.HOST/name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Real task source required: '+name)
        record['task_sources'][name] = sha(source)
    (AUDIT/'before').mkdir(parents=True, exist_ok=False)
    for name in (ADAPTER, RECORD):
        shutil.copy2(TARGET/name, AUDIT/'before'/name)
    io.save(AUDIT/'candidate'/RECORD, record)
    diff = ''.join(difflib.unified_diff((TARGET/ADAPTER).read_text().splitlines(True),
        candidate.read_text().splitlines(True), fromfile='before/'+ADAPTER, tofile='after/'+ADAPTER))
    (AUDIT/'adapter.diff').write_text(diff)
    io.save(AUDIT/'stage.json', dict(task_root=str(TARGET), shared_entries_modified=False,
        before={name:sha(TARGET/name) for name in (ADAPTER, RECORD)},
        after={name:sha(AUDIT/'candidate'/name) for name in (ADAPTER, RECORD)},
        shared_sources=old['shared_sources'], task_sources=record['task_sources'],
        scoring_unchanged=True, other_recovery_branches_unchanged=True, staged=time.time()))


def command(module, args, key, *, candidate, uuid=None):
    cmd = composed_command('shelf_drink', module, args, 'shelfv2_'+key,
                           candidate=False, uuid=uuid)
    if candidate:
        mounts = []
        for name in (ADAPTER, RECORD):
            mounts += ['--mount', f'type=bind,src={AUDIT}/candidate/{name},dst={CANON}/{name},readonly']
        cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')] = mounts
    return cmd


def run(mode, key, seed, candidate):
    out = AUDIT/'runs'/key
    out.mkdir(parents=True, exist_ok=False)
    guest = CANON/'shelf_continuous_v2/public_switch_v2/runs'/key
    locks = []
    uuid = None
    if mode == 'native':
        for gpu in range(8):
            path = Path('/workspace/lukun/arena_checkpoints_short14_20261018/queue/gpu_leases')/(socket.gethostname()+'_'+str(gpu)+'.lock')
            lock = path.open('r')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            if not io.idle(gpu):
                lock.close()
                continue
            # Unified parent's default admission is unchanged (<=1 MiB).
            memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if memory > 1:
                lock.close()
                continue
            locks.append(lock)
            uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            break
        if uuid is None:
            raise RuntimeError('No default-admitted isolated GPU; no sharing/stopping')
        # --seed is only the existing prepared-run key for this orchestration mode.
        prepared = CANON/'shelf_continuous_v2/public_switch_v2/runs'/str(seed)/'prepared/launch.json'
        module = 'short_tasks.agent_probe_v1.native_owner'
        args = ['--launch', str(prepared), '--output', str(guest/'native'), '--gpu', str(gpu), '--smoke']
    elif mode == 'prepare':
        module = 'short_tasks.agent_probe_v1.prepare_task'
        args = ['--task', 'failure_recovery_shelf_drink', '--seed', str(seed), '--output', str(guest/'prepared')]
    else:
        module = PACKAGE+'.shelf_continuous_v2.integration_tests'
        args = ['--output', str(guest/'contracts.json')]
    cmd = command(module, args, key, candidate=candidate, uuid=uuid)
    io.save(out/'command.json', dict(command=cmd, candidate=candidate, mode=mode,
        model_calls=0, shared_source_writes=False, started=time.time()))
    started = time.time()
    with (out/'stdout.log').open('x') as stdout, (out/'stderr.log').open('x') as stderr:
        rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr).returncode
    io.save(out/'exit.json', dict(returncode=rc, wall_seconds=time.time()-started))
    for lock in locks:
        lock.close()
    if rc:
        raise RuntimeError('Unified '+mode+' failed; see '+str(out))


def publish():
    stage = json.loads((AUDIT/'stage.json').read_text())
    before = json.loads((AUDIT/'runs/baseline/contracts.json').read_text())
    after = json.loads((AUDIT/'runs/candidate_contracts/contracts.json').read_text())
    if (before['ordinary_plans'] != after['ordinary_plans'] or before['coffee_plan'] != after['coffee_plan']):
        raise ValueError('Unrelated plans changed')
    native = json.loads((AUDIT/'runs/candidate_native/native/native_report.json').read_text())
    if native['status'] != 'completed' or native['cleanup_errors'] or native['initial_score'] != 0:
        raise ValueError('Candidate native acceptance required')
    if json.loads((AUDIT/'runs/candidate_native/exit.json').read_text())['returncode'] != 0:
        raise ValueError('Native process exit must be clean')
    for name, expected in stage['before'].items():
        if sha(TARGET/name) != expected or sha(io.HOST/name) != expected:
            raise ValueError('Concurrent task-entry change; refuse: '+name)
    for name, expected in stage['shared_sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Concurrent shared drift: '+name)
    # Do not touch any shared registry/preparer/native_owner or other tasks.
    for name, expected in stage['task_sources'].items():
        source = AUDIT/'candidate'/name if name == ADAPTER else io.HOST/name
        if sha(source) != expected:
            raise ValueError('Candidate dependency drift: '+name)
        dest = TARGET/name
        if name != ADAPTER and dest.exists() and sha(dest) != expected:
            raise ValueError('Unreviewed installed task-file change: '+name)
    for name in stage['task_sources']:
        if name == ADAPTER:
            continue
        dest = TARGET/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(io.HOST/name, dest)
    for base in (io.HOST, TARGET):
        for name in (ADAPTER, RECORD):
            # Same-directory rename: readers never see a partially written file.
            tmp = base/(name+'.shelf_v2_pending')
            shutil.copy2(AUDIT/'candidate'/name, tmp)
            tmp.replace(base/name)
    io.save(AUDIT/'published.json', dict(published=True, time=time.time(),
        task_root=str(TARGET), task_name='failure_recovery_shelf_drink',
        after={name:sha(TARGET/name) for name in (ADAPTER, RECORD)},
        shared_sources_unchanged=all(sha(ROOT/n)==h for n,h in stage['shared_sources'].items()),
        ordinary_plan_count=len(before['ordinary_plans']), ordinary_and_coffee_plans_unchanged=True))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('stage', 'contracts', 'prepare', 'native', 'publish'))
    p.add_argument('--key')
    p.add_argument('--seed', default='920')
    p.add_argument('--candidate', action='store_true')
    a = p.parse_args()
    if io.HERE != io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu', 'VM-0-10-ubuntu'):
        raise ValueError('Exact owned task on sim1/sim2 required')
    if a.mode == 'stage':
        stage()
    elif a.mode == 'publish':
        publish()
    else:
        if not a.key or not a.key.replace('_', '').isalnum():
            raise ValueError('Simple fresh run key required')
        run(a.mode, a.key, int(a.seed) if a.mode == 'prepare' else a.seed, a.candidate)

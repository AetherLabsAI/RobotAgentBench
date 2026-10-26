"""Autonomous queue of the coffee machine batch (API 1-1 … 1-4).

One long-running process: it derives the next job of every API from the artifacts, launches at most one Isaac job
per idle allowed GPU and never more than MAX_PARALLEL at once (user 2026-09-29: take idle cards on 202, at most two;
with no idle card it simply waits and looks again), records real handles, and runs the CPU follow-ups when a job
finishes. It never retries a failed job by itself, never takes a reserved card (gpu_policy), never shares a card,
and never edits physics or task rules.

Standard jobs (user 2026-09-29): three kinds, NO replay on a random layout.

    fixed-probe                    initial-state acceptance of the fixed initial state (source episode 1)
    source-run-1 … source-run-5    one replay of each of the five source episodes, each on its own recorded layout
    random-probe                   initial-state acceptance of the seed-7 random layout

A replay needs the accepted reset contract of its own layout, so source episodes 2 … 5 get a layout probe
(layout-probe-N) before their replay; it is the same acceptance job as fixed-probe on another source episode.
random-probe exists only where the approved definition names a reproducible sampler (1-3); for the other APIs the
queue records why it is not run instead of inventing a sampler.

A replay that fails is recorded with its reason; nothing is synthesized to make it pass.
State: artifacts/<batch>/queue-state.json (jobs, handles, results), queue.log.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO))
from evaluation.tasks.coffee_machine.tools import gpu_policy, interpreters          # noqa: E402
from evaluation.tasks.coffee_machine.specs import APIS, episodes                     # noqa: E402
from evaluation.tasks.coffee_machine.randomization import SUPPORTED as RANDOM_APIS   # noqa: E402

TOOLS = 'evaluation.tasks.coffee_machine.tools.'
SEED = 7
RUNS = 5
MAX_PARALLEL = 2          # default; --max-parallel raises it (user 2026-10-01: three cards for the left-hand batch)
POLL_S = 30
HANG_S = {'probe': 15 * 60, 'replay': 150 * 60}   # a probe takes ~1 min, a replay with liquid ~30 min; longer = hung (Kit start-up deadlock seen 2026-09-30)
LAUNCH_SPACING_S = 60
LIVE_TASK = os.environ.get('ARENA_LIVE_TASK', '/home/lingjun/robot_agent_arena/api-protocol/evaluation/tasks/coffee_machine')
ENV = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1', ARENA_PYTHON=interpreters.ISAAC, ARENA_LIVE_TASK=LIVE_TASK)
RANDOM_BLOCKED = ('not run: the approved definition table has no reproducible sampler for this API yet '
                  '(original generator to be supplied); no substitute sampler is invented')


def pipeline(api, runs=RUNS, random_probe=True):
    names = ['fixed-probe', 'source-run-1']
    for i in range(2, runs + 1):
        names += [f'layout-probe-{i}', f'source-run-{i}']
    return names + (['random-probe'] if random_probe else [])


def log(root, message):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
    with (root / 'queue.log').open('a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return Path(f'/proc/{pid}/status').exists() and 'zombie' not in Path(f'/proc/{pid}/status').read_text().lower()


def gpu_inventory(allowed):
    """Idle-GPU query. A slow or failing nvidia-smi yields no free GPU this round."""
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                       text=True, timeout=120)
        cards = {int(r.split(',')[0]): [x.strip() for x in r.split(',')[1:]] for r in rows.strip().splitlines()}
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=120)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return {}, {'query_failed': repr(error)}
    busy = {line.split(',')[0].strip() for line in apps.strip().splitlines() if line.strip()}
    # a card with memory in use or load but no listed process (another user's container) is not idle either
    busy |= {uuid for uuid, memory, load in cards.values() if int(memory) > 64 or int(load) > 0}
    # the same test the probe / replay guards apply per card: any listed process at all makes it busy
    for index, (uuid, memory, load) in cards.items():
        if index in allowed and uuid not in busy:
            try:
                report = subprocess.check_output(['nvidia-smi', '--id', str(index)], text=True, timeout=60)
            except (subprocess.SubprocessError, OSError):
                busy.add(uuid); continue
            if 'No running processes found' not in report:
                busy.add(uuid)
    return {i: v[0] for i, v in cards.items() if i in allowed}, busy


class Queue:
    def __init__(self, root, apis, gpus, ports, episode_lists=None, max_parallel=MAX_PARALLEL, random_probe=True, per_api_parallel=False):
        self.root, self.apis, self.gpus, self.ports = root, apis, gpus, list(ports)
        self.max_parallel, self.random_probe, self.per_api_parallel = max_parallel, random_probe, per_api_parallel
        self.state_path = root / 'queue-state.json'
        self.state = (read_json(self.state_path) if self.state_path.exists() else
                      dict(schema='arena.coffee_machine.batch_queue.v1', jobs={}, blocked={}, not_run={}))
        lists = episode_lists or {}
        for api in apis:
            for label in lists.get(api, []):
                if label not in episodes(api):
                    raise ValueError(f'{api}: {label} is not a packaged source episode')
        self.state['episodes'] = {api: list(lists.get(api) or episodes(api)[:RUNS]) for api in apis}
        self.state['pipeline'] = {api: pipeline(api, len(self.state['episodes'][api]), random_probe) for api in apis}
        self.state['max_parallel'] = max_parallel
        self.state['per_api_parallel'] = per_api_parallel
        self.state['gpu_policy'] = gpu_policy.describe()
        self.running = {}
        for job_id, j in self.state['jobs'].items():
            if j.get('status') == 'running':
                self.running[job_id] = None
                log(root, f'reattached {job_id} pid {j.get("pid")} ({"alive" if alive(j.get("pid")) else "gone"})')
        self.save()

    def save(self):
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(self.state_path)

    # ---------------- job helpers -----------------
    def job(self, job_id):
        j = self.state['jobs'].get(job_id)
        return None if j is not None and j.get('status') == 'preflight_failed' and not j.get('relaunched') else j

    def fresh_output(self, api, name):
        previous = self.state['jobs'].get(f'{api}:{name}')
        if previous is None:
            return name
        previous['relaunched'] = True
        previous['status'] = 'preflight_failed'
        self.state['jobs'][f'{api}:{name}:preflight-{int(previous["started_at"])}'] = previous
        return name + '-r' + str(int(time.time()))

    def done(self, job_id):
        j = self.job(job_id)
        return j is not None and j['status'] == 'completed'

    def finished(self, job_id):
        j = self.job(job_id)
        return j is not None and j['status'] in ('completed', 'error')

    def all_finished(self):
        return all(j.get('status') in ('completed', 'error', 'preflight_failed') for j in self.state['jobs'].values())

    def launch(self, job_id, api, kind, command, output, gpu, port=None, episode=None):
        out = self.root / api / output
        log_path = self.root / api / (output + '.queue.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as f:
            proc = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        self.state['jobs'][job_id] = dict(api=api, kind=kind, command=command, output=str(out), gpu=gpu, port=port, episode=episode,
                                          pid=proc.pid, started_at=time.time(), status='running')
        self.running[job_id] = proc
        self.save()
        log(self.root, f'launched {job_id} on GPU {gpu} pid {proc.pid}')

    def cpu(self, job_id, api, kind, command, output):
        """Synchronous CPU step; recorded like a job."""
        out = self.root / api / output
        log_path = self.root / api / (output.rstrip('/').replace('/', '_') + '.queue.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open('a') as f:
            result = subprocess.run(command, cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        self.state['jobs'][job_id] = dict(api=api, kind=kind, command=command, output=str(out), gpu=None, pid=None,
                                          started_at=started, finished_at=time.time(), returncode=result.returncode,
                                          status='completed' if result.returncode == 0 else 'error')
        self.save()
        log(self.root, f'cpu {job_id} -> {self.state["jobs"][job_id]["status"]}')
        return result.returncode == 0

    def poll(self):
        for job_id, proc in list(self.running.items()):
            j = self.state['jobs'][job_id]
            if proc is None:
                if alive(j.get('pid')):
                    continue
                out = Path(j['output'])
                name = 'manifest.json' if j['kind'] == 'probe' else 'job.json'
                ok = (out / name).exists() and read_json(out / name).get('status') == 'completed'
                code = 0 if ok else -1
                j['reattached_exit'] = 'inferred from artifacts'
            else:
                code = proc.poll()
                if code is None:
                    produced = (Path(j['output']) / ('manifest.json' if j['kind'] == 'probe' else 'job.json')).exists()
                    if time.time() - j['started_at'] > HANG_S[j['kind']] and (j['kind'] == 'replay' or not produced):
                        # hung: kill the whole process group, treat as infrastructure, relaunch once on a free card
                        import signal
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        j['status'] = 'preflight_failed'
                        j['preflight_error'] = 'hung for %d min; killed by the queue watchdog' % int((time.time() - j['started_at']) / 60)
                        j['finished_at'] = time.time()
                        del self.running[job_id]
                        self.save()
                        log(self.root, f'{job_id}: hung for {int((time.time() - j["started_at"]) / 60)} min, killed; will relaunch on a free GPU')
                    continue
            j['returncode'] = code
            j['finished_at'] = time.time()
            j['status'] = 'completed' if code == 0 else 'error'
            del self.running[job_id]
            self.save()
            log(self.root, f'finished {job_id} rc={code}')
            try:
                self.after(job_id, j)
            except Exception:
                j['follow_up_error'] = traceback.format_exc()
                j['status'] = 'error'
                self.save()
                log(self.root, f'follow-up failed for {job_id}')

    # ---------------- follow-ups (CPU) -----------------
    def after(self, job_id, j):
        api = j['api']
        out = Path(j['output'])
        if j['kind'] == 'probe':
            pre = Path(str(out) + '.preflight.json')
            if not (out / 'manifest.json').exists() and pre.exists() and read_json(pre).get('status') == 'error' and not out.exists():
                j['status'] = 'preflight_failed'
                j['preflight_error'] = str(read_json(pre).get('error', ''))
                self.save()
                log(self.root, f'{job_id}: preflight failed before any worker launched ({j["preflight_error"][:80]}); will relaunch once on a free GPU')
                return
            manifest = read_json(out / 'manifest.json') if (out / 'manifest.json').exists() else {}
            if 'accepted' not in manifest and 'GPU not idle' in str(manifest.get('error', '')):
                # the guard refused the card before any physics: infrastructure, not a result; relaunch on a free card
                j['status'] = 'preflight_failed'
                j['preflight_error'] = str(manifest['error'])
                self.save()
                log(self.root, f'{job_id}: guard refused a busy card before any physics ran; will relaunch on a free GPU')
                return
            ok = j['status'] == 'completed' and (out / 'audit.json').exists() and read_json(out / 'audit.json').get('passed') is True
            if not ok:
                j['status'] = 'error'
            for cache in ('cache', 'data', 'documents'):      # Kit run-time caches, not evidence
                shutil.rmtree(out / cache, ignore_errors=True)
        elif j['kind'] == 'replay':
            meta = read_json(out / 'job.json') if (out / 'job.json').exists() else {}
            j['replay_status'] = meta.get('status')
            j['report'] = meta.get('report')
            episodes_dir = sorted(p for p in (out / 'episodes').glob('*') if p.is_dir()) if (out / 'episodes').exists() else []
            if (meta.get('status') == 'error' and meta.get('runner_returncode') in (0, 1) and meta.get('source_unchanged') is False
                    and (meta.get('report') or {}).get('evaluation_valid')):
                changed = [rel for rel, h in meta['source_sha256'].items()
                           if (REPO / rel).exists() and hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != h]
                j['public_source_changed_during_run'] = changed
                meta['status'] = 'completed'
                log(self.root, f'{job_id}: files changed during the run ({changed}); physical result kept, audited separately')
            if meta.get('status') == 'completed' and len(episodes_dir) == 1 and (meta.get('report') or {}).get('evaluation_valid'):
                ok = self.cpu(job_id + ':audit', api, 'replay-audit',
                              [interpreters.ISAAC, '-m', TOOLS + 'audit_replay', '--episode', str(episodes_dir[0]), '--run', str(out / 'run')],
                              out.name + '/run/physical-audit')
                if ok:
                    audit = read_json(out / 'run/physical-audit.json')
                    steps = audit['control_steps']
                    marks = [row['step'] for row in audit.get('state_trace', [])][:4]
                    frames = sorted({0, *[min(steps, max(0, s)) for s in marks], steps})
                    if self.cpu(job_id + ':media', api, 'replay-media',
                                [interpreters.ISAAC, '-m', TOOLS + 'build_replay_media', '--job', str(out), '--steps', *map(str, frames)],
                                out.name + '/videos'):
                        self.cpu(job_id + ':prune', api, 'prune-frames',
                                 [interpreters.ISAAC, '-m', TOOLS + 'prune_replay_frames', str(out)], out.name + '/prune')
                else:
                    j['status'] = 'error'
            elif meta.get('status') == 'error' and meta.get('service_pid') is None and not (out / 'run').exists():
                j['status'] = 'preflight_failed'
                j['preflight_error'] = str(meta.get('error', ''))
                log(self.root, f'{job_id}: preflight failed before any service launched ({j["preflight_error"][:80]}); will relaunch once on a free GPU')
            else:
                j['status'] = 'error'
        self.publish(api, out.name)
        self.save()

    def publish(self, api, label):
        self.cpu(f'{api}:{label}:publish', api, 'publish-page',
                 [interpreters.ISAAC, '-m', TOOLS + 'publish_results', '--batch', str(self.root)], label + '/publish')

    # ---------------- per-API pipeline -----------------
    def next_jobs(self, api):
        S = lambda name: self.job(f'{api}:{name}')
        root = self.root / api
        if api in self.state['blocked']:
            return []
        names = self.state['episodes'][api]
        jobs = []
        parallel = self.per_api_parallel
        for index, episode in enumerate(names, start=1):
            probe = 'fixed-probe' if index == 1 else f'layout-probe-{index}'
            run = f'source-run-{index}'
            if S(probe) is None:
                if index > 1 and not self.finished(f'{api}:fixed-probe'):
                    break           # the fixed initial state (episode 1) is the precondition of everything else of this API
                jobs.append((probe, 'probe', lambda g, p, e=episode, n=probe: [
                    interpreters.ISAAC, '-m', TOOLS + 'run_guarded_probe', '--api', api, '--role', 'source', '--episode', e,
                    '--gpu', str(g), '--seed', str(SEED), '--output', str(root / n)], probe, episode))
                if not parallel:
                    return jobs
                continue
            if not self.finished(f'{api}:{probe}'):
                if parallel:
                    continue
                return []
            if not self.done(f'{api}:{probe}'):
                reason = self.probe_failure_reason(root / Path(S(probe)['output']).name, 'initial acceptance failed')
                if index == 1:
                    # the fixed initial state is the precondition of everything else of this API
                    return self.block(api, 'fixed initial state: ' + reason)
                self.skip(api, run, f'{probe}: {reason}')
                continue
            if S(run) is None and f'{api}:{run}' not in self.state['not_run']:
                bound = root / f'source-replay-{index}.json'
                if S(f'bind-{index}') is None:
                    probe_output = Path(S(probe)['output'])
                    ok = self.cpu(f'{api}:convert-{index}', api, 'convert',
                                  [interpreters.API, '-m', TOOLS + 'convert_trajectory', '--api', api, '--episode', episode,
                                   '--output', str(root / f'source-candidate-{index}.json')], f'source-candidate-{index}')
                    ok = ok and self.cpu(f'{api}:bind-{index}', api, 'bind',
                                         [interpreters.API, '-m', TOOLS + 'bind_reset', '--candidate', str(root / f'source-candidate-{index}.json'),
                                          '--probe', str(probe_output), '--output', str(bound)], f'source-replay-{index}')
                    if not ok:
                        self.skip(api, run, 'conversion or binding refused: ' + self.last_line(root / f'source-candidate-{index}.queue.log', root / f'source-replay-{index}.queue.log'))
                        continue
                if not self.done(f'{api}:bind-{index}'):
                    continue
                jobs.append((run, 'replay', lambda g, p, n=run, b=bound: [
                    interpreters.API, '-m', TOOLS + 'run_isolated_replay', '--api', api, '--gpu', str(g), '--port', str(p),
                    '--trajectory', str(b), '--seed', str(SEED), '--output', str(root / n)], run, episode))
                if not parallel:
                    return jobs
                continue
            if S(run) is not None and not self.finished(f'{api}:{run}'):
                if parallel:
                    continue
                return []
        if jobs or not self.random_probe:
            return jobs
        if any(not self.finished(f'{api}:{n}') and f'{api}:{n}' not in self.state['not_run'] for n in self.state['pipeline'][api] if n != 'random-probe'):
            return []
        if api not in RANDOM_APIS:
            self.skip(api, 'random-probe', RANDOM_BLOCKED)
            return []
        if S('random-probe') is None:
            return [('random-probe', 'probe', lambda g, p: [
                interpreters.ISAAC, '-m', TOOLS + 'run_guarded_probe', '--api', api, '--role', 'random',
                '--gpu', str(g), '--seed', str(SEED), '--output', str(root / 'random-probe')], 'random-probe', None)]
        return []

    def skip(self, api, name, reason):
        key = f'{api}:{name}'
        if key not in self.state['not_run']:
            self.state['not_run'][key] = reason
            self.save()
            log(self.root, f'{key} not run: {reason}')
            self.publish(api, name)

    @staticmethod
    def last_line(*logs):
        for log in logs:
            try:
                lines = [l.strip() for l in Path(log).read_text(errors='replace').splitlines() if l.strip()]
            except OSError:
                continue
            if lines and 'Error' in lines[-1]:
                return lines[-1][:400]
        return 'see the queue logs'

    @staticmethod
    def probe_failure_reason(output, default):
        try:
            manifest = read_json(output / 'manifest.json')
        except (OSError, ValueError):
            return default + ' (no probe manifest)'
        if manifest.get('accepted') is False:
            return default + ': ' + str(manifest.get('acceptance_error'))
        if manifest.get('error'):
            return 'probe did not run: ' + str(manifest['error'])[:200]
        if manifest.get('status') != 'completed':
            return 'probe ended at status ' + str(manifest.get('status'))
        return default

    def block(self, api, reason):
        if api not in self.state['blocked']:
            self.state['blocked'][api] = reason
            self.save()
            log(self.root, f'{api} blocked: {reason}')
            self.publish(api, 'blocked')
        return []

    # ---------------- main loop -----------------
    def reload_blocked(self):
        try:
            self.state['blocked'] = read_json(self.state_path).get('blocked', {})
        except Exception:
            return

    def run(self, allowed):
        waiting_logged = 0
        while True:
            self.reload_blocked()
            try:
                self.poll()
            except Exception:
                log(self.root, 'poll error: ' + traceback.format_exc().splitlines()[-1])
            pending = []
            for api in self.apis:
                if not self.per_api_parallel and any(j.get('status') == 'running' and j.get('api') == api for j in self.state['jobs'].values()):
                    continue
                try:
                    pending.extend((api, *job) for job in self.next_jobs(api))
                except Exception:
                    self.block(api, 'planner error: ' + traceback.format_exc().splitlines()[-1])
            in_use = {j['gpu'] for j in self.state['jobs'].values() if j.get('status') == 'running' and j.get('gpu') is not None}
            free = []
            if pending and len(in_use) < self.max_parallel:
                uuids, busy = gpu_inventory(allowed)
                if 'query_failed' in busy:
                    log(self.root, 'gpu query failed this round: ' + busy['query_failed'])
                free_now = [i for i, u in uuids.items() if u not in busy and i not in in_use]
                history = getattr(self, 'free_history', [])[-1:] + [set(free_now)]
                self.free_history = history
                free = [i for i in free_now if len(history) == 2 and all(i in h for h in history)]
                if not free_now and time.time() - waiting_logged > 1800:
                    log(self.root, f'no idle card among {sorted(allowed)}; waiting ({len(pending)} job(s) pending)')
                    waiting_logged = time.time()
                if time.time() - getattr(self, 'last_launch', 0) < LAUNCH_SPACING_S:
                    free = []
            else:
                self.free_history = []
            for api, name, kind, builder, output, episode in pending:
                if not free:
                    break
                if f'{api}:{name}' in self.state['jobs'] and self.job(f'{api}:{name}') is not None:
                    continue
                gpu = free[0]
                port = self.ports[gpu % len(self.ports)]
                fresh = self.fresh_output(api, name)
                command = [str(self.root / api / fresh) if x == str(self.root / api / name) else x for x in builder(gpu, port)]
                self.launch(f'{api}:{name}', api, kind, command, fresh, gpu, port, episode)
                self.last_launch = time.time()
                break
            if not self.running and not pending and self.all_finished():
                log(self.root, 'queue idle: nothing pending')
                return
            time.sleep(POLL_S)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--apis', nargs='+', default=list(APIS))
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--ports', type=int, nargs='+', default=[8180, 8181, 8182, 8183, 8184, 8185])
    p.add_argument('--episodes-file', type=Path, help='JSON {api: [packaged episode labels]} replacing the five displayed episodes')
    p.add_argument('--max-parallel', type=int, default=MAX_PARALLEL, help='GPU jobs at once (never more cards than --gpus)')
    p.add_argument('--no-random-probe', action='store_true', help='source replays only; no seed-7 random layout probe')
    p.add_argument('--per-api-parallel', action='store_true', help='independent episodes of one API may run on several cards at once')
    a = p.parse_args(argv)
    episode_lists = read_json(a.episodes_file) if a.episodes_file else None
    root = a.batch.resolve()
    gpu_policy.check(*a.gpus)
    unknown = [x for x in a.apis if x not in APIS]
    if unknown:
        raise ValueError('Not a coffee machine API: ' + ', '.join(unknown))
    if not root.is_relative_to((TASK / 'artifacts').resolve()):
        raise ValueError('Batch root must be task-owned')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'queue.pid').write_text(str(os.getpid()) + '\n')
    fingerprint = TASK / 'SNAPSHOT-HASHES.json'
    if fingerprint.is_file():
        frozen = read_json(fingerprint)
        if Path(frozen['snapshot']) != REPO:
            raise ValueError('This tree carries the fingerprint of another snapshot')
        (root / 'snapshot.json').write_text(json.dumps(dict(
            snapshot=frozen['snapshot'], created=frozen['created'], head=frozen['head'], file_count=frozen['file_count'], bytes=frozen['bytes'],
            manifest=str(fingerprint), manifest_sha256=hashlib.sha256(fingerprint.read_bytes()).hexdigest()), indent=2) + '\n')
    else:
        log(root, 'WARNING: running from a tree without a snapshot fingerprint (live working tree)')
    log(root, f'queue start pid {os.getpid()} apis {a.apis} gpus {a.gpus} max parallel {a.max_parallel} per-api parallel {a.per_api_parallel} random probe {not a.no_random_probe}')
    Queue(root, a.apis, a.gpus, a.ports, episode_lists, a.max_parallel, not a.no_random_probe, a.per_api_parallel).run(set(a.gpus))


if __name__ == '__main__':
    main()

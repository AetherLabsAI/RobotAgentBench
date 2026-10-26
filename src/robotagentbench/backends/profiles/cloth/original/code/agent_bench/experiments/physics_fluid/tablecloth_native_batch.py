"""Prepare 50+50 native cloth requests and resume explicitly diagnostic runs.

The old Newton replay batch is not compatible with native drive dynamics.
This entrypoint keeps target counts separate from attempts and measured capture
validity. Production remains blocked until physical, randomization and registered
task gates are implemented; a finished process or valid NPZ cannot release data.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def solver_log_errors(path):
    """PhysX can report lost contacts while the process still exits zero."""
    errors = []
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if re.search(r'(deformable surface contact buffer overflow|'
                         r'\[Error\].*\[omni\.physx\.plugin\])', line, re.I):
                # Keep the diagnostic identity, without repeating per-step logs.
                message = re.sub(r'^.*\[omni\.physx\.plugin\]\s*', '', line).strip()
                if message not in errors:
                    errors.append(message)
    return errors


def validate_native_capture(directory):
    """Check measured native state/command alignment, never action==state."""
    directory = Path(directory)
    reasons = []
    metrics = {}
    try:
        trial = json.loads((directory/'trial.json').read_text())
        if trial.get('failure'):
            reasons.append('simulation_incomplete_or_failed')
        if (trial.get('post_start_pose_or_velocity_writes') is not False
                or trial.get('attachment_count') != 0
                or trial.get('robot_parameters_unchanged') is not True):
            reasons.append('native_physics_contract_failed')
        with np.load(directory/'action.npz', allow_pickle=False) as action, \
                np.load(directory/'full_state.npz', allow_pickle=False) as state:
            target, actual = action['joint_target'], state['joint_position']
            at, st = action['time_s'], state['time_s']
            names, measured_names = action['joint_names'].tolist(), state['joint_names'].tolist()
            fps = trial['fps']
            if not isinstance(fps, (float, int)) or not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid action frequency')
            count = trial['frames']
            if (not isinstance(count, int) or count < 1 or target.shape != (count,len(names))
                    or actual.shape != (count,len(measured_names))
                    or at.shape != (count,) or st.shape != (count,)
                    or len(set(names)) != len(names) or len(set(measured_names)) != len(measured_names)
                    or not set(names).issubset(measured_names)):
                raise ValueError('Joint evidence dimensions or names do not match')
            if not np.allclose(at, np.arange(count)/fps, rtol=0, atol=1e-8):
                reasons.append('invalid_command_clock')
            if not np.allclose(st, at+1/fps, rtol=0, atol=1e-8):
                reasons.append('invalid_measured_state_clock')
            q, qd = state['cloth_q'], state['cloth_qd']
            if (q.ndim != 3 or q.shape[0] != count or q.shape[2] != 3
                    or not q.shape[1] or qd.shape != q.shape):
                raise ValueError('Invalid cloth capture dimensions')
            for value in (target,actual,at,st,q,qd,state['joint_velocity']):
                if not np.isfinite(value).all():
                    reasons.append('nonfinite_physical_capture')
                    break
            if not reasons:
                indices = [measured_names.index(n) for n in names]
                metrics['tracking_error_max_native_units'] = float(abs(actual[:,indices]-target).max())
            metrics.update(frames=count, duration_s=count/fps)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        reasons.append(f'evidence_unreadable:{type(exc).__name__}:{exc}')
    return dict(capture_valid=not reasons, reasons=sorted(set(reasons)), metrics=metrics,
                physical_success_verified=False, eligible_for_production=False)


def prepare(output, seed, count):
    from experiments.physics_fluid.tablecloth_native_randomization import build_batch
    plan = build_batch(seed, count)
    output.mkdir(parents=True, exist_ok=False)
    for ordinal, row in enumerate(plan['rows']):
        row['request_id'] = f"{row['workflow']}_{ordinal % count:03d}_s{row['seed']}"
    plan.update(target={'remove':count,'spread':count}, qualified={'remove':0,'spread':0})
    write_json(output/'plan.json', plan)
    write_json(output/'status.json', dict(phase='physics_and_owner_adapters_pending',
        target=plan['target'], qualified=plan['qualified'], planned=len(plan['rows']),
        production_ready=False, blockers=plan['missing_validation']))
    return plan


def run_diagnostics(root, command_file, limit, retry_failed=False):
    """Commands are argv arrays (foreground processes), never shell strings.

    Each receives {request}, {output}, {seed}, {layout_id}, {workflow}. The
    delivered producer must consume the request; this controller never rewrites
    the owner's injection or motion. Existing outputs are never overwritten.
    """
    plan = json.loads((root/'plan.json').read_text())
    commands = json.loads(command_file.read_text())
    for workflow in ('remove','spread'):
        command = commands.get(workflow)
        if (not isinstance(command,list) or not command
                or not all(isinstance(s,str) for s in command)
                or '--diagnostic-canonical' not in command
                or not any('{output}' in s for s in command)
                or '-d' in command or '--detach' in command):
            raise ValueError('Expected foreground diagnostic argv with an explicit output placeholder')
    with (root/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for row in plan['rows']:
            if len(completed) >= limit:
                break
            identifier = row['request_id']
            if Path(identifier).name != identifier or identifier in ('.','..'):
                raise ValueError('Unsafe request identifier')
            row_root = root/'attempts'/identifier
            attempts = sorted(row_root.glob('attempt_*')) if row_root.exists() else []
            if attempts:
                previous = attempts[-1]/'result.json'
                if previous.is_file():
                    old = json.loads(previous.read_text())
                    if old.get('capture',{}).get('capture_valid') or not retry_failed:
                        continue
                elif not retry_failed:
                    continue  # interrupted attempt is preserved until explicit retry
            attempt = row_root/f'attempt_{len(attempts):03d}'
            attempt.mkdir(parents=True, exist_ok=False)
            write_json(attempt/'request.json',row)
            substitutions = dict(row,request=str(attempt/'request.json'),output=str(attempt/'episode'))
            command = [arg.format_map(substitutions) for arg in commands[row['workflow']]]
            write_json(attempt/'command.json',command)
            with (attempt/'process.log').open('w') as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                    returncode = result.returncode
                except OSError as exc:
                    log.write(f'Process could not start: {exc}\n')
                    returncode = 127
            capture = validate_native_capture(attempt/'episode')
            errors = solver_log_errors(attempt/'process.log')
            if errors:
                capture['capture_valid'] = False
                capture['reasons'].append('physx_runtime_error')
            if returncode:
                capture['capture_valid'] = False
                capture['reasons'].append(f'process_exit:{returncode}')
            record = dict(request_id=identifier,returncode=returncode,capture=capture,
                solver_errors=errors,
                diagnostic_only=True,qualified=False)
            write_json(attempt/'result.json',record)
            completed.append(record)
            write_json(root/'status.json',dict(phase='diagnostic_runs_only',target=plan['target'],
                qualified={'remove':0,'spread':0},this_invocation=completed,production_ready=False,
                blockers=plan['missing_validation']))
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=50,help='Target per workflow')
    p = commands.add_parser('check-state')
    p.add_argument('episode',type=Path)
    p = commands.add_parser('run-diagnostic')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commands',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--retry-failed',action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output.resolve(),args.seed,args.count)
        print(f"Prepared {len(result['rows'])} requests; qualified data: 0")
    elif args.command == 'check-state':
        result = validate_native_capture(args.episode)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['capture_valid'] else 1)
    else:
        if args.limit < 1:
            parser.error('limit must be positive')
        results = run_diagnostics(args.root.resolve(),args.commands,args.limit,args.retry_failed)
        print(json.dumps(results,indent=2))
        raise SystemExit(1 if any(not r['capture']['capture_valid'] for r in results) else 0)


if __name__ == '__main__':
    main()
